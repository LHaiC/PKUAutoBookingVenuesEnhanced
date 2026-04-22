import argparse
import base64
import binascii
import io
import json
import math
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from itertools import combinations

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import uvicorn
from PIL import Image, UnidentifiedImageError

from captcha_matcher import Candidate, MatchError, match_targets, normalize_candidates
from captcha_vision import (
    ProposalSet,
    build_colored_text_strip,
    detect_colored_text_bboxes,
    generate_box_proposals,
    measure_box_size_consistency,
    refine_bbox_to_dark_pixels,
    validate_bbox,
    whiten_watermark,
)

app = FastAPI()
engine = None
OCR_CHAR_NORMALIZATION = {
    "イ": "八",
}


class ParseRequest(BaseModel):
    images: list[str] = Field(default_factory=list)
    targets: list[str] | None = None


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, _exc):
    return JSONResponse(status_code=400, content={"detail": "Invalid request body"})


def decode_data_uri(data_uri: str) -> bytes:
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return base64.b64decode(payload, validate=True)


@contextmanager
def temporary_image_file(image_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        path = tmp.name
    try:
        yield path
    finally:
        try:
            import os

            os.unlink(path)
        except FileNotFoundError:
            pass


class GlmOcrEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None
        self.model = None

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    @property
    def loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def recognize(self, image_bytes: bytes, targets: list[str] | None = None) -> str:
        if not self.loaded:
            raise RuntimeError("model not loaded")

        prompt = "Text Recognition:"
        with temporary_image_file(image_bytes) as image_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            output_ids = generated_ids[0][inputs["input_ids"].shape[1] :]
            return self.processor.decode(output_ids, skip_special_tokens=False)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _recognize_with_cache(recognizer, cache: dict[bytes, str], image_bytes: bytes) -> str:
    cached = cache.get(image_bytes)
    if cached is not None:
        return cached
    output = recognizer(image_bytes)
    cache[image_bytes] = output
    return output


def crop_box_image(image: Image.Image, box: list[int], padding: int = 5) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop(
        (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding),
        )
    )


def parse_model_output(output: str) -> list[str]:
    text = output.replace("识别结果：", "").replace("<|user|>", "")
    return re.findall(r"[\u4e00-\u9fff]", text)


def _strict_int(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("expected integer")
    return value


def _strict_float(value) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("expected number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected finite number")
    return number


def _parse_candidate_items(items: list) -> list[dict]:
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            candidates.append(
                {
                    "text": text,
                    "bbox": [_strict_int(v) for v in item.get("bbox", [])],
                    "confidence": _strict_float(item.get("confidence", 0.80)),
                }
            )
        except (TypeError, ValueError):
            continue
    return candidates


def _find_candidate_json_array(output: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", output):
        try:
            parsed, _end = decoder.raw_decode(output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            candidates = _parse_candidate_items(parsed)
            if candidates:
                return candidates
    return None


def normalize_ocr_text(text: str) -> str:
    for source, target in OCR_CHAR_NORMALIZATION.items():
        text = text.replace(source, target)
    return text


def parse_legacy_model_output(output: str) -> list[dict]:
    parsed = _find_candidate_json_array(output)
    if parsed is not None:
        return parsed

    result_line = re.search(r"识别结果\s*[:：]\s*([^\r\n]+)", output)
    if result_line:
        text_segment = normalize_ocr_text(result_line.group(1))
        chars = re.findall(r"[\u4e00-\u9fff]", text_segment)
    else:
        chars = []
        for line in output.replace("<|user|>", "\n").splitlines():
            text_segment = line.strip()
            if not text_segment:
                continue
            text_segment = text_segment.replace("场馆预约", "")
            text_segment = normalize_ocr_text(text_segment)
            if any(keyword in text_segment for keyword in ("请", "验证", "点击", "场馆", "预约", "完成", "安全")):
                continue
            if re.search(r"[:：]", text_segment):
                text_segment = re.split(r"[:：]", text_segment)[-1]
            chars.extend(re.findall(r"[\u4e00-\u9fff]", text_segment))
    return [{"text": char, "bbox": [], "confidence": 0.50} for char in chars]


def attach_colored_text_bboxes(
    image: Image.Image,
    candidates: list[dict],
    boxes: list[list[int]] | None = None,
    trim_to_boxes: bool = False,
    allow_index_attachment: bool = True,
) -> list[dict]:
    if not candidates:
        return candidates
    if all(len(item.get("bbox", [])) == 4 for item in candidates):
        return candidates
    if not allow_index_attachment:
        return candidates

    boxes = detect_colored_text_bboxes(image) if boxes is None else boxes
    if not boxes:
        return candidates

    sorted_box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    updated = []
    for ci, item in enumerate(candidates):
        item = dict(item)
        if len(item.get("bbox", [])) != 4 and ci < len(sorted_box_indices):
            item["bbox"] = boxes[sorted_box_indices[ci]]
        updated.append(item)

    if not trim_to_boxes:
        updated.extend(candidates[len(updated) :])
    return updated


def prepare_recognition_image_bytes(
    image: Image.Image,
    use_strip: bool,
    boxes: list[list[int]] | None = None,
) -> bytes:
    recognition_image = build_colored_text_strip(image, boxes=boxes) if use_strip else None
    if recognition_image is None:
        recognition_image = image
    return image_to_png_bytes(recognition_image)


def recognize_box_crops_legacy(
    image: Image.Image,
    boxes: list[list[int]],
    padding: int = 10,
    recognizer=None,
) -> list[Candidate]:
    recognizer = engine.recognize if recognizer is None else recognizer
    candidates = []
    for box in boxes:
        crop = crop_box_image(image, box, padding=padding)
        output = recognizer(image_to_png_bytes(crop))
        parsed = parse_legacy_model_output(output)
        for item in parsed:
            text = str(item.get("text", "")).strip()
            if len(text) == 1:
                candidates.append(
                    Candidate(
                        text=text,
                        bbox=[int(v) for v in box],
                        confidence=0.60,
                        original_bbox=[int(v) for v in box],
                    )
                )
                break
    return candidates


def candidates_from_legacy_output(
    image: Image.Image,
    output: str,
    boxes: list[list[int]],
    allow_index_attachment: bool = True,
) -> list[Candidate]:
    parsed = parse_legacy_model_output(output)
    candidates = attach_colored_text_bboxes(
        image,
        parsed,
        boxes=boxes,
        trim_to_boxes=True,
        allow_index_attachment=allow_index_attachment,
    )
    refined_items = []
    for item in candidates:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            item = dict(item)
            original_bbox = [int(v) for v in bbox]
            item["bbox"] = original_bbox
            item["original_bbox"] = original_bbox
        refined_items.append(item)
    return normalize_candidates(refined_items)


def expand_confusable_target_candidates(candidates: list[Candidate], targets: list[str]) -> list[Candidate]:
    expanded = list(candidates)
    if "三" in targets and "川" not in targets:
        for candidate in candidates:
            if candidate.text == "川":
                expanded.append(Candidate(text="三", bbox=list(candidate.bbox), confidence=0.50))
    if "心" in targets and "必" not in targets:
        for candidate in candidates:
            if candidate.text == "必":
                expanded.append(Candidate(text="心", bbox=list(candidate.bbox), confidence=0.50))
    return expanded


def _candidate_from_box(text: str, image: Image.Image, box: list[int], confidence: float = 0.50) -> Candidate:
    original_bbox = [int(value) for value in box]
    return Candidate(
        text=text,
        bbox=original_bbox,
        confidence=confidence,
        original_bbox=original_bbox,
    )


def infer_single_missing_target_candidates(
    image: Image.Image,
    targets: list[str],
    output: str,
    boxes: list[list[int]],
) -> list[Candidate]:
    if not targets or len(set(targets)) != len(targets):
        return []

    parsed = parse_legacy_model_output(output)
    chars = [
        str(item.get("text", "")).strip()
        for item in parsed
        if len(str(item.get("text", "")).strip()) == 1
    ]
    if len(boxes) != len(chars) + 1:
        return []

    target_counts = Counter(targets)
    parsed_counts = Counter(chars)
    missing_targets = [
        target
        for target in targets
        if parsed_counts[target] < target_counts[target]
    ]
    if len(missing_targets) != 1:
        return []
    missing_target = missing_targets[0]

    candidates: list[Candidate] = []
    left_box = 0
    right_box = len(boxes) - 1
    left_char = 0
    right_char = len(chars) - 1

    while (
        left_char <= right_char
        and left_box <= right_box
        and chars[left_char] in target_counts
        and chars[left_char] != missing_target
    ):
        candidates.append(_candidate_from_box(chars[left_char], image, boxes[left_box]))
        left_box += 1
        left_char += 1

    suffix: list[Candidate] = []
    while (
        right_char >= left_char
        and right_box >= left_box
        and chars[right_char] in target_counts
        and chars[right_char] != missing_target
    ):
        suffix.append(_candidate_from_box(chars[right_char], image, boxes[right_box]))
        right_box -= 1
        right_char -= 1

    middle_chars = chars[left_char : right_char + 1]
    if any(char in target_counts for char in middle_chars):
        return []
    if right_box - left_box + 1 != len(middle_chars) + 1:
        return []

    candidates.append(_candidate_from_box(missing_target, image, boxes[left_box]))
    candidates.extend(reversed(suffix))
    return candidates


def recognize_rotated_box_candidates_legacy(
    image: Image.Image,
    targets: list[str],
    boxes: list[list[int]],
    angles: tuple[int, ...] = (0, 90, 180, 270),
    recognizer=None,
) -> list[Candidate]:
    if not targets:
        return []

    recognizer = engine.recognize if recognizer is None else recognizer
    target_set = set(targets)
    candidates = []
    for box in boxes:
        crop = crop_box_image(image, box, padding=15)
        hits = set()
        raw_outputs = []
        for angle in angles:
            rotated = crop.rotate(angle, expand=True, fillcolor="white")
            output = recognizer(image_to_png_bytes(rotated))
            raw_outputs.append(output)
            for item in parse_legacy_model_output(output):
                text = str(item.get("text", "")).strip()
                if text in target_set:
                    hits.add(text)
        if len(hits) == 1:
            candidates.append(_candidate_from_box(next(iter(hits)), image, box))
            continue
        if "三" in target_set and "川" not in target_set and "111" in "\n".join(raw_outputs):
            candidates.append(_candidate_from_box("三", image, box))
    return candidates


def match_targets_from_views(
    targets: list[str],
    candidate_views: list[list[Candidate]],
    size: tuple[int, int],
) -> list[dict]:
    matched = []
    used_bboxes = set()

    for target in targets:
        selected = None
        for candidates in candidate_views:
            target_matches = [
                candidate
                for candidate in candidates
                if candidate.text == target and tuple(candidate.bbox) not in used_bboxes
            ]
            if len(target_matches) > 1:
                raise MatchError(f"ambiguous target: {target}")
            if target_matches:
                selected = target_matches[0]
                break
        if selected is None:
            raise MatchError(f"missing target: {target}")

        result = match_targets([target], [selected], size)[0]
        used_bboxes.add(tuple(selected.bbox))
        matched.append(result)

    return matched


def build_aligned_strip_candidates(
    image: Image.Image,
    output: str,
    boxes: list[list[int]],
    direct_candidates: list[Candidate],
    targets: list[str],
) -> list[Candidate]:
    chars = [
        str(item.get("text", "")).strip()
        for item in parse_legacy_model_output(output)
        if len(str(item.get("text", "")).strip()) == 1
    ]
    if not chars or len(chars) > len(boxes):
        return []

    box_to_index = {tuple(box): idx for idx, box in enumerate(boxes)}
    direct_labels: list[str | None] = [None] * len(boxes)
    for candidate in direct_candidates:
        idx = box_to_index.get(tuple(candidate.bbox))
        if idx is None or direct_labels[idx] is not None:
            continue
        direct_labels[idx] = candidate.text

    best_indices = None
    best_score = None
    for combo in combinations(range(len(boxes)), len(chars)):
        score = 0
        for char, idx in zip(chars, combo):
            label = direct_labels[idx]
            if label == char:
                score += 3
            elif label is not None:
                score += 1
        if best_score is None or score > best_score:
            best_score = score
            best_indices = combo

    if best_indices is None:
        return []

    aligned = [
            Candidate(
                text=char,
                bbox=[int(value) for value in boxes[idx]],
                confidence=0.60 if direct_labels[idx] == char else 0.50,
                original_bbox=[int(value) for value in boxes[idx]],
            )
        for char, idx in zip(chars, best_indices)
    ]

    missing_targets = [target for target in targets if target not in {candidate.text for candidate in aligned}]
    unused_indices = [idx for idx in range(len(boxes)) if idx not in best_indices]
    if len(missing_targets) == 1 and len(unused_indices) == 1:
        idx = unused_indices[0]
        aligned.insert(
            idx,
            Candidate(
                text=missing_targets[0],
                bbox=refine_bbox_to_dark_pixels(image, [int(value) for value in boxes[idx]]),
                confidence=0.50,
                original_bbox=[int(value) for value in boxes[idx]],
            ),
        )

    return aligned


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: dict[tuple[str, tuple[int, int, int, int]], Candidate] = {}
    for candidate in candidates:
        key = (candidate.text, tuple(candidate.bbox))
        existing = deduped.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            deduped[key] = candidate

    unique = list(deduped.values())
    if len(unique) < 2:
        return unique

    widths = sorted(_bbox_width(candidate.bbox) for candidate in unique)
    heights = sorted(_bbox_height(candidate.bbox) for candidate in unique)
    areas = sorted(_bbox_area(candidate.bbox) for candidate in unique)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    median_area = areas[len(areas) // 2]

    def candidate_quality(candidate: Candidate) -> float:
        width = _bbox_width(candidate.bbox)
        height = _bbox_height(candidate.bbox)
        area = _bbox_area(candidate.bbox)
        size_penalty = (
            abs(width - median_width) / max(1, median_width)
            + abs(height - median_height) / max(1, median_height)
            + abs(area - median_area) / max(1, median_area)
        )
        return candidate.confidence - 0.2 * size_penalty

    selected: list[Candidate] = []
    for candidate in unique:
        replaced = False
        for idx, existing in enumerate(selected):
            if candidate.text != existing.text:
                continue
            if _bbox_iou(candidate.bbox, existing.bbox) < 0.3 and _bbox_overlap_ratio_smaller(
                candidate.bbox, existing.bbox
            ) < 0.8:
                continue
            if candidate_quality(candidate) > candidate_quality(existing):
                selected[idx] = candidate
            replaced = True
            break
        if not replaced:
            selected.append(candidate)

    return selected


def _legacy_result_score(result: dict | None) -> float:
    if not result or result.get("error") != "ok":
        return float("-inf")
    results = result.get("results", [])
    if not results_have_consistent_box_sizes(results):
        return float("-inf")
    confidence_sum = sum(float(item.get("confidence", 0.0)) for item in results)
    size_score = measure_box_size_consistency([item.get("bbox", []) for item in results]).get("score", 0.0)
    return confidence_sum + float(size_score)


def choose_better_legacy_result(existing: dict | None, candidate: dict | None) -> dict | None:
    if existing is None:
        return candidate
    if candidate is None:
        return existing
    return candidate if _legacy_result_score(candidate) > _legacy_result_score(existing) else existing


def recognize_box_crop(image: Image.Image, box: list[int], target_set: set[str] | None, recognizer=None):
    recognizer = engine.recognize if recognizer is None else recognizer
    crop = crop_box_image(image, box)
    chars = parse_model_output(recognizer(image_to_png_bytes(crop)))
    if len(chars) != 1:
        return None
    text = chars[0]
    if target_set is not None and text not in target_set:
        return None
    return Candidate(text=text, bbox=box, confidence=0.80, original_bbox=list(box))


def recognize_box_crops(image: Image.Image, boxes: list[list[int]], targets: list[str] | None, recognizer=None):
    target_set = set(targets) if targets else None
    results = []
    for box in boxes:
        candidate = recognize_box_crop(image, box, target_set, recognizer=recognizer)
        if candidate is None:
            return []
        results.append(candidate)
    return results


def recognize_box_candidates_with_recovery(
    image: Image.Image,
    boxes: list[list[int]],
    targets: list[str] | None,
    recognizer=None,
) -> tuple[list[Candidate], list[list[int]]]:
    target_set = set(targets) if targets else None
    candidates = []
    failed_boxes = []
    for box in boxes:
        candidate = recognize_box_crop(image, box, target_set, recognizer=recognizer)
        if candidate is None:
            failed_boxes.append(box)
            continue
        candidates.append(candidate)
    return candidates, failed_boxes


def recognize_rotated_box_candidates(
    image: Image.Image,
    targets: list[str],
    boxes: list[list[int]],
    angles: tuple[int, ...] = (0, 90, 180, 270),
    recognizer=None,
) -> list[Candidate]:
    recognizer = engine.recognize if recognizer is None else recognizer
    target_set = set(targets)
    recovered = []
    for box in boxes:
        crop = crop_box_image(image, box)
        for angle in angles:
            rotated = crop.rotate(angle, expand=True, fillcolor="white")
            chars = parse_model_output(recognizer(image_to_png_bytes(rotated)))
            if len(chars) == 1 and chars[0] in target_set:
                recovered.append(Candidate(text=chars[0], bbox=box, confidence=0.50, original_bbox=list(box)))
                break
    return recovered


def score_proposal_set(
    image: Image.Image,
    proposal: ProposalSet,
    targets: list[str],
    candidates: list[Candidate],
) -> dict:
    size_stats = measure_box_size_consistency([candidate.bbox for candidate in candidates])
    try:
        matched = match_targets(targets, candidates, image.size, min_confidence=0.45)
        coverage = len(matched)
    except MatchError:
        matched = []
        coverage = 0
    score = (
        coverage * 10.0
        + sum(candidate.confidence for candidate in candidates)
        + size_stats["score"] * 2.0
        - max(0, len(proposal.boxes) - 3)
    )
    return {"score": score, "matched": matched, "size_stats": size_stats, "proposal": proposal}


def results_have_consistent_box_sizes(results: list[dict]) -> bool:
    if len(results) < 3:
        return True

    bboxes = [item.get("bbox", []) for item in results if len(item.get("bbox", [])) == 4]
    if len(bboxes) != len(results):
        return False

    widths = sorted(_bbox_width(bbox) for bbox in bboxes)
    heights = sorted(_bbox_height(bbox) for bbox in bboxes)
    areas = sorted(_bbox_area(bbox) for bbox in bboxes)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    median_area = areas[len(areas) // 2]

    for bbox in bboxes:
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        area = _bbox_area(bbox)
        if (
            width < median_width * 0.75
            or width > median_width * 1.35
            or height < median_height * 0.75
            or height > median_height * 1.35
            or area < median_area * 0.60
            or area > median_area * 1.60
        ):
            return False
    return True


def accept_solution(
    top_score: dict,
    runner_up_score: dict | None,
    min_margin: float = 0.25,
    expected_match_count: int = 3,
) -> tuple[bool, str]:
    matched = top_score.get("matched", [])
    if len(matched) != expected_match_count:
        return False, "incomplete_target_coverage"
    if runner_up_score is not None and top_score["score"] - runner_up_score["score"] < min_margin:
        return False, "ambiguous_top_score"
    if len({tuple(item["bbox"]) for item in matched}) != len(matched):
        return False, "duplicate_box_reuse"
    if not results_have_consistent_box_sizes(matched):
        return False, "inconsistent_box_size"
    return True, "accepted"


def build_solver_response(image: Image.Image, results: list[dict], error: str) -> dict:
    if error == "ok":
        results = normalize_result_bboxes(image, results)
    if error == "ok" and not results_have_safe_click_geometry(results):
        results = []
        error = "unsafe_click_geometry"
    return {
        "results": results,
        "error": error,
        "method": "glm_ocr_transformers_with_local_positioning",
        "image_size": list(image.size),
    }


def _bbox_width(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0])


def _bbox_height(bbox: list[int]) -> int:
    return max(0, bbox[3] - bbox[1])


def _bbox_area(bbox: list[int]) -> int:
    return _bbox_width(bbox) * _bbox_height(bbox)


def _bbox_center(bbox: list[int]) -> tuple[int, int]:
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def normalize_result_bboxes(image: Image.Image, results: list[dict]) -> list[dict]:
    if len(results) < 3:
        return results

    bboxes = [item.get("bbox", []) for item in results if len(item.get("bbox", [])) == 4]
    if len(bboxes) != len(results):
        return results

    widths = sorted(_bbox_width(bbox) for bbox in bboxes)
    heights = sorted(_bbox_height(bbox) for bbox in bboxes)
    areas = sorted(_bbox_area(bbox) for bbox in bboxes)
    median_width = widths[len(widths) // 2]
    median_height = heights[len(heights) // 2]
    median_area = areas[len(areas) // 2]

    restored = []
    for item in results:
        bbox = item.get("bbox", [])
        original_bbox = item.get("original_bbox", [])
        if len(bbox) != 4 or len(original_bbox) != 4:
            restored.append(item)
            continue

        refined_area = _bbox_area(bbox)
        original_area = _bbox_area(original_bbox)
        if original_area <= 0:
            restored.append(item)
            continue

        current_area = _bbox_area(bbox)
        refined_bbox = refine_bbox_to_dark_pixels(image, list(original_bbox))
        refined_area = _bbox_area(refined_bbox)
        too_small = current_area < original_area * 0.95 and (
            current_area < median_area * 0.75
            or current_area < original_area * 0.65
            or _bbox_width(bbox) < median_width * 0.8
            or _bbox_height(bbox) < median_height * 0.8
        )
        updated = dict(item)
        if too_small:
            updated["bbox"] = list(original_bbox)
            updated["x"], updated["y"] = _bbox_center(updated["bbox"])
            restored.append(updated)
            continue

        too_large = (
            _bbox_width(bbox) > median_width * 1.35
            or _bbox_height(bbox) > median_height * 1.35
            or _bbox_area(bbox) > median_area * 1.60
        )
        refined_is_plausible = (
            validate_bbox(refined_bbox, image.size)
            and refined_area >= median_area * 0.60
            and _bbox_width(refined_bbox) >= median_width * 0.75
            and _bbox_height(refined_bbox) >= median_height * 0.75
            and refined_area < _bbox_area(bbox)
        )
        if too_large and refined_is_plausible:
            updated["bbox"] = list(refined_bbox)
            updated["x"], updated["y"] = _bbox_center(updated["bbox"])
            restored.append(updated)
            continue

        updated["bbox"] = list(bbox)
        updated["x"], updated["y"] = _bbox_center(updated["bbox"])
        restored.append(updated)

    return restored


def _bbox_iou(b1: list[int], b2: list[int]) -> float:
    left = max(b1[0], b2[0])
    top = max(b1[1], b2[1])
    right = min(b1[2], b2[2])
    bottom = min(b1[3], b2[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - intersection
    return intersection / union


def _bbox_overlap_ratio_smaller(b1: list[int], b2: list[int]) -> float:
    left = max(b1[0], b2[0])
    top = max(b1[1], b2[1])
    right = min(b1[2], b2[2])
    bottom = min(b1[3], b2[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(max(1, _bbox_area(b1)), max(1, _bbox_area(b2)))
    return intersection / smaller


def results_have_safe_click_geometry(results: list[dict]) -> bool:
    centers = set()
    bboxes: list[list[int]] = []
    for item in results:
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            return False
        center = (item.get("x"), item.get("y"))
        if center in centers:
            return False
        centers.add(center)
        bboxes.append(bbox)

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bbox_iou(bboxes[i], bboxes[j]) > 0.7:
                return False
    return True


def _candidate_box_sets_for_legacy_fallback(image: Image.Image) -> list[list[list[int]]]:
    box_sets = []
    seen = set()

    for boxes in [detect_colored_text_bboxes(image), *[proposal.boxes for proposal in generate_box_proposals(image)]]:
        if not boxes:
            continue
        key = tuple(tuple(box) for box in boxes)
        if key in seen:
            continue
        seen.add(key)
        box_sets.append([list(box) for box in boxes])

    return box_sets


def solve_image_with_legacy_fallback(
    image: Image.Image,
    targets: list[str],
    initial_error: str,
) -> dict:
    working_image = whiten_watermark(image)
    box_sets = _candidate_box_sets_for_legacy_fallback(working_image)
    if not box_sets:
        return build_solver_response(image, [], initial_error)

    size = working_image.size
    original_output: str | None = None
    recognize_cache: dict[bytes, str] = {}
    cached_recognizer = lambda image_bytes: _recognize_with_cache(engine.recognize, recognize_cache, image_bytes)
    aggregate_box_candidates: list[Candidate] = []
    aggregate_aligned_strip_candidates: list[Candidate] = []
    aggregate_order_attached_candidates: list[Candidate] = []
    aggregate_rotated_candidates: list[Candidate] = []
    best_result: dict | None = None

    def get_original_output() -> str:
        nonlocal original_output
        if original_output is None:
            original_output = cached_recognizer(image_to_png_bytes(working_image))
        return original_output

    def consider(matched: list[dict]) -> bool:
        nonlocal best_result
        response = build_solver_response(image, matched, "ok")
        if response["error"] == "ok" and results_have_consistent_box_sizes(response["results"]):
            best_result = choose_better_legacy_result(best_result, response)
            return True
        return False

    for boxes in box_sets:
        allow_order_attachment = len(boxes) == len(targets)
        box_candidates = expand_confusable_target_candidates(
            recognize_box_crops_legacy(working_image, boxes, recognizer=cached_recognizer),
            targets,
        )
        aggregate_box_candidates.extend(box_candidates)
        try:
            direct_match = match_targets(targets, box_candidates, size)
            direct_match_succeeded = consider(direct_match)
        except MatchError:
            direct_match_succeeded = False
        if direct_match_succeeded and len(box_candidates) >= len(targets) and len(boxes) == len(targets):
            continue

        original_candidates = expand_confusable_target_candidates(
            candidates_from_legacy_output(
                working_image,
                get_original_output(),
                boxes,
                allow_index_attachment=False,
            ),
            targets,
        )
        try:
            consider(match_targets(targets, original_candidates, size))
        except MatchError:
            pass

        strip_output = cached_recognizer(prepare_recognition_image_bytes(working_image, use_strip=True, boxes=boxes))
        aligned_strip_candidates = expand_confusable_target_candidates(
            build_aligned_strip_candidates(
                working_image,
                strip_output,
                boxes,
                box_candidates,
                targets,
            ),
            targets,
        )
        aggregate_aligned_strip_candidates.extend(aligned_strip_candidates)
        strip_candidates = expand_confusable_target_candidates(
            candidates_from_legacy_output(
                working_image,
                strip_output,
                boxes,
                allow_index_attachment=allow_order_attachment,
            ),
            targets,
        )
        aggregate_order_attached_candidates.extend(strip_candidates)
        try:
            consider(
                match_targets_from_views(
                    targets,
                    [original_candidates, aligned_strip_candidates, strip_candidates, box_candidates],
                    size,
                )
            )
        except MatchError:
            pass

        # Last-resort recovery: some samples only succeed when the original OCR text
        # is aligned to left-to-right boxes by index. Keep this behind safer strip-
        # based matching so it does not become the default acceptance path.
        attached_original_candidates = expand_confusable_target_candidates(
            candidates_from_legacy_output(
                working_image,
                original_output,
                boxes,
                allow_index_attachment=allow_order_attachment,
            ),
            targets,
        )
        aggregate_order_attached_candidates.extend(attached_original_candidates)
        try:
            consider(
                match_targets_from_views(
                    targets,
                    [aligned_strip_candidates, strip_candidates, box_candidates, attached_original_candidates],
                    size,
                )
            )
        except MatchError:
            pass

        for output in (original_output, strip_output):
            if output is None:
                continue
            inferred_candidates = expand_confusable_target_candidates(
                infer_single_missing_target_candidates(working_image, targets, output, boxes),
                targets,
            )
            if not inferred_candidates:
                continue
            try:
                consider(match_targets(targets, inferred_candidates, size))
            except MatchError:
                continue

        rotated_candidates = expand_confusable_target_candidates(
            recognize_rotated_box_candidates_legacy(working_image, targets, boxes, recognizer=cached_recognizer),
            targets,
        )
        aggregate_rotated_candidates.extend(rotated_candidates)
        try:
            consider(
                match_targets_from_views(
                    targets,
                    [original_candidates, aligned_strip_candidates, strip_candidates, rotated_candidates, box_candidates],
                    size,
                )
            )
        except MatchError:
            continue

    try:
        consider(
            match_targets_from_views(
                targets,
                [
                    dedupe_candidates(aggregate_box_candidates),
                    dedupe_candidates(aggregate_aligned_strip_candidates),
                    dedupe_candidates(aggregate_order_attached_candidates),
                    dedupe_candidates(aggregate_rotated_candidates),
                ],
                size,
            )
        )
    except MatchError:
        pass

    if best_result is not None:
        return best_result

    return build_solver_response(image, [], initial_error)


def solve_image(image: Image.Image, targets: list[str]) -> dict:
    proposals = generate_box_proposals(image)
    scored = []
    recoverable = []
    recognize_cache: dict[bytes, str] = {}
    cached_recognizer = lambda image_bytes: _recognize_with_cache(engine.recognize, recognize_cache, image_bytes)
    for proposal in proposals:
        if not (3 <= len(proposal.boxes) <= 6):
            continue
        candidates, failed_boxes = recognize_box_candidates_with_recovery(
            image,
            proposal.boxes,
            targets,
            recognizer=cached_recognizer,
        )
        score = score_proposal_set(image, proposal, targets, candidates)
        scored.append(score)
        early_accepted, _ = accept_solution(score, None, expected_match_count=len(targets))
        if (
            early_accepted
            and not failed_boxes
            and len(proposal.boxes) == len(targets)
            and len(candidates) == len(targets)
        ):
            return build_solver_response(image, score["matched"], "ok")
        if failed_boxes:
            recoverable.append((score, candidates, failed_boxes))

    scored.sort(key=lambda item: item["score"], reverse=True)
    top = scored[0] if scored else {"score": 0.0, "matched": []}
    runner_up = scored[1] if len(scored) > 1 else None
    accepted, reason = accept_solution(top, runner_up, expected_match_count=len(targets))
    if accepted:
        return build_solver_response(image, top["matched"], "ok")

    rescored = list(scored)
    for base_score, candidates, failed_boxes in sorted(recoverable, key=lambda item: item[0]["score"], reverse=True)[:2]:
        rotated_candidates = recognize_rotated_box_candidates(
            image,
            targets,
            failed_boxes,
            recognizer=cached_recognizer,
        )
        if not rotated_candidates:
            continue
        rescored.append(
            score_proposal_set(
                image,
                base_score["proposal"],
                targets,
                candidates + rotated_candidates,
            )
        )

    rescored.sort(key=lambda item: item["score"], reverse=True)
    top = rescored[0] if rescored else {"score": 0.0, "matched": []}
    runner_up = rescored[1] if len(rescored) > 1 else None
    accepted, reason = accept_solution(top, runner_up, expected_match_count=len(targets))
    if accepted:
        return build_solver_response(image, top["matched"], "ok")
    return solve_image_with_legacy_fallback(image, targets, reason)


def health():
    return {
        "status": "ok",
        "model": "glm-ocr",
        "model_loaded": bool(getattr(engine, "loaded", False)),
        "model_path": getattr(engine, "model_path", None),
    }


def parse(req: ParseRequest):
    if engine is None or not getattr(engine, "loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")
    try:
        image_bytes = decode_data_uri(req.images[0])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc
    if req.targets:
        return solve_image(image, req.targets)
    return {
        "results": [],
        "image_size": list(image.size),
        "method": "glm_ocr_transformers_with_local_positioning",
    }


@app.get("/health")
def health_route():
    return health()


@app.post("/glmocr/parse")
@app.post("/ocr/parse")
async def parse_route(req: ParseRequest):
    return parse(req)


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR transformers captcha server")
    parser.add_argument("--model", default="models/GLM-OCR", help="Path to GLM-OCR model directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    args = parser.parse_args()

    global engine
    engine = GlmOcrEngine(args.model)
    engine.load()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
