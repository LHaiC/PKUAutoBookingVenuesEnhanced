import argparse
import base64
import binascii
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager

from PIL import Image, UnidentifiedImageError
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from captcha_matcher import Candidate, MatchError, match_targets, normalize_candidates
from captcha_vision import (
    build_colored_text_strip,
    decode_image,
    detect_colored_text_bboxes,
    filter_captcha_text_bboxes,
    image_size,
    refine_bbox_to_dark_pixels,
)


app = FastAPI()
engine = None
OCR_CHAR_NORMALIZATION = {
    "イ": "八",
}


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, _exc):
    return JSONResponse(status_code=400, content={"detail": "Invalid request body"})


class ParseRequest(BaseModel):
    images: list[str] = Field(default_factory=list)
    targets: list[str] | None = None


def decode_data_uri(data_uri: str) -> bytes:
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return base64.b64decode(payload, validate=True)


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


def parse_model_output(output: str) -> list[dict]:
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
) -> list[dict]:
    if not candidates:
        return candidates
    if all(len(item.get("bbox", [])) == 4 for item in candidates):
        return candidates

    boxes = detect_colored_text_bboxes(image) if boxes is None else boxes
    if not boxes:
        return candidates

    updated = []
    for item, bbox in zip(candidates[: len(boxes)], boxes):
        item = dict(item)
        if len(item.get("bbox", [])) != 4:
            item["bbox"] = bbox
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

    buf = io.BytesIO()
    recognition_image.save(buf, format="PNG")
    return buf.getvalue()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def crop_box_image(image: Image.Image, box: list[int], padding: int = 10) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop(
        (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(image.width, x2 + padding),
            min(image.height, y2 + padding),
        )
    )


def candidates_from_output(
    image: Image.Image,
    output: str,
    boxes: list[list[int]],
) -> list:
    parsed = parse_model_output(output)
    if boxes and len(parsed) < len(boxes):
        return []
    candidates = attach_colored_text_bboxes(
        image,
        parsed,
        boxes=boxes,
        trim_to_boxes=True,
    )
    refined_items = []
    for item in candidates:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            item = dict(item)
            item["bbox"] = refine_bbox_to_dark_pixels(image, [int(v) for v in bbox])
        refined_items.append(item)
    return normalize_candidates(refined_items)


def expand_confusable_target_candidates(candidates: list[Candidate], targets: list[str]) -> list[Candidate]:
    if "三" not in targets or "川" in targets:
        return candidates

    expanded = list(candidates)
    for candidate in candidates:
        if candidate.text == "川":
            expanded.append(Candidate(text="三", bbox=list(candidate.bbox), confidence=0.50))
    return expanded


def _candidate_from_box(text: str, image: Image.Image, box: list[int], confidence: float = 0.50) -> Candidate:
    return Candidate(
        text=text,
        bbox=refine_bbox_to_dark_pixels(image, [int(value) for value in box]),
        confidence=confidence,
    )


def infer_single_missing_target_candidates(
    image: Image.Image,
    targets: list[str],
    output: str,
    boxes: list[list[int]],
) -> list[Candidate]:
    if not targets or len(set(targets)) != len(targets):
        return []

    parsed = parse_model_output(output)
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


def recognize_rotated_box_candidates(
    image: Image.Image,
    targets: list[str],
    boxes: list[list[int]],
    angles: tuple[int, ...] = (0, 90, 180, 270),
) -> list[Candidate]:
    if not targets:
        return []

    target_set = set(targets)
    candidates = []
    for box in boxes:
        crop = crop_box_image(image, box)
        hits = set()
        raw_outputs = []
        for angle in angles:
            rotated = crop.rotate(angle, expand=True, fillcolor="white")
            output = engine.recognize(image_to_png_bytes(rotated), None)
            raw_outputs.append(output)
            for item in parse_model_output(output):
                text = str(item.get("text", "")).strip()
                if text in target_set:
                    hits.add(text)
        if len(hits) == 1:
            candidates.append(_candidate_from_box(next(iter(hits)), image, box))
            continue
        if "三" in target_set and "川" not in target_set:
            raw_text = "\n".join(raw_outputs)
            if "111" in raw_text:
                candidates.append(_candidate_from_box("三", image, box))
    return candidates


def recognize_box_crops(image, boxes):
    """
    Recognize each box region individually via GLM.

    Returns: list of dicts [{"text": char, "bbox": [x1,y1,x2,y2]}, ...]
            Returns empty list if any box fails to produce exactly 1 character.
    """
    if not boxes or engine is None or not engine.loaded:
        return []

    results = []
    for box in boxes:
        crop = crop_box_image(image, box, padding=5)
        crop_bytes = image_to_png_bytes(crop)
        output = engine.recognize(crop_bytes, None)

        chars = re.findall(r"[\u4e00-\u9fff]", output.replace("<|user|>", "").strip())
        if len(chars) != 1:
            return []  # fail: not exactly one Chinese char

        results.append({
            "text": chars[0],
            "bbox": box,
            "confidence": 0.80,
        })

    return results


def recognize_rotated_box_crops(image, targets, boxes):
    """
    Per-box OCR with rotation fallback. Tries 4 angles per box.
    Returns list of Candidate objects or empty list on failure.
    """
    if not boxes or engine is None or not engine.loaded:
        return []

    target_set = set(targets)
    candidates = []
    for box in boxes:
        crop = crop_box_image(image, box, padding=5)
        crop_bytes = image_to_png_bytes(crop)
        best_char = None
        for angle in (0, 90, 180, 270):
            rotated = crop.rotate(angle, expand=True, fillcolor="white")
            rotated_bytes = image_to_png_bytes(rotated)
            output = engine.recognize(rotated_bytes, None)
            chars = re.findall(r"[\u4e00-\u9fff]", output.replace("<|user|>", "").strip())
            if len(chars) == 1 and chars[0] in target_set:
                best_char = chars[0]
                break
        if best_char:
            from captcha_matcher import Candidate
            candidates.append(Candidate(
                text=best_char,
                bbox=box,
                confidence=0.80,
            ))
        else:
            return []  # fail whole fallback
    return candidates


def match_targets_from_views(
    targets: list[str],
    candidate_views: list[list],
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


def filter_watermark_boxes(boxes, image_size):
    """
    Remove boxes in the bottom-right watermark zone.
    Watermark "场馆预约" is in the bottom-right corner.
    Filter condition: box right >= 75% width AND box bottom >= 85% height
    """
    if not boxes:
        return boxes
    width, height = image_size
    threshold_x = 0.75
    threshold_y = 0.85
    return [
        box for box in boxes
        if not (box[2] >= width * threshold_x and box[3] >= height * threshold_y)
    ]


@contextmanager
def temporary_image_file(image_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        path = tmp.name
    try:
        yield path
    finally:
        try:
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

    def recognize(self, image_bytes: bytes, targets: list[str] | None) -> str:
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


def solve_image(image_bytes: bytes, targets: list[str] | None) -> dict:
    try:
        image = decode_image(image_bytes)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc
    if engine is None or not engine.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    size = image_size(image)
    detected_boxes = detect_colored_text_bboxes(image)
    boxes = filter_captcha_text_bboxes(detected_boxes, size) or detected_boxes

    if not boxes:
        return {
            "results": [],
            "error": "no_boxes_detected",
            "image_size": list(size),
            "method": "per_box_ocr",
            "raw_output": "",
        }

    # Filter watermark boxes from bottom-right corner
    filtered_boxes = filter_watermark_boxes(boxes, size)
    # Fall back to original if all filtered
    use_boxes = filtered_boxes if filtered_boxes else boxes

    if targets:
        # Main path: per-box OCR
        per_box_results = recognize_box_crops(image, use_boxes)
        if per_box_results and len(per_box_results) == len(use_boxes):
            from captcha_matcher import Candidate, match_targets
            try:
                matched = match_targets(targets, [
                    Candidate(text=r["text"], bbox=r["bbox"], confidence=r.get("confidence", 0.80))
                    for r in per_box_results
                ], size)
                return {
                    "results": matched,
                    "image_size": list(size),
                    "method": "per_box_ocr",
                    "raw_output": "",
                }
            except MatchError:
                pass

        # Fallback: rotated per-box OCR
        rotated_candidates = recognize_rotated_box_crops(image, targets, use_boxes)
        if rotated_candidates:
            from captcha_matcher import Candidate, match_targets
            try:
                matched = match_targets(targets, rotated_candidates, size)
                return {
                    "results": matched,
                    "image_size": list(size),
                    "method": "rotated_per_box_ocr",
                    "raw_output": "",
                }
            except MatchError:
                pass

        # All failed
        return {
            "results": [],
            "error": "unsafe_ocr_output",
            "detail": "per_box_ocr failed, rotated fallback also failed",
            "image_size": list(size),
            "method": "per_box_ocr",
            "raw_output": "",
        }

    else:
        # No targets: return all detected chars with boxes
        per_box_results = recognize_box_crops(image, use_boxes)
        return {
            "results": per_box_results,
            "image_size": list(size),
            "method": "per_box_ocr",
        }


@app.post("/glmocr/parse")
def parse(req: ParseRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")
    try:
        image_bytes = decode_data_uri(req.images[0])
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc
    return solve_image(image_bytes, req.targets)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None and engine.loaded,
        "model_path": None if engine is None else engine.model_path,
    }


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
