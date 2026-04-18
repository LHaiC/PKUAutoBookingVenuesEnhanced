import argparse
import base64
import binascii
import io
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager

from PIL import Image, UnidentifiedImageError
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from captcha_matcher import MatchError, match_targets, normalize_candidates
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

    if targets:
        original_output = engine.recognize(image_bytes, None)
        original_candidates = candidates_from_output(image, original_output, boxes)
        try:
            results = match_targets(targets, original_candidates, size)
            raw_output = original_output
        except MatchError as exc:
            recognition_bytes = prepare_recognition_image_bytes(image, use_strip=True, boxes=boxes)
            strip_output = engine.recognize(recognition_bytes, None)
            strip_candidates = candidates_from_output(image, strip_output, boxes)
            raw_outputs = [original_output, strip_output]
            try:
                results = match_targets_from_views(
                    targets,
                    [original_candidates, strip_candidates],
                    size,
                )
                raw_output = "\n".join(raw_outputs)
            except MatchError as fallback_exc:
                return {
                    "results": [],
                    "error": "unsafe_ocr_output",
                    "detail": str(fallback_exc),
                    "image_size": list(size),
                    "method": "glm_ocr_transformers_with_local_positioning",
                    "raw_output": original_output,
                    "raw_outputs": raw_outputs,
                }
    else:
        raw_output = engine.recognize(image_bytes, None)
        results = attach_colored_text_bboxes(image, parse_model_output(raw_output))

    return {
        "results": results,
        "image_size": list(size),
        "method": "glm_ocr_transformers_with_local_positioning",
        "raw_output": raw_output,
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
