import argparse
import base64
import binascii
import json
import os
import re
import tempfile
from contextlib import contextmanager

from PIL import UnidentifiedImageError
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from captcha_matcher import MatchError, match_targets, normalize_candidates
from captcha_vision import decode_image, image_size, refine_bbox_to_dark_pixels


app = FastAPI()
engine = None


class ParseRequest(BaseModel):
    images: list[str]
    targets: list[str] | None = None


def decode_data_uri(data_uri: str) -> bytes:
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return base64.b64decode(payload, validate=True)


def parse_model_output(output: str) -> list[dict]:
    json_match = re.search(r"\[.*\]", output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [
                    {
                        "text": str(item.get("text", "")).strip(),
                        "bbox": [int(v) for v in item.get("bbox", [])],
                        "confidence": float(item.get("confidence", 0.80)),
                    }
                    for item in parsed
                    if isinstance(item, dict)
                ]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    text_segment = re.split(r"[:：]", output)[-1]
    chars = re.findall(r"[\u4e00-\u9fff]", text_segment)
    return [{"text": char, "bbox": [], "confidence": 0.50} for char in chars]


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

        target_text = "、".join(targets or [])
        if target_text:
            prompt = (
                "请识别图片中的中文验证码字符，并只输出 JSON 数组。"
                f"必须包含这些目标字符：{target_text}。"
                '格式为 [{"text":"字","bbox":[x_min,y_min,x_max,y_max],"confidence":0.0}]。'
            )
        else:
            prompt = (
                "请识别图片中的中文验证码字符，并只输出 JSON 数组。"
                '格式为 [{"text":"字","bbox":[x_min,y_min,x_max,y_max],"confidence":0.0}]。'
            )

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
    if engine is None or not engine.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = decode_image(image_bytes)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc
    size = image_size(image)
    raw_output = engine.recognize(image_bytes, targets)
    raw_candidates = parse_model_output(raw_output)

    refined_items = []
    for item in raw_candidates:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            item = dict(item)
            item["bbox"] = refine_bbox_to_dark_pixels(image, [int(v) for v in bbox])
        refined_items.append(item)

    candidates = normalize_candidates(refined_items)
    if targets:
        try:
            results = match_targets(targets, candidates, size)
        except MatchError as exc:
            return {"results": [], "error": "unsafe_ocr_output", "detail": str(exc)}
    else:
        results = [
            {"text": candidate.text, "bbox": candidate.bbox, "confidence": candidate.confidence}
            for candidate in candidates
        ]

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
