import base64
import io
import re

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from PIL import Image

from captcha_matcher import Candidate, match_targets
from captcha_vision import ProposalSet, measure_box_size_consistency

app = FastAPI()
engine = None


class ParseRequest(BaseModel):
    images: list[str] = Field(default_factory=list)
    targets: list[str] | None = None


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request, _exc):
    return JSONResponse(status_code=400, content={"detail": "Invalid request body"})


def decode_data_uri(data_uri: str) -> bytes:
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return base64.b64decode(payload, validate=True)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


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


def recognize_box_crop(image: Image.Image, box: list[int], target_set: set[str] | None):
    crop = crop_box_image(image, box)
    chars = parse_model_output(engine.recognize(image_to_png_bytes(crop)))
    if len(chars) != 1:
        return None
    text = chars[0]
    if target_set is not None and text not in target_set:
        return None
    return Candidate(text=text, bbox=box, confidence=0.80)


def recognize_box_crops(image: Image.Image, boxes: list[list[int]], targets: list[str] | None):
    target_set = set(targets) if targets else None
    results = []
    for box in boxes:
        candidate = recognize_box_crop(image, box, target_set)
        if candidate is None:
            return []
        results.append(candidate)
    return results


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
    except Exception:
        matched = []
        coverage = 0
    score = (
        coverage * 10.0
        + sum(candidate.confidence for candidate in candidates)
        + size_stats["score"] * 2.0
        - max(0, len(proposal.boxes) - 3)
    )
    return {"score": score, "matched": matched, "size_stats": size_stats, "proposal": proposal}


def accept_solution(
    top_score: dict,
    runner_up_score: dict | None,
    min_margin: float = 0.25,
) -> tuple[bool, str]:
    matched = top_score.get("matched", [])
    if len(matched) != 3:
        return False, "incomplete_target_coverage"
    if runner_up_score is not None and top_score["score"] - runner_up_score["score"] < min_margin:
        return False, "ambiguous_top_score"
    if len({tuple(item["bbox"]) for item in matched}) != len(matched):
        return False, "duplicate_box_reuse"
    return True, "accepted"


def health():
    return {
        "status": "ok",
        "model": "glm-ocr",
        "model_loaded": bool(getattr(engine, "loaded", False)),
        "model_path": getattr(engine, "model_path", None),
    }


@app.post("/ocr/parse")
def parse(req: ParseRequest):
    if engine is None or not getattr(engine, "loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    raise HTTPException(status_code=501, detail="Not implemented yet")
