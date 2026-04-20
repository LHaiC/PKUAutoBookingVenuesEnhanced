"""
PaddleOCR PP-OCRv4 captcha solving server.
Replaces ocr_server_transformers.py (GLM OCR cascade).
"""
import io
import base64
import binascii
from PIL import Image, UnidentifiedImageError
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from captcha_vision import whiten_watermark
from captcha_matcher import Candidate, MatchError, match_targets, normalize_candidates

app = FastAPI()
ocr = None

class ParseRequest(BaseModel):
    images: list[str] = Field(default_factory=list)
    targets: list[str] | None = None

def decode_data_uri(data_uri: str) -> bytes:
    payload = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    return base64.b64decode(payload, validate=True)

def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def solve_image(image_bytes: bytes, targets: list[str] | None) -> dict:
    try:
        image = whiten_watermark(image_from_bytes(image_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    # Use PaddleOCR to detect and recognize text
    result = ocr.ocr(image, cls=True)

    # Parse PaddleOCR output: [ [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, confidence) ], ... ]
    candidates = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]
            if len(text) == 1:
                x1 = min(p[0] for p in box)
                y1 = min(p[1] for p in box)
                x2 = max(p[0] for p in box)
                y2 = max(p[1] for p in box)
                candidates.append(Candidate(text=text, bbox=[x1, y1, x2, y2], confidence=confidence))

    if targets:
        try:
            matched = match_targets(targets, candidates, image.size)
            return {
                "results": matched,
                "image_size": list(image.size),
                "method": "paddle_ocr_ppocrv4",
            }
        except MatchError as exc:
            return {
                "results": [],
                "error": str(exc),
                "image_size": list(image.size),
                "method": "paddle_ocr_ppocrv4",
            }
    else:
        return {
            "results": [{"text": c.text, "bbox": c.bbox, "confidence": c.confidence} for c in normalize_candidates(candidates)],
            "image_size": list(image.size),
            "method": "paddle_ocr_ppocrv4",
        }

@app.post("/ocr/parse")
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
    return {"status": "ok", "model": "paddleocr"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PaddleOCR captcha server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--use_angle_clf", type=bool, default=True, help="Use angle classifier")
    parser.add_argument("--det_db_thresh", type=float, default=0.3, help="Detection threshold")
    args = parser.parse_args()

    global ocr
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_clf=args.use_angle_clf,
        det_db_thresh=args.det_db_thresh,
        show_log=False,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()