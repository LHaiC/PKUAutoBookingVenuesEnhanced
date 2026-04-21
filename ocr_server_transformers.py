import base64

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
