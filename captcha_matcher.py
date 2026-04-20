from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


class MatchError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    text: str
    bbox: list[int]
    confidence: float = 1.0


def bbox_center(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_in_bounds(bbox: list[int], image_size: tuple[int, int]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _strict_int(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError('expected integer')
    return value


def _strict_float(value) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError('expected number')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('expected finite number')
    return number


def _normalize_bbox(item: dict) -> list[int] | None:
    bbox = item.get('bbox', [])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [_strict_int(v) for v in bbox]
        except (TypeError, ValueError):
            return None

    x = item.get('x')
    y = item.get('y')
    if isinstance(x, int) and not isinstance(x, bool) and isinstance(y, int) and not isinstance(y, bool):
        return [x, y, x + 1, y + 1]
    return None


def normalize_candidate(item: dict) -> Candidate | None:
    text = str(item.get('text', item.get('char', ''))).strip()
    if len(text) != 1:
        return None

    bbox = _normalize_bbox(item)
    if bbox is None:
        return None

    try:
        confidence = _strict_float(item.get('confidence', 1.0))
    except (TypeError, ValueError):
        return None
    return Candidate(text=text, bbox=bbox, confidence=confidence)


def normalize_candidates(items: Iterable[dict]) -> list[Candidate]:
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate = normalize_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidate_sort_key(candidate: Candidate):
    x1, y1, x2, y2 = candidate.bbox
    return (-candidate.confidence, bbox_area(candidate.bbox), x1, y1, x2, y2)


def candidate_to_result(candidate: Candidate) -> dict:
    x, y = bbox_center(candidate.bbox)
    return {
        'text': candidate.text,
        'bbox': list(candidate.bbox),
        'x': x,
        'y': y,
        'confidence': candidate.confidence,
    }


def match_targets(
    targets: list[str],
    candidates: list[Candidate],
    image_size: tuple[int, int],
    min_confidence: float = 0.50,
) -> list[dict]:
    if not targets:
        return []

    matched = []
    used_indexes: set[int] = set()

    for target in targets:
        high_conf = []
        low_conf = []
        out_of_bounds = []

        for idx, candidate in enumerate(candidates):
            if idx in used_indexes or candidate.text != target:
                continue
            if not bbox_in_bounds(candidate.bbox, image_size):
                out_of_bounds.append((idx, candidate))
                continue
            if not math.isfinite(candidate.confidence):
                continue
            if candidate.confidence < min_confidence:
                low_conf.append((idx, candidate))
                continue
            high_conf.append((idx, candidate))

        if not high_conf:
            if low_conf:
                raise MatchError(f'low confidence: {target}')
            if out_of_bounds:
                raise MatchError(f'bbox out of bounds: {target}')
            raise MatchError(f'missing target: {target}')

        idx, candidate = sorted((pair for pair in high_conf), key=lambda pair: candidate_sort_key(pair[1]))[0]
        used_indexes.add(idx)
        matched.append(candidate_to_result(candidate))

    return matched
