from dataclasses import dataclass


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


def bbox_in_bounds(bbox: list[int], image_size: tuple[int, int]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def normalize_candidate(item: dict) -> Candidate | None:
    text = str(item.get("text", "")).strip()
    bbox = item.get("bbox", [])
    if len(text) != 1 or len(bbox) != 4:
        return None
    try:
        confidence = float(item.get("confidence", 1.0))
        normalized_bbox = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return Candidate(text=text, bbox=normalized_bbox, confidence=confidence)


def normalize_candidates(items: list[dict]) -> list[Candidate]:
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate = normalize_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def match_targets(
    targets: list[str],
    candidates: list[Candidate],
    image_size: tuple[int, int],
    min_confidence: float = 0.50,
) -> list[dict]:
    matched = []
    used_indexes = set()

    for target in targets:
        target_matches = [
            (idx, candidate)
            for idx, candidate in enumerate(candidates)
            if candidate.text == target and idx not in used_indexes
        ]
        if not target_matches:
            raise MatchError(f"missing target: {target}")
        if len(target_matches) > 1:
            raise MatchError(f"ambiguous target: {target}")

        idx, candidate = target_matches[0]
        if candidate.confidence < min_confidence:
            raise MatchError(f"low confidence: {target}")
        if not bbox_in_bounds(candidate.bbox, image_size):
            raise MatchError(f"bbox out of bounds: {target}")

        used_indexes.add(idx)
        matched.append(
            {
                "text": candidate.text,
                "bbox": list(candidate.bbox),
                "confidence": candidate.confidence,
            }
        )

    return matched
