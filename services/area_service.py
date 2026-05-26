import io
import logging
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_PIXELS_PER_FOOT = 10.0


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def _shoelace_area(polygon: list[list[float]], width: int, height: int) -> float:
    """Compute pixel area of a polygon using the shoelace formula.

    Polygon coords are normalized [0-1]; they are scaled to pixel coords first.
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    coords = [(pt[0] * width, pt[1] * height) for pt in polygon]
    area = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def calculate_areas(
    image_url: str,
    segmentation_data: list[dict[str, Any]],
    pixels_per_foot: float = DEFAULT_PIXELS_PER_FOOT,
) -> dict[str, dict[str, Any]]:
    """Return mapping of segment label -> area info in sqft."""
    img = _download_image(image_url)
    width, height = img.size

    sq_pixels_per_sqft = pixels_per_foot ** 2
    result: dict[str, dict[str, Any]] = {}

    for segment in segmentation_data:
        label: str = segment["label"]
        polygon = segment.get("mask_polygon", [])
        pixel_area = _shoelace_area(polygon, width, height)
        area_sqft = round(pixel_area / sq_pixels_per_sqft, 2)
        result[label] = {"area_sqft": area_sqft, "unit": "sqft"}

    return result
