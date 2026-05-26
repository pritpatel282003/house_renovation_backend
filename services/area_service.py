import asyncio
import logging
from typing import Any

from services.image_utils import download_image

logger = logging.getLogger(__name__)

DEFAULT_PIXELS_PER_FOOT = 10.0


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


def _compute_areas(
    width: int,
    height: int,
    segmentation_data: list[dict[str, Any]],
    pixels_per_foot: float,
) -> dict[str, dict[str, Any]]:
    sq_pixels_per_sqft = pixels_per_foot**2
    result: dict[str, dict[str, Any]] = {}

    for segment in segmentation_data:
        label: str = segment["label"]
        polygon = segment.get("mask_polygon", [])
        pixel_area = _shoelace_area(polygon, width, height)
        area_sqft = round(pixel_area / sq_pixels_per_sqft, 2)
        result[label] = {"area_sqft": area_sqft, "unit": "sqft"}

    return result


async def calculate_areas(
    image_url: str,
    segmentation_data: list[dict[str, Any]],
    pixels_per_foot: float = DEFAULT_PIXELS_PER_FOOT,
) -> dict[str, dict[str, Any]]:
    """Return mapping of segment label -> area info in sqft."""
    img = await download_image(image_url, mode="RGB")
    width, height = img.size
    return await asyncio.to_thread(
        _compute_areas,
        width,
        height,
        segmentation_data,
        pixels_per_foot,
    )
