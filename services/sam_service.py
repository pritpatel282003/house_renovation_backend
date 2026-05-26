import io
import logging
import os
import base64
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")

MOCK_REGIONS: list[dict[str, Any]] = [
    {
        "label": "main_wall",
        "mask_polygon": [
            [0.05, 0.10], [0.95, 0.10], [0.95, 0.85], [0.05, 0.85],
        ],
        "bbox": [0.05, 0.10, 0.95, 0.85],
        "confidence": 0.93,
    },
    {
        "label": "window_left",
        "mask_polygon": [
            [0.10, 0.25], [0.30, 0.25], [0.30, 0.55], [0.10, 0.55],
        ],
        "bbox": [0.10, 0.25, 0.30, 0.55],
        "confidence": 0.90,
    },
    {
        "label": "window_right",
        "mask_polygon": [
            [0.60, 0.25], [0.80, 0.25], [0.80, 0.55], [0.60, 0.55],
        ],
        "bbox": [0.60, 0.25, 0.80, 0.55],
        "confidence": 0.89,
    },
    {
        "label": "balcony",
        "mask_polygon": [
            [0.15, 0.60], [0.85, 0.60], [0.85, 0.75], [0.15, 0.75],
        ],
        "bbox": [0.15, 0.60, 0.85, 0.75],
        "confidence": 0.87,
    },
    {
        "label": "gate_area",
        "mask_polygon": [
            [0.35, 0.75], [0.65, 0.75], [0.65, 0.98], [0.35, 0.98],
        ],
        "bbox": [0.35, 0.75, 0.65, 0.98],
        "confidence": 0.85,
    },
]


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_roboflow(img: Image.Image) -> list[dict[str, Any]] | None:
    """Call the Roboflow Hosted API for instance segmentation."""
    if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_ID:
        return None

    try:
        img_b64 = _image_to_base64(img)
        width, height = img.size

        url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 30,
            "format": "json",
        }

        resp = requests.post(
            url,
            params=params,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        predictions = data.get("predictions", [])
        if not predictions:
            logger.info("Roboflow returned no predictions")
            return None

        regions: list[dict[str, Any]] = []
        label_counts: dict[str, int] = {}

        for pred in predictions:
            class_name: str = pred.get("class", "region")
            confidence: float = pred.get("confidence", 0.0)

            count = label_counts.get(class_name, 0)
            label_counts[class_name] = count + 1
            label = f"{class_name}_{count}" if count > 0 else class_name

            points = pred.get("points", [])
            if points and len(points) >= 3:
                polygon = [
                    [round(pt["x"] / width, 4), round(pt["y"] / height, 4)]
                    for pt in points
                ]
            else:
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                w = pred.get("width", 0)
                h = pred.get("height", 0)
                x1n, y1n = (x - w / 2) / width, (y - h / 2) / height
                x2n, y2n = (x + w / 2) / width, (y + h / 2) / height
                polygon = [
                    [round(x1n, 4), round(y1n, 4)],
                    [round(x2n, 4), round(y1n, 4)],
                    [round(x2n, 4), round(y2n, 4)],
                    [round(x1n, 4), round(y2n, 4)],
                ]

            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            area_pixels = int(
                abs(bbox[2] - bbox[0]) * width * abs(bbox[3] - bbox[1]) * height
            )

            regions.append({
                "label": label,
                "mask_polygon": polygon,
                "bbox": [round(v, 4) for v in bbox],
                "area_pixels": area_pixels,
                "confidence": round(confidence, 4),
            })

        logger.info("Roboflow detected %d regions", len(regions))
        return regions

    except Exception as exc:
        logger.warning("Roboflow API call failed: %s", exc)
        return None


def segment_image(image_url: str) -> list[dict[str, Any]]:
    """Segment an exterior house image into labelled regions."""
    img = _download_image(image_url)
    width, height = img.size

    roboflow_result = _call_roboflow(img)
    if roboflow_result:
        return roboflow_result

    logger.info("Falling back to mock segmentation")
    return _mock_segments(width, height)


def _mock_segments(width: int, height: int) -> list[dict[str, Any]]:
    """Return mock segmentation data for development/testing."""
    logger.info("Using mock segmentation for %dx%d image", width, height)
    regions: list[dict[str, Any]] = []
    for region in MOCK_REGIONS:
        x1, y1, x2, y2 = region["bbox"]
        area_pixels = int((x2 - x1) * width * (y2 - y1) * height)
        regions.append({
            "label": region["label"],
            "mask_polygon": region["mask_polygon"],
            "bbox": region["bbox"],
            "area_pixels": area_pixels,
            "confidence": region["confidence"],
        })
    return regions
