import io
import cv2
import numpy as np
import requests
from PIL import Image

BLUR_THRESHOLD = 100.0


def check_blur(image_url: str) -> dict:
    """Download image and check if it's blurry using Laplacian variance."""
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()

        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return {"is_blurry": False, "score": 0, "error": "Could not decode image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            "is_blurry": bool(laplacian_var < BLUR_THRESHOLD),
            "score": float(round(laplacian_var, 2)),
            "threshold": float(BLUR_THRESHOLD),
        }
    except Exception as e:
        return {"is_blurry": False, "score": 0, "error": str(e)}
