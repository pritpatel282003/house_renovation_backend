import base64
import io
import logging
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

FEATHER_RADIUS = 3
TARGET_TEXTURE_PX = 128


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _scale_texture(texture: Image.Image, region_width: int) -> Image.Image:
    """Resize texture so tiles appear at a consistent physical scale."""
    scale = max(0.5, min(2.5, region_width / (TARGET_TEXTURE_PX * 4)))
    new_w = max(16, int(texture.width * scale))
    new_h = max(16, int(texture.height * scale))
    return texture.resize((new_w, new_h), Image.LANCZOS)


def _tile_texture(texture: Image.Image, width: int, height: int) -> Image.Image:
    """Tile a texture image to fill the given width x height area."""
    tiled = Image.new("RGBA", (width, height))
    tw, th = texture.size
    if tw <= 0 or th <= 0:
        return tiled
    for y in range(0, height, th):
        for x in range(0, width, tw):
            tiled.paste(texture, (x, y))
    return tiled.crop((0, 0, width, height))


def _denormalize_polygon(
    polygon: list[list[float]], width: int, height: int
) -> list[tuple[int, int]]:
    return [(int(pt[0] * width), int(pt[1] * height)) for pt in polygon]


def _build_feathered_mask(
    pixel_polygon: list[tuple[int, int]], width: int, height: int
) -> Image.Image:
    """Create a polygon mask with soft feathered edges."""
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(pixel_polygon, fill=255)
    if FEATHER_RADIUS > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=FEATHER_RADIUS))
    return mask


def _perspective_warp(
    tiled_arr: np.ndarray,
    polygon: list[list[float]],
    bbox: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Warp the tiled texture to match the polygon's perspective.

    Uses the four extreme points (top-left, top-right, bottom-right, bottom-left)
    of the polygon as destination corners.
    """
    bx1, by1, bx2, by2 = bbox
    box_w, box_h = bx2 - bx1, by2 - by1

    if len(polygon) < 4 or box_w <= 0 or box_h <= 0:
        return tiled_arr

    pts = np.array(
        [[pt[0] * img_width - bx1, pt[1] * img_height - by1] for pt in polygon],
        dtype=np.float32,
    )

    tl_idx = int(np.argmin(pts[:, 0] + pts[:, 1]))
    tr_idx = int(np.argmin(-pts[:, 0] + pts[:, 1]))
    br_idx = int(np.argmin(-pts[:, 0] - pts[:, 1]))
    bl_idx = int(np.argmin(pts[:, 0] - pts[:, 1]))

    if len({tl_idx, tr_idx, br_idx, bl_idx}) < 4:
        return tiled_arr

    dst_corners = np.array(
        [pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]], dtype=np.float32
    )

    src_corners = np.array(
        [[0, 0], [box_w - 1, 0], [box_w - 1, box_h - 1], [0, box_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(
        tiled_arr, M, (box_w, box_h), borderMode=cv2.BORDER_REFLECT_101
    )
    return warped


def _luminosity_blend(
    base_rgb: np.ndarray,
    texture_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend texture onto base while preserving the original lighting.

    Extracts the luminance channel from the original image and applies it
    to the texture so shadows, highlights, and ambient occlusion persist.
    """
    base_f = base_rgb.astype(np.float32)
    tex_f = texture_rgb.astype(np.float32)
    mask_f = mask.astype(np.float32) / 255.0

    luminance = (
        0.299 * base_f[:, :, 0]
        + 0.587 * base_f[:, :, 1]
        + 0.114 * base_f[:, :, 2]
    )
    mean_lum = luminance.mean()
    if mean_lum < 1.0:
        mean_lum = 1.0
    lum_ratio = luminance / mean_lum
    lum_ratio = np.clip(lum_ratio, 0.25, 2.5)

    lit_texture = tex_f * lum_ratio[:, :, np.newaxis]
    lit_texture = np.clip(lit_texture, 0, 255)

    mask_3d = mask_f[:, :, np.newaxis]
    result = base_f * (1.0 - mask_3d) + lit_texture * mask_3d
    return np.clip(result, 0, 255).astype(np.uint8)


def visualize(
    image_url: str,
    segmentation_data: list[dict[str, Any]],
    material_assignments: dict[str, str],
    materials: list[dict[str, Any]],
) -> str:
    """Apply material textures onto the original image and return base64 PNG."""
    base_img = _download_image(image_url)
    width, height = base_img.size

    base_rgb = np.array(base_img.convert("RGB"))
    result_rgb = base_rgb.copy()

    material_map: dict[str, dict[str, Any]] = {m["id"]: m for m in materials}

    sorted_segments = sorted(
        segmentation_data,
        key=lambda s: s.get("area_pixels", 0),
        reverse=True,
    )

    all_masks: list[Image.Image | None] = []
    all_polygons: list[list[tuple[int, int]]] = []
    for segment in sorted_segments:
        polygon = segment.get("mask_polygon", [])
        if polygon:
            px_poly = _denormalize_polygon(polygon, width, height)
            all_polygons.append(px_poly)
            all_masks.append(None)
        else:
            all_polygons.append([])
            all_masks.append(None)

    for seg_idx, segment in enumerate(sorted_segments):
        label: str = segment["label"]
        material_id = material_assignments.get(label)
        if not material_id:
            continue

        material = material_map.get(material_id)
        texture_url = (material or {}).get("texture_cloudinary_url") or (
            material or {}
        ).get("texture_url")
        if not material or not texture_url:
            logger.warning("No texture URL for material %s", material_id)
            continue

        try:
            texture_img = _download_image(texture_url)
        except Exception as exc:
            logger.warning("Failed to download texture for %s: %s", material_id, exc)
            continue

        polygon = segment.get("mask_polygon", [])
        if not polygon:
            continue

        pixel_polygon = all_polygons[seg_idx]

        x1, y1, x2, y2 = segment["bbox"]
        bx1, by1 = int(x1 * width), int(y1 * height)
        bx2, by2 = int(x2 * width), int(y2 * height)
        box_w, box_h = bx2 - bx1, by2 - by1
        if box_w <= 0 or box_h <= 0:
            continue

        scaled_texture = _scale_texture(texture_img, box_w)
        tiled = _tile_texture(scaled_texture, box_w, box_h)
        tiled_arr = np.array(tiled.convert("RGB"))

        warped = _perspective_warp(
            tiled_arr, polygon, (bx1, by1, bx2, by2), width, height
        )

        mask_full = _build_feathered_mask(pixel_polygon, width, height)

        mask_arr = np.array(mask_full)
        for later_idx in range(seg_idx + 1, len(sorted_segments)):
            later_poly = all_polygons[later_idx]
            if not later_poly:
                continue
            subtract = Image.new("L", (width, height), 0)
            ImageDraw.Draw(subtract).polygon(later_poly, fill=255)
            mask_arr = np.where(np.array(subtract) > 127, 0, mask_arr)
        mask_full = Image.fromarray(mask_arr)

        mask_region = np.array(mask_full.crop((bx1, by1, bx2, by2)))

        base_region = result_rgb[by1:by2, bx1:bx2]
        blended_region = _luminosity_blend(base_region, warped, mask_region)
        result_rgb[by1:by2, bx1:bx2] = blended_region

    result_img = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
