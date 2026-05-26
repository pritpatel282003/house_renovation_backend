import logging
from typing import Any

logger = logging.getLogger(__name__)


async def calculate_costs(
    area_data: dict[str, dict[str, Any]],
    material_assignments: dict[str, str],
    materials: list[dict[str, Any]],
    rate_overrides: dict[str, dict[str, float]] | None = None,
    wastage_percent: float = 12.0,
) -> dict[str, Any]:
    """Build a full cost breakdown for all assigned regions."""
    material_map: dict[str, dict[str, Any]] = {m["id"]: m for m in materials}
    overrides = rate_overrides or {}

    line_items: list[dict[str, Any]] = []
    subtotal_material = 0.0
    subtotal_labor = 0.0

    for region_label, material_id in material_assignments.items():
        area_info = area_data.get(region_label)
        if not area_info:
            logger.warning("No area data for region '%s' – skipping", region_label)
            continue

        material = material_map.get(material_id)
        if not material:
            logger.warning("Material '%s' not found – skipping", material_id)
            continue

        area_sqft: float = area_info["area_sqft"]
        coverage = material.get("coverage_per_unit", 1.0)
        unit = material.get("unit", "unit")

        mat_rate = material.get("material_rate_per_unit", material.get("material_rate", 0.0))
        lab_rate = material.get("labor_rate_per_unit", material.get("labor_rate", 0.0))

        region_overrides = overrides.get(material_id, {})
        mat_rate = region_overrides.get("material_rate", mat_rate)
        lab_rate = region_overrides.get("labor_rate", lab_rate)

        quantity = round(area_sqft / coverage, 2) if coverage else 0.0
        material_cost = round(quantity * mat_rate, 2)
        labor_cost = round(quantity * lab_rate, 2)
        total = round(material_cost + labor_cost, 2)

        line_items.append({
            "region": region_label,
            "material_id": material_id,
            "material_name": material.get("name", material_id),
            "area_sqft": area_sqft,
            "quantity": quantity,
            "unit": unit,
            "material_rate": mat_rate,
            "labor_rate": lab_rate,
            "material_cost": material_cost,
            "labor_cost": labor_cost,
            "total_cost": total,
        })

        subtotal_material += material_cost
        subtotal_labor += labor_cost

    subtotal = round(subtotal_material + subtotal_labor, 2)
    wastage_amount = round(subtotal * (wastage_percent / 100), 2)
    grand_total = round(subtotal + wastage_amount, 2)

    return {
        "line_items": line_items,
        "subtotal_material": round(subtotal_material, 2),
        "subtotal_labor": round(subtotal_labor, 2),
        "grand_total": grand_total,
        "wastage_percent": wastage_percent,
    }
