import asyncio
import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 0.75 * inch


def _generate_report_sync(project_data: dict[str, Any]) -> bytes:
    """Build a PDF report and return it as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=20,
        textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor("#16213e"),
    )
    normal_style = styles["Normal"]

    elements: list[Any] = []

    elements.append(Paragraph("Renovation Cost Estimate Report", title_style))

    project_title = project_data.get("title", "Untitled Project")
    elements.append(Paragraph(project_title, subtitle_style))

    date_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Date: {date_str}", normal_style))
    elements.append(Spacer(1, 20))

    usable_width = PAGE_WIDTH - 2 * MARGIN

    material_assignments = project_data.get("material_assignments", {})
    cost_data = project_data.get("cost_data", {})
    line_items = cost_data.get("line_items", [])

    if material_assignments and line_items:
        elements.append(Paragraph("Selected Materials", subtitle_style))

        mat_header = ["Region", "Material Applied", "Category", "Unit"]
        mat_table_data = [mat_header]
        for item in line_items:
            region = item.get("region", "").replace("_", " ").title()
            mat_table_data.append([
                region,
                item.get("material_name", "—"),
                item.get("unit", "—"),
                f'{item.get("area_sqft", 0):.1f} sqft',
            ])

        mat_col_widths = [usable_width * w for w in [0.25, 0.35, 0.15, 0.25]]
        mat_table = Table(mat_table_data, colWidths=mat_col_widths, repeatRows=1)
        mat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(mat_table)
        elements.append(Spacer(1, 20))

    if line_items:
        elements.append(Paragraph("Cost Breakdown", subtitle_style))

        cell_style = ParagraphStyle(
            "CellText", parent=normal_style, fontSize=8, leading=10
        )

        header = ["Region", "Material", "Area (sqft)", "Qty",
                  "Mat. Rate", "Labor Rate", "Total Cost"]
        table_data = [header]
        for item in line_items:
            region = item.get("region", "").replace("_", " ").title()
            table_data.append([
                Paragraph(region, cell_style),
                Paragraph(item.get("material_name", ""), cell_style),
                f"{item.get('area_sqft', 0):.1f}",
                f"{item.get('quantity', 0):.1f}",
                f"{item.get('material_rate', 0):,.0f}",
                f"{item.get('labor_rate', 0):,.0f}",
                f"{item.get('total_cost', 0):,.0f}",
            ])

        col_widths = [
            usable_width * w
            for w in [0.15, 0.25, 0.12, 0.10, 0.13, 0.12, 0.13]
        ]
        cost_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        cost_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f5")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(cost_table)
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("Summary", subtitle_style))

    subtotal_material = cost_data.get("subtotal_material", 0)
    subtotal_labor = cost_data.get("subtotal_labor", 0)
    wastage_pct = cost_data.get("wastage_percent", 0)
    grand_total = cost_data.get("grand_total", 0)

    summary_data = [
        ["Subtotal (Material)", f"INR {subtotal_material:,.2f}"],
        ["Subtotal (Labor)", f"INR {subtotal_labor:,.2f}"],
        ["Wastage", f"{wastage_pct}%"],
        ["Grand Total", f"INR {grand_total:,.2f}"],
    ]
    summary_table = Table(summary_data, colWidths=[usable_width * 0.5, usable_width * 0.5])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, -1), (-1, -1), 1.5, colors.HexColor("#1a1a2e")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(summary_table)

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#888888"),
        spaceBefore=30,
    )
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Note: Area estimates are approximate and derived from image analysis. "
        "Cost figures are advisory and not legally binding. Actual costs may vary "
        "based on site conditions, contractor rates, and material availability.",
        disclaimer_style,
    ))

    doc.build(elements)
    return buf.getvalue()


async def generate_report(project_data: dict[str, Any]) -> bytes:
    return await asyncio.to_thread(_generate_report_sync, project_data)
