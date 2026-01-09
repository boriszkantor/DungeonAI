"""PDF export functionality for character sheets.

This module provides functionality to export character sheets as PDF files
using the reportlab library.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dnd_manager.models.ecs import ActorEntity


def export_character_sheet(character: ActorEntity) -> bytes:
    """Generate a filled PDF character sheet for a character.
    
    Args:
        character: The ActorEntity to export.
    
    Returns:
        PDF file contents as bytes.
    
    Raises:
        ImportError: If reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as e:
        msg = "reportlab is required for PDF export. Install with: pip install reportlab"
        raise ImportError(msg) from e
    
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#8B2020"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#8B2020"),
        spaceAfter=6,
        spaceBefore=12,
    )
    
    # Character name and basic info
    elements.append(Paragraph(character.name, title_style))
    
    # Basic info table
    basic_info = [
        ["Race:", character.race or "Unknown"],
        ["Level:", character.level_display],
        ["Alignment:", character.alignment or "Unknown"],
        ["Background:", character.background or "None"],
    ]
    
    basic_table = Table(basic_info, colWidths=[1.5 * inch, 4 * inch])
    basic_table.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONT", (1, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(basic_table)
    elements.append(Spacer(1, 0.2 * inch))
    
    # HP, AC, Speed
    combat_stats = [
        ["HP:", f"{character.health.hp_current}/{character.health.hp_max}"],
        ["Armor Class:", str(character.ac)],
        ["Speed:", f"{character.movement.speed if character.movement else 30} ft."],
    ]
    
    combat_table = Table(combat_stats, colWidths=[1.5 * inch, 2 * inch])
    combat_table.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONT", (1, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CCCCCC")),
    ]))
    elements.append(combat_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Ability Scores
    elements.append(Paragraph("Ability Scores", heading_style))
    
    ability_data = [
        ["STR", "DEX", "CON", "INT", "WIS", "CHA"],
        [
            f"{character.stats.strength} ({character.stats.str_mod:+d})",
            f"{character.stats.dexterity} ({character.stats.dex_mod:+d})",
            f"{character.stats.constitution} ({character.stats.con_mod:+d})",
            f"{character.stats.intelligence} ({character.stats.int_mod:+d})",
            f"{character.stats.wisdom} ({character.stats.wis_mod:+d})",
            f"{character.stats.charisma} ({character.stats.cha_mod:+d})",
        ],
    ]
    
    ability_table = Table(ability_data, colWidths=[1 * inch] * 6)
    ability_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONT", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B2020")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(ability_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # Class Features
    if character.class_features and character.class_features.classes:
        elements.append(Paragraph("Classes & Features", heading_style))
        
        for cls in character.class_features.classes:
            class_text = f"<b>{cls.class_name}</b> Level {cls.level}"
            if cls.subclass_name:
                class_text += f" ({cls.subclass_name})"
            elements.append(Paragraph(class_text, styles["Normal"]))
        
        elements.append(Spacer(1, 0.1 * inch))
        
        # Features (limit to first 20 to avoid overflow)
        if character.class_features.features:
            feature_list = character.class_features.features[:20]
            for feature_name in feature_list:
                elements.append(Paragraph(f"• {feature_name}", styles["Normal"]))
            
            if len(character.class_features.features) > 20:
                elements.append(Paragraph(
                    f"... and {len(character.class_features.features) - 20} more features",
                    styles["Italic"]
                ))
        
        elements.append(Spacer(1, 0.2 * inch))
    
    # Inventory
    if character.inventory and character.inventory.items:
        elements.append(Paragraph("Equipment", heading_style))
        
        # Filter equipped items
        equipped = [item for item in character.inventory.items if item.equipped]
        other = [item for item in character.inventory.items if not item.equipped]
        
        if equipped:
            elements.append(Paragraph("<b>Equipped:</b>", styles["Normal"]))
            for item in equipped[:15]:  # Limit to first 15
                item_text = item.name
                if item.quantity > 1:
                    item_text += f" (×{item.quantity})"
                elements.append(Paragraph(f"• {item_text}", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        
        if other:
            elements.append(Paragraph("<b>Inventory:</b>", styles["Normal"]))
            for item in other[:15]:  # Limit to first 15
                item_text = item.name
                if item.quantity > 1:
                    item_text += f" (×{item.quantity})"
                elements.append(Paragraph(f"• {item_text}", styles["Normal"]))
        
        elements.append(Spacer(1, 0.2 * inch))
    
    # Spellcasting
    if character.spellbook:
        elements.append(Paragraph("Spellcasting", heading_style))
        
        spell_info = [
            ["Spellcasting Ability:", str(character.spellbook.spellcasting_ability or "None")],
            ["Spell Save DC:", str(character.spellbook.spell_save_dc)],
            ["Spell Attack Bonus:", f"+{character.spellbook.spell_attack_bonus}"],
        ]
        
        spell_table = Table(spell_info, colWidths=[2 * inch, 2 * inch])
        spell_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONT", (1, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(spell_table)
        elements.append(Spacer(1, 0.1 * inch))
        
        # Cantrips
        if character.spellbook.cantrips:
            elements.append(Paragraph("<b>Cantrips:</b>", styles["Normal"]))
            elements.append(Paragraph(", ".join(character.spellbook.cantrips), styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Prepared spells
        if character.spellbook.spells_prepared:
            elements.append(Paragraph("<b>Prepared Spells:</b>", styles["Normal"]))
            elements.append(Paragraph(", ".join(character.spellbook.spells_prepared[:30]), styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Spell slots
        if character.spellbook.spell_slots:
            elements.append(Paragraph("<b>Spell Slots:</b>", styles["Normal"]))
            slot_text = ", ".join(
                f"Level {level}: {current}/{max_slots}"
                for level, (current, max_slots) in sorted(character.spellbook.spell_slots.items())
            )
            elements.append(Paragraph(slot_text, styles["Normal"]))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def export_character_to_file(character: ActorEntity, filename: str) -> None:
    """Export a character sheet to a PDF file.
    
    Args:
        character: The ActorEntity to export.
        filename: Path to save the PDF file.
    """
    pdf_bytes = export_character_sheet(character)
    
    with open(filename, "wb") as f:
        f.write(pdf_bytes)


__all__ = [
    "export_character_sheet",
    "export_character_to_file",
]
