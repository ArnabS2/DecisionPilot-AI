"""
report_generator.py
DecisionPilot AI — Report Generator Module

Consumes structured output from analyzer.py and scorer.py to produce
formatted decision reports saved to outputs/reports/.
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional rich-PDF support via reportlab.  Falls back to plain-text if absent.
# ---------------------------------------------------------------------------
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPORTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    decision_context: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    recommendations: dict[str, Any] | None = None,
    simulations: dict[str, Any] | None = None,
    fmt: str = "pdf",
) -> str:
    """
    Generate a decision report and save it to outputs/reports/.

    Parameters
    ----------
    decision_context : str
        The original decision question / scenario entered by the user.
    analysis : dict
        Structured output from analyzer.py  — expected keys:
        ``summary``, ``options`` (list), ``risks`` (list), ``factors`` (list).
    scores : dict
        Structured output from scorer.py — expected keys:
        ``overall_score`` (float 0-100), ``confidence`` (float 0-1),
        ``dimension_scores`` (dict[str, float]), ``risk_level`` (str).
    recommendations : dict, optional
        Output from recommendation.py — expected key ``recommended_option`` (str),
        ``rationale`` (str), ``action_steps`` (list[str]).
    simulations : dict, optional
        Output from future_simulator.py — expected key ``scenarios`` (list of dicts
        with keys ``label``, ``probability``, ``outcome``).
    fmt : str
        ``"pdf"`` (default) or ``"txt"``.

    Returns
    -------
    str
        Absolute path to the saved report file.
    """
    fmt = fmt.lower().strip()
    if fmt == "pdf" and _REPORTLAB_AVAILABLE:
        return _generate_pdf(
            decision_context, analysis, scores, recommendations, simulations
        )
    return _generate_txt(
        decision_context, analysis, scores, recommendations, simulations
    )


def report_to_markdown(
    decision_context: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    recommendations: dict[str, Any] | None = None,
    simulations: dict[str, Any] | None = None,
) -> str:
    """
    Return the report as a Markdown string (used by Streamlit for in-app preview).
    Nothing is written to disk.
    """
    return _build_markdown(decision_context, analysis, scores, recommendations, simulations)


# ---------------------------------------------------------------------------
# Internal helpers — shared data assembly
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _safe_filename(label: str) -> str:
    """Sanitise decision context into a filesystem-safe slug."""
    slug = "".join(c if c.isalnum() or c in " _-" else "" for c in label)
    slug = slug.strip().replace(" ", "_")[:48]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"decision_report_{slug}_{ts}"


def _risk_colour(risk_level: str) -> str:
    mapping = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c", "critical": "#8e44ad"}
    return mapping.get(str(risk_level).lower(), "#555555")


# ---------------------------------------------------------------------------
# Markdown builder (shared by both in-app preview and .txt export)
# ---------------------------------------------------------------------------

def _build_markdown(
    decision_context: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    recommendations: dict[str, Any] | None,
    simulations: dict[str, Any] | None,
) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(f"{text}\n")

    def hr() -> None:
        lines.append("---\n")

    # ── Header ──────────────────────────────────────────────────────────────
    h(1, "🧭 DecisionPilot AI — Decision Report")
    p(f"*Generated: {_timestamp()}*")
    hr()

    # ── Decision context ────────────────────────────────────────────────────
    h(2, "Decision Context")
    p(decision_context)
    hr()

    # ── Scores ──────────────────────────────────────────────────────────────
    h(2, "Decision Score Summary")
    overall = scores.get("overall_score", 0)
    confidence = scores.get("confidence", 0)
    risk_level = scores.get("risk_level", "unknown")
    p(f"| Metric | Value |")
    p(f"|--------|-------|")
    p(f"| **Overall Score** | {overall:.1f} / 100 |")
    p(f"| **Confidence** | {confidence * 100:.1f}% |")
    p(f"| **Risk Level** | {risk_level.upper()} |")

    dim_scores: dict = scores.get("dimension_scores", {})
    if dim_scores:
        p("")
        h(3, "Dimension Scores")
        p("| Dimension | Score |")
        p("|-----------|-------|")
        for dim, val in dim_scores.items():
            bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
            p(f"| {dim} | {bar} {val:.1f} |")
    hr()

    # ── Analysis ────────────────────────────────────────────────────────────
    h(2, "Analysis Summary")
    summary = analysis.get("summary", "No summary provided.")
    p(summary)

    options: list = analysis.get("options", [])
    if options:
        h(3, "Options Evaluated")
        for i, opt in enumerate(options, 1):
            if isinstance(opt, dict):
                p(f"**{i}. {opt.get('name', f'Option {i}')}**  ")
                p(opt.get("description", ""))
            else:
                p(f"{i}. {opt}")

    factors: list = analysis.get("factors", [])
    if factors:
        h(3, "Key Decision Factors")
        for f_ in factors:
            if isinstance(f_, dict):
                p(f"- **{f_.get('name', '')}**: {f_.get('detail', '')}")
            else:
                p(f"- {f_}")

    risks: list = analysis.get("risks", [])
    if risks:
        h(3, "Risk Factors Identified")
        for r in risks:
            if isinstance(r, dict):
                severity = r.get("severity", "").upper()
                p(f"- ⚠️ **[{severity}]** {r.get('description', r)}")
            else:
                p(f"- ⚠️ {r}")
    hr()

    # ── Recommendations ─────────────────────────────────────────────────────
    if recommendations:
        h(2, "Recommendation")
        rec_option = recommendations.get("recommended_option", "")
        rationale = recommendations.get("rationale", "")
        steps: list = recommendations.get("action_steps", [])
        if rec_option:
            p(f"**Recommended Option:** {rec_option}")
        if rationale:
            p(f"\n{rationale}")
        if steps:
            h(3, "Action Steps")
            for i, step in enumerate(steps, 1):
                p(f"{i}. {step}")
        hr()

    # ── Future Simulations ──────────────────────────────────────────────────
    if simulations:
        scenarios: list = simulations.get("scenarios", [])
        if scenarios:
            h(2, "Future Scenario Simulations")
            p("| Scenario | Probability | Outcome |")
            p("|----------|-------------|---------|")
            for sc in scenarios:
                label = sc.get("label", "—")
                prob = sc.get("probability", 0)
                outcome = sc.get("outcome", "—")
                p(f"| {label} | {prob * 100:.0f}% | {outcome} |")
            hr()

    # ── Footer ───────────────────────────────────────────────────────────────
    p("*This report was generated by DecisionPilot AI. It is intended as a decision support tool, not a substitute for professional advice.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain-text (.txt) export
# ---------------------------------------------------------------------------

def _generate_txt(
    decision_context: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    recommendations: dict[str, Any] | None,
    simulations: dict[str, Any] | None,
) -> str:
    md = _build_markdown(decision_context, analysis, scores, recommendations, simulations)
    # Strip markdown symbols for clean plain text
    import re
    txt = re.sub(r"#+\s", "", md)
    txt = re.sub(r"\*\*(.+?)\*\*", r"\1", txt)
    txt = re.sub(r"\*(.+?)\*", r"\1", txt)
    txt = re.sub(r"\|", " | ", txt)

    slug = _safe_filename(decision_context)
    out_path = _REPORTS_DIR / f"{slug}.txt"
    out_path.write_text(txt, encoding="utf-8")
    return str(out_path)


# ---------------------------------------------------------------------------
# PDF export via ReportLab
# ---------------------------------------------------------------------------

def _generate_pdf(
    decision_context: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    recommendations: dict[str, Any] | None,
    simulations: dict[str, Any] | None,
) -> str:
    slug = _safe_filename(decision_context)
    out_path = _REPORTS_DIR / f"{slug}.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story: list = []

    # ── Custom styles ────────────────────────────────────────────────────────
    NAVY = colors.HexColor("#0d1b2a")
    ACCENT = colors.HexColor("#2563eb")
    MUTED = colors.HexColor("#64748b")
    LIGHT_BG = colors.HexColor("#f1f5f9")

    title_style = ParagraphStyle(
        "DPTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=NAVY,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "DPSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=MUTED,
        spaceAfter=16,
    )
    h2_style = ParagraphStyle(
        "DPH2",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h3_style = ParagraphStyle(
        "DPH3",
        parent=styles["Heading3"],
        fontSize=11,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "DPBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=NAVY,
        leading=15,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "DPBullet",
        parent=body_style,
        leftIndent=16,
        bulletIndent=4,
        spaceAfter=3,
    )
    footer_style = ParagraphStyle(
        "DPFooter",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MUTED,
        spaceBefore=20,
    )

    def add_hr():
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1")))
        story.append(Spacer(1, 8))

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("DecisionPilot AI", title_style))
    story.append(Paragraph(f"Decision Report &nbsp;·&nbsp; {_timestamp()}", subtitle_style))
    add_hr()

    # ── Decision context ─────────────────────────────────────────────────────
    story.append(Paragraph("Decision Context", h2_style))
    story.append(Paragraph(decision_context, body_style))
    add_hr()

    # ── Scores table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Decision Score Summary", h2_style))
    overall = scores.get("overall_score", 0)
    confidence = scores.get("confidence", 0)
    risk_level = scores.get("risk_level", "unknown")

    score_data = [
        ["Metric", "Value"],
        ["Overall Score", f"{overall:.1f} / 100"],
        ["Confidence", f"{confidence * 100:.1f}%"],
        ["Risk Level", risk_level.upper()],
    ]
    score_table = Table(score_data, colWidths=[8 * cm, 8 * cm])
    score_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(score_table)

    dim_scores: dict = scores.get("dimension_scores", {})
    if dim_scores:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Dimension Scores", h3_style))
        dim_data = [["Dimension", "Score", "Bar"]]
        for dim, val in dim_scores.items():
            bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
            dim_data.append([dim, f"{val:.1f}", bar])
        dim_table = Table(dim_data, colWidths=[6 * cm, 3 * cm, 7 * cm])
        dim_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(dim_table)
    add_hr()

    # ── Analysis ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Analysis Summary", h2_style))
    story.append(Paragraph(analysis.get("summary", "No summary provided."), body_style))

    options: list = analysis.get("options", [])
    if options:
        story.append(Paragraph("Options Evaluated", h3_style))
        for i, opt in enumerate(options, 1):
            if isinstance(opt, dict):
                story.append(Paragraph(f"<b>{i}. {opt.get('name', f'Option {i}')}</b>", bullet_style))
                story.append(Paragraph(opt.get("description", ""), bullet_style))
            else:
                story.append(Paragraph(f"{i}. {opt}", bullet_style))

    factors: list = analysis.get("factors", [])
    if factors:
        story.append(Paragraph("Key Decision Factors", h3_style))
        for f_ in factors:
            if isinstance(f_, dict):
                story.append(Paragraph(f"• <b>{f_.get('name', '')}</b>: {f_.get('detail', '')}", bullet_style))
            else:
                story.append(Paragraph(f"• {f_}", bullet_style))

    risks: list = analysis.get("risks", [])
    if risks:
        story.append(Paragraph("Risk Factors Identified", h3_style))
        for r in risks:
            if isinstance(r, dict):
                severity = r.get("severity", "").upper()
                story.append(Paragraph(f"⚠ <b>[{severity}]</b> {r.get('description', r)}", bullet_style))
            else:
                story.append(Paragraph(f"⚠ {r}", bullet_style))
    add_hr()

    # ── Recommendations ──────────────────────────────────────────────────────
    if recommendations:
        story.append(Paragraph("Recommendation", h2_style))
        rec_option = recommendations.get("recommended_option", "")
        rationale = recommendations.get("rationale", "")
        steps: list = recommendations.get("action_steps", [])
        if rec_option:
            story.append(Paragraph(f"<b>Recommended Option:</b> {rec_option}", body_style))
        if rationale:
            story.append(Paragraph(rationale, body_style))
        if steps:
            story.append(Paragraph("Action Steps", h3_style))
            for i, step in enumerate(steps, 1):
                story.append(Paragraph(f"{i}. {step}", bullet_style))
        add_hr()

    # ── Simulations ──────────────────────────────────────────────────────────
    if simulations:
        scenarios: list = simulations.get("scenarios", [])
        if scenarios:
            story.append(Paragraph("Future Scenario Simulations", h2_style))
            sim_data = [["Scenario", "Probability", "Outcome"]]
            for sc in scenarios:
                sim_data.append([
                    sc.get("label", "—"),
                    f"{sc.get('probability', 0) * 100:.0f}%",
                    sc.get("outcome", "—"),
                ])
            sim_table = Table(sim_data, colWidths=[5 * cm, 3.5 * cm, 7.5 * cm])
            sim_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            story.append(sim_table)
            add_hr()

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(
        Paragraph(
            "This report was generated by DecisionPilot AI. "
            "It is intended as a decision support tool, not a substitute for professional advice.",
            footer_style,
        )
    )

    doc.build(story)
    return str(out_path)
