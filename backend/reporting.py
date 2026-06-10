"""
PDF Report Generator — Unified AI Facial Intelligence System
Generates professional biometric + session analysis reports using ReportLab.
"""
import os
import time
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    HRFlowable, KeepTogether, PageBreak
)


# ── Colour palette ──────────────────────────────────────────────────
_DARK_BG     = colors.Color(0.10, 0.12, 0.15)
_HEADER_BG   = colors.Color(0.12, 0.45, 0.85)
_ROW_EVEN    = colors.Color(0.94, 0.96, 0.98)
_ROW_ODD     = colors.whitesmoke
_BORDER      = colors.Color(0.78, 0.80, 0.82)
_TEXT_MUTED   = colors.Color(0.45, 0.45, 0.50)
_GREEN       = colors.Color(0.13, 0.77, 0.37)
_RED         = colors.Color(0.90, 0.22, 0.21)
_ORANGE      = colors.Color(0.95, 0.55, 0.10)


class ReportGenerator:
    """Generate professional PDF reports from session data."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    #  PUBLIC API
    # ----------------------------------------------------------------
    def create_report(self, session_data: Dict[str, Any]) -> str:
        """
        Build a PDF from *session_data* and return the file path.

        Expected keys in session_data:
            total_frames   (int)
            avg_fps        (float)
            session_start  (str ISO / epoch)
            session_end    (str ISO / epoch)
            logs           (list[dict])   — per-frame log entries
            face_logs      (list[dict])   — enriched per-face snapshots
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"report_{timestamp}.pdf")

        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
        )

        styles = self._build_styles()
        story: list = []

        # ── Title ────────────────────────────────────────────────────
        story.append(Paragraph(
            "AI Facial Intelligence System — Session Report", styles["Title"]
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            styles["Subtitle"],
        ))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(
            width="100%", thickness=1.2,
            color=_HEADER_BG, spaceAfter=14,
        ))

        face_logs: list = session_data.get("face_logs", [])
        logs: list = session_data.get("logs", [])

        # ── 1. Biometric Summary ─────────────────────────────────────
        story.append(Paragraph("📄  Biometric Summary", styles["SectionHead"]))
        story.append(Spacer(1, 6))
        story.extend(self._biometric_summary(face_logs, styles))
        story.append(Spacer(1, 16))

        # ── 2. Session Metrics ───────────────────────────────────────
        story.append(Paragraph("[STATS]  Session Metrics", styles["SectionHead"]))
        story.append(Spacer(1, 6))
        story.extend(self._session_metrics(session_data, face_logs, styles))
        story.append(Spacer(1, 16))

        # ── 3. Security Summary ──────────────────────────────────────
        story.append(Paragraph("🛡️  Security / Liveness", styles["SectionHead"]))
        story.append(Spacer(1, 6))
        story.extend(self._security_summary(face_logs, styles))
        story.append(Spacer(1, 16))

        # ── 4. Detection Log (last 30 entries) ──────────────────────
        story.append(Paragraph("📋  Detection Log (Recent)", styles["SectionHead"]))
        story.append(Spacer(1, 6))
        story.extend(self._detection_log_table(logs, styles))
        story.append(Spacer(1, 16))

        # ── Footer ──────────────────────────────────────────────────
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=_BORDER, spaceBefore=10, spaceAfter=6,
        ))
        story.append(Paragraph(
            f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
            f"Total frames: {session_data.get('total_frames', 0)}  •  "
            f"Avg FPS: {session_data.get('avg_fps', 0):.1f}",
            styles["Footer"],
        ))

        doc.build(story)
        print(f"📄 PDF report saved: {filename}")
        return filename

    # ----------------------------------------------------------------
    #  SECTION BUILDERS
    # ----------------------------------------------------------------
    def _biometric_summary(self, face_logs: list, styles) -> list:
        """Table of unique identities seen during the session."""
        if not face_logs:
            return [Paragraph("No biometric data collected during this session.", styles["Muted"])]

        # Aggregate per person
        persons: Dict[str, Dict] = {}
        for entry in face_logs:
            name = entry.get("name", "Unknown")
            key = f"{name}_{entry.get('face_id', '?')}"
            if key not in persons:
                persons[key] = {
                    "name": name,
                    "face_id": entry.get("face_id", "—"),
                    "confidences": [],
                    "locked": False,
                    "gender": entry.get("registered_gender", "Unknown"),
                }
            conf = entry.get("recognition_confidence", 0)
            if conf:
                persons[key]["confidences"].append(conf)
            if entry.get("recognition_locked"):
                persons[key]["locked"] = True

        # Build table rows
        header = ["Name", "Track ID", "Gender", "Status", "Avg Confidence"]
        rows = [header]
        for p in persons.values():
            avg_conf = (
                f"{sum(p['confidences']) / len(p['confidences']):.1f}%"
                if p["confidences"] else "—"
            )
            status = "[OK] Verified" if p["locked"] or (p["confidences"] and max(p["confidences"]) > 80) else "❓ Unknown"
            rows.append([
                p["name"],
                str(p["face_id"]),
                p["gender"] or "Unknown",
                status,
                avg_conf,
            ])

        return [self._styled_table(rows)]

    def _session_metrics(self, session_data: dict, face_logs: list, styles) -> list:
        """Emotion, stress, blink, attention aggregates."""
        elements: list = []

        # Start / end times
        start = session_data.get("session_start", "—")
        end = session_data.get("session_end", "—")
        elements.append(Paragraph(
            f"<b>Session Start:</b> {start}  &nbsp;&nbsp;&nbsp; <b>Session End:</b> {end}",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 8))

        if not face_logs:
            elements.append(Paragraph("No session metric data available.", styles["Muted"]))
            return elements

        # — Emotion distribution
        emotions = [e.get("emotion", "").lower() for e in face_logs if e.get("emotion")]
        if emotions:
            counts = Counter(emotions)
            total = len(emotions)
            dominant = counts.most_common(1)[0]
            rows = [["Emotion", "Count", "Percentage"]]
            for emo, cnt in counts.most_common():
                rows.append([emo.capitalize(), str(cnt), f"{cnt/total*100:.1f}%"])
            elements.append(Paragraph(
                f"<b>Average Emotional State:</b> {dominant[0].capitalize()} "
                f"({dominant[1]}/{total}  —  {dominant[1]/total*100:.0f}%)",
                styles["Normal"],
            ))
            elements.append(Spacer(1, 4))
            elements.append(self._styled_table(rows, col_widths=[140, 70, 90]))
            elements.append(Spacer(1, 10))

        # — Stress
        stress_levels = [e.get("stress_level", "").upper() for e in face_logs if e.get("stress_level")]
        if stress_levels:
            counts_s = Counter(stress_levels)
            highest = max(counts_s.keys(), key=lambda k: {"LOW":1, "MEDIUM":2, "HIGH":3}.get(k, 0))
            elements.append(Paragraph(
                f"<b>Highest Recorded Stress Level:</b> {highest}",
                styles["Normal"],
            ))
            elements.append(Spacer(1, 6))

        # — Blink frequency
        blink_counts = [e.get("blink_count", 0) for e in face_logs if e.get("blink_count") is not None]
        if blink_counts:
            max_blink = max(blink_counts)
            avg_blink = sum(blink_counts) / len(blink_counts)
            elements.append(Paragraph(
                f"<b>Blink Frequency:</b> avg {avg_blink:.1f} blinks/sample, "
                f"peak {max_blink}",
                styles["Normal"],
            ))
            elements.append(Spacer(1, 6))

        # — Attention summary
        att_levels = [e.get("attention_level", "").upper() for e in face_logs if e.get("attention_level")]
        if att_levels:
            counts_a = Counter(att_levels)
            rows_a = [["Attention Level", "Count", "Percentage"]]
            total_a = len(att_levels)
            for lev, cnt in counts_a.most_common():
                rows_a.append([lev.capitalize(), str(cnt), f"{cnt/total_a*100:.1f}%"])
            elements.append(Paragraph("<b>Attention Level Summary:</b>", styles["Normal"]))
            elements.append(Spacer(1, 4))
            elements.append(self._styled_table(rows_a, col_widths=[140, 70, 90]))

        if not any([emotions, stress_levels, blink_counts, att_levels]):
            elements.append(Paragraph("No detailed metric data collected.", styles["Muted"]))

        return elements

    def _security_summary(self, face_logs: list, styles) -> list:
        """Anti-spoofing / liveness results."""
        spoof_entries = [e for e in face_logs if e.get("anti_spoof_status")]
        if not spoof_entries:
            return [Paragraph("No liveness data collected during this session.", styles["Muted"])]

        live_count = sum(1 for e in spoof_entries if e.get("anti_spoof_live"))
        spoof_count = len(spoof_entries) - live_count

        rows = [["Metric", "Value"]]
        rows.append(["Total Checks", str(len(spoof_entries))])
        rows.append(["[OK] Live Confirmations", str(live_count)])
        rows.append(["🚨 Spoof Detections", str(spoof_count)])
        if spoof_count > 0:
            reasons = [e.get("anti_spoof_reason", "N/A") for e in spoof_entries if not e.get("anti_spoof_live")]
            top_reason = Counter(reasons).most_common(1)[0][0] if reasons else "—"
            rows.append(["Top Failure Reason", top_reason])

        return [self._styled_table(rows, col_widths=[200, 200])]

    def _detection_log_table(self, logs: list, styles) -> list:
        """Last N detection log entries."""
        if not logs:
            return [Paragraph("No detection log entries recorded.", styles["Muted"])]

        recent = logs[-30:]
        header = ["Time", "Faces", "Details"]
        rows = [header]
        for entry in recent:
            rows.append([
                str(entry.get("time", "")),
                str(entry.get("face_count", 0)),
                str(entry.get("attributes", ""))[:60],
            ])
        return [self._styled_table(rows, col_widths=[80, 60, 300])]

    # ----------------------------------------------------------------
    #  HELPERS
    # ----------------------------------------------------------------
    def _styled_table(self, rows: list, col_widths=None) -> Table:
        """Return a nicely styled ReportLab Table."""
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        style_cmds = [
            # Header row
            ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_BG),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING",  (0, 0), (-1, 0), 8),
            # Body rows
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("TOPPADDING",  (0, 1), (-1, -1), 5),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",        (0, 0), (-1, -1), 0.5, _BORDER),
        ]
        # Alternate row colours
        for i in range(1, len(rows)):
            bg = _ROW_EVEN if i % 2 == 0 else _ROW_ODD
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

        table.setStyle(TableStyle(style_cmds))
        return table

    @staticmethod
    def _build_styles():
        """Custom paragraph styles."""
        base = getSampleStyleSheet()
        base.add(ParagraphStyle(
            name="Subtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=_TEXT_MUTED,
            alignment=TA_LEFT,
        ))
        base.add(ParagraphStyle(
            name="SectionHead",
            parent=base["Heading2"],
            fontSize=14,
            textColor=_HEADER_BG,
            spaceAfter=4,
            spaceBefore=10,
        ))
        base.add(ParagraphStyle(
            name="Muted",
            parent=base["Normal"],
            fontSize=10,
            textColor=_TEXT_MUTED,
            spaceAfter=6,
        ))
        base.add(ParagraphStyle(
            name="Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=_TEXT_MUTED,
            alignment=TA_CENTER,
        ))
        return base
