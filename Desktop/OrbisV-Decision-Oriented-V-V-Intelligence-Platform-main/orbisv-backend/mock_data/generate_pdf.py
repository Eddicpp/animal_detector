"""
mock_data/generate_pdf.py
Genera IRIS3_main.pdf — fonte di verita del progetto.
Output: projects/IRIS-3/main/IRIS3_main.pdf

Uso:
    python mock_data/generate_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pathlib import Path
import datetime

# ── Dati mock IRIS-3 ──────────────────────────────────────────────────────────

PROJECT = {
    "name": "IRIS-3",
    "version": "main-v1.0",
    "date": "2025-01-15",
    "customer": "ESA",
    "domain": "Optical Earth Observation Payload",
    "milestone": "CDR",
    "decision": "GO_WITH_CONSTRAINTS",
    "confidence": 78,
}

USER_NEEDS = [
    {"id": "UN-001", "description": "The system shall provide ground imagery with sufficient resolution to identify agricultural field boundaries", "domain": "optical", "priority": "critical"},
    {"id": "UN-002", "description": "The system shall operate continuously across orbital thermal cycles without performance degradation", "domain": "thermal", "priority": "critical"},
    {"id": "UN-003", "description": "The payload shall survive launch vibration and shock loads", "domain": "mechanical", "priority": "critical"},
    {"id": "UN-004", "description": "Imagery data shall be downlinked and processed within acceptable latency", "domain": "data_processing", "priority": "high"},
    {"id": "UN-005", "description": "The system shall maintain pointing accuracy sufficient for image geolocation", "domain": "optical", "priority": "critical"},
    {"id": "UN-006", "description": "Total payload mass shall fit within platform accommodation envelope", "domain": "mechanical", "priority": "high"},
    {"id": "UN-007", "description": "Power consumption shall not exceed platform supply allocation", "domain": "thermal", "priority": "high"},
]

REQUIREMENTS = [
    {"id": "L1-OPT-001", "level": "L1", "domain": "optical",          "description": "GSD ≤ 1.5m at nadir",                                    "acceptance": "GSD measured from calibration target < 1.5m",          "critical": True,  "parent_need": "UN-001", "compliance": 0.92},
    {"id": "L1-OPT-002", "level": "L1", "domain": "optical",          "description": "MTF at Nyquist ≥ 0.40",                                  "acceptance": "MTF measured at Nyquist ≥ 0.40",                       "critical": True,  "parent_need": "UN-001", "compliance": 0.88},
    {"id": "L1-OPT-003", "level": "L1", "domain": "optical",          "description": "SNR ≥ 80 dB at reference radiance",                      "acceptance": "SNR ≥ 80 dB under reference illumination",             "critical": False, "parent_need": "UN-001", "compliance": 0.95},
    {"id": "L1-THM-001", "level": "L1", "domain": "thermal",          "description": "Optical bench ΔT ≤ 15°C across orbital cycle",           "acceptance": "Max ΔT during TVT ≤ 15°C",                            "critical": True,  "parent_need": "UN-002", "compliance": 0.61},
    {"id": "L1-THM-002", "level": "L1", "domain": "thermal",          "description": "Detector temperature 200K ± 5K",                         "acceptance": "Detector temp stability verified in TVT",               "critical": True,  "parent_need": "UN-002", "compliance": 0.84},
    {"id": "L1-MEC-001", "level": "L1", "domain": "mechanical",       "description": "First natural frequency ≥ 80 Hz",                        "acceptance": "Modal analysis and sine sweep confirm f1 ≥ 80 Hz",      "critical": True,  "parent_need": "UN-003", "compliance": 0.93},
    {"id": "L1-MEC-002", "level": "L1", "domain": "mechanical",       "description": "Withstand random vibration 14.1 Grms (ECSS-E-ST-10-03)", "acceptance": "Random vibe test — no structural failure",             "critical": True,  "parent_need": "UN-003", "compliance": 0.90},
    {"id": "L1-MEC-003", "level": "L1", "domain": "mechanical",       "description": "Shock response per platform ICD",                         "acceptance": "SRS measured ≤ ICD limits at all frequencies",         "critical": False, "parent_need": "UN-003", "compliance": None},
    {"id": "L1-DPR-001", "level": "L1", "domain": "data_processing",  "description": "On-board compression ratio ≥ 4:1, PSNR ≥ 40 dB",        "acceptance": "Compression ratio and PSNR verified on reference set", "critical": False, "parent_need": "UN-004", "compliance": 0.68},
    {"id": "L1-DPR-002", "level": "L1", "domain": "data_processing",  "description": "End-to-end latency ≤ 500ms",                             "acceptance": "Latency measured in HIL test ≤ 500ms",                 "critical": False, "parent_need": "UN-004", "compliance": 0.82},
    {"id": "L1-ATT-001", "level": "L1", "domain": "optical",          "description": "Pointing knowledge error ≤ 0.005° (3σ)",                 "acceptance": "PKE verified by star tracker calibration",             "critical": True,  "parent_need": "UN-005", "compliance": 0.94},
    {"id": "L1-MAS-001", "level": "L1", "domain": "mechanical",       "description": "Total mass including harness ≤ 35 kg",                   "acceptance": "Mass measured at AIT ≤ 35 kg",                         "critical": False, "parent_need": "UN-006", "compliance": 0.97},
    {"id": "L1-PWR-001", "level": "L1", "domain": "thermal",          "description": "Peak power ≤ 180W during imaging mode",                  "acceptance": "Power measured during functional test ≤ 180W",         "critical": False, "parent_need": "UN-007", "compliance": 0.91},
    {"id": "L1-PWR-002", "level": "L1", "domain": "thermal",          "description": "Average power during eclipse ≤ 45W",                     "acceptance": "Average power during simulated eclipse ≤ 45W",         "critical": False, "parent_need": "UN-007", "compliance": 0.88},
]

SUBSYSTEMS = [
    {"id": "optical",     "score": 92, "status": "ok",   "detail": "MTF 0.42 > threshold 0.40",           "mass_kg": 8.2,  "volume_m3": 0.018},
    {"id": "thermal",     "score": 61, "status": "warn", "detail": "ΔT 14.8°C — margin < 5%",             "mass_kg": 3.1,  "volume_m3": 0.004},
    {"id": "mechanical",  "score": 88, "status": "ok",   "detail": "1st mode 112 Hz > 80 Hz req",         "mass_kg": 12.4, "volume_m3": 0.031},
    {"id": "structure",   "score": 91, "status": "ok",   "detail": "Mass 33.8 kg / budget 35 kg",         "mass_kg": 6.8,  "volume_m3": 0.022},
    {"id": "solar",       "score": 89, "status": "ok",   "detail": "Power margin +18%",                   "mass_kg": 1.8,  "volume_m3": 0.006},
    {"id": "startracker", "score": 67, "status": "warn", "detail": "Stray light analysis pending",        "mass_kg": 0.9,  "volume_m3": 0.001},
    {"id": "dataproc",    "score": 71, "status": "warn", "detail": "Compression ratio TBC",              "mass_kg": 1.0,  "volume_m3": 0.002},
]

EVIDENCES = [
    {"id": "SIM-OPT-001", "req": "L1-OPT-001", "type": "simulation", "tool": "MATLAB Optical Model",    "result": "GSD 1.23m",           "verdict": "PASS",        "date": "2024-09-12"},
    {"id": "TST-OPT-001", "req": "L1-OPT-001", "type": "test",       "tool": "Lab calibration target",  "result": "GSD 1.31m",           "verdict": "PASS",        "date": "2024-11-03"},
    {"id": "SIM-OPT-002", "req": "L1-OPT-002", "type": "simulation", "tool": "MATLAB Optical Model",    "result": "MTF 0.42",            "verdict": "PASS",        "date": "2024-09-15"},
    {"id": "SIM-OPT-003", "req": "L1-OPT-003", "type": "simulation", "tool": "STK Radiometric Model",   "result": "SNR 84.2 dB",         "verdict": "PASS",        "date": "2024-10-01"},
    {"id": "TST-OPT-003", "req": "L1-OPT-003", "type": "test",       "tool": "Integrating sphere",      "result": "SNR 81.7 dB",         "verdict": "PASS",        "date": "2024-11-20"},
    {"id": "SIM-THM-001", "req": "L1-THM-001", "type": "simulation", "tool": "MATLAB Thermal Model",    "result": "ΔT 14.8°C",           "verdict": "CONDITIONAL", "date": "2024-08-20"},
    {"id": "SIM-THM-002", "req": "L1-THM-002", "type": "simulation", "tool": "MATLAB Thermal Model",    "result": "198.3K ± 3.1K",       "verdict": "PASS",        "date": "2024-08-22"},
    {"id": "TST-THM-002", "req": "L1-THM-002", "type": "test",       "tool": "Thermal Vacuum Chamber",  "result": "199.1K ± 4.2K",       "verdict": "PASS",        "date": "2024-12-05"},
    {"id": "SIM-MEC-001", "req": "L1-MEC-001", "type": "simulation", "tool": "FEM NASTRAN",             "result": "f1 = 112 Hz",         "verdict": "PASS",        "date": "2024-07-15"},
    {"id": "TST-MEC-001", "req": "L1-MEC-001", "type": "test",       "tool": "Vibration test facility", "result": "f1 = 108 Hz",         "verdict": "PASS",        "date": "2024-10-28"},
    {"id": "TST-MEC-002", "req": "L1-MEC-002", "type": "test",       "tool": "Vibration test facility", "result": "14.1 Grms — PASS",    "verdict": "PASS",        "date": "2024-10-30"},
    {"id": "SIM-DPR-001", "req": "L1-DPR-001", "type": "simulation", "tool": "Python pipeline",         "result": "Ratio 3.8:1 PSNR 39.2dB","verdict": "CONDITIONAL","date": "2024-11-10"},
    {"id": "TST-DPR-002", "req": "L1-DPR-002", "type": "test",       "tool": "HIL test bench",          "result": "Latency 420ms",       "verdict": "PASS",        "date": "2024-12-01"},
    {"id": "TST-ATT-001", "req": "L1-ATT-001", "type": "test",       "tool": "Star tracker calib bench","result": "PKE 0.0028°",         "verdict": "PASS",        "date": "2024-09-30"},
    {"id": "TST-MAS-001", "req": "L1-MAS-001", "type": "test",       "tool": "Precision scale AIT",     "result": "34.2 kg",             "verdict": "PASS",        "date": "2024-12-10"},
    {"id": "TST-PWR-001", "req": "L1-PWR-001", "type": "test",       "tool": "Power analyzer AIT",      "result": "162W peak",           "verdict": "PASS",        "date": "2024-11-25"},
    {"id": "SIM-PWR-002", "req": "L1-PWR-002", "type": "simulation", "tool": "STK Power Budget",        "result": "38W average",         "verdict": "PASS",        "date": "2024-10-15"},
]

NCRS = [
    {"id": "NCR-2024-031", "title": "Thermal gradient margin < 5% on optical bench",       "severity": "major", "req": "L1-THM-001", "owner": "Thermal team",        "opened": "2024-11-15"},
    {"id": "NCR-2024-038", "title": "On-board compression ratio 3.8:1 below 4:1 req",      "severity": "minor", "req": "L1-DPR-001", "owner": "Data processing team", "opened": "2024-11-12"},
    {"id": "NCR-2024-041", "title": "Shock test not yet performed — evidence missing",      "severity": "major", "req": "L1-MEC-003", "owner": "AIT team",            "opened": "2024-12-02"},
]

ECSS_OPEN = [
    {"id": "THM-V-001", "standard": "ECSS-E-ST-31C",      "clause": "§5.4", "activity": "Thermal balance test to validate TMM",                        "status": "open"},
    {"id": "THM-V-002", "standard": "ECSS-E-ST-31C",      "clause": "§5.5", "activity": "Thermal vacuum test at qualification margins (4 cycles min)", "status": "open"},
    {"id": "MEC-V-004", "standard": "ECSS-E-ST-32C Rev.1","clause": "§9",   "activity": "Shock test per launcher SRS ICD",                             "status": "open"},
    {"id": "STR-V-003", "standard": "ECSS-E-ST-32C Rev.1","clause": "§10",  "activity": "Acoustic test at qualification level +3dB",                   "status": "pending"},
    {"id": "DPR-V-001", "standard": "ECSS-E-ST-40C",      "clause": "§5.3", "activity": "On-board compression algorithm validation",                   "status": "open"},
]

# ── Builder ───────────────────────────────────────────────────────────────────

def build_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("title",   fontSize=16, fontName="Helvetica-Bold", spaceAfter=4)
    style_h1      = ParagraphStyle("h1",      fontSize=13, fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=12, textColor=colors.HexColor("#1d4ed8"))
    style_h2      = ParagraphStyle("h2",      fontSize=10, fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=6)
    style_body    = ParagraphStyle("body",    fontSize=8,  fontName="Helvetica",      spaceAfter=2, leading=12)
    style_mono    = ParagraphStyle("mono",    fontSize=7.5,fontName="Courier",        spaceAfter=1)
    style_warning = ParagraphStyle("warning", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#b45309"))
    style_fail    = ParagraphStyle("fail",    fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#dc2626"))
    style_ok      = ParagraphStyle("ok",      fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#16a34a"))

    story = []

    # ── HEADER ──
    story.append(Paragraph(f"PROJECT DOCUMENT — {PROJECT['name']}", style_title))
    story.append(Paragraph(f"Version: {PROJECT['version']}  |  Date: {PROJECT['date']}  |  Milestone: {PROJECT['milestone']}", style_body))
    story.append(Paragraph(f"Customer: {PROJECT['customer']}  |  Domain: {PROJECT['domain']}", style_body))
    story.append(Paragraph(f"Decision: {PROJECT['decision']}  |  Confidence: {PROJECT['confidence']}%", style_body))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", style_body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1d4ed8"), spaceAfter=8))

    # ── USER NEEDS ──
    story.append(Paragraph("1. USER NEEDS", style_h1))
    for n in USER_NEEDS:
        story.append(Paragraph(f"<b>{n['id']}</b> [{n['priority'].upper()}] — {n['description']}", style_body))
    story.append(Spacer(1, 6))

    # ── REQUIREMENTS ──
    story.append(Paragraph("2. REQUIREMENTS", style_h1))
    current_domain = None
    for r in REQUIREMENTS:
        if r["domain"] != current_domain:
            current_domain = r["domain"]
            story.append(Paragraph(f"Domain: {current_domain.upper()}", style_h2))
        compliance_str = f"{int(r['compliance']*100)}%" if r["compliance"] is not None else "N/A"
        critical_str   = " [CRITICAL]" if r["critical"] else ""
        style_c = style_fail if (r["compliance"] or 1) < 0.65 else style_warning if (r["compliance"] or 1) < 0.80 else style_body
        story.append(Paragraph(
            f"<b>{r['id']}</b>{critical_str} | Need: {r['parent_need']} | Compliance: {compliance_str}",
            style_c
        ))
        story.append(Paragraph(f"  Req: {r['description']}", style_mono))
        story.append(Paragraph(f"  Acceptance: {r['acceptance']}", style_mono))
    story.append(Spacer(1, 6))

    # ── SUBSYSTEMS ──
    story.append(Paragraph("3. SUBSYSTEM STATUS", style_h1))
    sub_data = [["Subsystem", "Score", "Status", "Detail", "Mass (kg)", "Volume (m³)"]]
    for s in SUBSYSTEMS:
        score_str = f"{s['score']}%"
        sub_data.append([s["id"], score_str, s["status"].upper(), s["detail"], str(s["mass_kg"]), str(s["volume_m3"])])
    total_mass = sum(s["mass_kg"] for s in SUBSYSTEMS)
    sub_data.append(["TOTAL", "", "", "", f"{total_mass:.1f} kg", ""])

    t = Table(sub_data, colWidths=[3*cm, 1.5*cm, 2*cm, 6*cm, 2*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cbd5e1")),
        ("BACKGROUND",  (0,-1), (-1,-1), colors.HexColor("#f1f5f9")),
        ("FONTNAME",    (0,-1), (-1,-1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # ── V&V EVIDENCE ──
    story.append(Paragraph("4. V&V EVIDENCE", style_h1))
    ev_data = [["Evidence ID", "Requirement", "Type", "Tool", "Result", "Verdict", "Date"]]
    for e in EVIDENCES:
        ev_data.append([e["id"], e["req"], e["type"], e["tool"], e["result"], e["verdict"], e["date"]])
    t2 = Table(ev_data, colWidths=[2.5*cm, 2.5*cm, 2*cm, 3.5*cm, 3*cm, 2*cm, 2*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 6.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    # ── OPEN NCRs ──
    story.append(Paragraph("5. OPEN NON-CONFORMANCE REPORTS", style_h1))
    for n in NCRS:
        sty = style_fail if n["severity"] == "major" else style_warning
        story.append(Paragraph(
            f"<b>{n['id']}</b> [{n['severity'].upper()}] — {n['title']}", sty
        ))
        story.append(Paragraph(f"  Requirement: {n['req']} | Owner: {n['owner']} | Opened: {n['opened']}", style_mono))
    story.append(Spacer(1, 6))

    # ── ECSS OPEN ITEMS ──
    story.append(Paragraph("6. ECSS OPEN VALIDATION ITEMS", style_h1))
    for e in ECSS_OPEN:
        sty = style_fail if e["status"] == "open" else style_warning
        story.append(Paragraph(
            f"<b>{e['id']}</b> | {e['standard']} {e['clause']} | [{e['status'].upper()}]", sty
        ))
        story.append(Paragraph(f"  Activity: {e['activity']}", style_mono))
    story.append(Spacer(1, 6))

    # ── SYSTEM CONSTRAINTS (per LLM) ──
    story.append(Paragraph("7. SYSTEM CONSTRAINTS", style_h1))
    story.append(Paragraph("Mass budget: 35.0 kg total | Current: 34.2 kg | Margin: 0.8 kg", style_body))
    story.append(Paragraph("Power budget imaging: 180W max | Current peak: 162W | Margin: 18W", style_body))
    story.append(Paragraph("Thermal ΔT limit: 15°C | Current: 14.8°C | Margin: 0.2°C [CRITICAL]", style_fail))
    story.append(Paragraph("First natural frequency: 80 Hz min | Current: 108 Hz | Margin: 28 Hz", style_ok))
    story.append(Paragraph("Volume envelope: platform ICD defines max 0.10 m³ | Current: 0.084 m³ | Margin: 0.016 m³", style_body))

    doc.build(story)
    print(f"PDF generato: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "projects" / "IRIS-3" / "main" / "IRIS3_main.pdf"
    build_pdf(out)