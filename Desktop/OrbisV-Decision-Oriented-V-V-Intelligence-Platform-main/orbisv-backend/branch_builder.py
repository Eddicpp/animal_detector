"""
branch_builder.py
Crea un branch del progetto applicando il risultato cascade del LLM.
Genera: meta.json, diff.json, cascade_result.json, PDF branch.
"""

import json
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm


# Stato base subsistemi IRIS-3 (main)
BASE_SUBSYSTEMS = {
    "optical":     {"score": 92, "status": "ok",   "detail": "MTF 0.42 > threshold 0.40",    "mass_kg": 8.2,  "volume_m3": 0.018},
    "thermal":     {"score": 61, "status": "warn",  "detail": "ΔT 14.8°C — margin < 5%",     "mass_kg": 3.1,  "volume_m3": 0.004},
    "mechanical":  {"score": 88, "status": "ok",   "detail": "1st mode 112 Hz > 80 Hz req",  "mass_kg": 12.4, "volume_m3": 0.031},
    "structure":   {"score": 91, "status": "ok",   "detail": "Mass 33.8 kg / budget 35 kg",  "mass_kg": 6.8,  "volume_m3": 0.022},
    "solar":       {"score": 89, "status": "ok",   "detail": "Power margin +18%",             "mass_kg": 1.8,  "volume_m3": 0.006},
    "startracker": {"score": 67, "status": "warn",  "detail": "Stray light analysis pending", "mass_kg": 0.9,  "volume_m3": 0.001},
    "dataproc":    {"score": 71, "status": "warn",  "detail": "Compression ratio TBC",        "mass_kg": 1.0,  "volume_m3": 0.002},
}

BASE_DECISION   = "GO_WITH_CONSTRAINTS"
BASE_CONFIDENCE = 78


def apply_cascade_to_subsystems(base: dict, cascade_result: dict) -> dict:
    """
    Aggiorna subsistemi in base agli issues identificati dal LLM.
    """
    updated = {k: dict(v) for k, v in base.items()}
    affected = cascade_result.get("affected_subsystems", [])
    issues   = cascade_result.get("cascade_issues", [])

    for sub_id in affected:
        if sub_id in updated:
            # trova severity peggiore per questo subsistema
            sub_issues = [i for i in issues if i.get("subsystem") == sub_id]
            severities = [i.get("severity", "minor") for i in sub_issues]

            if "critical" in severities:
                updated[sub_id]["status"] = "fail"
                updated[sub_id]["score"]  = max(0, updated[sub_id]["score"] - 35)
            elif "major" in severities:
                updated[sub_id]["status"] = "warn"
                updated[sub_id]["score"]  = max(0, updated[sub_id]["score"] - 20)
            else:
                updated[sub_id]["score"]  = max(0, updated[sub_id]["score"] - 8)

            # aggiungi nota dal LLM
            if sub_issues:
                updated[sub_id]["detail"] = sub_issues[0].get("title", updated[sub_id]["detail"])

    return updated


def compute_diff(base_subs: dict, new_subs: dict,
                 base_decision: str, new_decision: str,
                 base_conf: int, new_conf: int,
                 cascade_result: dict) -> dict:
    """
    Calcola diff tra main e branch.
    """
    subsystem_diffs = {}
    for sid, base_data in base_subs.items():
        new_data = new_subs.get(sid, base_data)
        if base_data != new_data:
            subsystem_diffs[sid] = {
                "before": base_data,
                "after":  new_data,
                "score_delta": new_data["score"] - base_data["score"],
                "status_changed": base_data["status"] != new_data["status"],
            }

    return {
        "subsystems":            subsystem_diffs,
        "decision_before":       base_decision,
        "decision_after":        new_decision,
        "confidence_before":     base_conf,
        "confidence_after":      new_conf,
        "new_cascade_issues":    cascade_result.get("cascade_issues", []),
        "new_missing_tests":     cascade_result.get("missing_tests", []),
        "affected_requirements": cascade_result.get("affected_requirements", []),
    }


def build_branch_pdf(project_id: str, branch_id: str,
                     output_path: Path,
                     change_request: str,
                     cascade_result: dict,
                     new_subsystems: dict,
                     new_decision: str,
                     new_confidence: int):
    """
    Genera PDF del branch con modifiche evidenziate.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    style_title   = ParagraphStyle("t",  fontSize=15, fontName="Helvetica-Bold", spaceAfter=4)
    style_h1      = ParagraphStyle("h1", fontSize=12, fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10, textColor=colors.HexColor("#7c3aed"))
    style_h2      = ParagraphStyle("h2", fontSize=10, fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=6)
    style_body    = ParagraphStyle("b",  fontSize=8,  fontName="Helvetica",      spaceAfter=2, leading=12)
    style_mono    = ParagraphStyle("m",  fontSize=7.5,fontName="Courier",        spaceAfter=1)
    style_changed = ParagraphStyle("c",  fontSize=8,  fontName="Helvetica-Bold", textColor=colors.HexColor("#7c3aed"), spaceAfter=2)
    style_critical= ParagraphStyle("cr", fontSize=8,  fontName="Helvetica-Bold", textColor=colors.HexColor("#dc2626"), spaceAfter=2)
    style_major   = ParagraphStyle("ma", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#b45309"), spaceAfter=2)
    style_ok      = ParagraphStyle("ok", fontSize=8,  fontName="Helvetica",      textColor=colors.HexColor("#16a34a"), spaceAfter=2)

    story = []
    affected_subs = cascade_result.get("affected_subsystems", [])

    # ── HEADER ──
    story.append(Paragraph(f"BRANCH DOCUMENT — {project_id}", style_title))
    story.append(Paragraph(f"Branch: {branch_id}  |  Base: main-v1.0", style_body))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", style_body))
    story.append(Paragraph(f"Decision: {new_decision}  |  Confidence: {new_confidence}%", style_changed))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#7c3aed"), spaceAfter=8))

    # ── CHANGE REQUEST ──
    story.append(Paragraph("CHANGE REQUEST", style_h1))
    story.append(Paragraph(change_request.strip(), style_body))
    story.append(Spacer(1, 6))

    # ── CASCADE ISSUES ──
    story.append(Paragraph("CASCADE ISSUES IDENTIFIED", style_h1))
    issues = cascade_result.get("cascade_issues", [])
    if not issues:
        story.append(Paragraph("No cascade issues identified.", style_ok))
    else:
        for issue in issues:
            sty = style_critical if issue.get("severity") == "critical" else style_major
            story.append(Paragraph(
                f"[{issue.get('severity','').upper()}] {issue.get('id','')} — {issue.get('title','')}",
                sty
            ))
            story.append(Paragraph(f"  Subsystem: {issue.get('subsystem','')} | Req: {issue.get('requirement','N/A')} | Type: {issue.get('type','')}", style_mono))
            story.append(Paragraph(f"  {issue.get('description','')}", style_body))
    story.append(Spacer(1, 6))

    # ── MISSING TESTS ──
    story.append(Paragraph("MISSING ECSS TESTS — NOW REQUIRED", style_h1))
    tests = cascade_result.get("missing_tests", [])
    if not tests:
        story.append(Paragraph("No additional tests required.", style_ok))
    else:
        for t in tests:
            story.append(Paragraph(
                f"{t.get('ecss_standard','')} {t.get('clause','')} — {t.get('activity','')}",
                style_critical
            ))
            story.append(Paragraph(f"  Reason: {t.get('reason','')}", style_mono))
    story.append(Spacer(1, 6))

    # ── SUBSYSTEM STATUS (aggiornato) ──
    story.append(Paragraph("SUBSYSTEM STATUS — UPDATED", style_h1))
    sub_data = [["Subsystem", "Score", "Status", "Detail", "Changed"]]
    for sid, data in new_subsystems.items():
        changed = "⚠ MODIFIED" if sid in affected_subs else "—"
        sub_data.append([sid, f"{data['score']}%", data["status"].upper(), data["detail"], changed])

    t = Table(sub_data, colWidths=[3*cm, 1.5*cm, 2*cm, 7*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#4c1d95")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f5f3ff"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#c4b5fd")),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # ── AFFECTED REQUIREMENTS ──
    story.append(Paragraph("AFFECTED REQUIREMENTS", style_h1))
    affected_reqs = cascade_result.get("affected_requirements", [])
    if affected_reqs:
        for req in affected_reqs:
            story.append(Paragraph(f"• {req} — verification evidence invalidated, re-analysis required", style_critical))
    else:
        story.append(Paragraph("No requirements directly invalidated.", style_ok))

    doc.build(story)
    print(f"Branch PDF generato: {output_path}")


def build_branch(project_dir: Path,
                 branch_id: str,
                 change_request: str,
                 cascade_result: dict,
                 author: str = "unknown") -> Path:
    """
    Entry point principale.
    Crea la cartella branch con tutti i file.
    """
    branch_dir = project_dir / "branches" / branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)

    project_id = project_dir.name

    # calcola nuovi subsistemi
    new_subs = apply_cascade_to_subsystems(BASE_SUBSYSTEMS, cascade_result)

    # calcola nuova decisione
    conf_delta  = cascade_result.get("confidence_delta", 0)
    new_conf    = max(0, min(100, BASE_CONFIDENCE + conf_delta))
    new_decision= cascade_result.get("go_nogo_impact", BASE_DECISION)

    # diff
    diff = compute_diff(
        BASE_SUBSYSTEMS, new_subs,
        BASE_DECISION, new_decision,
        BASE_CONFIDENCE, new_conf,
        cascade_result,
    )

    # meta
    meta = {
        "branch_id":      branch_id,
        "project_id":     project_id,
        "base_version":   "main-v1.0",
        "change_request": change_request,
        "author":         author,
        "created_at":     datetime.now().isoformat(),
        "go_nogo_impact": new_decision,
        "confidence_before": BASE_CONFIDENCE,
        "confidence_after":  new_conf,
        "issues_count":   len(cascade_result.get("cascade_issues", [])),
        "missing_tests_count": len(cascade_result.get("missing_tests", [])),
    }

    # salva JSON
    with open(branch_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(branch_dir / "diff.json", "w") as f:
        json.dump(diff, f, indent=2)
    with open(branch_dir / "cascade_result.json", "w") as f:
        json.dump(cascade_result, f, indent=2)

    # genera PDF branch
    pdf_path = branch_dir / f"{project_id}_{branch_id}.pdf"
    build_branch_pdf(
        project_id, branch_id, pdf_path,
        change_request, cascade_result,
        new_subs, new_decision, new_conf,
    )

    return branch_dir
