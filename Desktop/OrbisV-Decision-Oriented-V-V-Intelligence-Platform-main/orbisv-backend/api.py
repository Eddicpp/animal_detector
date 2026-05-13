"""
api.py
FastAPI backend per OrbisV.

Avvio:
    pip install fastapi uvicorn pypdf httpx reportlab
    uvicorn api:app --reload --port 8000
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ollama_engine import analyze_change
from branch_builder import build_branch

# ── Setup ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="OrbisV API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECTS_DIR = Path(__file__).parent / "projects"


# ── Models ────────────────────────────────────────────────────────────────────

class ChangeRequest(BaseModel):
    change_request: str
    author: str = "unknown"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_project_dir(project_id: str) -> Path:
    d = PROJECTS_DIR / project_id
    if not d.exists():
        raise HTTPException(404, f"Project {project_id} not found")
    return d


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# -- PDF download
@app.get("/api/project/{project_id}/pdf")
def download_pdf(project_id: str):
    proj = get_project_dir(project_id)
    pdf  = proj / "main" / "IRIS3_main.pdf"
    if not pdf.exists():
        raise HTTPException(404, "PDF not found — run generate_pdf.py first")
    return FileResponse(
        path=str(pdf),
        media_type="application/pdf",
        filename=f"{project_id}_main.pdf",
    )


# -- Analyze change request → crea branch
@app.post("/api/project/{project_id}/analyze")
def analyze(project_id: str, body: ChangeRequest):
    proj    = get_project_dir(project_id)
    pdf     = proj / "main" / "IRIS3_main.pdf"

    if not pdf.exists():
        raise HTTPException(404, "PDF not found — run generate_pdf.py first")

    # chiama ollama
    try:
        result = analyze_change(pdf, body.change_request)
    except Exception as e:
        raise HTTPException(500, f"Ollama error: {e}")

    # crea branch
    branch_id = f"branch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    branch    = build_branch(proj, branch_id, body.change_request, result, body.author)

    return {
        "branch_id":   branch_id,
        "branch_path": str(branch),
        "result":      result,
    }


# -- Lista branches
@app.get("/api/project/{project_id}/branches")
def list_branches(project_id: str):
    proj     = get_project_dir(project_id)
    branches = proj / "branches"
    if not branches.exists():
        return {"branches": []}

    out = []
    for d in sorted(branches.iterdir(), reverse=True):
        if d.is_dir():
            meta = load_json(d / "meta.json")
            out.append({
                "id":             d.name,
                "change_request": meta.get("change_request", ""),
                "author":         meta.get("author", ""),
                "created_at":     meta.get("created_at", ""),
                "go_nogo_impact": meta.get("go_nogo_impact", ""),
                "issues_count":   meta.get("issues_count", 0),
            })
    return {"branches": out}


# -- Dati singolo branch
@app.get("/api/project/{project_id}/branch/{branch_id}")
def get_branch(project_id: str, branch_id: str):
    proj   = get_project_dir(project_id)
    branch = proj / "branches" / branch_id
    if not branch.exists():
        raise HTTPException(404, f"Branch {branch_id} not found")

    return {
        "meta":     load_json(branch / "meta.json"),
        "diff":     load_json(branch / "diff.json"),
        "cascade":  load_json(branch / "cascade_result.json"),
    }


# -- PDF branch scaricabile
@app.get("/api/project/{project_id}/branch/{branch_id}/pdf")
def download_branch_pdf(project_id: str, branch_id: str):
    proj   = get_project_dir(project_id)
    pdf    = proj / "branches" / branch_id / f"{project_id}_{branch_id}.pdf"
    if not pdf.exists():
        raise HTTPException(404, "Branch PDF not found")
    return FileResponse(
        path=str(pdf),
        media_type="application/pdf",
        filename=f"{project_id}_{branch_id}.pdf",
    )


# -- Merge branch → main
@app.post("/api/project/{project_id}/branch/{branch_id}/merge")
def merge_branch(project_id: str, branch_id: str):
    proj   = get_project_dir(project_id)
    branch = proj / "branches" / branch_id
    if not branch.exists():
        raise HTTPException(404, f"Branch {branch_id} not found")

    # backup main
    main     = proj / "main"
    backup   = proj / f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(main, backup)

    # copia PDF branch su main
    branch_pdf = branch / f"{project_id}_{branch_id}.pdf"
    if branch_pdf.exists():
        shutil.copy(branch_pdf, main / f"{project_id}_main.pdf")

    # aggiorna meta main
    cascade = load_json(branch / "cascade_result.json")
    meta    = load_json(branch / "meta.json")
    with open(main / "merge_log.json", "w") as f:
        json.dump({
            "merged_branch":  branch_id,
            "merged_at":      datetime.now().isoformat(),
            "change_request": meta.get("change_request", ""),
            "issues_resolved": cascade.get("cascade_issues", []),
        }, f, indent=2)

    return {"status": "merged", "backup": str(backup)}
