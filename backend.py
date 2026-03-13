from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import shutil
import uuid

from main import CitizenDashboardApp, DashboardResponse


app = FastAPI(
    title="Citizen's Dashboard API",
    description="FastAPI backend for interactive exploration of parliamentary bills.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


dashboards: Dict[str, CitizenDashboardApp] = {}
pdf_registry: Dict[str, dict] = {}
active_pdf_id: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "citizens-dashboard"
    pdf_id: Optional[str] = None


class QuestionResponse(BaseModel):
    one_liner: str
    paragraph: str
    impact_points: List[str]
    key_sections: List[str]
    acts_referenced: List[str]


def _resolve_dashboard(pdf_id: Optional[str]) -> CitizenDashboardApp:
    target_id = pdf_id or active_pdf_id
    if not target_id:
        raise HTTPException(status_code=400, detail="No document indexed yet. Upload a PDF first.")
    dashboard = dashboards.get(target_id)
    if dashboard is None:
        raise HTTPException(status_code=404, detail="Selected PDF is not available.")
    return dashboard


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    """
    Upload and index a new PDF. Blocks until indexing completes so the
    front-end can start asking questions once /status reports ready.
    """
    global active_pdf_id

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    pdf_id = str(uuid.uuid4())[:8]
    safe_name = os.path.basename(file.filename)
    saved_name = f"{pdf_id}_{safe_name}"
    saved_path = os.path.join(uploads_dir, saved_name)

    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        dashboard = CitizenDashboardApp(saved_path)
        dashboards[pdf_id] = dashboard
        pdf_registry[pdf_id] = {
            "pdf_id": pdf_id,
            "filename": safe_name,
            "stored_path": saved_path,
        }
        active_pdf_id = pdf_id

        # Stats are optional; the current front-end can handle null.
        return {
            "message": "PDF uploaded and indexed successfully.",
            "pdf_id": pdf_id,
            "active_pdf_id": active_pdf_id,
            "pdf_path": saved_path,
            "pdfs": list(pdf_registry.values()),
            "stats": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        file.file.close()


@app.get("/pdfs")
def list_pdfs() -> dict:
    return {
        "active_pdf_id": active_pdf_id,
        "pdfs": list(pdf_registry.values()),
    }


@app.post("/pdfs/{pdf_id}/select")
def select_pdf(pdf_id: str) -> dict:
    global active_pdf_id
    if pdf_id not in dashboards:
        raise HTTPException(status_code=404, detail="PDF not found.")
    active_pdf_id = pdf_id
    return {"message": "Active PDF updated.", "active_pdf_id": active_pdf_id}


@app.delete("/pdfs/{pdf_id}")
def delete_pdf(pdf_id: str) -> dict:
    global active_pdf_id
    record = pdf_registry.get(pdf_id)
    if not record:
        raise HTTPException(status_code=404, detail="PDF not found.")

    dashboards.pop(pdf_id, None)
    pdf_registry.pop(pdf_id, None)
    stored_path = record.get("stored_path")
    if stored_path and os.path.exists(stored_path):
        try:
            os.remove(stored_path)
        except OSError:
            pass

    if active_pdf_id == pdf_id:
        active_pdf_id = next(iter(dashboards), None)

    return {
        "message": "PDF removed.",
        "active_pdf_id": active_pdf_id,
        "pdfs": list(pdf_registry.values()),
    }


@app.get("/status")
def status() -> dict:
    """Simple polling endpoint used by the front-end to know when indexing is done."""
    return {"ready": bool(active_pdf_id and active_pdf_id in dashboards), "active_pdf_id": active_pdf_id}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest) -> QuestionResponse:
    """
    Question endpoint used by the new HTML front-end.
    Returns a structured payload matching the UI expectations.
    """
    dashboard = _resolve_dashboard(payload.pdf_id)

    answer: DashboardResponse = dashboard.ask_structured(
        payload.question,
        thread_id=f"{payload.thread_id or 'citizens-dashboard'}:{payload.pdf_id or active_pdf_id}",
    )

    return QuestionResponse(
        one_liner=answer.summary,
        paragraph=answer.paragraph,
        impact_points=answer.impact_points,
        key_sections=answer.key_clauses,
        acts_referenced=answer.acts_referenced,
    )


static_dir = os.path.join(os.path.dirname(__file__), "static")
index_path = os.path.join(static_dir, "index.html")


@app.get("/")
def index() -> FileResponse:
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

