from __future__ import annotations
import hashlib, json, os, re, shutil, subprocess, sys, uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, File, Form, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from .database import Base, engine, SessionLocal, get_db
from . import models, schemas
from chromabook_engine.advanced_mapping import DEFAULT_PARAMS, PARAMETER_UI, validate_params

DATA_ROOT = Path(os.getenv("CHROMABOOK_DATA_DIR", str(ROOT / "var"))).resolve()
STORAGE_ROOT = DATA_ROOT / "storage"
PALETTE_PATH = ROOT / "data" / "main-21.palette.json"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
ADMIN_TOKEN = os.getenv("CHROMABOOK_ADMIN_TOKEN", "").strip()

@asynccontextmanager
async def lifespan(app):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="ChromaBook Studio", version="0.7.0",
              description="Self-hosted image review, versioned library and book production workspace.",
              lifespan=lifespan)

def uid() -> str:
    return str(uuid.uuid4())

def utcnow():
    return datetime.now(timezone.utc)

def guard(authorization: Optional[str] = Header(default=None)):
    if ADMIN_TOKEN:
        expected = f"Bearer {ADMIN_TOKEN}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Admin token required.")
    return True

def audit(db: Session, entity_type: str, entity_id: str, action: str, detail: str = ""):
    db.add(models.AuditEvent(id=uid(), entity_type=entity_type, entity_id=entity_id,
                             action=action, detail=detail[:4000]))

def serialize_project(p):
    return {"id": p.id, "name": p.name, "description": p.description, "created_at": p.created_at.isoformat()}

def latest_run(asset: models.Asset):
    return max(asset.runs, key=lambda x: x.created_at) if asset.runs else None

def _run_payload(run: models.ProcessingRun, asset_id: str | None = None):
    metrics = None
    if run.output_dir:
        mp = Path(run.output_dir) / "engine_comparison_metrics.json"
        if mp.is_file():
            try:
                metrics = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
    try:
        params = json.loads(run.params_json or "{}")
    except Exception:
        params = {}
    stability = None
    if run.output_dir:
        sp = Path(run.output_dir) / "stability_metrics.json"
        if sp.is_file():
            try:
                stability = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                stability = None
    return {"id": run.id, "state": run.state, "engine": run.engine,
            "confidence": run.confidence, "error": run.error,
            "params": params, "metrics": metrics, "stability":stability,
            "preview_base_url": (f"/api/assets/{asset_id}/runs/{run.id}/preview/" if asset_id and run.state == "completed" else None),
            "created_at": run.created_at.isoformat()}

def serialize_asset(a: models.Asset):
    run = latest_run(a)
    return {"id": a.id, "project_id": a.project_id, "logical_name": a.logical_name,
            "filename": a.filename, "version": a.version, "parent_asset_id": a.parent_asset_id,
            "status": a.status, "sha256": a.sha256, "created_at": a.created_at.isoformat(),
            "latest_run": _run_payload(run, a.id) if run else None,
            "run_history": [_run_payload(x, a.id) for x in sorted(a.runs, key=lambda x:x.created_at, reverse=True)],
            "preview_url": f"/api/assets/{a.id}/preview/color_preview.png" if run and run.state == "completed" else None}

def serialize_book(b):
    return {"id": b.id, "project_id": b.project_id, "title": b.title, "subtitle": b.subtitle,
            "status": b.status, "error": b.error, "created_at": b.created_at.isoformat(),
            "book_url": f"/api/books/{b.id}/download/book" if b.book_pdf_path else None,
            "answer_key_url": f"/api/books/{b.id}/download/answer-key" if b.answer_key_path else None,
            "report_url": f"/api/books/{b.id}/download/report" if b.report_path else None}

def _safe_file(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.is_relative_to(STORAGE_ROOT):
        raise HTTPException(status_code=404, detail="File not found.")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return p

def _process_asset(asset_id: str, run_id: str):
    db = SessionLocal()
    try:
        asset = db.get(models.Asset, asset_id)
        run = db.get(models.ProcessingRun, run_id)
        if not asset or not run:
            return
        run.state = "running"; run.started_at = utcnow()
        asset.status = "processing"
        db.commit()
        out_dir = STORAGE_ROOT / "projects" / asset.project_id / "assets" / asset.id / "runs" / run.id
        out_dir.mkdir(parents=True, exist_ok=True)
        source = Path(asset.storage_path)
        run_config = json.loads(run.params_json or "{}")
        config_path = out_dir / "mapping_config.json"
        config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "tools" / "process_advanced_image.py"),
               str(source), str(out_dir), "--palette", str(PALETTE_PATH), "--config", str(config_path)]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Image processing failed")[-8000:])
        summary = json.loads((out_dir / "asset_summary.json").read_text(encoding="utf-8"))
        run.state = "completed"; run.output_dir = str(out_dir)
        run.engine = str(summary.get("engine", ""))
        run.confidence = str(summary.get("confidence", ""))
        run.finished_at = utcnow(); asset.status = "review"
        audit(db, "asset", asset.id, "processed", f"run={run.id}; engine={run.engine}; confidence={run.confidence}")
        db.commit()
    except Exception as exc:
        db.rollback()
        run = db.get(models.ProcessingRun, run_id)
        asset = db.get(models.Asset, asset_id)
        if run:
            run.state = "failed"; run.error = str(exc)[-8000:]; run.finished_at = utcnow()
        if asset:
            asset.status = "failed"
        if run or asset:
            db.commit()
    finally:
        db.close()

def _build_book(book_id: str):
    db = SessionLocal()
    try:
        book = db.get(models.Book, book_id)
        if not book:
            return
        book.status = "building"; db.commit()
        from chromabook_engine import Palette
        from chromabook_engine.layout import SIDE_BY_SIDE, assemble_book
        import numpy as np
        ids = json.loads(book.asset_ids_json)
        arts = []
        for aid in ids:
            asset = db.get(models.Asset, aid)
            if not asset or asset.status != "approved":
                raise RuntimeError(f"Asset {aid} is not approved.")
            run = latest_run(asset)
            if not run or run.state != "completed":
                raise RuntimeError(f"Asset {aid} has no completed processing run.")
            labels_path = Path(run.output_dir) / "labels.npy"
            if not labels_path.is_file():
                raise RuntimeError(f"Labels missing for asset {aid}.")
            arts.append({"name": asset.logical_name, "labels": np.load(labels_path)})
        palette = Palette.from_file(PALETTE_PATH)
        if len(palette) != 21:
            raise RuntimeError("The frozen production palette must contain exactly 21 colours.")
        out_dir = STORAGE_ROOT / "projects" / book.project_id / "books" / book.id
        result = assemble_book(out_dir, book.title, book.subtitle, arts, palette, SIDE_BY_SIDE,
                               min_cell_mm=3.0, collection="CHROMABOOK")
        report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
        if not report.get("preflight", {}).get("passed"):
            raise RuntimeError("Book preflight failed; inspect the book report before release.")
        book.book_pdf_path = str(Path(result["book"]).resolve())
        book.answer_key_path = str(Path(result["answer_key"]).resolve())
        book.report_path = str(Path(result["report"]).resolve())
        book.status = "ready"; book.error = None
        audit(db, "book", book.id, "built", f"assets={len(arts)}; palette=21")
        db.commit()
    except Exception as exc:
        db.rollback()
        book = db.get(models.Book, book_id)
        if book:
            book.status = "failed"; book.error = str(exc)[-8000:]; db.commit()
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse((ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return {"status": "ok", "service": "chromabook-studio", "version": app.version}

@app.get("/api/dashboard", dependencies=[Depends(guard)])
def dashboard_data(db: Session = Depends(get_db)):
    return {"projects": db.scalar(select(func.count()).select_from(models.Project)) or 0,
            "assets": db.scalar(select(func.count()).select_from(models.Asset)) or 0,
            "needs_review": db.scalar(select(func.count()).select_from(models.Asset).where(models.Asset.status == "review")) or 0,
            "books": db.scalar(select(func.count()).select_from(models.Book)) or 0,
            "palette": {"id": "main-21", "colors": 21, "locked": True}}

@app.get("/api/mapping/defaults", dependencies=[Depends(guard)])
def mapping_defaults():
    return {"schema":"chromabook-advanced-mapping-v2", "defaults":DEFAULT_PARAMS,
            "controls":PARAMETER_UI, "engines":[
                {"id":"A","name":"A — Legacy Mean-Lab / ΔE76"},
                {"id":"B","name":"B — Cell-wise CIEDE2000"},
                {"id":"C","name":"C — Edge-Aware ICM"},
                {"id":"D","name":"D — SLIC Region-Aware ICM"},
            ], "palette":{"id":"main-21","colors":21,"editable":False}}

@app.get("/api/projects", dependencies=[Depends(guard)])
def list_projects(db: Session = Depends(get_db)):
    return [serialize_project(p) for p in db.scalars(select(models.Project).order_by(models.Project.created_at.desc())).all()]

@app.post("/api/projects", status_code=201, dependencies=[Depends(guard)])
def create_project(payload: schemas.ProjectCreate, db: Session = Depends(get_db)):
    p = models.Project(id=uid(), name=payload.name.strip(), description=payload.description)
    db.add(p); audit(db, "project", p.id, "created", p.name); db.commit()
    return serialize_project(p)

def _store_upload(db, project_id: str, upload: UploadFile, logical_name: str, parent_asset_id: str | None = None):
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found.")
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Only PNG, JPG, JPEG and WEBP images are supported.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
    if not data:
        raise HTTPException(400, "Empty upload.")
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "The uploaded file is not a valid image.")
    logical_name = (logical_name or Path(upload.filename or "artwork").stem).strip()
    logical_name = re.sub(r"[^A-Za-z0-9 _.-]", "_", logical_name)[:180].strip(" .") or "artwork"
    version_q = select(func.max(models.Asset.version)).where(
        models.Asset.project_id == project_id, models.Asset.logical_name == logical_name)
    version = (db.scalar(version_q) or 0) + 1
    aid = uid()
    dest_dir = STORAGE_ROOT / "projects" / project_id / "assets" / aid
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(upload.filename or f"upload{suffix}").name)[:180]
    dest = dest_dir / f"source{suffix}"
    dest.write_bytes(data)
    asset = models.Asset(id=aid, project_id=project_id, logical_name=logical_name,
                         filename=filename, storage_path=str(dest.resolve()),
                         sha256=hashlib.sha256(data).hexdigest(), version=version,
                         parent_asset_id=parent_asset_id, status="uploaded")
    db.add(asset); audit(db, "asset", aid, "uploaded", f"name={logical_name}; version={version}")
    db.commit()
    return asset

@app.post("/api/projects/{project_id}/assets", status_code=201, dependencies=[Depends(guard)])
def upload_asset(project_id: str, file: UploadFile = File(...), logical_name: str = Form(""),
                 db: Session = Depends(get_db)):
    return serialize_asset(_store_upload(db, project_id, file, logical_name))

@app.post("/api/assets/{asset_id}/revisions", status_code=201, dependencies=[Depends(guard)])
def upload_revision(asset_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    old = db.get(models.Asset, asset_id)
    if not old:
        raise HTTPException(404, "Asset not found.")
    new = _store_upload(db, old.project_id, file, old.logical_name, parent_asset_id=old.id)
    return serialize_asset(new)

@app.get("/api/assets", dependencies=[Depends(guard)])
def list_assets(project_id: str | None = None, db: Session = Depends(get_db)):
    q = select(models.Asset).order_by(models.Asset.created_at.desc())
    if project_id:
        q = q.where(models.Asset.project_id == project_id)
    return [serialize_asset(a) for a in db.scalars(q).all()]

@app.post("/api/assets/{asset_id}/process", status_code=202, dependencies=[Depends(guard)])
def process_asset(asset_id: str, payload: schemas.ProcessRequest | None = None,
                  db: Session = Depends(get_db)):
    asset = db.get(models.Asset, asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found.")
    if asset.status == "processing":
        raise HTTPException(409, "Asset is already processing.")
    raw_params = payload.params if payload else {}
    selected_engine = payload.selected_engine if payload else "D"
    try:
        params = validate_params(raw_params)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc))
    run_config = {"schema":"chromabook-advanced-mapping-v2",
                  "params":params, "selected_engine":selected_engine,
                  "run_stability":bool(payload.run_stability) if payload else False}
    run = models.ProcessingRun(id=uid(), asset_id=asset.id, state="queued",
                               engine=selected_engine,
                               params_json=json.dumps(run_config, ensure_ascii=False))
    asset.status = "processing"; db.add(run)
    audit(db, "asset", asset.id, "processing_queued",
          f"run={run.id}; engine={selected_engine}; schema=chromabook-advanced-mapping-v2")
    db.commit()
    return {"asset_id": asset.id, "run_id": run.id, "state": "queued",
            "selected_engine":selected_engine, "params":params, "run_stability":run_config["run_stability"]}

@app.post("/api/assets/{asset_id}/review", dependencies=[Depends(guard)])
def review_asset(asset_id: str, payload: schemas.ReviewAction, db: Session = Depends(get_db)):
    asset = db.get(models.Asset, asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found.")
    run = latest_run(asset)
    if not run or run.state != "completed":
        raise HTTPException(409, "A completed processing run is required before review.")
    asset.status = "approved" if payload.action == "approve" else "rejected"
    audit(db, "asset", asset.id, payload.action, payload.note)
    db.commit()
    return serialize_asset(asset)

@app.get("/api/assets/{asset_id}/runs/{run_id}/preview/{filename:path}", dependencies=[Depends(guard)])
def processing_run_preview(asset_id: str, run_id: str, filename: str, db: Session = Depends(get_db)):
    allowed = {"color_preview.png", "numbered_preview.png", "asset_summary.json",
               "engine_comparison_metrics.json", "config.json", "render_preflight.json", "stability_metrics.json"}
    safe_engine_preview = bool(re.fullmatch(r"engine_[ABCD]/(color_preview|numbered_preview|metrics)\.(png|json)", filename))
    if filename not in allowed and not safe_engine_preview:
        raise HTTPException(404, "Preview not found.")
    asset = db.get(models.Asset, asset_id)
    run = db.get(models.ProcessingRun, run_id)
    if not asset or not run or run.asset_id != asset.id or run.state != "completed":
        raise HTTPException(404, "Processing run not found.")
    p = _safe_file(Path(run.output_dir) / filename)
    media = "image/png" if p.suffix == ".png" else "application/json"
    return FileResponse(p, media_type=media, filename=p.name)

@app.get("/api/assets/{asset_id}/preview/{filename:path}", dependencies=[Depends(guard)])
def asset_preview(asset_id: str, filename: str, db: Session = Depends(get_db)):
    allowed = {"color_preview.png", "numbered_preview.png", "asset_summary.json",
               "engine_comparison_metrics.json", "config.json", "render_preflight.json", "stability_metrics.json"}
    safe_engine_preview = bool(re.fullmatch(r"engine_[ABCD]/(color_preview|numbered_preview|metrics)\.(png|json)", filename))
    if filename not in allowed and not safe_engine_preview:
        raise HTTPException(404, "Preview not found.")
    asset = db.get(models.Asset, asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found.")
    run = latest_run(asset)
    if not run or run.state != "completed":
        raise HTTPException(404, "No completed preview.")
    p = _safe_file(Path(run.output_dir) / filename)
    media = "image/png" if p.suffix == ".png" else ("text/markdown" if p.suffix == ".md" else "application/json")
    return FileResponse(p, media_type=media, filename=p.name)

@app.post("/api/books", status_code=202, dependencies=[Depends(guard)])
def create_book(payload: schemas.BookCreate, db: Session = Depends(get_db)):
    project = db.get(models.Project, payload.project_id)
    if not project:
        raise HTTPException(404, "Project not found.")
    if len(set(payload.asset_ids)) != len(payload.asset_ids):
        raise HTTPException(400, "Duplicate asset IDs are not allowed.")
    assets = [db.get(models.Asset, aid) for aid in payload.asset_ids]
    if any(not a or a.project_id != project.id for a in assets):
        raise HTTPException(400, "All selected assets must belong to this project.")
    if any(a.status != "approved" for a in assets):
        raise HTTPException(409, "Only approved assets can be assembled into a book.")
    book = models.Book(id=uid(), project_id=project.id, title=payload.title.strip(),
                       subtitle=payload.subtitle.strip(), asset_ids_json=json.dumps(payload.asset_ids),
                       status="queued")
    db.add(book); audit(db, "book", book.id, "build_queued", f"assets={len(assets)}"); db.commit()
    return serialize_book(book)

@app.get("/api/books", dependencies=[Depends(guard)])
def list_books(project_id: str | None = None, db: Session = Depends(get_db)):
    q = select(models.Book).order_by(models.Book.created_at.desc())
    if project_id: q = q.where(models.Book.project_id == project_id)
    return [serialize_book(b) for b in db.scalars(q).all()]

@app.get("/api/books/{book_id}/download/{kind}", dependencies=[Depends(guard)])
def download_book(book_id: str, kind: str, db: Session = Depends(get_db)):
    book = db.get(models.Book, book_id)
    if not book: raise HTTPException(404, "Book not found.")
    field = {"book": book.book_pdf_path, "answer-key": book.answer_key_path, "report": book.report_path}.get(kind)
    if not field: raise HTTPException(404, "Requested artifact is not ready.")
    p = _safe_file(field)
    return FileResponse(p, filename=p.name)

@app.get("/api/audit", dependencies=[Depends(guard)])
def audit_log(limit: int = 100, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 500))
    rows = db.scalars(select(models.AuditEvent).order_by(models.AuditEvent.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "entity_type": r.entity_type, "entity_id": r.entity_id,
             "action": r.action, "detail": r.detail, "created_at": r.created_at.isoformat()} for r in rows]
