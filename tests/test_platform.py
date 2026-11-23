from __future__ import annotations
import io, json, uuid
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app, SessionLocal, STORAGE_ROOT, _build_book
from app import models

def png_bytes():
    im = Image.new("RGB", (160, 160), "white")
    for x in range(30, 130):
        for y in range(30, 130):
            im.putpixel((x, y), (220, 80, 100))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()

def test_project_upload_revision_and_approval_gate():
    with TestClient(app) as client:
        p = client.post("/api/projects", json={"name": f"Test {uuid.uuid4().hex[:6]}"})
        assert p.status_code == 201
        pid = p.json()["id"]
        upload = client.post(f"/api/projects/{pid}/assets",
                             files={"file": ("flower.png", png_bytes(), "image/png")},
                             data={"logical_name": "Flower"})
        assert upload.status_code == 201
        a1 = upload.json()
        assert a1["version"] == 1 and a1["status"] == "uploaded"
        revision = client.post(f"/api/assets/{a1['id']}/revisions",
                               files={"file": ("flower2.png", png_bytes(), "image/png")})
        assert revision.status_code == 201
        assert revision.json()["version"] == 2
        assert revision.json()["parent_asset_id"] == a1["id"]
        review = client.post(f"/api/assets/{a1['id']}/review",
                             json={"action": "approve", "note": "test"})
        assert review.status_code == 409

def test_book_build_from_approved_asset():
    with TestClient(app) as client:
        p = client.post("/api/projects", json={"name": f"Book test {uuid.uuid4().hex[:6]}"})
        pid = p.json()["id"]
        upload = client.post(f"/api/projects/{pid}/assets",
                             files={"file": ("simple.png", png_bytes(), "image/png")},
                             data={"logical_name": "Simple"})
        aid = upload.json()["id"]
        out = STORAGE_ROOT / "test-fixtures" / uuid.uuid4().hex
        out.mkdir(parents=True, exist_ok=True)
        # 21-colour index matrix; one simple region makes a valid print asset for assembly.
        labels = np.zeros((12, 12), dtype=np.uint8)
        labels[2:10, 2:10] = 4
        np.save(out / "labels.npy", labels)
        (out / "asset_summary.json").write_text(json.dumps({"engine": "D", "confidence": "high"}))
        db = SessionLocal()
        try:
            asset = db.get(models.Asset, aid)
            run = models.ProcessingRun(id=str(uuid.uuid4()), asset_id=aid, state="completed",
                                       output_dir=str(out), engine="D", confidence="high")
            asset.runs.append(run)
            asset.status = "approved"
            db.commit()
        finally:
            db.close()
        created = client.post("/api/books", json={"project_id": pid, "asset_ids": [aid],
                                                   "title": "Test Book", "subtitle": "API test"})
        assert created.status_code == 202
        bid = created.json()["id"]
        _build_book(bid)
        books = client.get(f"/api/books?project_id={pid}").json()
        book = next(x for x in books if x["id"] == bid)
        assert book["status"] == "ready", book.get("error")
        assert book["book_url"] and book["answer_key_url"] and book["report_url"]


def test_database_backed_worker_queue_recovers_interrupted_jobs():
    from app import main as main_module
    from app.worker import claim_job, recover_interrupted_jobs
    with TestClient(app) as client:
        p = client.post("/api/projects", json={"name": f"Queue test {uuid.uuid4().hex[:6]}"})
        pid = p.json()["id"]
        upload = client.post(f"/api/projects/{pid}/assets",
                             files={"file": ("queued.png", png_bytes(), "image/png")},
                             data={"logical_name": "Queued"})
        aid = upload.json()["id"]
        queued = client.post(f"/api/assets/{aid}/process")
        assert queued.status_code == 202
        claimed = claim_job()
        assert claimed == ("process", aid, queued.json()["run_id"])
        db = SessionLocal()
        try:
            run = db.get(models.ProcessingRun, queued.json()["run_id"])
            assert run.state == "running"
        finally:
            db.close()
        recover_interrupted_jobs()
        db = SessionLocal()
        try:
            run = db.get(models.ProcessingRun, queued.json()["run_id"])
            asset = db.get(models.Asset, aid)
            assert run.state == "queued"
            assert asset.status == "uploaded"
        finally:
            db.close()

def test_admin_token_protects_api_when_configured(monkeypatch):
    from app import main as main_module
    monkeypatch.setattr(main_module, "ADMIN_TOKEN", "test-secret")
    with TestClient(app) as client:
        assert client.get("/api/projects").status_code == 401
        assert client.get("/api/projects", headers={"Authorization": "Bearer test-secret"}).status_code == 200
    monkeypatch.setattr(main_module, "ADMIN_TOKEN", "")
