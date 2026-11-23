from __future__ import annotations
import logging, os, signal, time
from sqlalchemy import select
from .database import Base, engine, SessionLocal
from . import models
from .main import _process_asset, _build_book

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("chromabook.worker")
STOP = False

def stop(*_):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def recover_interrupted_jobs():
    """Requeue jobs left in a running state after a worker/container crash."""
    db = SessionLocal()
    try:
        runs = db.scalars(select(models.ProcessingRun).where(models.ProcessingRun.state == "running")).all()
        for run in runs:
            run.state = "queued"
            run.error = "Automatically requeued after worker restart."
            asset = db.get(models.Asset, run.asset_id)
            if asset:
                asset.status = "uploaded"
        books = db.scalars(select(models.Book).where(models.Book.status == "building")).all()
        for book in books:
            book.status = "queued"
            book.error = "Automatically requeued after worker restart."
        db.commit()
        if runs or books:
            log.warning("Recovered %s processing jobs and %s book builds", len(runs), len(books))
    finally:
        db.close()

def claim_job():
    db = SessionLocal()
    try:
        # PostgreSQL uses row locks/skip-locked; SQLite is supported for single-worker testing.
        run = db.scalar(select(models.ProcessingRun).where(models.ProcessingRun.state == "queued")
                        .order_by(models.ProcessingRun.created_at).with_for_update(skip_locked=True).limit(1))
        if run:
            run.state = "running"
            run.started_at = models.utcnow()
            db.commit()
            return ("process", run.asset_id, run.id)
        book = db.scalar(select(models.Book).where(models.Book.status == "queued")
                         .order_by(models.Book.created_at).with_for_update(skip_locked=True).limit(1))
        if book:
            book.status = "building"
            db.commit()
            return ("book", book.id, None)
        db.rollback()
        return None
    finally:
        db.close()

def main():
    Base.metadata.create_all(bind=engine)
    recover_interrupted_jobs()
    log.info("ChromaBook worker started")
    while not STOP:
        job = claim_job()
        if not job:
            time.sleep(float(os.getenv("WORKER_POLL_SECONDS", "2")))
            continue
        kind, first, second = job
        log.info("Starting %s job %s", kind, second or first)
        if kind == "process":
            _process_asset(first, second)
        else:
            _build_book(first)
        log.info("Finished %s job %s", kind, second or first)
    log.info("Worker stopped gracefully")

if __name__ == "__main__":
    main()
