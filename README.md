# ChromaBook Studio — Image Engine, Book Studio & Self-hosted Platform

Turns an image into a **true discrete grid** (one palette index per cell) and renders a colour
preview and a printable numbered grid from that *same label matrix*. Four mapping engines
(A legacy mean-Lab / dE76, B cell-wise CIEDE2000, C edge-aware ICM, D SLIC region-aware ICM)
are ported from `ChromaBook_Advanced_Mapping_Lab_v2.ipynb` and verified against it.

The repository now includes the image engine, print optimisation, book assembly, a browser management UI, a versioned project library, a PostgreSQL-backed worker queue and a Docker Compose server deployment.

## Install and run
    pip install -r requirements.txt
    export PYTHONPATH=src            # or: pip install -e .
    python tests/run_tests.py        # full test suite (pytest -q also works)
    python -m chromabook_engine.cli run photo.jpg --palette data/main-21.palette.json --out out --cols 36 --stability

## Library use
    from chromabook_engine import Pipeline, Palette, PipelineConfig, load_image
    from chromabook_engine.export import export_asset
    pal = Palette.from_file("data/main-21.palette.json")
    res = Pipeline().run(load_image("photo.jpg"), pal)         # engines A-D
    labels = res.engines["C"].result.labels                    # palette INDICES, (rows, cols)
    export_asset(res, "C", "out/photo_C")                      # full asset package

## Guarantees enforced by tests
- Printed colour IDs (`"01"`..`"21"`) are strings read from the palette; array index and ID never mix.
- Preview and numbered grid are pure functions of one label matrix; cells are uniform and square.
- The normalised original is read-only; the source file SHA256 is recorded; runs are reproducible.
- Crop and content zoom change the pixels the engines see; display zoom never changes labels.
- Each stage has its own cache key; render settings never recompute mapping.
- Output matches the notebook's own code exactly (`tests/test_notebook_parity.py` executes the .ipynb).

See `docs/PHASE1_REPORT.md` for measured results and known limits, `docs/ARCHITECTURE.md` for the module map.

## Phase 2 (optimisation)
    python -m chromabook_engine.cli optimize photo.jpg --palette data/main-21.palette.json --out out
Writes `optimization_report.json/.md` with diagnostics, framing proposals, every candidate, the ranked finalists and a confidence level. See `docs/PHASE2_REPORT.md`.

## Layout engine and PDFs
    python tools/run_book.py photo.jpg out/     # both book types: optimisation, layout report, numbered + answer-key PDFs
See `docs/LAYOUT_ENGINE.md` for the exact A4 geometry, guide allocation and the parrot results.

## Phase 5 — Book Studio (new assembly workflow)

The Book Studio assembly layer now supports a standalone cover page, multi-artwork interior assembly, right-to-left slot filling (right slot first for side-by-side layouts), folio numbers, a separate answer-key PDF, and a book-level JSON/Markdown report. The final unused slot is intentionally left blank. The answer-key grid numbers select black or white text using relative luminance for contrast.

The palette is locked to `data/main-21.palette.json` (21 colours). `tools/build_collection.py` refuses to run if that palette does not contain exactly 21 colours.

Create a UTF-8 manifest, for example:

```json
{
  "collection": "CHROMABOOK",
  "title": "BIRDS IN BLOOM",
  "subtitle": "A Color-by-Number Collection of Birds & Flowers",
  "images": ["images/bird_01.jpg", "images/bird_02.jpg", "images/owl.jpg"]
}
```

Then run from the project root:

```bash
python tools/build_collection.py manifest.json output/birds
```

Default layout is A4 landscape side-by-side, with the right slot filled first. Use `--full-page` for A4 portrait, one artwork per page. Output includes `book.pdf`, `answer_key.pdf`, `book_report.json`, `book_report.md`, and per-image optimization reports under `assets/`.

Preflight validates palette IDs, label indices, physical cell size (minimum 3 mm), conservative number fit, artwork placement, page accounting and blank-slot count. English cover text is supported in this pass; Persian shaping/RTL typography remains a later enhancement and is not claimed as complete.


## Phase 3 — Management UI

A FastAPI-served browser dashboard is available at `http://127.0.0.1:8000/`. It supports project creation, image upload, processing/review/approval/rejection, revision uploads, preview access, book assembly and downloads. All `/api/*` endpoints accept a bearer admin token when `CHROMABOOK_ADMIN_TOKEN` is set.

## Phase 4 — Database & Library

SQLAlchemy models store projects, immutable asset versions, processing runs, books and audit events. PostgreSQL is used in the server deployment; SQLite is available for local development. Original uploads and generated artifacts are stored in a persistent filesystem volume. Alembic migration `0001_initial` initializes the schema.

## Phase 6 — Deployment

Dockerfile and Docker Compose provide PostgreSQL, the web/API service and a separate database-backed worker. The worker consumes queued image-processing and book-build jobs and requeues interrupted jobs after restart. The default web port is bound to loopback and should be exposed through an HTTPS reverse proxy or SSH tunnel.

See:
- `docs/SERVER_DEPLOYMENT.md` — installation, configuration, backups and operations.
- `docs/SERVER_ACCEPTANCE_CHECKLIST.md` — server test plan.
- `docs/nginx.example.conf` — reverse-proxy example.

The first server release is a single-host deployment baseline. It does not yet include SSO/roles, distributed object storage, high availability, or a distributed broker.


## Phase 5.1 — Interactive Hybrid Image Optimization

The web UI now runs the exact Advanced Mapping Lab v2 preprocessing and mapping pipeline. Automatic processing uses Notebook v2 defaults; users can then adjust only Notebook v2 parameters, compare engines A/B/C/D, reprocess from the original upload, and retain each run's config and metrics. The optional ±1 L* stability test is also available. The application palette remains locked to `data/main-21.palette.json` (exactly 21 colours).

See `docs/HYBRID_MAPPING.md` for the parameter contract, workflow and API.
