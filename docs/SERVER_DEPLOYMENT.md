# ChromaBook Studio — Self-hosted Server Deployment

This release adds the Phase 3 management UI, Phase 4 versioned project library/database, and Phase 6 container deployment.

## 1. What is included

- Browser-based management dashboard at `/`.
- Project creation and per-project artwork library.
- PNG/JPG/JPEG/WEBP upload with a configurable size cap (default 25 MB).
- SHA-256 source fingerprint and immutable upload versions. Re-uploading an artwork under the same logical name creates a new version and preserves the earlier source.
- Processing queue stored in the database and consumed by a separate worker container.
- Notebook v2 processing reports, four-engine colour/numbered previews, and metric comparison.
- Review workflow: `uploaded → processing → review → approved/rejected`.
- Automatic processing uses Notebook v2 defaults; manual Hybrid reruns use only the 29 Notebook v2 controls and always restart from the original upload.\n- Only approved assets can be assembled into a book.
- Book creation generates a separate-cover numbered PDF, separate answer-key PDF and JSON/Markdown book report.
- Audit events for project, upload, processing, review and book-build actions.
- PostgreSQL database and persistent storage volumes in Docker Compose.
- Admin-token protection for all `/api/*` routes; health endpoint is public.
- Alembic initial schema migration.

The palette is frozen at `data/main-21.palette.json`. The processing and book-building paths reject any palette that is not exactly 21 colours.

## 2. Deploy with Docker Compose (recommended)

Requirements: Docker Engine and the Docker Compose v2 plugin.

```bash
cp .env.example .env
```

Generate two secrets and paste the values into `.env`. Use hexadecimal values so the database URL does not need URL escaping:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

Set:
- `POSTGRES_PASSWORD` to the first generated value.
- `CHROMABOOK_ADMIN_TOKEN` to the second generated value.

Start the stack:

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f web worker
```

The stack contains:
- `db`: PostgreSQL 16 with a persistent volume.
- `web`: FastAPI server, UI, API and schema migration.
- `worker`: persistent database-backed processing/build queue consumer.

The web port is bound to `127.0.0.1:8000` by default. This is intentional: do not expose the unencrypted application port directly to the public internet.

### Access from your own computer for a first server test

From your computer, open an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-server
```

Then open `http://127.0.0.1:8000` in your browser. Paste the `CHROMABOOK_ADMIN_TOKEN` value into the token field in the page and click Save.

### Public access

Put Nginx, Caddy or another reverse proxy in front of the loopback-bound web port and configure HTTPS/TLS. An example Nginx server block is in `docs/nginx.example.conf`. Do not disable the admin token on an internet-facing host.

## 3. Local development without Docker

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

Use SQLite for a local smoke test:

```bash
export CHROMABOOK_DATA_DIR=./var
export CHROMABOOK_ADMIN_TOKEN=local-test-token
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and run:

```bash
export CHROMABOOK_DATA_DIR=./var
export CHROMABOOK_ADMIN_TOKEN=local-test-token
python -m app.worker
```

On Windows PowerShell, use `$env:CHROMABOOK_DATA_DIR="./var"` and `$env:CHROMABOOK_ADMIN_TOKEN="local-test-token"`.

## 4. First-use workflow

1. Open the dashboard and enter the admin token.
2. Create a project.
3. Upload one image. Use PNG, JPG/JPEG or WEBP, at most 25 MB by default.
4. Press **Process**. The web API adds a job to the database; the worker performs the CPU-intensive processing.
5. Wait for the asset to move to **review**. Refreshing the browser is safe; the job state is stored in PostgreSQL.
6. Inspect the color preview, numbered preview, confidence and report.
7. Approve or reject the image. A rejected image cannot be included in a book.
8. To revise an image, use **Upload revision**. The earlier version remains in the library; the new version has a higher version number and a parent link.
9. Select approved images, enter a book title/subtitle and build the book.
10. Download the numbered book, separate answer key and preflight report when status becomes **ready**.

The initial book workflow uses the Phase 5 A4 landscape, two-slot layout and right-slot-first reading order. The cover is a separate first page; the answer key is a separate PDF. The existing 21-colour palette is not editable through the UI.

## 5. Persistence and backups

Persistent data is held in two Docker volumes:
- PostgreSQL data volume: project metadata, versions, processing runs, book records and audit events.
- `chromabook_data`: original uploads, processing outputs, labels, previews and generated books.

Back up both. A database-only backup does not include the uploaded images or generated PDFs.

PostgreSQL backup:

```bash
docker compose exec -T db pg_dump -U chromabook -d chromabook > chromabook-db.sql
```

For the file volume, identify its actual name with `docker volume ls`, then archive it while the stack is stopped or while using a consistent filesystem snapshot:

```bash
docker compose stop web worker
docker run --rm -v <compose-project>_chromabook_data:/data -v "$PWD":/backup alpine \
  tar -czf /backup/chromabook-files.tar.gz -C /data .
docker compose start web worker
```

Restore both the database dump and the file archive together. Keep the same `CHROMABOOK_ADMIN_TOKEN` and database credentials in a secure password manager.

## 6. Operational notes and current boundaries

- The worker queue is persisted in PostgreSQL. Jobs left in `running`/`building` after a worker restart are requeued at worker startup.
- Run one worker replica for this release. The queue claim uses database row locking on PostgreSQL; if scaling workers later, load-test and monitor CPU/RAM first.
- The app currently keeps the original upload and every revision. No automatic retention or deletion policy is enabled.
- The initial Alembic migration is provided. Future schema changes must be added as new Alembic revisions; do not edit the initial migration after a server has applied it.
- This is a single-host deployment baseline, not a high-availability cluster. Use external backups, HTTPS, host firewalling and monitoring before production sales use.
- Persian/RTL book typography, multi-user roles, SSO, billing, cloud object storage, automatic email notifications and a distributed task broker are not included in this server-test release.
- The server worker processes the full A/B/C/D optimizer configuration. It can be CPU-intensive; start with a small number of images and monitor resource use.


## 7. Hybrid Mapping Controls (Phase 5.1)

The dashboard's Hybrid Mapping Controls are derived from the uploaded `ChromaBook_Advanced_Mapping_Lab_v2.ipynb`:
- Preprocess: max side, crop, illumination correction/scale, CLAHE, denoise method/strength, fine/coarse sharpening.
- Edges: automatic Canny, manual Canny low/high, morphological close iterations.
- Mapping: grid columns, samples per cell, trimmed outliers, smoothness λ, boundary β, ICM iterations, SLIC segments/compactness, region balance, numbered-cell pixels and line width.
- Engines A/B/C/D are all run for each processing job. The selected engine supplies the primary labels used by book assembly.
- Metric comparison includes ΔE00 mean/median/p95, SSIM, edge F1/precision/recall, used colours, singleton/tiny component ratios and transition rate.
- The optional ±1 L* stability test is available for engines B/C/D.
- Every run stores its own Notebook v2 configuration, selected engine and output artifacts. The original source and older processing runs are not overwritten.

The UI intentionally does not expose Notebook's sample palette upload/editor. The server always uses the project's immutable 21-colour palette.
