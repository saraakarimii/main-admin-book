import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express, { type NextFunction, type Request, type Response } from 'express';
import multer from 'multer';
import sharp from 'sharp';
import {
  ALLOWED_SUFFIXES,
  MAX_UPLOAD_BYTES,
  PALETTE_PATH,
  ROOT,
  STORAGE_ROOT,
  ensureStorageDirs,
  safeStorageFile,
  uid,
} from './server/store.js';
import {
  DEFAULT_PARAMS,
  ENGINE_NAMES,
  PARAMETER_UI,
  loadPalette,
  runImageAndSave,
  validateParams,
} from './server/mapping.js';
import { SIDE_BY_SIDE, assembleBook, type BookArtworkInput } from './server/book.js';
import { requireAuth, type AuthRequest } from './src/middleware/auth.js';
import {
  addAuditEvent,
  assignAssetsToProject,
  createAssetRecord,
  createBookRecord,
  createProcessingRun,
  createProjectForUser,
  deleteAssetsForUser,
  findDuplicateAssetBySha,
  getAssetForUser,
  getAssetSystem,
  getBookForUser,
  getBookSystem,
  getDashboardStats,
  getNextAssetVersion,
  getProjectForUser,
  getProjectOwnerUserId,
  getRunById,
  getRunsForAsset,
  listAssetsWithRuns,
  listAuditEventsForUser,
  listBooksForUser,
  listProjectsForUser,
  listRunsForUser,
  removeAssetsFromProject,
  resolveAssetOwnerUserId,
  serializeAssetRow,
  serializeBookRow,
  serializeProjectRow,
  updateAssetStatus,
  updateBookRecord,
  updateProcessingRun,
} from './src/db/queries.js';

ensureStorageDirs();

const app = express();
app.use(express.json({ limit: '2mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES + 1024 },
});

function sendError(res: Response, status: number, detail: string) {
  res.status(status).json({ detail });
}

// --- Background job runners backed by Cloud SQL ---

async function executeProcessJob(assetId: string, runId: string): Promise<void> {
  try {
    const asset = await getAssetSystem(assetId);
    const run = await getRunById(runId);
    if (!asset || !run) return;

    await updateProcessingRun(runId, { state: 'running', startedAt: new Date() });
    await updateAssetStatus(assetId, 'processing');

    const outDir = path.join(STORAGE_ROOT, 'assets', asset.id, 'runs', run.id);
    fs.mkdirSync(outDir, { recursive: true });
    const runConfig = JSON.parse(run.paramsJson || '{}') as {
      params?: Record<string, unknown>;
      selected_engine?: string;
      run_stability?: boolean;
    };
    fs.writeFileSync(path.join(outDir, 'mapping_config.json'), JSON.stringify(runConfig, null, 2), 'utf-8');

    const res = await runImageAndSave(
      asset.storagePath,
      outDir,
      runConfig.params,
      runConfig.selected_engine || 'D',
      Boolean(runConfig.run_stability),
    );

    await updateProcessingRun(runId, {
      state: 'completed',
      outputDir: outDir,
      engine: res.engine,
      confidence: res.confidence,
      finishedAt: new Date(),
    });
    await updateAssetStatus(assetId, 'review');

    const ownerUserId = await resolveAssetOwnerUserId(asset);
    if (ownerUserId !== null) {
      await addAuditEvent(
        uid(),
        ownerUserId,
        'asset',
        asset.id,
        'processed',
        `run=${run.id}; engine=${res.engine}; confidence=${res.confidence}`,
      );
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    try {
      await updateProcessingRun(runId, {
        state: 'failed',
        error: msg.slice(-8000),
        finishedAt: new Date(),
      });
      await updateAssetStatus(assetId, 'failed');
    } catch (updateErr) {
      console.error('Failed to persist processing failure state:', updateErr);
    }
  }
}

async function executeBookJob(bookId: string): Promise<void> {
  try {
    const book = await getBookSystem(bookId);
    if (!book) return;

    await updateBookRecord(bookId, { status: 'building' });
    const ids = JSON.parse(book.assetIdsJson) as string[];
    const arts: BookArtworkInput[] = [];

    for (const aid of ids) {
      const asset = await getAssetSystem(aid);
      if (!asset || asset.status !== 'approved' || asset.projectId !== book.projectId) {
        throw new Error(`Asset ${aid} is not approved for this project.`);
      }
      const runs = await getRunsForAsset(asset.id);
      const run = runs[0] ?? null;
      if (!run || run.state !== 'completed' || !run.outputDir) {
        throw new Error(`Asset ${aid} has no completed processing run.`);
      }
      const labelsJsonPath = path.join(run.outputDir, 'labels.json');
      if (!fs.existsSync(labelsJsonPath)) {
        throw new Error(`Labels missing for asset ${aid}.`);
      }
      const parsed = JSON.parse(fs.readFileSync(labelsJsonPath, 'utf-8')) as {
        rows: number;
        cols: number;
        data: number[];
      };
      arts.push({
        name: asset.logicalName,
        rows: parsed.rows,
        cols: parsed.cols,
        labels: Uint8Array.from(parsed.data),
      });
    }

    const palette = loadPalette(PALETTE_PATH);
    if (palette.length !== 21) {
      throw new Error('The frozen production palette must contain exactly 21 colours.');
    }

    const outDir = path.join(STORAGE_ROOT, 'projects', book.projectId, 'books', book.id);
    const result = await assembleBook(
      outDir,
      book.title,
      book.subtitle,
      arts,
      palette,
      SIDE_BY_SIDE,
      3.0,
      'CHROMABOOK',
    );
    if (!result.preflight.passed) {
      throw new Error('Book preflight failed; inspect the book report before release.');
    }

    await updateBookRecord(bookId, {
      bookPdfPath: path.resolve(result.book),
      answerKeyPath: path.resolve(result.answer_key),
      reportPath: path.resolve(result.report),
      status: 'ready',
      error: null,
    });

    const ownerUserId = await getProjectOwnerUserId(book.projectId);
    if (ownerUserId !== null) {
      await addAuditEvent(uid(), ownerUserId, 'book', book.id, 'built', `assets=${arts.length}; palette=21`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    try {
      await updateBookRecord(bookId, {
        status: 'failed',
        error: msg.slice(-8000),
      });
    } catch (updateErr) {
      console.error('Failed to persist book failure state:', updateErr);
    }
  }
}

// --- Static & Health Routes ---

app.use(express.static(path.join(ROOT, 'public')));

app.get('/', (_req, res) => {
  res.sendFile(path.join(ROOT, 'public', 'index.html'));
});

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'chromabook-studio', version: '0.7.0' });
});

// --- Authenticated API Routes ---

app.get('/api/dashboard', requireAuth, async (req: AuthRequest, res) => {
  try {
    const stats = await getDashboardStats(req.dbUser!.id);
    res.json({
      ...stats,
      palette: { id: 'main-21', colors: 21, locked: true },
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load dashboard.');
  }
});

app.get('/api/mapping/defaults', requireAuth, (_req: AuthRequest, res) => {
  res.json({
    schema: 'chromabook-advanced-mapping-v2',
    defaults: DEFAULT_PARAMS,
    controls: PARAMETER_UI,
    engines: [
      { id: 'A', name: ENGINE_NAMES.A },
      { id: 'B', name: ENGINE_NAMES.B },
      { id: 'C', name: ENGINE_NAMES.C },
      { id: 'D', name: ENGINE_NAMES.D },
    ],
    palette: { id: 'main-21', colors: 21, editable: false },
  });
});

app.get('/api/system-status', requireAuth, async (req: AuthRequest, res) => {
  try {
    const palette = loadPalette(PALETTE_PATH);
    const stats = await getDashboardStats(req.dbUser!.id);
    res.json({
      service: 'ChromaBook Studio',
      version: '0.7.0',
      status: 'Operational',
      upload: {
        formats: ['PNG', 'JPG', 'JPEG', 'WEBP'],
        max_upload_mb: Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024)),
        independent_inbox_enabled: true,
      },
      palette: {
        id: 'main-21',
        count: palette.length,
        locked: true,
        colors: palette.map((c) => ({ id: c.id, name: c.name, hex: c.hex })),
      },
      engines: Object.entries(ENGINE_NAMES).map(([id, name]) => ({ id, name, status: 'Ready' })),
      workspace: stats,
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load system status.');
  }
});

app.get('/api/projects', requireAuth, async (req: AuthRequest, res) => {
  try {
    const rows = await listProjectsForUser(req.dbUser!.id);
    res.json(rows);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load projects.');
  }
});

app.post('/api/projects', requireAuth, async (req: AuthRequest, res) => {
  try {
    const body = (req.body || {}) as { name?: unknown; description?: unknown };
    const name = typeof body.name === 'string' ? body.name.trim() : '';
    if (!name || name.length > 160) {
      return sendError(res, 422, 'Project name must be between 1 and 160 characters.');
    }
    const description = typeof body.description === 'string' ? body.description.slice(0, 3000) : '';
    const project = await createProjectForUser(uid(), req.dbUser!.id, name, description);
    await addAuditEvent(uid(), req.dbUser!.id, 'project', project.id, 'created', project.name);
    res.status(201).json(serializeProjectRow(project, { assetCount: 0, approvedCount: 0 }));
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to create project.');
  }
});

app.post('/api/projects/:projectId/assign-assets', requireAuth, async (req: AuthRequest, res) => {
  try {
    const projectId = String(req.params.projectId);
    const project = await getProjectForUser(projectId, req.dbUser!.id);
    if (!project) return sendError(res, 404, 'Project not found.');

    const body = (req.body || {}) as { asset_ids?: unknown };
    const rawIds = Array.isArray(body.asset_ids) ? body.asset_ids : [];
    const assetIds = Array.from(new Set(rawIds.map((x) => String(x).trim()).filter(Boolean)));
    if (assetIds.length === 0) {
      return sendError(res, 422, 'Select at least one artwork to assign.');
    }

    const found = await Promise.all(assetIds.map((id) => getAssetForUser(id, req.dbUser!.id)));
    if (found.some((x) => !x)) {
      return sendError(res, 404, 'One or more selected artworks could not be found.');
    }

    const updated = await assignAssetsToProject(project.id, req.dbUser!.id, assetIds);
    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'project',
      project.id,
      'assigned_assets',
      `project=${project.name}; count=${updated.length}; assets=${assetIds.join(',')}`,
    );

    res.json({
      project_id: project.id,
      project_name: project.name,
      assigned_count: updated.length,
      asset_ids: updated.map((a) => a.id),
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to assign artworks.');
  }
});

app.post('/api/projects/:projectId/remove-assets', requireAuth, async (req: AuthRequest, res) => {
  try {
    const projectId = String(req.params.projectId);
    const project = await getProjectForUser(projectId, req.dbUser!.id);
    if (!project) return sendError(res, 404, 'Project not found.');

    const body = (req.body || {}) as { asset_ids?: unknown };
    const rawIds = Array.isArray(body.asset_ids) ? body.asset_ids : [];
    const assetIds = Array.from(new Set(rawIds.map((x) => String(x).trim()).filter(Boolean)));
    if (assetIds.length === 0) {
      return sendError(res, 422, 'Select at least one artwork to remove from this project.');
    }

    const updated = await removeAssetsFromProject(project.id, req.dbUser!.id, assetIds);
    if (updated.length === 0) {
      return sendError(res, 404, 'No matching artworks found in this project.');
    }

    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'project',
      project.id,
      'removed_assets',
      `project=${project.name}; count=${updated.length}; assets=${updated.map((a) => a.id).join(',')}`,
    );

    res.json({
      project_id: project.id,
      project_name: project.name,
      removed_count: updated.length,
      asset_ids: updated.map((a) => a.id),
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to remove artworks from project.');
  }
});

app.delete('/api/projects/:projectId/assets/:assetId', requireAuth, async (req: AuthRequest, res) => {
  try {
    const projectId = String(req.params.projectId);
    const assetId = String(req.params.assetId);
    const project = await getProjectForUser(projectId, req.dbUser!.id);
    if (!project) return sendError(res, 404, 'Project not found.');

    const updated = await removeAssetsFromProject(project.id, req.dbUser!.id, [assetId]);
    if (updated.length === 0) {
      return sendError(res, 404, 'Artwork not found in this project.');
    }

    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'project',
      project.id,
      'removed_asset',
      `project=${project.name}; asset=${assetId}`,
    );

    res.json({
      project_id: project.id,
      project_name: project.name,
      removed_count: 1,
      asset_id: assetId,
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to remove artwork from project.');
  }
});

async function storeUpload(
  projectId: string | null,
  userId: number,
  file: Express.Multer.File | undefined,
  rawLogicalName: string,
  parentAssetId: string | null = null,
) {
  if (projectId !== null) {
    const project = await getProjectForUser(projectId, userId);
    if (!project) {
      throw Object.assign(new Error('Project not found.'), { status: 404 });
    }
  }
  if (!file) {
    throw Object.assign(new Error('Empty upload.'), { status: 400 });
  }
  const suffix = path.extname(file.originalname || '').toLowerCase();
  if (!ALLOWED_SUFFIXES.has(suffix)) {
    throw Object.assign(new Error('Only PNG, JPG, JPEG and WEBP images are supported.'), { status: 415 });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    throw Object.assign(
      new Error(`Upload exceeds ${Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024))} MB.`),
      { status: 413 },
    );
  }
  if (!file.buffer || file.buffer.length === 0) {
    throw Object.assign(new Error('Empty upload.'), { status: 400 });
  }

  let imgWidth = 0;
  let imgHeight = 0;
  try {
    const meta = await sharp(file.buffer).metadata();
    if (!meta.width || !meta.height) {
      throw new Error('Invalid image dimensions');
    }
    imgWidth = meta.width;
    imgHeight = meta.height;
  } catch {
    throw Object.assign(new Error('The uploaded file is not a valid image.'), { status: 415 });
  }

  const baseName = (rawLogicalName || path.parse(file.originalname || 'artwork').name).trim();
  const logicalName =
    baseName.replace(/[^A-Za-z0-9 _.-]/g, '_').slice(0, 180).replace(/^[ .]+|[ .]+$/g, '') || 'artwork';

  const sha256 = crypto.createHash('sha256').update(file.buffer).digest('hex');
  const duplicateMatch = await findDuplicateAssetBySha(userId, sha256);

  const version = await getNextAssetVersion(userId, projectId, logicalName);
  const aid = uid();
  // Store uploads under user/asset directory so project assignment never moves or duplicates files
  const destDir = path.join(STORAGE_ROOT, 'users', String(userId), 'assets', aid);
  fs.mkdirSync(destDir, { recursive: true });
  const filename = path
    .basename(file.originalname || `upload${suffix}`)
    .replace(/[^A-Za-z0-9_.-]/g, '_')
    .slice(0, 180);
  const dest = path.join(destDir, `source${suffix}`);
  fs.writeFileSync(dest, file.buffer);

  const asset = await createAssetRecord({
    id: aid,
    userId,
    projectId,
    logicalName,
    filename,
    storagePath: path.resolve(dest),
    sha256,
    width: imgWidth,
    height: imgHeight,
    fileSize: file.buffer.length,
    version,
    parentAssetId,
    status: 'uploaded',
  });
  await addAuditEvent(
    uid(),
    userId,
    'asset',
    aid,
    'uploaded',
    `name=${logicalName}; version=${version}; assigned=${projectId ? 'project' : 'inbox'}`,
  );
  return { asset, isDuplicate: Boolean(duplicateMatch) };
}

// Independent Artwork Inbox upload (no project required)
app.post('/api/assets/upload', requireAuth, upload.single('file'), async (req: AuthRequest, res) => {
  try {
    const logicalName = typeof req.body?.logical_name === 'string' ? req.body.logical_name : '';
    const { asset, isDuplicate } = await storeUpload(null, req.dbUser!.id, req.file, logicalName, null);
    res.status(201).json(serializeAssetRow(asset, [], { projectName: null, isDuplicate }));
  } catch (err) {
    const status = (err as { status?: number }).status || 500;
    const msg = err instanceof Error ? err.message : 'Upload failed.';
    sendError(res, status, msg);
  }
});

// Direct project-scoped upload (preserved for backwards compatibility)
app.post('/api/projects/:projectId/assets', requireAuth, upload.single('file'), async (req: AuthRequest, res) => {
  try {
    const projectId = String(req.params.projectId);
    const project = await getProjectForUser(projectId, req.dbUser!.id);
    if (!project) return sendError(res, 404, 'Project not found.');
    const logicalName = typeof req.body?.logical_name === 'string' ? req.body.logical_name : '';
    const { asset, isDuplicate } = await storeUpload(projectId, req.dbUser!.id, req.file, logicalName, null);
    res.status(201).json(serializeAssetRow(asset, [], { projectName: project.name, isDuplicate }));
  } catch (err) {
    const status = (err as { status?: number }).status || 500;
    const msg = err instanceof Error ? err.message : 'Upload failed.';
    sendError(res, status, msg);
  }
});

app.post('/api/assets/:assetId/revisions', requireAuth, upload.single('file'), async (req: AuthRequest, res) => {
  try {
    const found = await getAssetForUser(String(req.params.assetId), req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    const { asset, isDuplicate } = await storeUpload(
      found.asset.projectId,
      req.dbUser!.id,
      req.file,
      found.asset.logicalName,
      found.asset.id,
    );
    res.status(201).json(serializeAssetRow(asset, [], { projectName: found.projectName, isDuplicate }));
  } catch (err) {
    const status = (err as { status?: number }).status || 500;
    const msg = err instanceof Error ? err.message : 'Revision upload failed.';
    sendError(res, status, msg);
  }
});

app.get('/api/assets', requireAuth, async (req: AuthRequest, res) => {
  try {
    const projectId = typeof req.query.project_id === 'string' && req.query.project_id ? req.query.project_id : null;
    const status = typeof req.query.status === 'string' && req.query.status ? req.query.status : null;
    const list = await listAssetsWithRuns(req.dbUser!.id, { projectId, status });
    res.json(list);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load assets.');
  }
});

function isInsideStorageRoot(targetPath: string): boolean {
  const root = path.resolve(STORAGE_ROOT);
  const resolved = path.resolve(targetPath);
  return resolved.startsWith(`${root}${path.sep}`);
}

function cleanupDeletedAssetFiles(
  deletedAssets: Array<{ id: string; userId: number | null; storagePath: string }>,
  runOutputDirs: string[],
) {
  for (const dir of runOutputDirs) {
    if (isInsideStorageRoot(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
  for (const asset of deletedAssets) {
    const assetRunsRoot = path.join(STORAGE_ROOT, 'assets', asset.id);
    if (isInsideStorageRoot(assetRunsRoot)) {
      fs.rmSync(assetRunsRoot, { recursive: true, force: true });
    }
    if (asset.userId !== null) {
      const userAssetDir = path.join(STORAGE_ROOT, 'users', String(asset.userId), 'assets', asset.id);
      if (isInsideStorageRoot(userAssetDir)) {
        fs.rmSync(userAssetDir, { recursive: true, force: true });
      }
    }
    if (asset.storagePath && isInsideStorageRoot(asset.storagePath)) {
      fs.rmSync(asset.storagePath, { force: true });
    }
  }
}

app.post('/api/assets/delete', requireAuth, async (req: AuthRequest, res) => {
  try {
    const body = (req.body || {}) as { asset_ids?: unknown };
    const rawIds = Array.isArray(body.asset_ids) ? body.asset_ids : [];
    const assetIds = Array.from(new Set(rawIds.map((x) => String(x).trim()).filter(Boolean)));
    if (assetIds.length === 0) {
      return sendError(res, 422, 'Select at least one artwork to delete.');
    }

    const found = await Promise.all(assetIds.map((id) => getAssetForUser(id, req.dbUser!.id)));
    const owned = found.filter((x): x is NonNullable<typeof x> => Boolean(x));
    if (owned.length === 0) {
      return sendError(res, 404, 'No matching artworks found.');
    }
    if (owned.some((x) => x.asset.status === 'processing')) {
      return sendError(res, 409, 'Cannot delete an artwork while processing is in progress.');
    }

    const { deletedAssets, runOutputDirs } = await deleteAssetsForUser(
      req.dbUser!.id,
      owned.map((x) => x.asset.id),
    );
    cleanupDeletedAssetFiles(deletedAssets, runOutputDirs);

    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'asset',
      deletedAssets[0]?.id || assetIds[0],
      'deleted',
      `count=${deletedAssets.length}; assets=${deletedAssets.map((a) => `${a.id}:${a.logicalName}`).join(',')}`,
    );

    res.json({
      deleted_count: deletedAssets.length,
      asset_ids: deletedAssets.map((a) => a.id),
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to delete artworks.');
  }
});

app.delete('/api/assets/:assetId', requireAuth, async (req: AuthRequest, res) => {
  try {
    const assetId = String(req.params.assetId);
    const found = await getAssetForUser(assetId, req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    if (found.asset.status === 'processing') {
      return sendError(res, 409, 'Cannot delete an artwork while processing is in progress.');
    }

    const { deletedAssets, runOutputDirs } = await deleteAssetsForUser(req.dbUser!.id, [assetId]);
    cleanupDeletedAssetFiles(deletedAssets, runOutputDirs);

    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'asset',
      assetId,
      'deleted',
      `name=${found.asset.logicalName}; version=${found.asset.version}`,
    );

    res.json({
      deleted_count: deletedAssets.length,
      asset_id: assetId,
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to delete artwork.');
  }
});

app.get('/api/assets/:assetId/source', requireAuth, async (req: AuthRequest, res) => {
  try {
    const found = await getAssetForUser(String(req.params.assetId), req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    const safePath = safeStorageFile(found.asset.storagePath);
    if (!safePath) return sendError(res, 404, 'Source image not found.');
    res.sendFile(safePath);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load source image.');
  }
});

app.post('/api/assets/:assetId/process', requireAuth, async (req: AuthRequest, res) => {
  try {
    const found = await getAssetForUser(String(req.params.assetId), req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    const asset = found.asset;
    if (asset.status === 'processing') return sendError(res, 409, 'Asset is already processing.');

    const payload = (req.body || {}) as {
      params?: Record<string, unknown>;
      selected_engine?: string;
      run_stability?: boolean;
    };
    const selectedEngine = (payload.selected_engine || 'D').toUpperCase();
    if (!['A', 'B', 'C', 'D'].includes(selectedEngine)) {
      return sendError(res, 422, 'selected_engine must be A, B, C or D.');
    }

    let params;
    try {
      params = validateParams(payload.params || {});
    } catch (err) {
      return sendError(res, 422, err instanceof Error ? err.message : 'Invalid parameters.');
    }

    const runStability = Boolean(payload.run_stability);
    const runConfig = {
      schema: 'chromabook-advanced-mapping-v2',
      params,
      selected_engine: selectedEngine,
      run_stability: runStability,
    };

    const run = await createProcessingRun({
      id: uid(),
      assetId: asset.id,
      state: 'queued',
      engine: selectedEngine,
      paramsJson: JSON.stringify(runConfig),
    });
    await updateAssetStatus(asset.id, 'processing');
    await addAuditEvent(
      uid(),
      req.dbUser!.id,
      'asset',
      asset.id,
      'processing_queued',
      `run=${run.id}; engine=${selectedEngine}; schema=chromabook-advanced-mapping-v2`,
    );

    setImmediate(() => {
      void executeProcessJob(asset.id, run.id);
    });

    res.status(202).json({
      asset_id: asset.id,
      run_id: run.id,
      state: 'queued',
      selected_engine: selectedEngine,
      params,
      run_stability: runStability,
    });
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to queue processing.');
  }
});

app.post('/api/assets/:assetId/review', requireAuth, async (req: AuthRequest, res) => {
  try {
    const found = await getAssetForUser(String(req.params.assetId), req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    const asset = found.asset;
    const runs = await getRunsForAsset(asset.id);
    const latest = runs[0] ?? null;
    if (!latest || latest.state !== 'completed') {
      return sendError(res, 409, 'A completed processing run is required before review.');
    }
    const body = (req.body || {}) as { action?: string; note?: string };
    if (body.action !== 'approve' && body.action !== 'reject') {
      return sendError(res, 422, 'action must be approve or reject.');
    }
    const nextStatus = body.action === 'approve' ? 'approved' : 'rejected';
    await updateAssetStatus(asset.id, nextStatus);
    asset.status = nextStatus;
    const note = typeof body.note === 'string' ? body.note.slice(0, 2000) : '';
    await addAuditEvent(uid(), req.dbUser!.id, 'asset', asset.id, body.action, note);
    res.json(serializeAssetRow(asset, runs, { projectName: found.projectName }));
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to save review.');
  }
});

const ALLOWED_PREVIEW_FILES = new Set([
  'color_preview.png',
  'numbered_preview.png',
  'asset_summary.json',
  'engine_comparison_metrics.json',
  'config.json',
  'render_preflight.json',
  'stability_metrics.json',
]);

function isAllowedPreviewName(filename: string): boolean {
  if (ALLOWED_PREVIEW_FILES.has(filename)) return true;
  return /^engine_[ABCD]\/(color_preview|numbered_preview|metrics)\.(png|json)$/.test(filename);
}

app.get(/^\/api\/assets\/([^/]+)\/runs\/([^/]+)\/preview\/(.+)$/, requireAuth, async (req: AuthRequest, res) => {
  try {
    const assetId = req.params[0];
    const runId = req.params[1];
    const filename = req.params[2];
    if (!isAllowedPreviewName(filename)) return sendError(res, 404, 'Preview not found.');

    const found = await getAssetForUser(assetId, req.dbUser!.id);
    const run = await getRunById(runId);
    if (!found || !run || run.assetId !== found.asset.id || run.state !== 'completed' || !run.outputDir) {
      return sendError(res, 404, 'Processing run not found.');
    }
    const safePath = safeStorageFile(path.join(run.outputDir, filename));
    if (!safePath) return sendError(res, 404, 'File not found.');
    res.type(path.extname(safePath) === '.png' ? 'image/png' : 'application/json');
    res.sendFile(safePath);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to fetch preview.');
  }
});

app.get(/^\/api\/assets\/([^/]+)\/preview\/(.+)$/, requireAuth, async (req: AuthRequest, res) => {
  try {
    const assetId = req.params[0];
    const filename = req.params[1];
    if (!isAllowedPreviewName(filename)) return sendError(res, 404, 'Preview not found.');

    const found = await getAssetForUser(assetId, req.dbUser!.id);
    if (!found) return sendError(res, 404, 'Asset not found.');
    const runs = await getRunsForAsset(found.asset.id);
    const run = runs[0] ?? null;
    if (!run || run.state !== 'completed' || !run.outputDir) {
      return sendError(res, 404, 'No completed preview.');
    }
    const safePath = safeStorageFile(path.join(run.outputDir, filename));
    if (!safePath) return sendError(res, 404, 'File not found.');
    const ext = path.extname(safePath);
    res.type(ext === '.png' ? 'image/png' : ext === '.md' ? 'text/markdown' : 'application/json');
    res.sendFile(safePath);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to fetch preview.');
  }
});

app.get('/api/runs', requireAuth, async (req: AuthRequest, res) => {
  try {
    const rawLimit = Number.parseInt(String(req.query.limit || '100'), 10);
    const limit = Number.isFinite(rawLimit) ? Math.max(1, Math.min(rawLimit, 500)) : 100;
    const rows = await listRunsForUser(req.dbUser!.id, limit);
    res.json(rows);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load processing history.');
  }
});

app.post('/api/books', requireAuth, async (req: AuthRequest, res) => {
  try {
    const body = (req.body || {}) as {
      project_id?: string;
      asset_ids?: string[];
      title?: string;
      subtitle?: string;
    };
    const project =
      typeof body.project_id === 'string'
        ? await getProjectForUser(body.project_id, req.dbUser!.id)
        : null;
    if (!project) return sendError(res, 404, 'Project not found.');

    const assetIds = Array.isArray(body.asset_ids) ? body.asset_ids : [];
    if (assetIds.length === 0 || assetIds.length > 80) {
      return sendError(res, 422, 'Select between 1 and 80 approved assets.');
    }
    if (new Set(assetIds).size !== assetIds.length) {
      return sendError(res, 400, 'Duplicate asset IDs are not allowed.');
    }
    const selectedAssets = await Promise.all(assetIds.map((id) => getAssetForUser(id, req.dbUser!.id)));
    if (selectedAssets.some((found) => !found || found.asset.projectId !== project.id)) {
      return sendError(res, 400, 'All selected assets must belong to this project.');
    }
    if (selectedAssets.some((found) => found!.asset.status !== 'approved')) {
      return sendError(res, 409, 'Only approved assets can be assembled into a book.');
    }

    const title = typeof body.title === 'string' ? body.title.trim() : '';
    if (!title || title.length > 180) {
      return sendError(res, 422, 'Book title must be between 1 and 180 characters.');
    }
    const subtitle = typeof body.subtitle === 'string' ? body.subtitle.trim().slice(0, 300) : '';

    const book = await createBookRecord({
      id: uid(),
      projectId: project.id,
      title,
      subtitle,
      assetIdsJson: JSON.stringify(assetIds),
      status: 'queued',
    });
    await addAuditEvent(uid(), req.dbUser!.id, 'book', book.id, 'build_queued', `assets=${selectedAssets.length}`);

    setImmediate(() => {
      void executeBookJob(book.id);
    });

    res.status(202).json(serializeBookRow(book, project.name));
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to create book.');
  }
});

app.get('/api/books', requireAuth, async (req: AuthRequest, res) => {
  try {
    const projectId = typeof req.query.project_id === 'string' ? req.query.project_id : null;
    const rows = await listBooksForUser(req.dbUser!.id, projectId);
    res.json(rows);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load books.');
  }
});

app.get('/api/books/:bookId/download/:kind', requireAuth, async (req: AuthRequest, res) => {
  try {
    const book = await getBookForUser(String(req.params.bookId), req.dbUser!.id);
    if (!book) return sendError(res, 404, 'Book not found.');
    const kind = String(req.params.kind);
    const filePath =
      kind === 'book'
        ? book.bookPdfPath
        : kind === 'answer-key'
          ? book.answerKeyPath
          : kind === 'report'
            ? book.reportPath
            : null;
    if (!filePath) return sendError(res, 404, 'Requested artifact is not ready.');
    const safePath = safeStorageFile(filePath);
    if (!safePath) return sendError(res, 404, 'File not found.');
    res.download(safePath, path.basename(safePath));
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to download artifact.');
  }
});

app.get('/api/audit', requireAuth, async (req: AuthRequest, res) => {
  try {
    const rawLimit = Number.parseInt(String(req.query.limit || '100'), 10);
    const limit = Number.isFinite(rawLimit) ? Math.max(1, Math.min(rawLimit, 500)) : 100;
    const rows = await listAuditEventsForUser(req.dbUser!.id, limit);
    res.json(
      rows.map((r) => ({
        id: r.id,
        entity_type: r.entityType,
        entity_id: r.entityId,
        action: r.action,
        detail: r.detail,
        created_at: r.createdAt.toISOString(),
      })),
    );
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to load audit log.');
  }
});

// Express error handler (including multer file size errors)
app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (err && typeof err === 'object' && 'code' in err && (err as { code?: string }).code === 'LIMIT_FILE_SIZE') {
    res.status(413).json({ detail: `Upload exceeds ${Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024))} MB.` });
    return;
  }
  const msg = err instanceof Error ? err.message : 'Internal server error.';
  res.status(500).json({ detail: msg });
});

const PORT = Number.parseInt(process.env.PORT || '3000', 10) || 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`ChromaBook Studio listening on http://0.0.0.0:${PORT}`);
});
