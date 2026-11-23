#!/usr/bin/env bash
# Run one Python worker per image from a shell parent (avoids fork-after-NumPy issues).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAST_ARGS=()
if [[ "${1:-}" == "--fast" ]]; then FAST_ARGS+=(--fast); shift; fi
PREVIEW_SIDE="${1:-180}"
FINAL_SIDE="${2:-480}"
ENGINES="${3:-BCD}"
while IFS=$'\t' read -r image outdir; do
  printf "\n=== Processing %s ===\n" "$image"
  if [[ ${#FAST_ARGS[@]} -gt 0 ]]; then
    PYTHONPATH="$ROOT/src" python "$ROOT/tools/process_image.py" "$ROOT/$image" "$ROOT/$outdir" --fast
  else
    PYTHONPATH="$ROOT/src" python "$ROOT/tools/process_image.py" "$ROOT/$image" "$ROOT/$outdir" \
      --preview-side "$PREVIEW_SIDE" --final-side "$FINAL_SIDE" --engines "$ENGINES"
  fi
done < <(python - "$ROOT/inputs/collections.json" "$ROOT/outputs" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]).parent
out=Path(sys.argv[2])
manifest=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for collection in manifest['collections']:
    for i,item in enumerate(collection['images'],1):
        stem=Path(item['path']).stem
        outdir=out/collection['id']/'images'/f'{i:02d}_{stem}'
        print(f"{item['path']}\t{outdir}")
PY
)
