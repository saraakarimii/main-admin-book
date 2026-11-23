"""Build a themed English ChromaBook collection from a JSON manifest.
Usage: python tools/build_collection.py manifest.json output_dir [--full-page]
Manifest: {"title":..., "subtitle":..., "images":["path1", ...]}
"""
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chromabook_engine import load_image, Palette
from chromabook_engine.optimize import optimize, OptimizerConfig, write_report
from chromabook_engine.layout import LayoutPrintSpec, SIDE_BY_SIDE, FULL_PAGE, assemble_book

manifest_path=Path(sys.argv[1]); out=Path(sys.argv[2]); full='--full-page' in sys.argv[3:]
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
root=manifest_path.parent
images=[(root/Path(x)).resolve() if not Path(x).is_absolute() else Path(x) for x in manifest['images']]
if not images: raise SystemExit('Manifest must contain at least one image.')
base=Path(__file__).resolve().parents[1]
palette=Palette.from_file(base/'data'/'main-21.palette.json')
if len(palette)!=21: raise SystemExit(f'Expected the locked 21-colour palette; found {len(palette)}.')
layout=FULL_PAGE if full else SIDE_BY_SIDE
arts=[]
for i,path in enumerate(images,1):
    print(f'[{i}/{len(images)}] Optimizing {path.name}',flush=True)
    src=load_image(path)
    spec=LayoutPrintSpec(layouts=(layout,),palette_size=len(palette))
    rep=optimize(src,palette,spec,OptimizerConfig(final_side=900))
    itemdir=out/'assets'/f'{i:02d}_{path.stem}'
    write_report(rep,itemdir)
    arts.append({'name':path.stem,'labels':rep.labels})
result=assemble_book(out,manifest['title'],manifest['subtitle'],arts,palette,layout,
                     min_cell_mm=3.0,collection=manifest.get('collection','CHROMABOOK'))
print(json.dumps(result,indent=2))
