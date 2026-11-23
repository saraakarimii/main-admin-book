"""Assemble finished per-image labels into the three books and their reports."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chromabook_engine import Palette  # noqa: E402
from chromabook_engine.layout import SIDE_BY_SIDE, assemble_book  # noqa: E402


def main():
    manifest=json.loads((ROOT/'inputs/collections.json').read_text(encoding='utf-8'))
    palette=Palette.from_file(ROOT/manifest['palette'])
    if len(palette)!=21: raise RuntimeError('Palette must remain exactly 21 colours')
    out_root=ROOT/'outputs'; out_root.mkdir(exist_ok=True)
    summary={'version':'0.4.0','palette':palette.snapshot(),'collections':[]}
    t=time.perf_counter()
    for collection in manifest['collections']:
        c_out=out_root/collection['id']; artworks=[]; image_rows=[]
        for i,item in enumerate(collection['images'],1):
            stem=Path(item['path']).stem
            img_out=c_out/'images'/f'{i:02d}_{stem}'
            asset=json.loads((img_out/'asset_summary.json').read_text(encoding='utf-8'))
            labels=np.load(img_out/'labels.npy')
            artworks.append({'name':item['name'],'labels':labels,'source_path':str(ROOT/item['path'])})
            image_rows.append({'name':item['name'],'source':item['path'],'grid':asset['grid'],
                               'colors_used':asset['colors_used'],'engine':asset['engine'],
                               'confidence':asset['confidence'],'elapsed_s':asset['elapsed_s'],
                               'warnings':item.get('warnings',[])})
        warnings=[w for x in collection['images'] for w in x.get('warnings',[])]
        result=assemble_book(c_out,artworks,palette,collection['title'],collection['subtitle'],
                             collection['id'],ROOT/collection['cover'],
                             theme_color=tuple(collection['theme_color']),layout=SIDE_BY_SIDE,
                             language='English',min_cell_mm=3.0,max_id_chars=3,warnings=warnings)
        summary['collections'].append({
            'id':collection['id'],'title':collection['title'],'subtitle':collection['subtitle'],
            'image_count':len(artworks),'images':image_rows,
            'numbered_pdf':str(result['numbered_pdf'].relative_to(ROOT)),
            'answer_key_pdf':str(result['answer_key_pdf'].relative_to(ROOT)),
            'book_report':str((c_out/'book_report.json').relative_to(ROOT)),
            'preflight_passed':result['preflight']['passed'],
            'content_pages':result['preflight']['pages'],'blank_slots':result['preflight']['blank_slots']})
        print(f"Built {collection['title']}: {len(artworks)} images, {result['preflight']['pages']} content pages + cover, "
              f"{result['preflight']['blank_slots']} blank slot(s), preflight PASS")
    summary['elapsed_s']=round(time.perf_counter()-t,2)
    (out_root/'build_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"Build summary: {out_root/'build_summary.json'}")

if __name__=='__main__': main()
