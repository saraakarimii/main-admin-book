"""Book-level PDF assembly, covers, folios and preflight (English edition)."""
from __future__ import annotations
import json, math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from pypdf import PdfReader, PdfWriter
from .geometry import MM
from .pdf import build_pdf


def _cover(path, layout, title, subtitle, collection="CHROMABOOK"):
    c = canvas.Canvas(str(path), pagesize=(layout.page_w*MM, layout.page_h*MM))
    w, h = layout.page_w*MM, layout.page_h*MM
    c.setFillColor(HexColor('#17152E')); c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(HexColor('#F7F1E8')); c.setFont('Helvetica-Bold', 13)
    c.drawCentredString(w/2, h*0.82, collection.upper())
    c.setStrokeColor(HexColor('#D6B66B')); c.setLineWidth(1.2)
    c.line(w*0.2, h*0.77, w*0.8, h*0.77)
    max_width=w*0.78
    size=30
    while size>16 and stringWidth(title, 'Helvetica-Bold', size)>max_width: size-=1
    c.setFillColor(HexColor('#FFFFFF')); c.setFont('Helvetica-Bold', size)
    c.drawCentredString(w/2, h*0.57, title.upper())
    c.setFillColor(HexColor('#E6DDBF')); c.setFont('Helvetica', 13)
    # Keep subtitle on one/two lines without clipping.
    words=subtitle.split(); lines=[]; line=''
    for word in words:
        test=(line+' '+word).strip()
        if stringWidth(test,'Helvetica',13)>max_width and line:
            lines.append(line); line=word
        else: line=test
    if line: lines.append(line)
    for i, ln in enumerate(lines[:3]): c.drawCentredString(w/2, h*0.49-i*18, ln)
    c.setFillColor(HexColor('#D6B66B')); c.setFont('Helvetica', 9)
    c.drawCentredString(w/2, h*0.12, 'COLOR-BY-NUMBER COLLECTION')
    c.showPage(); c.save()


def _add_folios(input_pdf, output_pdf, layout, start=1):
    src=PdfReader(str(input_pdf)); writer=PdfWriter()
    for idx, page in enumerate(src.pages, start=start):
        from io import BytesIO
        buff=BytesIO(); c=canvas.Canvas(buff, pagesize=(layout.page_w*MM, layout.page_h*MM))
        c.setFillColor(HexColor('#333333')); c.setFont('Helvetica', 8)
        c.drawCentredString(layout.page_w*MM/2, 6*MM, str(idx))
        c.save(); buff.seek(0)
        page.merge_page(PdfReader(buff).pages[0]); writer.add_page(page)
    with open(output_pdf,'wb') as f: writer.write(f)
    return len(src.pages)


def preflight_book(artworks, palette, layout, min_cell_mm=3.0):
    """Validate palette IDs, artwork indices, physical cell size, number fit and slot accounting."""
    from .geometry import all_plans, GuideSpec
    import numpy as np
    ids_ok = all(1 <= len(str(cid)) <= 3 for cid in palette.ids)
    images=[]
    for art in artworks:
        labels=np.asarray(art["labels"])
        valid=labels.ndim==2 and labels.size>0 and int(labels.min())>=0 and int(labels.max())<len(palette)
        plans=all_plans(labels.shape[1],labels.shape[0],len(np.unique(labels)),(layout,),GuideSpec(),min_cell_mm) if valid else []
        valid_plans=[p for p in plans if p.meets_min_cell]
        plan=max(valid_plans,key=lambda p:p.cell_mm) if valid_plans else None
        max_id=max((len(str(x)) for x in palette.ids),default=0)
        # PDF uses Helvetica at 0.52 cell; conservative width check against cell width.
        from reportlab.pdfbase.pdfmetrics import stringWidth
        number_fits=bool(plan) and all(stringWidth(str(cid),'Helvetica',0.52*plan.cell_mm*MM)<=plan.cell_mm*MM*0.92 for cid in palette.ids)
        images.append({'name':art.get('name','unnamed'),'valid_palette_indices':valid,
                       'cell_mm':round(plan.cell_mm,3) if plan else None,
                       'cell_size_pass':bool(plan),'number_fits':bool(number_fits),
                       'margins_within_slot':bool(plan and plan.boxes['artwork'][0]>=-1e-8 and plan.boxes['artwork'][1]>=-1e-8)})
    pages=math.ceil(len(artworks)/layout.slots) if artworks else 0
    blank_slots=pages*layout.slots-len(artworks)
    passed=bool(artworks) and ids_ok and all(x['valid_palette_indices'] and x['cell_size_pass'] and x['number_fits'] and x['margins_within_slot'] for x in images)
    return {'passed':passed,'palette':{'count':len(palette),'ids':palette.ids,'ids_valid':ids_ok},
            'images':images,'pages':pages,'blank_slots':blank_slots,'minimum_cell_mm':min_cell_mm,
            'reading_order':layout.reading_order,'slots_per_page':layout.slots}


def assemble_book(output_dir, title, subtitle, artworks, palette, layout, min_cell_mm=3.0, collection='CHROMABOOK'):
    """artworks: [{name, labels}]. Creates cover + numbered interior and separate answer key."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    cover=out/'cover.pdf'; _cover(cover,layout,title,subtitle,collection)
    raw_numbered=out/'_interior_numbered_raw.pdf'; raw_key=out/'_answer_key_raw.pdf'
    placed=build_pdf(raw_numbered,layout,artworks,palette,min_cell_mm=min_cell_mm,colored=False)
    build_pdf(raw_key,layout,artworks,palette,min_cell_mm=min_cell_mm,colored=True)
    interior=out/'interior_numbered.pdf'; pages=_add_folios(raw_numbered,interior,layout,start=1)
    # Cover is its own first page in the book PDF; interior folios start at 1 by design.
    writer=PdfWriter(); writer.append(str(cover)); writer.append(str(interior))
    book=out/'book.pdf'
    with open(book,'wb') as f: writer.write(f)
    answer=out/'answer_key.pdf'; _add_folios(raw_key,answer,layout,start=1)
    raw_numbered.unlink(missing_ok=True); raw_key.unlink(missing_ok=True)
    preflight=preflight_book(artworks,palette,layout,min_cell_mm)
    report={
      'title':title,'subtitle':subtitle,'collection':collection,'language':'en',
      'palette':{'id':palette.palette_id,'version':palette.version,'color_count':len(palette),'ids':palette.ids},
      'layout':layout.name,'page_size_mm':[layout.page_w,layout.page_h],
      'artwork_count':len(artworks),'interior_pages':pages,'book_pages':pages+1,
      'cover_is_separate_page':True,'answer_key':'separate PDF',
      'reading_order':layout.reading_order,'slots_per_page':layout.slots,
      'blank_slot_rule':'unused final slot remains blank; no placeholder artwork is inserted',
      'placements':[{'name':p['name'],'page':p['page'],'slot_index_left_to_right':p['slot_index_left_to_right'],
                     'grid':[int(p['plan'].cols),int(p['plan'].rows)],'cell_mm':round(p['plan'].cell_mm,3)} for p in placed],
      'preflight':{**preflight,'palette_unchanged':len(palette)==21,'all_artworks_placed':len(placed)==len(artworks),
                   'page_count_consistent':pages==math.ceil(len(artworks)/layout.slots)}
    }
    report['preflight']['passed']=bool(preflight['passed'] and report['preflight']['palette_unchanged'] and report['preflight']['all_artworks_placed'] and report['preflight']['page_count_consistent'])
    (out/'book_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'book_report.md').write_text('# Book report\n\n'+ '\n'.join(f'- **{k}:** {v}' for k,v in report.items() if k not in ('placements','preflight'))+'\n\n## Preflight\n'+'\n'.join(f'- {k}: {v}' for k,v in report['preflight'].items())+'\n',encoding='utf-8')
    return {'book':str(book),'answer_key':str(answer),'report':str(out/'book_report.json'),'preflight':report['preflight']}
