"""Build the editable memo, then typeset the saved DOCX content as a PDF."""

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph as WordParagraph
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table as PdfTable, TableStyle

HERE = Path(__file__).resolve().parent
IMAGE = HERE.parent / 'manuscript' / 'figures' / 'decision_tradeoff.png'
DOCX = HERE / 'Decision_Memo.docx'
PDF = HERE / 'Decision_Memo.pdf'
INK, MUTED, TEAL, PALE, BORDER = '182932', '425963', '106F68', 'EDF5F5', 'D9D9D9'


def fill(cell, color):
    props = cell._tc.get_or_add_tcPr()
    shade = OxmlElement('w:shd')
    shade.set(qn('w:fill'), color)
    props.append(shade)


def word_table(doc, rows, widths):
    table = doc.add_table(rows=0, cols=len(widths))
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for col, width in zip(table.columns, widths):
        col.width = Inches(width)
    props = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:'+edge)
        for key, value in {'val':'single', 'sz':'4', 'color':BORDER}.items():
            border.set(qn('w:'+key),value)
        borders.append(border)
    props.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge in ('top','left','bottom','right'):
        e=OxmlElement('w:'+edge); e.set(qn('w:w'),'115'); e.set(qn('w:type'),'dxa'); margins.append(e)
    props.append(margins)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        if i == 0:
            header = OxmlElement('w:tblHeader')
            table.rows[0]._tr.get_or_add_trPr().append(header)
        for cell, label, width in zip(cells,row,widths):
            cell.width=Inches(width)
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            fill(cell, '164C52' if i==0 else ('F1F7F7' if i%2==0 else 'FFFFFF'))
            p = cell.paragraphs[0]
            p.paragraph_format.space_after=Pt(0)
            run=p.add_run(label)
            run.bold=(i==0)
            run.font.size=Pt(10)
            run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else INK)
    doc.add_paragraph('',style='Memo Spacer')


def create_word():
    d = Document()
    d.core_properties.title = 'Review Policy for Sensor Outages'
    d.core_properties.author = 'James Jean'
    d.core_properties.subject = 'Operational recommendation based on a controlled synthetic benchmark'
    d.core_properties.comments = ''
    s = d.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin, s.bottom_margin = Inches(.67), Inches(.68)
    s.left_margin, s.right_margin = Inches(.72), Inches(.72)
    from docx.enum.style import WD_STYLE_TYPE
    for name in ('Memo Eyebrow','Memo Lead','Memo Caption','Memo Spacer'):
        d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
    specs = {
        'Normal':(10.8,INK,False,1.15,0,7),
        'Title':(31,'000000',True,1.02,0,7),
        'Subtitle':(11,'000000',False,1.13,0,12),
        'Heading 1':(16,'000000',True,1.10,7,5),
        'Memo Eyebrow':(9,'000000',True,1.10,0,8),
        'Memo Lead':(12,INK,True,1.15,0,10),
        'Memo Caption':(9,MUTED,False,1.14,3,7),
        'Memo Spacer':(2,INK,False,1,0,4),
    }
    for name,(size,color,bold,line,above,below) in specs.items():
        st=d.styles[name]; st.font.name='Arial'; st.font.size=Pt(size)
        st.font.color.rgb=RGBColor.from_string(color); st.font.bold=bold; st.font.italic=False
        fmt=st.paragraph_format; fmt.line_spacing=line; fmt.space_before=Pt(above); fmt.space_after=Pt(below)
        if name in ('Title','Subtitle','Heading 1','Memo Eyebrow'):
            fmt.keep_with_next=True
        for node in st.element.xpath('./w:pPr/w:pBdr'):
            node.getparent().remove(node)
    p=d.add_paragraph
    p('OPERATIONS DECISION MEMO  /  NOVEMBER 2026','Memo Eyebrow')
    p('Review Policy for Sensor Outages','Title')
    p('For telemetry operations and reliability leads | James Jean','Subtitle')
    p('Advance the confidence rule to shadow evaluation. Keep it out of live decisions until missed-event risk and review capacity are measured.','Memo Lead')
    p('When one of two synchronized channels goes missing, an automated event label may still be useful. The question is when to trust it and when to send the window to a person. I compared those choices on the same 1,800 later synthetic windows from 30 held-out device instances.')
    p('What the two options trade','Heading 1')
    word_table(d, [
        ('Sustained dropout','Issue every label','Use frozen confidence rule'),
        ('Automated labels','1,800 / 1,800','1,438 / 1,800 (79.9%)'),
        ('Wrong automated labels','431 / 1,800 (23.9%)','274 / 1,438 (19.1%)'),
        ('Windows sent to review','0','362 / 1,800 (20.1%)'),
    ],[2.25,1.94,2.87])
    p('Of the 362 deferred items, 157 would have received a wrong label and 205 a correct one. That does not say how many a reviewer would get right. Of the 274 wrong labels the rule still issued, 260 were missed events.')
    picture=p().add_run().add_picture(str(IMAGE),width=Inches(6.9))
    picture._inline.docPr.set('descr','Sustained dropout: issuing all produces 431 wrong labels. The fixed rule issues 274 wrong labels and defers 362 windows. Illustrative costs cross at 0.434 units per deferral.')
    p('Figure 1. Same 1,800 windows in both options. Deferred labels are counterfactual model outputs. Costs are assumptions, not dollars or measured reviewer outcomes.','Memo Caption')
    d.add_page_break()
    p('VALIDATION PLAN','Memo Eyebrow')
    p('What to Test Before Rollout','Title')
    p('Budget the review queue','Heading 1')
    p('Per 1,000 windows, the fixed rule would defer about 122 clean, 130 brief-dropout, or 201 sustained-dropout windows if the corresponding benchmark condition repeated. Use the actual mix, arrival volume and review duration for staffing. Stress-test bursts and long outages separately.')
    p('The simplified cost model values a wrong automated label at one unit. At the benchmark weight of 0.18 units per deferral, the rule costs 0.188 units per window versus 0.239 for issue-all. The rule breaks even at 0.434 units per deferral. Reviewer errors, delays and unequal error costs can change that comparison.')
    p('How to run the shadow test','Heading 1')
    p('Keep existing alerts active. Log scores, masks, outage types, decisions and queue timestamps. On later time blocks and held-out devices, compare the frozen rule with issue-all and an explicit outage escalation rule. Adjudicate a stratified sample of issued and deferred items, including high-confidence no-event labels.')
    p('Agree on decision gates before collecting results','Heading 1')
    word_table(d, [
        ('Gate','Advance only when','If the gate fails'),
        ('Missed events','The upper uncertainty bound is below the pre-agreed missed-event limit.','Escalate outages and investigate confident misses.'),
        ('Review capacity','Workload and queue age stay within agreed limits during outage bursts.','Keep shadow mode and revise queue or fallback policy.'),
        ('Evidence coverage','Common loss types, unseen devices, and later periods have enough adjudicated examples.','Collect more labels; keep live behavior unchanged.'),
    ],[1.32,3.14,2.6])
    p('If the team changes a cutoff, use new validation data and a fresh holdout. The synthetic benchmark gives a comparison and an audit trail; it does not establish field accuracy, staffing levels or savings.','Memo Caption')
    d.save(DOCX)


def markup(p):
    result = []
    for run in p.runs:
        text = escape(run.text).replace('\n','<br/>')
        if run.bold: text = '<b>'+text+'</b>'
        if run.italic: text = '<i>'+text+'</i>'
        result.append(text)
    return ''.join(result)


def pdf_from_saved_word():
    # Render this memo's Word styles and OOXML content without a desktop Office dependency.
    d=Document(DOCX)
    styles={}
    for name in ('Normal','Title','Subtitle','Heading 1','Memo Eyebrow','Memo Lead','Memo Caption','Memo Spacer'):
        style=d.styles[name]; font=style.font; fmt=style.paragraph_format
        styles[name]=ParagraphStyle(name,fontName='Helvetica-Bold' if font.bold else 'Helvetica',
            fontSize=font.size.pt,leading=font.size.pt*(fmt.line_spacing or 1.12),
            spaceBefore=fmt.space_before.pt,spaceAfter=fmt.space_after.pt,
            textColor=HexColor('#'+str(font.color.rgb)),alignment=TA_LEFT,
            keepWithNext=bool(fmt.keep_with_next),allowWidows=0,allowOrphans=0)
    table_normal=ParagraphStyle('Table body',fontName='Helvetica',fontSize=9.7,leading=12,textColor=HexColor('#'+INK))
    table_header=ParagraphStyle('Table header',parent=table_normal,fontName='Helvetica-Bold',textColor=white)
    story=[]
    for child in d._element.body:
        if child.tag==qn('w:p'):
            p=WordParagraph(child,d)
            if child.xpath('.//w:br[@w:type="page"]'):
                story.append(PageBreak())
                continue
            found=child.xpath('.//a:blip')
            if found:
                rel=found[0].get(qn('r:embed'))
                blob=d.part.related_parts[rel].blob
                image=Image(BytesIO(blob))
                image.drawWidth=6.9*72
                image.drawHeight=image.imageHeight/image.imageWidth*image.drawWidth
                story.append(image)
            elif p.text or p.style.name=='Memo Spacer':
                story.append(Paragraph(markup(p) or '&#160;',styles[p.style.name]))
        if child.tag==qn('w:tbl'):
            t=Table(child,d)
            data=[[Paragraph(escape(c.text),table_header if i==0 else table_normal) for c in row.cells]
                  for i,row in enumerate(t.rows)]
            widths=[col.width.pt for col in t.columns]
            tab=PdfTable(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),.45,HexColor('#'+BORDER)),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#F1F7F7')]),
                ('BACKGROUND',(0,0),(-1,0),HexColor('#164C52')),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ]))
            story.append(tab)
    sec=d.sections[0]
    pdf=SimpleDocTemplate(str(PDF),pagesize=(sec.page_width.pt,sec.page_height.pt),
        leftMargin=sec.left_margin.pt,rightMargin=sec.right_margin.pt,
        topMargin=sec.top_margin.pt,bottomMargin=sec.bottom_margin.pt,
        title=d.core_properties.title,author=d.core_properties.author,
        subject=d.core_properties.subject,pageCompression=1)
    def footer(canvas,document):
        canvas.saveState()
        canvas.setFillColor(HexColor('#'+MUTED)); canvas.setFont('Helvetica',8.5)
        canvas.drawString(sec.left_margin.pt+6,26,'SENSOR OUTAGE DECISION MEMO')
        canvas.drawRightString(sec.page_width.pt-sec.right_margin.pt-6,26,str(document.page))
        canvas.restoreState()
    pdf.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__ == '__main__':
    create_word()
    pdf_from_saved_word()
