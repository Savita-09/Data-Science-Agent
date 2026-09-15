"""One evidence-based report outline rendered as self-contained HTML and paginated PDF."""
import html
import io
import json
import math
from pathlib import Path
import reportlab
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from app.tools.data import json_safe
from app.tools.dashboard import dataset_dashboard, dataset_relationship


def prepare_report(result, *, filename=None, request=None, frame=None):
    """Enrich a saved snapshot without changing its measurements or training again."""
    prepared = dict(result)
    prepared['report_metadata'] = {**result.get('report_metadata', {}), **({'filename': filename} if filename else {}), **({'request': request} if request else {})}
    if frame is not None and 'report_eda' not in prepared:
        dashboard = dataset_dashboard(frame, result['plan']['target'])
        columns = dashboard['relationship_columns']
        if len(columns) >= 2:
            y = result['plan']['target'] if result['plan']['target'] in columns else columns[1]
            x = next(column for column in columns if column != y)
            dashboard['relationship'] = dataset_relationship(frame, x, y)
        prepared['report_eda'] = dashboard
    return json_safe(prepared)


def fmt(value):
    if value is None: return 'N/A'
    if isinstance(value, float): return f'{value:,.5g}' if math.isfinite(value) else 'N/A'
    if isinstance(value, int): return f'{value:,}'
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False)
    return str(value)


def report_outline(result):
    p, c, f = result['profile'], result['cleaning'], result['features']
    models, ev, shap, insights = (result[k] for k in ('models', 'evaluation', 'explainability', 'insights'))
    meta = result.get('report_metadata', {})
    request = meta.get('request', {})
    eda = result.get('report_eda', result['eda'])
    sections = []
    def section(title, paragraphs=(), table=None, charts=()):
        sections.append({'title': title, 'paragraphs': list(paragraphs), 'table': table, 'charts': list(charts)})
    section('1. Problem and plan', [result['plan']['summary'], f"Task: {result['plan']['task']}. Target: {result['plan']['target'] or 'No target (clustering)'}."],
            [['Setting', 'Value'], ['Dataset', meta.get('filename', 'Saved dataset')], ['Analysis ID', result['id']], ['Random seed', request.get('seed', 'Not recorded')], ['CV folds', models.get('cv_folds', request.get('cv_folds'))], ['Tuning configurations per model', request.get('tuning_iterations', 'Not recorded')]])
    if result['plan'].get('reasoning'): section('Optional planning rationale', [result['plan']['reasoning']])
    section('2. Executive summary', [insights['summary'], 'Model rankings use cross-validation; held-out results measure a separate test partition. Correlation and SHAP describe associations, not causal effects.'])
    section('3. Dataset profile and quality', ['Quality statistics describe the original uploaded dataset.'],
            [['Measure', 'Value'], ['Rows', p['rows']], ['Columns', p['column_count']], ['Complete cells (%)', p['completeness']], ['Missing cells', p['missing_cells']], ['Duplicate rows', p['duplicate_rows']]])
    missing = [column for column in p['columns'] if column['missing']]
    if missing:
        section('Missing values', [f'{len(missing)} columns have missing data. Chart shows up to 15 columns with the most missing cells.'],
                charts=[{'kind':'bar', 'title':'Missing cells by column', 'items':[(v['name'],v['missing']) for v in sorted(missing,key=lambda v:v['missing'],reverse=True)[:15]]}])
    section('Column statistics', ['All recorded columns are listed. Long display values are shortened; the JSON export retains the stored values.'],
            [['Column', 'Type / unique', 'Missing', 'Mean / range']] + [[v['name'], f"{v['type']} / {v['unique']}", f"{v['missing']} ({v['missing_pct']}%)", f"{fmt(v.get('mean'))} / {fmt(v.get('min'))} to {fmt(v.get('max'))}" if v['type']=='numeric' else 'N/A'] for v in p['columns']])
    preview = p.get('preview', [])[:5]
    preview_cols = [v['name'] for v in p['columns']]
    for start in range(0, len(preview_cols), 4):
        cols = preview_cols[start:start+4]
        section('Dataset preview (first 5 rows)' if start == 0 else 'Dataset preview - continued columns', table=[['Row']+cols]+[[i+1]+[row.get(col) for col in cols] for i,row in enumerate(preview)])
    section('4. Exploratory data analysis (EDA)', [
        'EDA below describes the original upload before cleaning.' if 'report_eda' in result else 'Legacy EDA below describes cleaned data; stored distributions may omit infrequent categories.',
        eda.get('note', ''),
        'Distributions summarize observed values. Outliers use the 1.5 x IQR rule. Charts are descriptive and do not establish causes.'
    ])
    distributions = eda.get('distributions', [])
    for distribution in distributions:
        outliers = distribution.get('outliers')
        counts = sum(bin['count'] for bin in distribution['bins'])
        kind = distribution.get('kind', next((v['type'] for v in p['columns'] if v['name']==distribution['column']), 'categorical'))
        section('Distribution: '+distribution['column'], [f"{kind.capitalize()} column; {counts:,} observations represented. Outliers: {fmt(outliers)}."],
                table=([['Category / interval','Records']]+[[v['label'],v['count']] for v in distribution['bins']]) if any(len(v['label'])>32 for v in distribution['bins']) else None,
                charts=[{'kind':'bar','title':distribution['column']+' - records','items':[(v['label'],v['count']) for v in distribution['bins']]}])
    corr = eda.get('correlations', {'columns':[], 'values':[]})
    if len(corr['columns']) >= 2:
        # Bounded map remains legible on A4. Legacy snapshots can contain up to 30 columns.
        count = min(12, len(corr['columns']))
        columns = corr['columns'][:count]
        values = [row[:count] for row in corr['values'][:count]]
        pairs = sorted([(columns[i],columns[j],value) for i,row in enumerate(values) for j,value in enumerate(row) if j>i and isinstance(value,(float,int)) and math.isfinite(value)],key=lambda v:abs(v[2]),reverse=True)[:10]
        section('Numeric correlations', ['Pearson correlation for up to 12 numeric columns. Gray cells are undefined. Strongest associations are ranked by absolute correlation.'],
                table=[['Column A','Column B','Correlation']]+[list(row) for row in pairs], charts=[{'kind':'heatmap','title':'Numeric correlation map','columns':columns,'values':values}])
    relation = eda.get('relationship')
    if relation and relation.get('points'):
        section('Numeric relationship', [f"{relation['x']} versus {relation['y']}. Showing {relation['sample_rows']:,} points from {relation['valid_rows']:,} valid pairs; {relation['omitted_rows']:,} missing pairs excluded. Pearson r: {fmt(relation['correlation'])}. Sample points are reproducible; correlation uses all valid pairs."], charts=[{'kind':'scatter','title':relation['x']+' vs '+relation['y'], **relation}])
    section('5. Cleaning and feature engineering', [f"Rows before cleaning: {c['original_rows']:,}. Removed: {c['removed_rows']:,}. Usable rows: {c['clean_rows']:,}.", *c['actions'], *f['transformations']],
            [['Selected feature']]+[[name] for name in f['selected']])
    if f['excluded']: section('Excluded features and reasons', table=[['Feature','Reason']]+[[item['column'],item['reason']] for item in f['excluded']])
    section('6. Cross-validation leaderboard', [f"Selected model: {models['winner']}. Selection metric: {models['selection_metric']}. CV scores are used for selection, not the test scores."],
            [['Model / family','CV mean / std','Status / seconds','Best parameters']]+[[v['name']+' / '+v.get('family',''),f"{fmt(v.get('cv_score'))} / {fmt(v.get('cv_std'))}",v['status']+' / '+fmt(v.get('seconds')),v.get('parameters',v.get('error','N/A'))] for v in models['leaderboard']])
    section('7. Held-out evaluation', [f"Training rows: {ev['train_rows']:,}; test rows: {ev['test_rows']:,}. Baseline results provide a reference for the selected model."],
            [['Metric','Selected model','Baseline','Training']]+[[key, value, ev['baseline'].get(key), ev.get('training_metrics',{}).get(key)] for key,value in ev['metrics'].items() if not isinstance(value,(dict,list))])
    confusion = ev['metrics'].get('confusion_matrix')
    if confusion:
        classes = ev['classes']
        # Small blocks avoid wide tables for up to 30 classes.
        for start in range(0, len(classes), 5):
            section('Confusion matrix' if start==0 else 'Confusion matrix - continued predicted classes', ['Rows are actual classes; columns are predicted classes.'],
                    [['Actual / predicted']+classes[start:start+5]]+[[classes[i]]+row[start:start+5] for i,row in enumerate(confusion)])
    section('8. SHAP explainability', [shap.get('note',shap.get('reason','No explanation available.')), f"Method: {shap.get('method','N/A')}. Explained held-out rows: {shap.get('sample_rows',0)}. Units: {shap.get('units','N/A')}."])
    if shap.get('global'):
        section('Global feature importance', ['Top 20 features by mean absolute SHAP over the explained sample. Full names and values are listed below the chart.'],
                table=[['Feature','Mean absolute SHAP']]+[[v['feature'],v['mean_abs_shap']] for v in shap['global'][:20]],
                charts=[{'kind':'bar','title':'Mean absolute SHAP','items':[(v['feature'],v['mean_abs_shap']) for v in shap['global'][:20]]}])
    for local in shap.get('local',[])[:3]:
        section(f"Local explanation: row {local['row']}", [f"Output: {local['output']}. Base {fmt(local['base_value'])} + contribution sum {fmt(local['sum_contributions'])} = prediction {fmt(local['prediction'])}. Additivity residual: {fmt(local['additivity_error'])}."],
                [['Feature','Contribution']]+[[v['feature'],v['contribution']] for v in local.get('contributions',[])[:10]])
    section('9. Recommendations and limitations', insights['recommendations'] + ev['warnings'])
    if insights.get('llm_commentary'): section('Optional LLM commentary - advisory', [insights['llm_commentary'], *insights.get('llm_recommendations',[])])
    section('10. Using the saved model', ['Use the saved preprocessing pipeline with the same feature names. New-row predictions are available through the prediction API. Download the model, cleaned data, and test predictions separately from Reports.', f"Prediction endpoint: POST /api/analyses/{result['id']}/predict", 'Validate on independent data before operational use. Load only trusted Joblib model files.'],
            [['Input feature','Type']]+[[v['name'],v['type']] for v in result.get('model_schema',[])])
    return sections


def chart_drawing(chart):
    width = 510
    ink, blue = colors.HexColor('#1d3354'), colors.HexColor('#416dcc')
    if chart['kind']=='bar':
        items = chart['items']; height = max(80, len(items)*25+25)
        drawing=Drawing(width,height)
        largest=max((abs(value) for _,value in items),default=1) or 1
        if not items: drawing.add(String(10,30,'No observed values.',fillColor=ink,fontSize=10))
        for i,(label,value) in enumerate(items):
            y=height-25-i*25
            display=str(label); display=display if len(display)<=32 else display[:29]+'...'
            drawing.add(String(0,y+3,display,fillColor=ink,fontSize=8))
            drawing.add(Rect(186,y,max(0.5,abs(value)/largest*265),14,fillColor=blue,strokeColor=None))
            drawing.add(String(459,y+3,fmt(value),fillColor=ink,fontSize=8))
        return drawing
    if chart['kind']=='heatmap':
        cols=chart['columns']; n=len(cols); cell=27; left=180; bottom=30; height=n*cell+bottom+18
        drawing=Drawing(width,height)
        for i,name in enumerate(cols):
            y=height-30-i*cell
            label=f'{i+1}. {name}'; label=label if len(label)<33 else label[:30]+'...'
            drawing.add(String(0,y+9,label,fontSize=8,fillColor=ink))
            drawing.add(String(left+i*cell+9,10,str(i+1),fontSize=8,fillColor=ink))
            for j,value in enumerate(chart['values'][i]):
                defined=value is not None and math.isfinite(value)
                intensity=abs(value) if defined else 0
                base=colors.HexColor('#456ccb') if defined and value>=0 else colors.HexColor('#9969b4')
                fill=colors.linearlyInterpolatedColor(colors.HexColor('#edf1f7'),base,0,1,intensity) if defined else colors.HexColor('#d8dde6')
                drawing.add(Rect(left+j*cell,y,cell-2,cell-2,fillColor=fill,strokeColor=None))
                drawing.add(String(left+j*cell+(cell-2)/2,y+9,f'{value:.1f}' if defined else '-',textAnchor='middle',fontSize=6,fillColor=colors.white if intensity>.65 else ink))
        return drawing
    drawing=Drawing(width,280)
    points=chart['points']; bounds=chart['bounds']; left,bottom,plot_w,plot_h=50,38,435,210
    domains=[]
    for axis in ('x','y'):
        low,high=bounds[axis]; pad=(high-low)*.04 or abs(low)*.04 or 1
        domains.append((low-pad,high+pad))
    for i in range(5):
        part=i/4
        drawing.add(Line(left,bottom+part*plot_h,left+plot_w,bottom+part*plot_h,strokeColor=colors.HexColor('#d8e2f0')))
        drawing.add(String(left-6,bottom+part*plot_h,fmt(domains[1][0]+part*(domains[1][1]-domains[1][0])),fontSize=7,textAnchor='end',fillColor=ink))
        drawing.add(String(left+part*plot_w,bottom-14,fmt(domains[0][0]+part*(domains[0][1]-domains[0][0])),fontSize=7,textAnchor='middle',fillColor=ink))
    for point in points:
        x=left+(point['x']-domains[0][0])/(domains[0][1]-domains[0][0])*plot_w
        y=bottom+(point['y']-domains[1][0])/(domains[1][1]-domains[1][0])*plot_h
        drawing.add(Circle(x,y,1.8,fillColor=colors.HexColor('#329c91'),strokeColor=None))
    drawing.add(String(left,265,chart['y'][:65],fontSize=9,fillColor=ink))
    drawing.add(String(left+plot_w/2,4,chart['x'][:65],fontSize=9,textAnchor='middle',fillColor=ink))
    return drawing


def render_html(result):
    escape=lambda value:html.escape(fmt(value))
    sections=report_outline(result)
    title=result.get('report_metadata',{}).get('filename','Saved dataset')
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Main analysis report</title><style>
    *{box-sizing:border-box}body{font:15px/1.65 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:40px 28px;color:#203452;background:#f8faff}header{padding:32px;border-radius:16px;background:#132746;color:white}h1{font-size:32px;margin:4px 0}h2{font-size:22px;color:#234d87;margin:0 0 16px}section{background:white;border:1px solid #d9e3f2;border-radius:12px;margin:22px 0;padding:26px;break-inside:auto}p,td,th{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:12px;table-layout:fixed}td,th{border:1px solid #dce4ee;padding:9px;text-align:left;vertical-align:top}th{background:#eaf0fa}thead{display:table-header-group}figure{margin:20px 0;break-inside:avoid}svg{width:100%;max-height:650px;height:auto;display:block}figcaption{font-weight:600;margin-bottom:12px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}a{color:#365fa7}.brand{font-size:12px;letter-spacing:2px}footer{color:#65758e;font-size:12px}@media print{body{background:white;padding:0}header{background:white;color:#162d4d;border-bottom:3px solid #3155a0}nav{display:none}section{border:0;padding:12px 0}h2{break-after:avoid}tr{break-inside:avoid}}
    </style></head><body>'''
    document+=f'<header><span class="brand">AUTODS STUDIO / MAIN REPORT</span><h1>{escape(title)}</h1><p>From dataset to evaluation, explanations and next steps.</p><small>Analysis {escape(result["id"])}</small></header><nav>'
    document+=''.join(f'<a href="#section-{i}">{escape(s["title"])}</a>' for i,s in enumerate(sections) if s['title'][0].isdigit())+'</nav>'
    for i,section in enumerate(sections):
        document+=f'<section id="section-{i}"><h2>{escape(section["title"])}</h2>'
        document+=''.join('<p>'+escape(p)+'</p>' for p in section['paragraphs'] if p)
        for chart in section['charts']:
            svg=renderSVG.drawToString(chart_drawing(chart))
            svg=svg[svg.index('<svg'):]
            document+='<figure><figcaption>'+escape(chart['title'])+'</figcaption>'+svg+'</figure>'
        if section['table']:
            rows=section['table']
            document+='<table><thead><tr>'+''.join('<th>'+escape(v)+'</th>' for v in rows[0])+'</tr></thead><tbody>'
            document+=''.join('<tr>'+''.join('<td>'+escape(fmt(v)[:500])+'</td>' for v in row)+'</tr>' for row in rows[1:])+'</tbody></table>'
        document+='</section>'
    return document+'<footer>Generated from saved Python measurements. Raw previews may contain private data. Review this report before sharing.</footer></body></html>'


def render_pdf(result):
    styles=getSampleStyleSheet()
    font_dir=Path(reportlab.__file__).parent/'fonts'
    if 'ReportVera' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('ReportVera',str(font_dir/'Vera.ttf')))
        pdfmetrics.registerFont(TTFont('ReportVeraBold',str(font_dir/'VeraBd.ttf')))
    for name in ('Normal','BodyText','Title','Heading1','Heading2'):
        styles[name].fontName='ReportVeraBold' if name in ('Title','Heading1','Heading2') else 'ReportVera'
    styles['BodyText'].fontSize=8;styles['BodyText'].leading=12;styles['BodyText'].splitLongWords=True
    styles['Heading2'].fontSize=13;styles['Heading2'].leading=18;styles['Heading2'].spaceBefore=14
    styles['Title'].fontSize=22;styles['Title'].leading=28
    def text(value,style='BodyText',limit=None):
        value=fmt(value).replace('\u2013','-').replace('\u2014','-')
        return Paragraph(html.escape(value[:limit] if limit else value),styles[style])
    title=result.get('report_metadata',{}).get('filename','Saved dataset')
    story=[text('AutoDS Studio | Main report','Title'),text(title,'Heading2'),text('Analysis '+result['id']),Spacer(1,16)]
    for section in report_outline(result):
        introduction=[text(section['title'],'Heading2')]+[text(value) for value in section['paragraphs'] if value]
        if section['charts']:
            for chart in section['charts']:
                story.append(KeepTogether(introduction+[text(chart['title']),Spacer(1,8),chart_drawing(chart),Spacer(1,12)]))
                introduction=[]
        else:
            if section['table']:
                for paragraph in introduction: paragraph.keepWithNext=True
                story.extend(introduction)
            else:
                story.append(KeepTogether(introduction))
        if section['table']:
            rows=section['table']; widths=[(A4[0]-84)/len(rows[0])]*len(rows[0])
            item=Table([[text(value,limit=500) for value in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
            item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eefb')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c9d6e8')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([item,Spacer(1,12)])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#61718a'))
        canvas.drawString(42,22,'AutoDS Studio | Dataset to decisions');canvas.drawRightString(A4[0]-42,22,f'Page {doc.page}')
    buffer=io.BytesIO()
    SimpleDocTemplate(buffer,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=40,title='Main analysis report - '+title,author='AutoDS Studio').build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()


def build_reports(result, destination):
    (destination/'report.html').write_text(render_html(result),encoding='utf-8')
    (destination/'report.pdf').write_bytes(render_pdf(result))
    (destination/'report.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False),encoding='utf-8')
    return {'formats':['html','pdf','json','model'],'note':'Complete dataset, EDA, modeling, evaluation and explainability report. Load only trusted server-generated Joblib model files.'}
