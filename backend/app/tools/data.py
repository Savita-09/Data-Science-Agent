import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def json_safe(value):
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (pd.Timestamp,Path)): return str(value)
    return value

def frame_records(frame): return json_safe(frame.astype(object).where(frame.notna(), None).to_dict(orient='records'))

def export_csv(frame, destination):
    """Neutralize spreadsheet formulas in downloads without changing model inputs."""
    def safe(value):
        return "'" + value if isinstance(value, str) and re.match(r'^\s*[=+@-]', value) else value
    exported = frame.copy()
    for column in exported.select_dtypes(exclude='number'):
        exported[column] = exported[column].map(safe)
    exported.columns = [safe(str(column)) for column in exported.columns]
    exported.to_csv(destination, index=False)

def read_upload(content, filename, settings):
    if len(content) > settings.max_upload_mb * 1024**2: raise ValueError('Dataset exceeds the upload size limit.')
    if filename.lower().endswith('.csv'):
        import csv
        header = next(csv.reader(io.StringIO(content.decode('utf-8-sig'))), [])
        if len(set(header)) != len(header) or any(not h.strip() for h in header): raise ValueError('CSV column names must be non-empty and unique.')
        frame = pd.read_csv(io.BytesIO(content), nrows=settings.max_rows+1)
    elif filename.lower().endswith('.json'):
        rows = json.loads(content)
        if not isinstance(rows,list) or not rows or any(not isinstance(row,dict) for row in rows): raise ValueError('JSON must contain a non-empty array of flat row objects.')
        if any(isinstance(v,(list,dict)) for row in rows for v in row.values()): raise ValueError('Nested JSON values are unsupported.')
        frame = pd.DataFrame(rows)
    else: raise ValueError('Upload a UTF-8 CSV or flat JSON dataset.')
    if not 20 <= len(frame) <= settings.max_rows: raise ValueError(f'Use 20–{settings.max_rows:,} rows.')
    if not 1 <= len(frame.columns) <= settings.max_columns: raise ValueError(f'Use 1–{settings.max_columns} columns.')
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any() or any(not c or len(c)>200 for c in frame.columns): raise ValueError('Invalid or duplicate column names.')
    if any(frame[c].astype(str).str.len().max()>20000 for c in frame): raise ValueError('A cell exceeds 20,000 characters.')
    return frame.replace([np.inf,-np.inf],np.nan)

def describe(frame):
    columns=[]
    for name in frame:
        s=frame[name]; numeric=pd.api.types.is_numeric_dtype(s)
        values=s.dropna()
        record={'name':name,'type':'numeric' if numeric else 'categorical','missing':int(s.isna().sum()),'unique':int(s.nunique()),'missing_pct':round(float(s.isna().mean()*100),2)}
        if numeric and len(values):
            record.update(json_safe({'mean':values.mean(),'std':values.std(),'min':values.min(),'max':values.max(),'median':values.median()}))
        else: record['top']=[{'value':str(k),'count':int(v)} for k,v in values.astype(str).value_counts().head(8).items()]
        columns.append(record)
    cells=frame.size
    return {'rows':len(frame),'columns':columns,'column_count':len(frame.columns),'missing_cells':int(frame.isna().sum().sum()),'completeness':round(100*(1-frame.isna().sum().sum()/max(cells,1)),2),'duplicate_rows':int(frame.duplicated().sum()),'preview':frame_records(frame.head(12))}

def eda(frame, target=None):
    distributions=[]
    for name in frame.columns[:40]:
        values=frame[name].dropna()
        if pd.api.types.is_numeric_dtype(values) and len(values):
            counts,edges=np.histogram(values,bins=min(12,max(1,values.nunique())))
            bins=[{'label':f'{edges[i]:.3g}–{edges[i+1]:.3g}','count':int(v)} for i,v in enumerate(counts)]
            lo,hi=values.quantile([.25,.75]);outliers=int(((values<lo-1.5*(hi-lo))|(values>hi+1.5*(hi-lo))).sum())
        else:
            bins=[{'label':str(k),'count':int(v)} for k,v in values.astype(str).value_counts().head(12).items()];outliers=None
        distributions.append({'column':name,'bins':bins,'outliers':outliers})
    numeric=frame.select_dtypes(include='number').iloc[:,:30]
    corr=numeric.corr().fillna(0)
    return {'distributions':distributions,'correlations':{'columns':list(corr.columns),'values':json_safe(corr.to_numpy())},'target':target,'note':'Descriptive EDA uses the uploaded data. It does not select predictors using test outcomes; associations are not causal effects.'}

def synthetic_churn(rows=600, seed=42):
    rng=np.random.default_rng(seed)
    tenure=rng.integers(1,72,rows); monthly=rng.uniform(25,120,rows).round(2); support=rng.poisson(2,rows)
    contract=rng.choice(['Monthly','Annual','Two year'],rows,p=[.55,.3,.15])
    score=-1.7+1.05*(contract=='Monthly')+0.65*support-.052*tenure+.018*(monthly-65)
    probability=1/(1+np.exp(-score))
    return pd.DataFrame({'customer_id':[f'C{i:05d}' for i in range(rows)],'tenure_months':tenure,'monthly_charge':monthly,'support_calls':support,'contract':contract,'autopay':rng.choice(['Yes','No'],rows),'region':rng.choice(['North','South','East','West'],rows),'churn':np.where(rng.random(rows)<probability,'Yes','No')})

def fingerprint(sha, request): return hashlib.sha256((sha+json.dumps(request,sort_keys=True)+'pipeline-v2').encode()).hexdigest()

def identifier(name): return bool(re.search(r'(^id$|_id$|^index$|^unnamed:|^name$|email|phone)', name, re.I))
