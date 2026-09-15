import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.tools.data import identifier

def choose_features(train, target, requested=None, max_features=40):
    available=[c for c in train if c!=target]
    if requested is not None:
        unknown=set(requested)-set(available)
        if unknown: raise ValueError(f'Unknown features or target included as a feature: {sorted(unknown)}')
        available=list(dict.fromkeys(requested))
    selected=[]; excluded=[]; warnings=[]
    y=train[target] if target else None
    for name in available:
        series=train[name]
        reason=None
        if identifier(name): reason='identifier or personal-name column'
        elif series.isna().mean()>.95: reason='more than 95% missing in training data'
        elif series.nunique()<=1: reason='constant in training data'
        elif not pd.api.types.is_numeric_dtype(series) and series.nunique()>40: reason='high-cardinality text; requires a dedicated text pipeline'
        elif y is not None:
            valid=series.notna()&y.notna()
            if valid.any() and (series[valid].astype(str)==y[valid].astype(str)).all(): reason='exact copy of the target (leakage)'
            elif pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(series) and abs(series.corr(y))>.9999: reason='near-perfect target proxy (possible leakage)'
        if reason: excluded.append({'column':name,'reason':reason})
        else: selected.append(name)
    if len(selected)>max_features: raise ValueError(f'{len(selected)} usable features exceed the limit of {max_features}. Select a smaller feature set.')
    if not selected: raise ValueError('No usable predictors remain. Review identifiers, constant columns, missing data, and target leakage.')
    for item in excluded:
        if 'leakage' in item['reason']: warnings.append(f"Excluded {item['column']}: {item['reason']}.")
    warnings.append('Review availability at prediction time: automated checks cannot detect every post-outcome feature or repeated-entity leak.')
    return selected,excluded,warnings

def make_preprocessor(frame):
    numeric=list(frame.select_dtypes(include='number').columns)
    categorical=[c for c in frame if c not in numeric]
    transformers=[]
    if numeric: transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),('scale',StandardScaler())]),numeric))
    if categorical: transformers.append(('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False,max_categories=20))]),categorical))
    return ColumnTransformer(transformers,remainder='drop',sparse_threshold=0)

def clean_frame(frame):
    result=frame.copy().replace([np.inf,-np.inf],np.nan)
    for name in result.select_dtypes(exclude='number'):
        result[name]=result[name].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).replace('',np.nan)
    return result.drop_duplicates()

def normalize_prediction(rows, feature_schema):
    frame=pd.DataFrame(rows)
    expected=[c['name'] for c in feature_schema]
    unknown=set(frame.columns)-set(expected)
    if unknown: raise ValueError(f'Unknown input columns: {sorted(unknown)}')
    for col in feature_schema:
        name=col['name']
        if name not in frame: frame[name]=np.nan
        if col['type']=='numeric':
            values=pd.to_numeric(frame[name],errors='coerce')
            invalid=frame[name].notna()&values.isna()
            if invalid.any() or np.isinf(values).any(): raise ValueError(f'{name} must contain finite numbers or null.')
            frame[name]=values
        else: frame[name]=frame[name].map(lambda v:str(v).strip() if pd.notna(v) else np.nan)
    return frame[expected]
