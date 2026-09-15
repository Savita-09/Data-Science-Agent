"""Bounded, descriptive charts computed from the uploaded dataset, independent of training."""
import numpy as np
import pandas as pd
from app.tools.data import describe, identifier, json_safe


def dataset_dashboard(frame, target=None):
    frame = frame.replace([np.inf, -np.inf], np.nan)
    profile = describe(frame)
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    categorical = [c for c in frame if c not in numeric]
    distributions = []
    for name in frame:
        values = frame[name].dropna()
        bins, outliers = [], None
        if name in numeric:
            if len(values):
                counts, edges = np.histogram(values.astype(float), bins=min(16, max(1, values.nunique())))
                bins = [{'label': f'{edges[i]:.12g} – {edges[i+1]:.12g}', 'low': float(edges[i]), 'high': float(edges[i+1]), 'count': int(count)} for i, count in enumerate(counts)]
                q1, median, q3 = values.quantile([.25, .5, .75])
                iqr = q3 - q1
                outliers = int(((values < q1-1.5*iqr) | (values > q3+1.5*iqr)).sum())
                stats = {'min': values.min(), 'max': values.max(), 'mean': values.mean(), 'median': median, 'q1': q1, 'q3': q3, 'std': values.std()}
            else:
                stats = {}
        else:
            counts = values.astype(str).value_counts()
            bins = [{'label': str(label), 'count': int(count)} for label, count in counts.head(10).items()]
            if len(counts) > 10:
                bins.append({'label': 'Remaining categories (combined)', 'count': int(counts.iloc[10:].sum()), 'other': True})
            stats = {}
        distributions.append({'column': name, 'kind': 'numeric' if name in numeric else 'categorical', 'count': len(values), 'missing': int(frame[name].isna().sum()), 'unique': int(values.nunique()), 'bins': bins, 'outliers': outliers, 'stats': stats})

    # IDs can be inspected in distributions, but are not useful default relationships.
    useful = [c for c in numeric if not identifier(c)]
    corr_columns = useful[:12]
    if target in useful and target not in corr_columns:
        corr_columns = corr_columns[:11] + [target]
    corr = frame[corr_columns].corr(min_periods=3)
    pairs = [{'left': left, 'right': right, 'value': float(corr.loc[left, right])}
             for i, left in enumerate(corr_columns) for right in corr_columns[i+1:]
             if pd.notna(corr.loc[left, right])]
    pairs.sort(key=lambda pair: abs(pair['value']), reverse=True)
    # Treat booleans as categories consistently in the dashboard profile.
    for column in profile['columns']:
        column['type'] = 'numeric' if column['name'] in numeric else 'categorical'
    return json_safe({
        **profile, 'source': 'uploaded', 'target': target if target in frame else None,
        'numeric_columns': numeric, 'relationship_columns': useful,
        'types': [{'label': 'Numeric', 'count': len(numeric)}, {'label': 'Categorical', 'count': len(categorical)}],
        'distributions': distributions,
        'missing': sorted([{'column': c['name'], 'count': c['missing'], 'percent': c['missing_pct']} for c in profile['columns']], key=lambda c: c['count'], reverse=True),
        'correlations': {'columns': corr_columns, 'values': corr.to_numpy()},
        'top_correlations': pairs[:5],
        'note': 'Charts describe the uploaded dataset before training cleanup. Missing values are excluded from distributions and pairwise correlations. Associations do not establish causation.',
    })


def dataset_relationship(frame, x, y, limit=600):
    if x not in frame or y not in frame:
        raise ValueError('Choose existing columns for both axes.')
    if x == y:
        raise ValueError('Choose two different numeric columns.')
    if any(not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]) for c in (x, y)):
        raise ValueError('Scatter plots require numeric columns on both axes.')
    pairs = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    sample = pairs.sample(n=limit, random_state=42).sort_index() if len(pairs) > limit else pairs
    correlation = pairs[x].corr(pairs[y]) if len(pairs) >= 3 and pairs[x].nunique() > 1 and pairs[y].nunique() > 1 else None
    return json_safe({
        'x': x, 'y': y, 'total_rows': len(frame), 'valid_rows': len(pairs), 'omitted_rows': len(frame)-len(pairs),
        'sampled': len(sample) < len(pairs), 'sample_rows': len(sample), 'correlation': correlation,
        'points': [{'row': int(index)+1, 'x': float(row[x]), 'y': float(row[y])} for index, row in sample.iterrows()],
        'bounds': {'x': [pairs[x].min(), pairs[x].max()], 'y': [pairs[y].min(), pairs[y].max()]} if len(pairs) else None,
    })
