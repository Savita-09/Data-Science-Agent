"""Shared problem validation for the form, API, and training agents."""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.preprocessing.pipeline import clean_frame
from app.tools.data import identifier

MAX_CLASSES = 30
COMMON_TARGETS = ('churn', 'target', 'label', 'outcome', 'sales', 'final_result')


def infer_task(series):
    values = series.dropna()
    numeric = pd.to_numeric(values, errors='coerce')
    if not len(values) or numeric.isna().any():
        return 'classification'
    unique = numeric.nunique()
    integral = np.isclose(numeric.to_numpy(dtype=float) % 1, 0).all()
    return 'classification' if unique <= 2 or (integral and unique <= 20) else 'regression'


def resolve_problem(frame, request):
    goal = request.get('goal', '').lower()
    task = request.get('task', 'auto')
    target = request.get('target')
    warnings = []
    if re.search(r'forecast|time.series|next.month|next.week', goal):
        raise ValueError('Forecasting requires chronological validation; choose an independent-row classification, regression, or clustering problem.')
    if task == 'auto' and re.search(r'cluster|segment|group customers|unsupervised', goal):
        task = 'clustering'
    if task == 'clustering':
        target = None
    else:
        if not target:
            candidates = [c for c in frame if not identifier(c)]
            mentions = [c for c in candidates if re.search(r'\b(?:predict|classify|estimate)\s+(?:the\s+)?' + re.escape(c).replace('_', r'[_ ]') + r'\b', goal, re.I)]
            conventional = [c for c in candidates if c.lower() in COMMON_TARGETS]
            inferred = mentions if len(mentions) == 1 else conventional
            if len(inferred) != 1:
                raise ValueError('Choose the target column you want to predict. The business problem does not identify one unambiguously; an ID or the last column will not be guessed.')
            target = inferred[0]
            warnings.append(f'Target inferred as {target}. Review this choice before using the results.')
        if target not in frame:
            raise ValueError('The target column does not exist in this dataset.')
        if identifier(target):
            raise ValueError(f'"{target}" is an identifier. Choose the measured outcome you want to predict, such as a category or numeric measurement.')
        if task == 'auto':
            task = infer_task(frame[target])
    return {'task': task, 'target': target, 'summary': request.get('goal', ''), 'source': 'deterministic task detection', 'warnings': warnings}


def validate_target(frame, task, target):
    if len(frame) < 20:
        raise ValueError(f'Only {len(frame)} usable distinct rows remain. At least 20 are needed after cleaning and removing missing targets.')
    if task == 'clustering':
        return {}
    values = frame[target]
    if task == 'regression':
        numeric = pd.to_numeric(values, errors='coerce')
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f'Regression target "{target}" must contain finite numbers. Choose Classification for category labels.')
        if numeric.nunique() < 2:
            raise ValueError(f'Target "{target}" is constant. Choose an outcome with at least two different values.')
        return {'distinct_values': int(numeric.nunique())}
    counts = values.astype(str).value_counts()
    if len(counts) < 2:
        raise ValueError(f'Target "{target}" has only one class after cleaning. Add examples of another class or choose a different target.')
    if len(counts) > MAX_CLASSES:
        suggestion = ' This looks like a numeric measurement; select Regression or Detect automatically.' if infer_task(values) == 'regression' else ' Choose a meaningful outcome column or explicitly regroup labels before uploading.'
        raise ValueError(f'Target "{target}" has {len(counts)} distinct labels; this workspace supports up to {MAX_CLASSES} classes.' + suggestion)
    rare = counts[counts < 3]
    if len(rare):
        examples = ', '.join(f'{str(label)[:60]!r}: {int(count)}' for label, count in rare.head(5).items())
        raise ValueError(f'Target "{target}" has labels with fewer than 3 usable rows ({examples}). Each label needs 2 training rows for cross-validation and 1 held-out row. Add observations, choose another target, or explicitly regroup these labels; no labels are silently removed.')
    return {'class_count': len(counts), 'smallest_class': int(counts.min())}


def split_supervised(frame, task, target, seed):
    if task != 'classification':
        train, test = train_test_split(frame, test_size=.2, random_state=seed)
        return train, test, []
    labels = frame[target].astype(str)
    counts = labels.value_counts()
    # Keep the established stratified split whenever it supports all classes.
    try:
        train, test = train_test_split(frame, test_size=.2, random_state=seed, stratify=labels)
        train_counts = train[target].astype(str).value_counts().reindex(counts.index, fill_value=0)
        test_counts = test[target].astype(str).value_counts().reindex(counts.index, fill_value=0)
        if train_counts.min() >= 2 and test_counts.min() >= 1:
            return train, test, []
    except ValueError:
        pass
    rng = np.random.default_rng(seed)
    train_positions, test_positions = [], []
    for label in sorted(counts.index):
        positions = rng.permutation(np.flatnonzero(labels.to_numpy() == label))
        test_count = min(max(1, round(len(positions) * .2)), len(positions) - 2)
        test_positions.extend(positions[:test_count])
        train_positions.extend(positions[test_count:])
    train = frame.iloc[rng.permutation(train_positions)]
    test = frame.iloc[rng.permutation(test_positions)]
    return train, test, ['Adjusted the stratified holdout to retain at least 2 training rows and 1 test row per class. Rare-class estimates have high uncertainty.']


def assess_problem(frame, request):
    plan = None
    try:
        cleaned = clean_frame(frame)
        plan = resolve_problem(cleaned, request)
        if plan['target']:
            cleaned = cleaned.dropna(subset=[plan['target']])
        details = validate_target(cleaned, plan['task'], plan['target'])
        warnings = list(plan['warnings'])
        folds = request.get('cv_folds', 3)
        if plan['task'] == 'classification':
            train, _, split_warnings = split_supervised(cleaned, plan['task'], plan['target'], request.get('seed', 42))
            folds = min(folds, int(train[plan['target']].astype(str).value_counts().min()))
            warnings.extend(split_warnings)
            if details['smallest_class'] < 5:
                warnings.append(f'Small classes are supported with {folds}-fold validation; collect more labeled examples for reliable estimates.')
        return {'valid': True, 'task': plan['task'], 'target': plan['target'], 'usable_rows': len(cleaned), 'cv_folds': folds, 'warnings': warnings, 'errors': [], **details}
    except ValueError as error:
        return {'valid': False, 'task': plan['task'] if plan else None, 'target': plan['target'] if plan else request.get('target'), 'warnings': [], 'errors': [str(error)]}
