# API documentation

Base URL: `http://127.0.0.1:8000`. Interactive schema: `/docs`. Machine-readable schema: `/openapi.json` (also exported beside this document).

All `/api` endpoints except `/api/health` require `X-API-Key` when configured. In production the server refuses to start without a key of at least 32 characters. Without a key, development API access is restricted to loopback. Keep keys in headers, never URL parameters. API errors have a `detail` field; schema-validation errors contain a list of field errors.

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | HTTP health, LLM/auth availability, public limits, `groq_configured`, and `groq_model`. No credential values. Configuration availability is not a live provider readiness check. |
| POST | `/api/datasets` | Multipart `file`: bounded UTF-8 CSV or array-of-flat-objects JSON. Returns 201, dataset UUID and profile. |
| POST | `/api/datasets/sample` | Creates a reproducible 600-row synthetic churn dataset. |
| GET | `/api/datasets` | Latest 100 uploaded datasets with filename, row/column counts, and creation time. |
| GET | `/api/datasets/{dataset_id}` | Dataset metadata and preview. |
| GET | `/api/datasets/{dataset_id}/eda?target=Price` | Original-upload dashboard: profile, numeric/category distributions, missingness, type counts, and up to 12 numeric correlations. Optional target is included in the correlation map when numeric. No training required. |
| GET | `/api/datasets/{dataset_id}/relationship?x=area&y=Price` | Numeric scatter: at most 600 seeded sample points, full valid-pair count/correlation/bounds, and excluded missing pairs. Unknown, categorical, or identical axes return 422. |
| POST | `/api/datasets/{dataset_id}/validate` | Check goal, task, target, cleaned class counts, and feasible CV folds before enqueueing. Returns `valid`, `task`, `target`, `errors`, and `warnings`. |
| POST | `/api/analyses` | Validate and enqueue an analysis; returns 202 and status/events URLs. |
| GET | `/api/analyses` | Latest 100 analyses, newest first, including `dataset_filename` and the original request. |
| GET | `/api/analyses/{id}` | Status, `dataset_filename`, request, checkpoint summaries, and completed result. Prediction previews capped at 50. |
| GET | `/api/analyses/{id}/events?after=0` | SSE stream of durable progress events and a terminal `done` event. |
| POST | `/api/analyses/{id}/cancel` | Request cancellation or cancel a queued job. |
| POST | `/api/analyses/{id}/retry` | Requeue a failed, timed-out, interrupted, or cancelled analysis, preserving checkpoints. |
| GET | `/api/analyses/{id}/artifacts/{kind}` | Download `html`, `pdf`, `json`, `model`, `predictions`, or `cleaned`. Completed jobs only. HTML/PDF are the complete main dataset report, including EDA charts. |
| POST | `/api/analyses/{id}/predict` | Apply saved preprocessing and model to up to 200 new rows. |
| GET | `/api/analyses/{id}/chat` | Latest 30 saved question/answer pairs. |
| POST | `/api/analyses/{id}/chat` | Groq, configured LLM, or local report assistant for a completed analysis. |

## Minimal Python client

```python
import os
import time
import httpx

headers = {'X-API-Key': os.environ['APP_API_KEY']} if os.getenv('APP_API_KEY') else {}
with httpx.Client(base_url='http://127.0.0.1:8000', headers=headers, timeout=30) as client:
    response = client.post('/api/datasets/sample')
    response.raise_for_status()
    dataset = response.json()
    response = client.post('/api/analyses', json={
        'dataset_id': dataset['id'],
        'goal': 'Predict churn and identify associated risk factors.',
        'task': 'auto',
        'target': 'churn',
        'include_dl': True,
        'use_llm': False,
        'cv_folds': 3,
        'tuning_iterations': 3,
        'time_budget_seconds': 600,
        'seed': 42,
        'quick': False,
    })
    response.raise_for_status()
    analysis_id = response.json()['id']
    while True:
        response = client.get(f'/api/analyses/{analysis_id}')
        response.raise_for_status()
        job = response.json()
        if job['status'] not in ('queued', 'running'):
            break
        time.sleep(2)
    if job['status'] != 'completed':
        raise RuntimeError(job['error'])
    print(job['result']['evaluation']['metrics'])

    example = {c['name']: c['example'] for c in job['result']['model_schema']}
    response = client.post(f'/api/analyses/{analysis_id}/predict', json={'rows': [example]})
    response.raise_for_status()
    print(response.json())

    response = client.get(f'/api/analyses/{analysis_id}/artifacts/pdf')
    response.raise_for_status()
    with open('analysis-report.pdf', 'wb') as destination:
        destination.write(response.content)
```

To upload a file, use `client.post('/api/datasets', files={'file': ('customers.csv', content, 'text/csv')})` where `content` is the file's bytes. Dataset size limits are enforced before a job is created. A minimum of 20 usable distinct rows is required; classification needs 2–30 classes with at least three usable rows per class after cleaning. The split reserves two training observations and one test observation per class, and CV folds decrease automatically when needed. Labels with fewer observations get a specific error and are never silently dropped or merged.

The validation endpoint accepts `goal`, `task`, `target`, `cv_folds`, and `seed`. Identifiers cannot be targets. Inference uses a clearly named or conventional outcome; otherwise the user must choose a target, rather than guessing the last column. Auto mode detects continuous numeric targets, including numeric strings, as regression. Explicit classification of a high-cardinality continuous target returns guidance to change task. Submission and retry repeat validation so invalid settings do not enter the worker queue.

## Analysis options and results

`task` is `auto`, `classification`, `regression`, or `clustering`. `target` is ignored for clustering; for supervised tasks omission invokes a heuristic. `features` optionally supplies a list of column names; missing/invalid/unsafe choices fail or are excluded with reasons. `include_dl` enables eligible MLP candidates. `quick` reduces estimator size and the UI uses one tuning configuration; direct API callers control `tuning_iterations` separately. CV folds range from 2 to 5, and tuning/time budgets must satisfy both schema and configured server limits. Each request gets an immutable dataset/request fingerprint.

Job states: `queued`, `running`, `completed`, `failed`, `cancelled`, `timed_out`, `interrupted`. `error` explains a terminal failure. `checkpoints` provide stage outputs and durations. Completed `result` includes `plan`, `profile`, `cleaning`, `eda`, `features`, `models`, `evaluation`, `explainability`, `insights`, and `model_schema`. `models.leaderboard` records candidate status, family, CV mean/std, selected parameters, and duration. Failed candidate entries expose their failure without inventing a score.

Prediction inputs must use the feature names in `model_schema`. Unknown columns are rejected; absent feature values are imputed using the saved training pipeline. Numeric fields reject invalid/non-finite values. Output classification probabilities align with the returned `classes` order. Regression returns numeric predictions; clustering returns arbitrary cluster IDs. Probabilities are not automatically calibrated or tuned for business cost. Local SHAP examples currently cover held-out rows from the analysis; the prediction endpoint does not compute new-row SHAP.

## Event stream

Use authenticated `fetch` streaming or an HTTP client for SSE, since browser `EventSource` cannot set the application header. Resume with the last received numeric event ID as `after`. The frontend falls back to status polling if streaming is unavailable. A progress item has the following shape:

```text
id: 42
event: progress
data: {"id":42,"analysis_id":"<uuid>","stage":"training","message":"Training xgboost with 3-fold cross-validation.","created":1789400000.0}

event: done
data: {"status":"completed"}
```

The connection sends heartbeat comments while waiting. Configure a reverse proxy to disable SSE buffering and allow idle streaming. Polling `GET /api/analyses/{id}` remains a supported fallback.

## Chat, downloads, and errors

Report downloads (`html`, `pdf`, `json`) render from the saved analysis using the current report layout. Legacy runs are enriched with EDA computed from their original saved CSV; no training, LLM calls, or rewriting of their stored result occurs. New runs persist `report_metadata` (dataset filename and request settings) and `report_eda` (original-upload dashboard plus an optional numeric relationship) alongside the existing result fields. These exports include the dataset profile, first five preview rows, cleaning, feature engineering, leaderboard, baseline/test/train metrics, confusion matrix when available, global/local SHAP, recommendations, and input schema.

HTML embeds SVG charts without external assets. PDF includes paginated tables and vector charts. Correlation maps include up to 12 numeric columns; default scatter plots contain at most 600 reproducibly sampled points with full valid-pair correlation. Category charts combine categories beyond the ten most frequent into a remaining-categories bin. JSON retains original stored values; long display values in report tables may be shortened. Model and CSV artifact downloads continue to serve their saved files.

Groq chat body: `{"question":"Which features matter?","provider":"groq"}`. The server uses `GROQ_API_KEY`, `GROQ_MODEL` (default `openai/gpt-oss-20b`), and `GROQ_TIMEOUT_SECONDS` (default 30 per HTTP operation). It sends the trimmed question, up to four saved turns, and an allowlist of analysis facts: goal, task/target, model, evaluation/baseline metrics, test count, warnings, and ten global feature importance entries. Raw previews, predictions, and local SHAP rows are excluded. Successful responses have `answer` and `source` (for example `Groq · openai/gpt-oss-20b`) and are persisted. Failed requests create no chat entry.

Use `provider: "local"` for the deterministic assistant, or `provider: "configured"` for the generic `LLM_*` provider. For compatibility, omitting `provider` maps `use_llm: true` to `configured`, otherwise to `local`; an explicit provider takes precedence. Questions must contain 2–2000 characters and cannot be whitespace-only. Chat does not execute Python, mutate analysis results, or retrieve external data.

Groq network errors, rate limits, and server failures receive at most three attempts. Retry delays are capped at five seconds. Missing configuration returns 422; credential/model or invalid reply failures return 502; exhausted rate limits return 429; exhausted transport/server failures return 503. Provider response bodies and credentials are not reflected in these errors. Empty or token-truncated replies are rejected.

CSV exports escape formula-leading string values/headers for safer spreadsheet opening. This escaping does not change the trained model or original canonical dataset. Full JSON reports include original-valued previews and held-out predictions. All artifacts are private to the shared application workspace; authentication does not distinguish individual users.

Common status codes: 401 invalid application key, 403 non-local unauthenticated development access, 404 unknown UUID/artifact, 409 invalid job transition or result not ready, 413 oversized request body, 422 invalid input/configuration, 429 full submission queue or Groq rate limit, 502 invalid LLM chat response, 503 exhausted Groq transport/server failures. Model computation errors after enqueue are recorded in the job's status/error and durable event log.
