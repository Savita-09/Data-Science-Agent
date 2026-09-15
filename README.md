# Autonomous AI Data Scientist

A runnable, single-workspace data science application. Upload a CSV or flat JSON dataset, describe a business problem, and inspect a durable Python workflow from planning to downloadable reports and saved-model predictions.

The React dashboard talks to FastAPI. A separate supervised worker trains real scikit-learn, XGBoost, and optional multilayer neural-network candidates. Pandas, NumPy, and SHAP calculate the results. An optional LLM supplies narrative reasoning only; no model metrics are invented and no LLM-generated code is executed.

The repository contains the complete FastAPI application, training worker, React frontend, tests, API documentation, and Docker configuration. Dataset processing and model calculations run in Python; external LLM calls are limited to optional reasoning and chat.

## Get the code

```bash
git clone https://github.com/Savita-09/Data-Science-Agent.git
cd Data-Science-Agent
```

GitHub hosts the source code. Running the full app requires a Python server and training worker; GitHub Pages cannot run this backend. Use the local launcher or Docker Compose instructions below.

For a Vercel dashboard with a separate Python service, follow [Vercel deployment](docs/VERCEL.md). Import the repository with `frontend` as the root directory.

## Quick start — Windows PowerShell

Prerequisites: Python 3.12 and Node.js 22.12+ with npm. Allow roughly 2 GB for Python dependencies and at least 4 GB available RAM. Commands start in this directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
Set-Location frontend
npm ci
npm run build
Set-Location ..
.\.venv\Scripts\python.exe scripts/run-local.py
```

Open **http://127.0.0.1:8000**. The launcher starts both the API and training worker; Ctrl+C stops them. In this development mode, unauthenticated requests are restricted to the local machine. If PowerShell blocks `npm.ps1`, use `npm.cmd` instead.

Choose **New Analysis → Try synthetic customer churn**, enter a goal such as “Predict customer churn and identify associated risk factors,” select `churn` as the target, and run the analysis. ML and neural-network candidates are enabled by default. Quick mode reduces the search budget. The sample is reproducible synthetic data, not a claim about a real business.

## Quick start — Linux/macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
cp .env.example .env
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python scripts/run-local.py
```

The Python 3.12/Windows and Node 26 local workflow was tested. Linux/macOS commands and the Docker image have not been executed in this environment.

## Docker

Install Docker with Compose, copy `.env.example` to `.env`, and set `APP_API_KEY` to a random secret of at least 32 characters. Generate one locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"` and keep it private. Do not reuse an LLM provider key as the application key.

```bash
docker compose up --build -d
docker compose logs -f api worker
```

Open http://localhost:8000, choose the connection icon, and enter the application key. Compose enforces production authentication, binds the host port to loopback, uses non-root containers, limits CPU/memory, and persists SQLite and artifacts in named volumes. The worker is a separate service. Use a TLS reverse proxy and an identity-aware access gateway before exposing the application beyond a trusted workspace. This is shared-workspace authentication, not tenant isolation.

`docker compose down` stops services and retains named volumes. Back up `analysis-data` and `analysis-projects` together while services are stopped. Never remove these volumes unless their contents are intentionally disposable. Docker was unavailable on the development machine, so image build and container execution remain deployment checks.

## Groq AI Chat

Set these server-side variables in `.env`, then restart the application:

```dotenv
GROQ_API_KEY=your-private-groq-key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TIMEOUT_SECONDS=30
```

Open **AI Chat** and select a completed analysis. Chat uses Groq automatically. The server sends your question, up to four recent conversation turns, the analysis goal, and bounded aggregate metrics/global feature importance to Groq. Raw dataset rows, prediction previews, and local SHAP examples are excluded. Free-text goals and chat messages can themselves contain private information. The provider key stays on the server; `.env` is excluded from Git and Docker build context.

API clients can request the local report assistant with `provider: "local"`, which needs no external API. Missing credentials and provider errors are displayed explicitly; Groq failures do not create a fabricated reply or silently switch providers. Chat uses its own Groq settings independently of the analysis form's optional LLM setting. Models must be available to your Groq account; see [Groq models](https://console.groq.com/docs/models) and [API compatibility](https://console.groq.com/docs/openai).

## Optional analysis LLM reasoning

Set these server-side variables in `.env`, then restart both services:

```dotenv
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-private-provider-key
LLM_MODEL=your-provider-model-name
```

The provider must support a compatible `POST /chat/completions` interface and JSON response mode. The application appends `/chat/completions` to the base URL. Explicitly enable LLM sharing for an analysis. The provider receives the business goal, schema names, and aggregate measurements; dataset rows are not included by the application. Schema names and free-text goals can still contain sensitive information, so review what you enter. The UI never receives the provider key. Legacy API chat clients can also request this provider with `provider: "configured"`.

Without these variables the complete numerical workflow works offline after dependency installation, with deterministic planning, recommendations, and local report chat. Provider failures are retried and planning/insights fall back to deterministic guidance. Optional commentary is labeled as advisory. Groq chat was separately verified with a live connection; the generic analysis provider remains covered by mocked transport tests.

## Main dataset report

Open **Reports** after an analysis completes. The main report brings together the problem, dataset profile, five-row preview, interactive EDA dashboard, cleaning, feature engineering, model leaderboard, held-out and baseline metrics, SHAP, recommendations, and prediction schema. Use the report contents links to move between sections.

**Main report (PDF)** and **Main report (HTML)** export the complete report with embedded distribution charts for every column, missing-value charts when applicable, a numeric correlation map, a bounded numeric scatter plot when available, and SHAP charts. EDA describes the original upload; cleaning and model evaluation describe the saved workflow. HTML is self-contained; PDF is paginated. Long chart labels have accompanying tables where needed. JSON retains the full stored values.

Existing saved analyses receive the current report layout on download without retraining or changing their saved measurements. New analyses also save complete reports as workflow artifacts. Downloads can take a few seconds to render, especially for wide datasets. Reports include data previews and sampled scatter values; review them before sharing.

## What is implemented

- Automatic or explicit classification, regression, and clustering; target inference is shown for review.
- Target validation before training; uploading a replacement clears the previous goal, task, and target. A goal entered before the first upload is kept. Identifiers cannot be selected as outcomes, continuous numeric targets use regression in auto mode, and small classes use an adjusted split/CV with explicit warnings. Failed analyses have an **Edit analysis setup** action and completed analyses have **Run again**, both restoring the saved dataset. The business problem is optional in the UI when a target is selected: a visible default goal follows that target. Disabled runs display a reason. Saved runs show their filename and target; old sample descriptions are flagged without rewriting reports.
- Profiling, basic cleaning, EDA, identifier/proxy checks, bounded categorical encoding, and numeric imputation/scaling.
- Dummy baseline; linear models, Random Forest/Extra Trees, XGBoost; optional two-hidden-layer MLP; KMeans/MiniBatch KMeans for clustering.
- Cross-validation and randomized hyperparameter search; model selection based on validation scores, followed by a separate held-out evaluation.
- Global and local SHAP, measured business summaries, an automatic leaderboard, and critic warnings.
- HTML/PDF/JSON reports, downloadable model and CSV artifacts, and a saved-pipeline prediction API.
- SQLite queue/events/chat, atomic agent checkpoints, per-model cache reuse on retries, cancellation, timeout and memory supervision.
- Dashboard, New Analysis, Data Profile, EDA, Models, Explainability, Reports, and AI Chat; live event stream with polling fallback.
- EDA works directly from the uploaded dataset library, even with no saved analyses: selectable numeric histograms/category bars, data-type donut, missing-value bars, a scatter plot with axis selectors, and a correlation heatmap with clickable associations. Python computes the statistics; at most 600 reproducibly sampled points are plotted, with full-data correlation and axis ranges.

The supported workload is **independent-row tabular data**. Neural networks here are scikit-learn MLPs. This release does not train transformer models, NLP embedding pipelines, image/audio networks, time-series forecasting, or grouped/longitudinal validation. It does not connect directly to databases or arbitrary URLs. Those require additional ingestion and task-specific validation. It is not a multi-tenant SaaS, and no general-purpose app can promise to solve every dataset or improve every model.

## Configuration and operations

The deployed Render demo uses **public access without an API key**. `config.render.yaml` enables `public_demo: true` with temporary storage. Visitors share the demo's uploads, analyses, chats, and reports; use sample or non-sensitive data only. Public mode automatically uses separate `public-demo` subdirectories, leaving private workspaces untouched. Local and private production defaults retain API-key protection. See [deployment and access modes](docs/VERCEL.md).

Defaults in `config.yaml`: 20 MB per upload, 50,000 rows, 100 columns, 40 selected features, 512 encoded features, 20 queued/running analyses, 900 seconds per job, 2,048 MB process-tree RSS, three CV folds, three tuning configurations per candidate, and ten local SHAP examples. The worker executes one job at a time and estimators use one thread. Dataset-size limits fail with an actionable error instead of silently sampling the training data.

`MAX_JOB_SECONDS`, `MAX_MEMORY_MB`, `DATA_ROOT`, `PROJECT_ROOT`, `APP_CONFIG`, authentication, CORS, and LLM settings can be supplied through environment variables. Request limits cannot exceed server limits. Request `seed` and `cv_folds` default to 42 and 3 in the API schema. Increase worker/container resources together if you raise limits. Resource checks occur about every half second; they do not provide an OS sandbox or exact real-time enforcement. Docker adds an outer resource boundary.

For frontend development, start the API and worker using the launcher, then run `npm run dev` inside `frontend` in another terminal. Vite runs at http://localhost:5174 and proxies `/api` to port 8000. Run `npm run build` after frontend edits to update the UI served by FastAPI.

If an analysis stays queued, ensure the worker is running. If a worker is interrupted, the analysis becomes `interrupted` when its heartbeat expires; choose **Resume from checkpoints**. For a timeout, rerunning uses completed checkpoints and candidate caches. Start a new analysis with a larger permitted budget when a single stage cannot finish within the budget. Inspect the job's error and server-side `projects/<id>/error.log` for failures. Never share that log publicly without reviewing it.

## Verification

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -q
Set-Location ..
# Keep the API and worker running in another terminal:
.\.venv\Scripts\python.exe scripts/smoke_e2e.py
```

Use `.venv/bin/python` on Linux/macOS. The live smoke test creates a synthetic dataset, submits a real worker job, waits for completion, predicts a new row, downloads all artifact types, and checks report chat. If authentication is enabled, set `APP_API_KEY` in the test process environment. Smoke tests intentionally retain the synthetic analysis so it can be inspected in the dashboard.

See [test results](docs/TEST_RESULTS.md), [architecture](docs/ARCHITECTURE.md), and [API documentation](docs/API.md). Interactive OpenAPI documentation is served at **http://127.0.0.1:8000/docs**.
