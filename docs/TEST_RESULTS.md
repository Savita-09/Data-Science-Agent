# Verification results

The application was checked locally on Windows with Python 3.12 and Node.js 26. No user datasets, credentials, generated reports, or machine-specific verification records are included in this repository.

## Automated checks

- Backend: **45 tests passed**, including synthetic classification, regression, and clustering workflows; saved-model predictions; target validation; EDA statistics; reports; Groq error handling; authentication; and worker reliability.
- Frontend: **10 regression checks passed**. TypeScript and the production Vite build passed.
- Report pagination refinements: all **3 focused report tests passed** again after the final layout changes.
- Report generation was checked for HTML escaping, measured EDA counts, PDF sections, and refreshing legacy downloads without retraining or changing stored evidence.
- Groq transport and chat persistence were verified with invented metrics in an isolated synthetic workspace; regular tests use mocked provider responses and need no API key.

## UI and document checks

The browser was checked for dataset-specific EDA charts, the five-row Data Profile preview, main-report navigation, PDF download preparation, and Groq configuration. Exported PDF pages were rendered and visually inspected for table pagination, chart labels, and section headings. No user report artifacts are published here.

## Reproduce

From `backend`, run `python -B -m pytest -q -p no:cacheprovider`. From `frontend`, run `npm ci`, `npm run build`, and `node --experimental-strip-types --test tests/setup.test.ts`.

For a complete local HTTP/worker smoke test, build the frontend, start `scripts/run-local.py`, then run `scripts/smoke_e2e.py` in another terminal. This creates a synthetic sample analysis in the chosen workspace.

## Deployment limits

Local tests are not a security audit or a load test. Docker build/runtime and Linux-host behavior must be verified on the deployment host. The app is a single shared workspace, not a multitenant service. Add appropriate HTTPS, access controls, backups, and resource monitoring before external use.
