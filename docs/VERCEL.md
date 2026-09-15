# Vercel deployment

Deploy the React dashboard from this repository with **Root Directory: `frontend`**, the **Vite** preset, `npm ci`, `npm run build`, and output directory `dist`. The configuration lives in `frontend/vercel.json`. Connect the `main` branch for automatic deployments.

## Connect the Python backend

The app requires a running FastAPI service and training worker sharing a writable data directory and model directory. The existing SQLite queue and worker are not a standalone Vercel Function deployment. A dashboard deployment by itself cannot perform analyses. See [Vercel function limits](https://vercel.com/docs/functions/limitations).

1. Run the API and worker on a host with enough memory and persistent storage, using the Docker Compose or local launcher instructions in the main README.
2. Expose the API over HTTPS with `APP_ENV=production` and a random `APP_API_KEY` of at least 32 characters.
3. Set backend `CORS_ORIGINS` to the exact dashboard origin, such as `https://your-project.vercel.app`. Include additional preview origins explicitly only when needed.
4. Set Vercel's `VITE_API_BASE_URL` to the public backend URL, such as `https://api.example.com`, and redeploy. This URL is public; it is not a secret. Alternatively, enter the backend URL in the dashboard's **API connection** dialog for the current browser tab.
5. Enter the backend's application key in **API connection**. The browser stores it in session storage. Do not put an application key or provider key in a `VITE_` variable or a URL.

Uploads, progress, chat, predictions, and report downloads all use the configured backend. The app appends `/api` when needed and requires HTTPS when the dashboard uses HTTPS. Empty backend URLs keep the existing same-origin local/Docker behavior.

Keep `GROQ_API_KEY` and other provider credentials on the Python host. They do not belong in the Vercel frontend environment.

## No-cost demonstration using your own computer

An existing computer can supply backend compute and storage without a hosting subscription. A temporary HTTPS tunnel can connect it to the Vercel dashboard. This requires the computer, API, worker, and tunnel to remain running, and it is a demonstration deployment rather than an always-on production host. [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) have no uptime guarantee, do not support SSE, and change address when recreated; polling keeps analysis progress available.

Use an isolated workspace for this mode. Create a private, ignored `.env.hosted` file with:

```dotenv
APP_ENV=production
APP_API_KEY=replace-with-a-random-private-key-of-at-least-32-characters
DATA_ROOT=./.hosted/data
PROJECT_ROOT=./.hosted/projects
CORS_ORIGINS=https://your-project.vercel.app
GROQ_API_KEY=
LLM_API_KEY=
```

Run `python scripts/run-local.py --port 8001 --env-file .env.hosted`. This explicitly chosen environment file disables implicit loading of the normal development `.env`. Keep the API bound to loopback, verify authentication, then connect an HTTPS tunnel to port 8001 and use its URL in the dashboard. The isolated directories start without your existing local datasets or analyses. Stop the tunnel to remove public access; stopping the launcher stops both Python services.

For an always-on deployment, use a persistent Python host and a stable HTTPS address. Free frontend hosting does not provide persistent ML compute or storage for this architecture.

## Free Render cloud demo

`render.yaml` defines a **Free** Docker web service using `Dockerfile.backend`. The container launches the API and worker together on port 10000, and generates a private application key during Blueprint creation. Import the repository as a Render Blueprint after reviewing the Free plan. The API health endpoint is `/api/health`.

`config.render.yaml` limits the demo to 5 MB uploads, 5,000 rows, 40 columns, 20 selected features, two queued/running jobs, and a 384 MB training-process budget within the host's memory limit. Start with the synthetic churn sample and Quick mode. Large ML/DL tasks may fail with a resource-limit error. This is a demo configuration, not a production capacity guarantee.

Render's free filesystem is temporary. Uploads, SQLite records, models, and reports are lost when the instance sleeps, restarts, or redeploys. The frontend displays this limitation when connected to this configuration. Download outputs you need to retain. The free database plan also has an expiry, so it is not used as a substitute for durable storage. See [Render free service limits](https://render.com/docs/free).

Keep the service on the Free plan and review account usage limits. No paid disk, database, or worker service is configured by this Blueprint. Set a zero spending limit where available; Render may charge usage overages when a payment method is present. This setup does not add a payment method.

After Render reports the backend healthy, set `VITE_API_BASE_URL` in Vercel to its HTTPS service URL and redeploy. `CORS_ORIGINS` in the Blueprint is the current dashboard origin. Enter the generated `APP_API_KEY` in the dashboard's API connection dialog. Add `GROQ_API_KEY` privately in Render's environment settings if you want Groq chat; the Blueprint does not contain a provider credential.

### Connection recovery

The dashboard retries the public health endpoint for up to 100 seconds while the free instance wakes up. It shows a connecting/starting message, then loads the workspace automatically. Health probes omit the application key so startup does not depend on a CORS preflight. Network failures during later API calls start another connection check; uploads, training requests, and chat are never automatically resubmitted. If the server is still unavailable, use **Retry connection** or review **API connection**. A 401 means the server is reachable but needs the correct application key.

Use the production dashboard at `https://data-science-agent-rosy.vercel.app`. A Vercel preview has a different origin and needs its own explicit backend CORS entry and API URL configuration. Do not use a wildcard CORS policy or publish the application key to solve connection errors.

The deployed Render service ignores `frontend/**` and `docs/**` in its Build Filters. Dashboard-only updates therefore deploy on Vercel without restarting the Python service and discarding its temporary workspace.
