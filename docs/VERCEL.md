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
