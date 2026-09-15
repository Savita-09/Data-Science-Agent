export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

export class BackendUnavailableError extends Error {
  constructor() {
    super('The Python server is temporarily unreachable. A free server may need about a minute to wake up. Reconnect, then try your action again.');
    this.name = 'BackendUnavailableError';
  }
}

export function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const finish = () => { signal.removeEventListener('abort', cancel); resolve(); };
    const timer = setTimeout(finish, ms);
    const cancel = () => { clearTimeout(timer); reject(signal.reason); };
    signal.addEventListener('abort', cancel, {once: true});
  });
}

/** One attempt only: never automatically replay uploads, training, or chat requests. */
export async function requestJson<T>(url: string, options: RequestInit = {}, fetcher: typeof fetch = fetch): Promise<T> {
  let response: Response;
  try {
    response = await fetcher(url, options);
  } catch (error) {
    if (options.signal?.aborted) throw options.signal.reason;
    throw new BackendUnavailableError();
  }
  // A sleeping/restarting host may return HTML (sometimes with HTTP 200).
  if ([502, 503, 504].includes(response.status)) throw new BackendUnavailableError();
  const json = response.headers.get('content-type')?.includes('application/json');
  if (!response.ok) {
    let detail = response.statusText || 'Request failed';
    if (json) {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    }
    throw new ApiError(`${response.status}: ${detail}`, response.status);
  }
  if (!json) throw new BackendUnavailableError();
  return response.json();
}

type ConnectionOptions = {
  signal: AbortSignal;
  onRetry?: () => void;
  fetcher?: typeof fetch;
  timeoutMs?: number;
  attemptMs?: number;
  retryMs?: number;
};

/** Wake up the public health endpoint without sending an API key or triggering a preflight. */
export async function waitForBackend<T>(url: string, {
  signal, onRetry, fetcher = fetch, timeoutMs = 100_000, attemptMs = 10_000, retryMs = 2_500,
}: ConnectionOptions): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    signal.throwIfAborted();
    const attempt = new AbortController();
    const cancel = () => attempt.abort(signal.reason);
    signal.addEventListener('abort', cancel, {once: true});
    const timer = setTimeout(() => attempt.abort(), Math.min(attemptMs, deadline - Date.now()));
    try {
      return await requestJson<T>(url, {signal: attempt.signal, cache: 'no-store', credentials: 'omit'}, fetcher);
    } catch (error) {
      signal.throwIfAborted();
      if (!(error instanceof BackendUnavailableError) && !attempt.signal.aborted) throw error;
      onRetry?.();
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', cancel);
    }
    const remaining = deadline - Date.now();
    if (remaining > 0) await abortableDelay(Math.min(retryMs, remaining), signal);
  }
  throw new BackendUnavailableError();
}
