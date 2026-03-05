const API_BASE = 'http://localhost:8420';

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  params?: Record<string, string | number>,
): Promise<T> {
  const url = new URL(path, API_BASE);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
    });
  }
  const resp = await fetch(url.toString(), {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`${resp.status}: ${text}`);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

export const apiGet = <T>(path: string, params?: Record<string, string | number>) =>
  request<T>('GET', path, undefined, params);

export const apiPost = <T>(path: string, body?: unknown) =>
  request<T>('POST', path, body);

export const apiPut = <T>(path: string, body: unknown) =>
  request<T>('PUT', path, body);

export const apiDelete = <T>(path: string, params?: Record<string, string | number>) =>
  request<T>('DELETE', path, undefined, params);
