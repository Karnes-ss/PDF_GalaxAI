import type { GraphResponse, QueryResponse } from '../types/scholar';

export const backendBaseUrl: string =
  (import.meta.env.VITE_BACKEND_BASE as string | undefined) ?? (import.meta.env.DEV ? '' : 'http://127.0.0.1:8000');

export function apiUrl(path: string): string {
  return `${backendBaseUrl}${path}`;
}

export function fileUrl(paperId: string): string {
  return `${backendBaseUrl}/files/${paperId}.pdf`;
}

export async function fetchGraph(): Promise<GraphResponse> {
  const r = await fetch(apiUrl('/api/papers'));
  if (!r.ok) throw new Error(`GET /api/papers failed: ${r.status}`);
  return r.json();
}

async function parseErrorResponse(response: Response, defaultMessage: string): Promise<Error> {
  const contentType = response.headers.get('Content-Type') || '';
  let detail = defaultMessage;

  try {
    if (contentType.includes('application/json')) {
      const body = await response.json();
      if (body?.detail) {
        detail = String(body.detail);
      } else if (body?.message) {
        detail = String(body.message);
      }
    } else {
      const text = await response.text();
      if (text) {
        detail = text;
      }
    }
  } catch {
    // ignore parse errors, fall back to default message
  }

  return new Error(detail || defaultMessage);
}

export async function uploadPdf(file: File): Promise<void> {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(apiUrl('/api/papers/upload'), { method: 'POST', body: fd });
  if (!r.ok) throw await parseErrorResponse(r, `POST /api/papers/upload failed: ${r.status}`);
}

export async function deletePaper(paperId: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/papers/${paperId}`), { method: 'DELETE' });
  if (!r.ok) throw await parseErrorResponse(r, `DELETE /api/papers/${paperId} failed: ${r.status}`);
}

export async function queryLocal(prompt: string): Promise<QueryResponse> {
  const r = await fetch(apiUrl('/api/query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: prompt, provider: 'local' }),
  });
  if (!r.ok) throw new Error(`POST /api/query failed: ${r.status}`);
  return r.json();
}

export async function queryWithProvider(prompt: string, provider: 'local' | 'gemini'): Promise<QueryResponse> {
  const r = await fetch(apiUrl('/api/query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: prompt, provider }),
  });
  if (!r.ok) throw new Error(`POST /api/query failed: ${r.status}`);
  return r.json();
}