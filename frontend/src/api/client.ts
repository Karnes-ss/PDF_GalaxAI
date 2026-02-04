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

export async function uploadPdf(file: File): Promise<void> {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(apiUrl('/api/papers/upload'), { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`POST /api/papers/upload failed: ${r.status}`);
}

export async function queryLocal(prompt: string): Promise<QueryResponse> {
  const r = await fetch(apiUrl('/api/query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // ⚠️ CRITICAL CHANGE: 后端 api.py 中的 ChatBody 定义字段为 'question'，必须匹配
    body: JSON.stringify({ question: prompt }),
  });
  if (!r.ok) throw new Error(`POST /api/query failed: ${r.status}`);
  return r.json();
}