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

export type OcrMode = 'auto' | 'force' | 'off';

export async function uploadPdf(file: File, ocrMode?: OcrMode): Promise<void> {
  const fd = new FormData();
  fd.append('file', file);
  if (ocrMode) fd.append('ocr_mode', ocrMode);
  const r = await fetch(apiUrl('/api/papers/upload'), { method: 'POST', body: fd });
  if (!r.ok) throw await parseErrorResponse(r, `POST /api/papers/upload failed: ${r.status}`);
}

export async function deletePaper(
  paperId: string,
  options: { purgeSource?: boolean } = {},
): Promise<void> {
  const purge = options.purgeSource ?? true;
  const r = await fetch(apiUrl(`/api/papers/${paperId}?purge_source=${purge ? '1' : '0'}`), { method: 'DELETE' });
  if (!r.ok) throw await parseErrorResponse(r, `DELETE /api/papers/${paperId} failed: ${r.status}`);
}

export type ReprocessResult = {
  success: boolean;
  paper_id: string;
  ocr_mode: string;
  parser?: string;
  text_len: number;
  abstract_len: number;
  keywords: string[];
};

export type IndexStatus = {
  current_model: string;
  last_indexed_model: string | null;
  needs_reindex: boolean;
  total_papers: number;
};

export async function getIndexStatus(): Promise<IndexStatus> {
  const r = await fetch(apiUrl('/api/admin/index-status'));
  if (!r.ok) throw await parseErrorResponse(r, `GET /api/admin/index-status failed: ${r.status}`);
  return r.json();
}

export type ReindexResult = {
  success: boolean;
  total: number;
  success_count?: number;  // 兼容字段
  failed: number;
  model: string;
};

export async function reindexAll(): Promise<ReindexResult & { success: number }> {
  const r = await fetch(apiUrl('/api/admin/reindex'), { method: 'POST' });
  if (!r.ok) throw await parseErrorResponse(r, `POST /api/admin/reindex failed: ${r.status}`);
  return r.json();
}

export type VisionQueryResponse = {
  answer: string;
  description: string;
  cites: string[];
  cite_details: { paper_id: string; chunk_id: string; snippet: string; page?: number | null }[];
  provider_used: string;
  status: string;
};

export async function queryVision(params: {
  image: File | Blob;
  question: string;
  provider: string;
  paperId?: string;
  mode?: 'auto' | 'chat' | 'rag';
}): Promise<VisionQueryResponse> {
  const fd = new FormData();
  fd.append('image', params.image, 'screenshot.png');
  fd.append('question', params.question);
  fd.append('provider', params.provider);
  if (params.paperId) fd.append('paper_id', params.paperId);
  if (params.mode) fd.append('mode', params.mode);
  const r = await fetch(apiUrl('/api/query_vision'), { method: 'POST', body: fd });
  if (!r.ok) throw await parseErrorResponse(r, `POST /api/query_vision failed: ${r.status}`);
  return r.json();
}

export type ParserKind = 'default' | 'mineru';

export async function reprocessPaper(
  paperId: string,
  options: { ocrMode?: OcrMode; parser?: ParserKind } | OcrMode = {},
): Promise<ReprocessResult> {
  // 兼容老调用 reprocessPaper(id, 'force')
  const opts: { ocrMode?: OcrMode; parser?: ParserKind } =
    typeof options === 'string' ? { ocrMode: options } : options;
  const body: Record<string, unknown> = {};
  if (opts.ocrMode) body.ocr_mode = opts.ocrMode;
  if (opts.parser) body.parser = opts.parser;
  const r = await fetch(apiUrl(`/api/papers/${paperId}/reprocess`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw await parseErrorResponse(r, `POST /api/papers/${paperId}/reprocess failed: ${r.status}`);
  return r.json();
}

export type MinerUStatus = {
  available: boolean;
  cmd?: string;
  backend?: string;
  lang?: string;
  error?: string;
};

export async function getMinerUStatus(): Promise<MinerUStatus> {
  const r = await fetch(apiUrl('/api/mineru/status'));
  if (!r.ok) throw await parseErrorResponse(r, `GET /api/mineru/status failed: ${r.status}`);
  return r.json();
}

export type PolishSummaryResponse = {
  success: boolean;
  paper_id: string;
  summary: string;
  provider_used: string;
};

export async function polishSummary(
  paperId: string,
  provider: string
): Promise<PolishSummaryResponse> {
  const r = await fetch(apiUrl(`/api/papers/${paperId}/polish-summary`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });
  if (!r.ok) throw await parseErrorResponse(r, `润色摘要失败（HTTP ${r.status}）`);
  return r.json();
}

export async function clearPolishedSummary(paperId: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/papers/${paperId}/polish-summary`), { method: 'DELETE' });
  if (!r.ok) throw await parseErrorResponse(r, `清除润色摘要失败（HTTP ${r.status}）`);
}

export type ChatTurn = { role: 'user' | 'ai'; text: string };

/** 问答模式：auto=自适应，chat=强制通用对话（不检索），rag=强制基于文献 */
export type AssistantMode = 'auto' | 'chat' | 'rag';

export async function queryWithProvider(
  prompt: string,
  provider: string,
  history?: ChatTurn[],
  mode: AssistantMode = 'auto',
): Promise<QueryResponse> {
  const r = await fetch(apiUrl('/api/query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: prompt,
      provider,
      history: history || [],
      mode,
    }),
  });
  if (!r.ok) throw new Error(`POST /api/query failed: ${r.status}`);
  return r.json();
}

/** @deprecated 保留向后兼容，建议改用 queryWithProvider */
export async function queryLocal(prompt: string, history?: ChatTurn[]): Promise<QueryResponse> {
  return queryWithProvider(prompt, 'local', history);
}

// ------------------------------------------------------------------ //
// 模型注册表（内置 + 自定义）
// ------------------------------------------------------------------ //

export type LLMProtocol = 'openai' | 'gemini';

export type ModelConfig = {
  id: string;
  label: string;
  protocol: LLMProtocol;
  base_url: string;
  model: string;
  api_key: string;           // 列表接口永远为空串，不泄露明文
  api_key_mask: string;
  has_api_key: boolean;
  /** 网络模式：''|'auto' | 'direct' | 'http(s)://...' | 'socks5(h)://...' */
  proxy: string;
  builtin: boolean;
  requires_key: boolean;
};

export type ModelsListResponse = {
  models: ModelConfig[];
  default_provider: string;
};

export type ModelCreatePayload = {
  label: string;
  protocol: LLMProtocol;
  base_url?: string;
  model: string;
  api_key?: string;
  proxy?: string;
};

export type ModelTestResult = {
  ok: boolean;
  message: string;
  sample?: string;
  proxy_mode?: string;   // 后端回写本次使用的网络模式，便于调试
};

export async function listModels(): Promise<ModelsListResponse> {
  const r = await fetch(apiUrl('/api/models'));
  if (!r.ok) throw await parseErrorResponse(r, `获取模型列表失败（HTTP ${r.status}）`);
  return r.json();
}

export async function createModel(payload: ModelCreatePayload): Promise<ModelConfig> {
  const r = await fetch(apiUrl('/api/models'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw await parseErrorResponse(r, `创建模型失败（HTTP ${r.status}）`);
  const data = await r.json();
  return data.model as ModelConfig;
}

export async function updateModel(
  modelId: string,
  payload: Partial<ModelCreatePayload>
): Promise<ModelConfig> {
  const r = await fetch(apiUrl(`/api/models/${modelId}`), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw await parseErrorResponse(r, `更新模型失败（HTTP ${r.status}）`);
  const data = await r.json();
  return data.model as ModelConfig;
}

export async function deleteModel(modelId: string): Promise<void> {
  const r = await fetch(apiUrl(`/api/models/${modelId}`), { method: 'DELETE' });
  if (!r.ok) throw await parseErrorResponse(r, `删除模型失败（HTTP ${r.status}）`);
}

export async function testModelAdhoc(
  payload: ModelCreatePayload
): Promise<ModelTestResult> {
  const r = await fetch(apiUrl('/api/models/test'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw await parseErrorResponse(r, `测试连接失败（HTTP ${r.status}）`);
  return r.json();
}

export async function testModel(modelId: string): Promise<ModelTestResult> {
  const r = await fetch(apiUrl(`/api/models/${modelId}/test`), { method: 'POST' });
  if (!r.ok) throw await parseErrorResponse(r, `测试连接失败（HTTP ${r.status}）`);
  return r.json();
}