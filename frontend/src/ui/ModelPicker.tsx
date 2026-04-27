import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  Cloud,
  Cpu,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
  Zap,
} from 'lucide-react';

import {
  deleteModel,
  listModels,
  testModel,
  type ModelConfig,
} from '../api/client';
import ModelManager from './ModelManager';

type Props = {
  currentProviderId: string;
  onProviderChange: (id: string) => void;
};

type TestState = { ok: boolean; message: string } | undefined;

function splitGroups(models: ModelConfig[]) {
  const builtins = models.filter((m) => m.builtin);
  const customs = models.filter((m) => !m.builtin);
  return { builtins, customs };
}

function protocolBadge(proto: string) {
  return proto === 'gemini' ? 'Gemini' : 'OpenAI';
}

/** 返回给列表行用的 proxy 文案。localhost 不显示。 */
function proxyHint(m: ModelConfig): string | null {
  const url = m.base_url || '';
  if (/^(https?:)?\/\/(127\.0\.0\.1|localhost)/i.test(url)) return null;
  const raw = (m.proxy || '').trim().toLowerCase();
  if (!raw || raw === 'auto') return 'auto';
  if (raw === 'direct') return 'direct';
  return 'proxy';
}

function iconFor(m: ModelConfig) {
  if (m.id === 'local') return <Cpu className="w-3.5 h-3.5 text-emerald-300" />;
  if (m.id === 'gemini') return <Sparkles className="w-3.5 h-3.5 text-amber-300" />;
  if (m.protocol === 'gemini') return <Cloud className="w-3.5 h-3.5 text-amber-200" />;
  return <Zap className="w-3.5 h-3.5 text-indigo-300" />;
}

export default function ModelPicker({ currentProviderId, onProviderChange }: Props) {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [dialogMode, setDialogMode] = useState<
    null | { mode: 'create' } | { mode: 'edit'; target: ModelConfig }
  >(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, TestState>>({});

  const popRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await listModels();
      setModels(data.models);
      if (!data.models.find((m) => m.id === currentProviderId)) {
        onProviderChange('local');
      }
    } catch (e) {
      console.error('[model-picker] load failed:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (popRef.current?.contains(t)) return;
      if (btnRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const current = useMemo(
    () => models.find((m) => m.id === currentProviderId),
    [models, currentProviderId]
  );
  const currentLabel = current?.label || '本地 Ollama';
  const currentModel = current?.model || '';

  const handleDelete = async (m: ModelConfig) => {
    if (!confirm(`确定删除模型「${m.label}」？此操作不可撤销。`)) return;
    try {
      await deleteModel(m.id);
      await refresh();
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const handleTest = async (m: ModelConfig) => {
    setTesting(m.id);
    setTestResult((prev) => ({ ...prev, [m.id]: undefined }));
    try {
      const res = await testModel(m.id);
      setTestResult((prev) => ({ ...prev, [m.id]: res }));
    } catch (e) {
      setTestResult((prev) => ({
        ...prev,
        [m.id]: { ok: false, message: (e as Error).message },
      }));
    } finally {
      setTesting(null);
    }
  };

  const { builtins, customs } = splitGroups(models);

  return (
    <div className="relative">
      {/* ───────── 触发器：圆润 pill，带当前模型 + model id ───────── */}
      <button
        ref={btnRef}
        onClick={() => setOpen((v) => !v)}
        className={`group flex items-center gap-2 text-[11px] pl-1.5 pr-2 py-1 rounded-full
                    border border-white/15 bg-white/5 hover:bg-white/10
                    text-slate-100 max-w-[240px] transition-all
                    ${open ? 'ring-2 ring-indigo-500/40 border-indigo-400/50 bg-white/10' : ''}`}
        title={current ? `当前模型：${current.label}（${current.model}）` : '选择 AI 模型'}
      >
        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
          {current ? iconFor(current) : <Cpu className="w-3.5 h-3.5 text-emerald-300" />}
        </span>
        <span className="flex flex-col items-start leading-tight min-w-0">
          <span className="truncate max-w-[140px] font-medium text-slate-100">
            {currentLabel}
          </span>
          {currentModel && (
            <span className="truncate max-w-[140px] font-mono text-[9px] text-slate-400">
              {currentModel}
            </span>
          )}
        </span>
        <ChevronDown
          className={`w-3 h-3 opacity-60 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* ───────── 下拉面板 ───────── */}
      {open && (
        <div
          ref={popRef}
          className="absolute right-0 top-[calc(100%+8px)] z-[60] w-[360px] rounded-2xl
                     border border-white/10 bg-slate-950/95 backdrop-blur-xl
                     shadow-2xl shadow-black/60 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* 头部 */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-gradient-to-r from-indigo-500/10 via-transparent to-transparent">
            <div>
              <div className="text-[11px] font-semibold text-slate-100 tracking-wide">
                AI 模型
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                选择用于问答 / 润色的推理后端
              </div>
            </div>
            <button
              onClick={refresh}
              disabled={loading}
              className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-100 transition-colors"
              title="重新加载"
            >
              {loading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {/* 列表 */}
          <div className="max-h-[380px] overflow-y-auto scrollbar-hide py-2">
            {builtins.length > 0 && <GroupHeader>内置</GroupHeader>}
            {builtins.map((m) => (
              <ModelRow
                key={m.id}
                m={m}
                active={m.id === currentProviderId}
                testing={testing === m.id}
                testResult={testResult[m.id]}
                onSelect={() => {
                  onProviderChange(m.id);
                  setOpen(false);
                }}
                onTest={() => handleTest(m)}
                onEdit={null}
                onDelete={null}
              />
            ))}

            {customs.length > 0 && <GroupHeader>自定义</GroupHeader>}
            {customs.map((m) => (
              <ModelRow
                key={m.id}
                m={m}
                active={m.id === currentProviderId}
                testing={testing === m.id}
                testResult={testResult[m.id]}
                onSelect={() => {
                  onProviderChange(m.id);
                  setOpen(false);
                }}
                onTest={() => handleTest(m)}
                onEdit={() => setDialogMode({ mode: 'edit', target: m })}
                onDelete={() => handleDelete(m)}
              />
            ))}

            {customs.length === 0 && (
              <div className="mx-3 my-2 px-3 py-2.5 rounded-lg border border-dashed border-white/10 text-[10.5px] text-slate-500 leading-relaxed">
                还没有自定义模型。可以接入 DeepSeek / Kimi / GLM /
                OpenRouter / LM Studio 等任意 OpenAI 兼容或 Gemini 协议服务。
              </div>
            )}
          </div>

          {/* 底部：新增按钮 */}
          <div className="border-t border-white/5 p-2 bg-black/30">
            <button
              onClick={() => {
                setDialogMode({ mode: 'create' });
                setOpen(false);
              }}
              className="group w-full flex items-center justify-center gap-2 text-xs font-medium
                         px-3 py-2.5 rounded-xl
                         bg-gradient-to-br from-indigo-600 to-indigo-500
                         hover:from-indigo-500 hover:to-indigo-400
                         text-white shadow-lg shadow-indigo-600/20
                         transition-all active:scale-[0.98]"
            >
              <Plus className="w-4 h-4 transition-transform group-hover:rotate-90" />
              添加模型
            </button>
          </div>
        </div>
      )}

      {dialogMode && (
        <ModelManager
          mode={dialogMode.mode}
          initial={dialogMode.mode === 'edit' ? dialogMode.target : undefined}
          onClose={() => setDialogMode(null)}
          onSaved={async (saved) => {
            setDialogMode(null);
            await refresh();
            onProviderChange(saved.id);
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 小组件                                                              */
/* ------------------------------------------------------------------ */

function GroupHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 pt-1.5 pb-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">
      {children}
    </div>
  );
}

function ModelRow(props: {
  m: ModelConfig;
  active: boolean;
  testing: boolean;
  testResult: TestState;
  onSelect: () => void;
  onTest: () => void;
  onEdit: null | (() => void);
  onDelete: null | (() => void);
}) {
  const { m, active, testing, testResult, onSelect, onTest, onEdit, onDelete } = props;

  return (
    <div
      className={`group relative mx-2 my-0.5 rounded-xl cursor-pointer transition-colors
                  ${
                    active
                      ? 'bg-indigo-500/15 ring-1 ring-indigo-400/30'
                      : 'hover:bg-white/[0.04]'
                  }`}
      onClick={onSelect}
    >
      {active && (
        <span className="absolute left-0 top-2 bottom-2 w-[2px] rounded-r bg-indigo-400" />
      )}

      <div className="flex items-center gap-2.5 px-3 py-2">
        <span
          className={`flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center
                      ${
                        active
                          ? 'bg-indigo-500/25 border border-indigo-400/30'
                          : 'bg-white/5 border border-white/10'
                      }`}
        >
          {iconFor(m)}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs font-medium truncate ${
                active ? 'text-white' : 'text-slate-100'
              }`}
            >
              {m.label}
            </span>
            <span className="text-[9px] px-1.5 py-px rounded border border-white/10 text-slate-400 flex-shrink-0">
              {protocolBadge(m.protocol)}
            </span>
            {m.builtin && (
              <span className="text-[9px] text-slate-500 flex-shrink-0">官方</span>
            )}
          </div>
          <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5">
            <span className="font-mono truncate">{m.model || '—'}</span>
            {(() => {
              const hint = proxyHint(m);
              if (!hint) return null;
              const color =
                hint === 'direct'
                  ? 'text-emerald-300/80'
                  : hint === 'proxy'
                  ? 'text-indigo-300/80'
                  : 'text-slate-500';
              const label =
                hint === 'direct' ? '直连' : hint === 'proxy' ? '代理' : '自动';
              return (
                <>
                  <span className="text-slate-700">·</span>
                  <span className={`${color} flex-shrink-0`} title={`网络模式：${label}`}>
                    {label}
                  </span>
                </>
              );
            })()}
            {!m.builtin && (
              <>
                <span className="text-slate-700">·</span>
                {m.has_api_key ? (
                  <span className="font-mono text-slate-500 truncate">
                    {m.api_key_mask}
                  </span>
                ) : (
                  <span className="text-amber-300/80">未设置 Key</span>
                )}
              </>
            )}
          </div>

          {testResult && (
            <div
              className={`mt-1 text-[10px] flex items-center gap-1 ${
                testResult.ok ? 'text-emerald-300' : 'text-rose-300'
              }`}
            >
              <span>{testResult.ok ? '✓' : '✗'}</span>
              <span className="truncate">{testResult.message}</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <IconBtn
            title="测试连接"
            onClick={(e) => {
              e.stopPropagation();
              onTest();
            }}
            disabled={testing}
          >
            {testing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Zap className="w-3.5 h-3.5" />
            )}
          </IconBtn>
          {onEdit && (
            <IconBtn
              title="编辑"
              onClick={(e) => {
                e.stopPropagation();
                onEdit();
              }}
            >
              <Pencil className="w-3.5 h-3.5" />
            </IconBtn>
          )}
          {onDelete && (
            <IconBtn
              title="删除"
              danger
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </IconBtn>
          )}
        </div>
      </div>
    </div>
  );
}

function IconBtn(props: {
  children: React.ReactNode;
  onClick: (e: React.MouseEvent) => void;
  title: string;
  disabled?: boolean;
  danger?: boolean;
}) {
  const { children, onClick, title, disabled, danger } = props;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-1.5 rounded-md transition-colors
                  ${
                    danger
                      ? 'text-slate-400 hover:bg-rose-500/15 hover:text-rose-300'
                      : 'text-slate-400 hover:bg-white/10 hover:text-slate-100'
                  }
                  disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      {children}
    </button>
  );
}
