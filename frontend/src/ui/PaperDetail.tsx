import React, { useState, useEffect } from 'react';
import { BookOpen, FileText, X, Search, Sparkles, Loader2, RotateCcw, Paperclip, ScanText, Trash2 } from 'lucide-react';
import type { Edge, Paper } from '../types/scholar';
import {
  polishSummary,
  clearPolishedSummary,
  reprocessPaper,
  getMinerUStatus,
  type OcrMode,
  type ParserKind,
} from '../api/client';

interface Props {
  selectedPaper: Paper;
  edges: Edge[];
  onClose: () => void;
  onOpenReader: (p: Paper) => void;
  onDelete: (p: Paper, options?: { purgeSource?: boolean }) => void;
  onAddToChat?: (p: Paper) => void;
  onSummaryUpdated?: (paperId: string, summary: string) => void;
  onReprocessed?: (paperId: string) => void;
  screenPosition: { x: number; y: number };
  aiChatWidth: number;
  llmProvider: string;   // 模型 id：local / gemini / cm_xxx
}

export default function PaperDetail({ selectedPaper, edges, onClose, onOpenReader, onDelete, onAddToChat, onSummaryUpdated, onReprocessed, screenPosition, aiChatWidth, llmProvider }: Props) {
  const linked = edges.filter((e) => e.source === selectedPaper.id || e.target === selectedPaper.id);
  const linkCount = linked.length;

  const [panelStyle, setPanelStyle] = useState({});
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [startMousePos, setStartMousePos] = useState({ x: 0, y: 0 });

  // 润色摘要的本地状态：用于立即回显 + loading + 错误提示
  const [polishedLocal, setPolishedLocal] = useState<string>(selectedPaper.llmSummary || '');
  const [polishing, setPolishing] = useState<boolean>(false);
  const [polishError, setPolishError] = useState<string>('');

  // 重新识别（单篇 OCR 覆盖）的本地状态
  const [reprocessing, setReprocessing] = useState<boolean>(false);
  const [reprocessMenuOpen, setReprocessMenuOpen] = useState<boolean>(false);
  const [reprocessError, setReprocessError] = useState<string>('');
  const [minerUAvailable, setMinerUAvailable] = useState<boolean | null>(null);
  const [deleteMenuOpen, setDeleteMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    // 菜单第一次打开前预检查一次 MinerU 是否可用，避免用户点下去才发现没装
    let cancelled = false;
    if (reprocessMenuOpen && minerUAvailable === null) {
      getMinerUStatus()
        .then((s) => !cancelled && setMinerUAvailable(!!s.available))
        .catch(() => !cancelled && setMinerUAvailable(false));
    }
    return () => {
      cancelled = true;
    };
  }, [reprocessMenuOpen, minerUAvailable]);

  // 切换选中的论文时，同步本地润色缓存
  useEffect(() => {
    setPolishedLocal(selectedPaper.llmSummary || '');
    setPolishError('');
    setPolishing(false);
    setDeleteMenuOpen(false);
  }, [selectedPaper.id, selectedPaper.llmSummary]);

  const displaySummary = polishedLocal || selectedPaper.abstract || selectedPaper.firstSentence || '暂无摘要内容...';

  const handlePolish = async () => {
    if (polishing) return;
    setPolishing(true);
    setPolishError('');
    try {
      const res = await polishSummary(selectedPaper.id, llmProvider);
      setPolishedLocal(res.summary);
      onSummaryUpdated?.(selectedPaper.id, res.summary);
    } catch (e: any) {
      setPolishError(e?.message || 'AI 润色失败');
    } finally {
      setPolishing(false);
    }
  };

  const handleResetSummary = async () => {
    if (polishing) return;
    setPolishing(true);
    setPolishError('');
    try {
      await clearPolishedSummary(selectedPaper.id);
      setPolishedLocal('');
      onSummaryUpdated?.(selectedPaper.id, '');
    } catch (e: any) {
      setPolishError(e?.message || '恢复原摘要失败');
    } finally {
      setPolishing(false);
    }
  };

  const handleReprocess = async (
    mode: OcrMode | 'mineru',
  ) => {
    if (reprocessing) return;
    setReprocessMenuOpen(false);

    let hint: string;
    let payload: { ocrMode?: OcrMode; parser?: ParserKind };
    if (mode === 'mineru') {
      hint =
        '将用 MinerU 对该文档做版面/公式/表格识别（输出 Markdown + LaTeX）。\n\n' +
        '首次使用需确保 mineru CLI 已安装（后端日志有提示）。\n' +
        '解析较慢：GPU 每页 1~3 秒，CPU 可能 20~60 秒/页。\n\n继续吗？';
      payload = { parser: 'mineru' };
    } else if (mode === 'force') {
      hint = '将对此文档逐页强制 OCR（速度较慢但对扫描件/公式页效果更好）。\n继续吗？';
      payload = { ocrMode: mode };
    } else if (mode === 'off') {
      hint = '将只用 PDF 原生文本层重新解析（最快，适合不是扫描件的文档）。\n继续吗？';
      payload = { ocrMode: mode };
    } else {
      hint = '将按自动模式重新识别此文档（逐页判质量，差页再 OCR）。\n继续吗？';
      payload = { ocrMode: mode };
    }
    if (!window.confirm(hint)) return;

    setReprocessing(true);
    setReprocessError('');
    try {
      const res = await reprocessPaper(selectedPaper.id, payload);
      console.log('[reprocess] done', res);
      onReprocessed?.(selectedPaper.id);
    } catch (e: any) {
      setReprocessError(e?.message || '重新识别失败');
    } finally {
      setReprocessing(false);
    }
  };

  useEffect(() => {
    const calculatePosition = () => {
      const panelWidth = 420;
      // More accurate height calculation based on component structure
      const panelHeight = 400; // Increased from 300 to account for actual content
      const aiChatWidthLocal = aiChatWidth;
      const padding = 10;

      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;

      const rightUsableEdge = windowWidth - aiChatWidthLocal;
      const leftUsableEdge = 80; // Left sidebar width

      // Define possible positions with preference order
      const possiblePositions = [
        // Bottom-right (preferred when click point is in top-left area)
        { x: screenPosition.x + 20, y: screenPosition.y + 20, priority: 1 },
        // Bottom-left (preferred when click point is in top-right area)
        { x: screenPosition.x - panelWidth - 20, y: screenPosition.y + 20, priority: 2 },
        // Top-right (fallback when bottom is blocked)
        { x: screenPosition.x + 20, y: screenPosition.y - panelHeight - 20, priority: 3 },
        // Top-left (last resort)
        { x: screenPosition.x - panelWidth - 20, y: screenPosition.y - panelHeight - 20, priority: 4 },
      ];

      // Filter positions that fit within bounds
      const validPositions = possiblePositions.filter(pos => {
        const clampedLeft = Math.max(leftUsableEdge + padding, Math.min(pos.x, rightUsableEdge - panelWidth - padding));
        const clampedTop = Math.max(padding, Math.min(pos.y, windowHeight - panelHeight - padding));
        return clampedLeft === pos.x && clampedTop === pos.y; // Only if no clamping was needed
      });

      let finalLeft: number;
      let finalTop: number;

      if (validPositions.length > 0) {
        // Use the highest priority valid position
        const bestPos = validPositions.reduce((best, current) =>
          current.priority < best.priority ? current : best
        );
        finalLeft = bestPos.x;
        finalTop = bestPos.y;
      } else {
        // Fallback: clamp the preferred position
        finalLeft = Math.max(leftUsableEdge + padding, Math.min(possiblePositions[0].x, rightUsableEdge - panelWidth - padding));
        finalTop = Math.max(padding, Math.min(possiblePositions[0].y, windowHeight - panelHeight - padding));
      }

      setPanelStyle({
        borderColor: selectedPaper.color,
        left: finalLeft,
        top: finalTop,
      });
    };

    calculatePosition();
    window.addEventListener('resize', calculatePosition);
    return () => window.removeEventListener('resize', calculatePosition);
  }, [screenPosition, selectedPaper.color]);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const newLeft = e.clientX - dragOffset.x;
      const newTop = e.clientY - dragOffset.y;

      // Apply boundary checks to the new position
      const panelWidth = 420;
      const panelHeight = 300;
      const aiChatWidthLocal = aiChatWidth;
      const padding = 10;
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const dragRightBoundary = windowWidth; // Allow dragging across the full window width

      let constrainedLeft = Math.max(padding, Math.min(newLeft, dragRightBoundary - panelWidth - padding));
      let constrainedTop = Math.max(padding, Math.min(newTop, windowHeight - panelHeight - padding));

      setPanelStyle((prevStyle) => ({
        ...prevStyle,
        left: constrainedLeft,
        top: constrainedTop,
      }));
    };

    const onMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [isDragging, dragOffset, aiChatWidth]);
  return (
    <div
      className="absolute w-[420px] glass rounded-2xl p-6 border-l-4 animate-in fade-in slide-in-from-left-4 z-40 flex flex-col gap-4 shadow-2xl shadow-black/50"
      style={{ ...panelStyle, cursor: isDragging ? 'grabbing' : 'grab' }}
      onMouseDown={(e) => {
        setIsDragging(true);
        setStartMousePos({ x: e.clientX, y: e.clientY });
        const currentLeft = (panelStyle as React.CSSProperties).left as number;
        const currentTop = (panelStyle as React.CSSProperties).top as number;
        setDragOffset({ x: e.clientX - currentLeft, y: e.clientY - currentTop });
      }}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1 mr-4 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider text-black/80"
              style={{ backgroundColor: selectedPaper.color }}
            >
              {selectedPaper.field}
            </span>
            <span className="text-[10px] text-slate-500 font-mono">CONF: {(selectedPaper.confidence * 100).toFixed(0)}%</span>
          </div>
          <h3 className="font-bold text-lg leading-tight text-white/90 break-words break-all" title={selectedPaper.title}>
            {selectedPaper.displayTitle}
          </h3>
        </div>
        <button onClick={onClose} className="shrink-0 p-1 hover:bg-white/10 rounded-lg transition-colors">
          <X className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      <div className="bg-black/20 rounded-xl p-3 max-h-[140px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[10px] uppercase tracking-wider font-bold text-slate-500">
            {polishedLocal ? 'AI 润色摘要' : '自动摘要'}
          </span>
          <div className="flex items-center gap-1">
            {polishedLocal && !polishing && (
              <button
                onClick={handleResetSummary}
                className="text-[10px] text-slate-400 hover:text-slate-200 flex items-center gap-1 px-1.5 py-0.5 rounded hover:bg-white/5 transition-colors"
                title="恢复为自动抽取的摘要"
              >
                <RotateCcw className="w-3 h-3" /> 还原
              </button>
            )}
            <button
              onClick={handlePolish}
              disabled={polishing}
              className="text-[10px] flex items-center gap-1 px-2 py-0.5 rounded border border-fuchsia-400/40 text-fuchsia-200 hover:bg-fuchsia-500/10 transition-colors disabled:opacity-60 disabled:cursor-wait"
              title={`用当前选择的模型（${llmProvider}）生成一段润色摘要`}
            >
              {polishing ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" /> 润色中…
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3" /> {polishedLocal ? '重新润色' : 'AI 润色'}
                </>
              )}
            </button>
          </div>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{displaySummary}</p>
        {polishError && (
          <p className="mt-2 text-[10px] text-red-300/90 break-all">{polishError}</p>
        )}
      </div>

      {selectedPaper.keywords && selectedPaper.keywords.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedPaper.keywords.slice(0, 5).map((kw, i) => (
            <span key={i} className="text-[10px] px-2 py-1 rounded bg-white/5 border border-white/10 text-slate-300">
              #{kw}
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between pt-2 border-t border-white/10">
        <div className="flex gap-4">
          <div className="text-xs text-slate-400 flex items-center gap-1.5" title="关联连线数">
            <BookOpen className="w-3 h-3 text-indigo-400" /> <span className="font-mono">{linkCount}</span>
          </div>
          <div className="text-xs text-slate-400 flex items-center gap-1.5" title="文件类型">
            <FileText className="w-3 h-3 text-emerald-400" /> <span className="font-mono">PDF</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onAddToChat && (
            <button
              onClick={() => onAddToChat(selectedPaper)}
              className="bg-white/5 hover:bg-white/10 border border-white/10 hover:border-indigo-400/50 text-slate-200 hover:text-white text-xs font-semibold px-3 py-2 rounded-lg transition-all flex items-center gap-1.5 active:scale-95"
              title="把这篇加入右侧对话框，方便针对它提问（也可在星图中右键/Ctrl+Click 节点）"
            >
              <Paperclip className="w-3 h-3" />
              加入对话
            </button>
          )}
          <button
            onClick={() => onOpenReader(selectedPaper)}
            className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-4 py-2 rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-indigo-600/20 active:scale-95"
          >
            <Search className="w-3 h-3" />
            深度阅读
          </button>
        </div>
      </div>

      <div className="pt-2 border-t border-white/10 flex items-center justify-between gap-3">
        <div className="relative">
          <button
            onClick={() => (reprocessing ? null : setReprocessMenuOpen((v) => !v))}
            disabled={reprocessing}
            className="text-xs px-3 py-1.5 rounded-md border border-indigo-400/40 text-indigo-200 hover:bg-indigo-500/10 transition-colors flex items-center gap-1.5 disabled:opacity-60 disabled:cursor-not-allowed"
            title="如果觉得这篇识别得不太对（公式、扫描件、乱码），可以单独重新解析一下"
          >
            {reprocessing ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                重新识别中…
              </>
            ) : (
              <>
                <ScanText className="w-3 h-3" />
                重新识别
              </>
            )}
          </button>
          {reprocessMenuOpen && !reprocessing && (
            <>
              <div
                className="fixed inset-0 z-[998]"
                onClick={() => setReprocessMenuOpen(false)}
              />
              <div className="absolute bottom-full left-0 mb-1 w-64 rounded-lg bg-slate-900/95 border border-white/10 shadow-xl backdrop-blur p-1 z-[999]">
                <button
                  onClick={() => handleReprocess('mineru')}
                  disabled={minerUAvailable === false}
                  className={`w-full text-left text-xs px-3 py-2 rounded-md ${
                    minerUAvailable === false
                      ? 'opacity-40 cursor-not-allowed'
                      : 'hover:bg-amber-500/15 text-slate-100'
                  }`}
                  title={
                    minerUAvailable === false
                      ? '未检测到 mineru 命令。请参考 README 安装后重启后端'
                      : 'MinerU：针对公式、表格、版面的学术 PDF 解析器'
                  }
                >
                  <div className="font-semibold flex items-center gap-1.5">
                    MinerU（公式/表格最优）
                    {minerUAvailable === true && (
                      <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-400/30">可用</span>
                    )}
                    {minerUAvailable === false && (
                      <span className="text-[9px] px-1 py-0.5 rounded bg-slate-500/20 text-slate-400 border border-slate-400/30">未安装</span>
                    )}
                    {minerUAvailable === null && (
                      <span className="text-[9px] px-1 py-0.5 rounded bg-slate-500/20 text-slate-400 border border-slate-400/30">检查中…</span>
                    )}
                  </div>
                  <div className="text-[10px] text-slate-400">输出 Markdown + LaTeX 公式，理科文献首选</div>
                </button>
                <div className="my-1 border-t border-white/5" />
                <button
                  onClick={() => handleReprocess('force')}
                  className="w-full text-left text-xs px-3 py-2 rounded-md hover:bg-indigo-500/15 text-slate-100"
                >
                  <div className="font-semibold">强制 OCR</div>
                  <div className="text-[10px] text-slate-400">逐页重跑 OCR，适合扫描件</div>
                </button>
                <button
                  onClick={() => handleReprocess('auto')}
                  className="w-full text-left text-xs px-3 py-2 rounded-md hover:bg-white/5 text-slate-200"
                >
                  <div className="font-semibold">自动</div>
                  <div className="text-[10px] text-slate-400">按质量自适应（默认）</div>
                </button>
                <button
                  onClick={() => handleReprocess('off')}
                  className="w-full text-left text-xs px-3 py-2 rounded-md hover:bg-white/5 text-slate-200"
                >
                  <div className="font-semibold">仅原生文本层</div>
                  <div className="text-[10px] text-slate-400">最快，确定不是扫描件时用</div>
                </button>
              </div>
            </>
          )}
          {reprocessError && (
            <div className="absolute bottom-full left-0 mb-1 w-72 text-[11px] text-red-300 bg-red-500/10 border border-red-500/30 rounded-md px-2 py-1">
              {reprocessError}
            </div>
          )}
        </div>
        <div className="relative">
          <button
            onClick={() => setDeleteMenuOpen((v) => !v)}
            className="text-xs px-3 py-1.5 rounded-md border border-red-500/40 text-red-300 hover:bg-red-500/10 transition-colors flex items-center gap-1.5"
            title="删除方式"
          >
            <Trash2 className="w-3.5 h-3.5" />
            删除文献
          </button>
          {deleteMenuOpen && (
            <>
              <div
                className="fixed inset-0 z-[998]"
                onClick={() => setDeleteMenuOpen(false)}
              />
              <div className="absolute right-0 bottom-full mb-1 w-72 rounded-lg bg-slate-900/95 border border-white/10 shadow-xl backdrop-blur p-1 z-[999]">
                <button
                  onClick={() => {
                    setDeleteMenuOpen(false);
                    onDelete(selectedPaper, { purgeSource: true });
                  }}
                  className="w-full text-left text-xs px-3 py-2 rounded-md hover:bg-red-500/15 text-red-200"
                >
                  <div className="font-semibold">彻底删除（推荐）</div>
                  <div className="text-[10px] text-slate-400">
                    删除知识库记录 + inbox 源文件，重启后不会回流
                  </div>
                </button>
                <button
                  onClick={() => {
                    setDeleteMenuOpen(false);
                    onDelete(selectedPaper, { purgeSource: false });
                  }}
                  className="w-full text-left text-xs px-3 py-2 rounded-md hover:bg-white/5 text-slate-200"
                >
                  <div className="font-semibold">仅从知识库移除</div>
                  <div className="text-[10px] text-slate-400">
                    保留 inbox 源文件；以后仍可再次导入
                  </div>
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
