import { useState, useEffect, useRef } from 'react';
import { Activity, BookOpen, Compass, FileText, MessageSquare, Paperclip, Search, Send, Sparkles, Upload, X, Zap } from 'lucide-react';
import { deletePaper, fetchGraph, fileUrl, queryWithProvider, uploadPdf, type AssistantMode } from '../api/client';
import GalaxyArea from './GalaxyArea';
import IndexAdmin from './IndexAdmin';
import MarkdownMessage from './MarkdownMessage';
import ModelPicker from './ModelPicker';
import PaperDetail from './PaperDetail';
import ReaderModal from './ReaderModal';
import type { Edge, Paper } from '../types/scholar';

const LLM_PROVIDER_STORAGE_KEY = 'scholar:llm-provider';
const ASSISTANT_MODE_STORAGE_KEY = 'scholar:assistant-mode';

export function App() {
  type CiteDetail = { paper_id: string; chunk_id?: string; snippet?: string; page?: number | null };
  type ChatMsg = { role: string; text: string; cites?: string[]; citeDetails?: CiteDetail[] };
  const normalizeAnswerText = (raw: string, details: CiteDetail[] = []) => {
    const firstIndexByPaper = new Map<string, number>();
    details.forEach((d, i) => {
      if (!d?.paper_id || firstIndexByPaper.has(d.paper_id)) return;
      firstIndexByPaper.set(d.paper_id, i + 1);
    });
    let t = raw || '';
    t = t.replace(/\[CITE:([^\]]+)\]/gi, (_m, pid) => {
      const idx = firstIndexByPaper.get(String(pid).trim());
      return idx ? `[参考${idx}]` : '';
    });
    t = t.replace(/(?:\(|（)?\s*paper_id\s*=\s*([A-Za-z0-9_-]+)\s*(?:\)|）)?/gi, (_m, pid) => {
      const idx = firstIndexByPaper.get(String(pid).trim());
      return idx ? `[参考${idx}]` : '';
    });
    t = t.replace(/(?:\(|（)?\s*chunk_id\s*=\s*([A-Za-z0-9_-]+)\s*(?:\)|）)?/gi, '');
    t = t.replace(/\n{3,}/g, '\n\n').replace(/[ \t]{2,}/g, ' ').trim();
    return t;
  };
  const citationTerms = (text: string) => {
    const raw = (text || '').toLowerCase();
    const zh = raw.match(/[\u4e00-\u9fa5]{3,}/g) || [];
    const en = raw.match(/[a-z][a-z0-9_-]{4,}/g) || [];
    const stop = new Set([
      'this',
      'that',
      'with',
      'from',
      'have',
      'about',
      'which',
      'there',
      'their',
      'paper',
      'chunk',
      'reference',
      'according',
      'based',
    ]);
    return Array.from(new Set([...zh, ...en].filter((x) => !stop.has(x)))).slice(0, 24);
  };
  const scoreCitationForBlock = (block: string, hit: CiteDetail) => {
    const blockTerms = citationTerms(block);
    const snippetTerms = new Set(citationTerms(hit.snippet || ''));
    if (!blockTerms.length || !snippetTerms.size) return 0;
    return blockTerms.reduce((score, term) => score + (snippetTerms.has(term) ? 1 : 0), 0);
  };
  const citationIndexesForBlock = (block: string, cards: CiteDetail[], fallbackIdx: number) => {
    if (!cards.length) return [];
    const ranked = cards
      .map((hit, idx) => ({ idx, score: scoreCitationForBlock(block, hit) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .map((x) => x.idx);
    if (ranked.length) return ranked;
    return [Math.min(fallbackIdx, cards.length - 1)];
  };
  const [papers, setPapers] = useState<Paper[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [readerPaper, setReaderPaper] = useState<Paper | null>(null);
  const [searchText, setSearchText] = useState('');
  const [highlightedSearchPaperId, setHighlightedSearchPaperId] = useState<string | null>(null);
  const [selectedPaperScreenPosition, setSelectedPaperScreenPosition] = useState<{ x: number; y: number } | null>(null);
  const [chat, setChat] = useState<ChatMsg[]>([
    { role: 'ai', text: '欢迎进入本地 Scholar 星系。请上传 PDF 文献以生成你的专属知识星云。' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [llmProvider, setLlmProvider] = useState<string>(() => {
    try {
      return localStorage.getItem(LLM_PROVIDER_STORAGE_KEY) || 'local';
    } catch {
      return 'local';
    }
  });

  const [assistantMode, setAssistantMode] = useState<AssistantMode>(() => {
    try {
      const v = localStorage.getItem(ASSISTANT_MODE_STORAGE_KEY);
      if (v === 'chat' || v === 'rag' || v === 'auto') return v;
    } catch {
      // ignore
    }
    return 'auto';
  });

  useEffect(() => {
    try {
      localStorage.setItem(LLM_PROVIDER_STORAGE_KEY, llmProvider);
    } catch {
      // ignore quota / private mode errors
    }
  }, [llmProvider]);

  useEffect(() => {
    try {
      localStorage.setItem(ASSISTANT_MODE_STORAGE_KEY, assistantMode);
    } catch {
      // ignore
    }
  }, [assistantMode]);
  const [activeHighlights, setActiveHighlights] = useState<string[]>([]);
  // 对话框附加的文献（右键节点 / 点击「加入对话」加进来）
  const [attachedPapers, setAttachedPapers] = useState<Paper[]>([]);
  const [readerHighlightText, setReaderHighlightText] = useState<string>('');
  const [readerHighlightPage, setReaderHighlightPage] = useState<number | null>(null);

  const handleAddPaperToChat = (p: Paper) => {
    setAttachedPapers((prev) => {
      if (prev.some((x) => x.id === p.id)) return prev;
      return [...prev, p];
    });
  };

  const handleRemoveAttached = (id: string) => {
    setAttachedPapers((prev) => prev.filter((p) => p.id !== id));
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const htmlPortalTargetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Some logic if needed
  }, []);

  const refreshGraph = async (silent = false) => {
    try {
      const g = await fetchGraph();
      setPapers(g.papers);
      setEdges(g.edges);
      if (!silent) {
        setChat((prev) => [...prev, { role: 'ai', text: '🔄 已刷新星云数据。' }]);
      }
    } catch (error) {
      console.error('Failed to refresh graph:', error);
      setChat((prev) => [...prev, { role: 'ai', text: '⚠️ 未连接到本地后端：请确保 server.py 正在运行 (端口 8000)。' }]);
    }
  };

  useEffect(() => {
    refreshGraph(true);
  }, []);

  useEffect(() => {
    if (activeHighlights.length === 0) return;
    const t = window.setTimeout(() => setActiveHighlights([]), 12000);
    return () => window.clearTimeout(t);
  }, [activeHighlights]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fileName = f.name;

    setLoading(true);
    try {
      // Clear states before upload
      setSelectedPaper(null);
      setSelectedPaperScreenPosition(null);
      setActiveHighlights([]);
      setHighlightedSearchPaperId(null);
      setSearchText('');

      console.log(`[upload] Starting upload of ${fileName}`);
      await uploadPdf(f);
      console.log(`[upload] File uploaded successfully`);

      // Wait longer for backend processing and clustering
      console.log(`[upload] Waiting 2.5 seconds for backend processing...`);
      await new Promise(resolve => setTimeout(resolve, 2500));

      // Refresh from server with retries
      console.log(`[upload] Refreshing graph`);
      let g = null;
      let lastError = null;
      for (let retries = 3; retries > 0; retries--) {
        try {
          g = await fetchGraph();
          break;
        } catch (err) {
          lastError = err;
          if (retries > 1) {
            console.log(`[upload] Graph refresh failed, retrying... (${retries - 1} left)`);
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      
      if (!g) {
        throw lastError || new Error('Failed to refresh graph');
      }
      
      console.log(`[upload] Graph refreshed, papers count: ${g.papers.length}`);
      setPapers(g.papers);
      setEdges(g.edges);
      
      setChat((prev) => [...prev, { role: 'ai', text: `✅ 成功接入文献: ${fileName}。已完成空间映射与关键词连边。` }]);
    } catch (error) {
      console.error('Upload failed:', error);
      setChat((prev) => [...prev, { role: 'ai', text: `❌ 上传失败: ${error instanceof Error ? error.message : '未知错误'}` }]);
    } finally {
      setLoading(false);
      e.target.value = '';
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userVisibleMsg = input;
    setInput('');

    // 有附加文献时，把文献清单前置到实际发给后端的 question 里，方便命中 RAG 与标题匹配
    const attachedSnapshot = attachedPapers;
    const backendMsg = attachedSnapshot.length
      ? (
        '【已选定以下文献作为重点参考】\n'
        + attachedSnapshot
            .map((p, i) => `${i + 1}. 《${p.displayTitle || p.title || p.filename}》(paper_id=${p.id})`)
            .join('\n')
        + '\n\n用户问题：\n'
        + userVisibleMsg
      )
      : userVisibleMsg;

    // 聊天气泡里只展示用户原话 + 附加文献 chip，不展示 prompt 前缀
    setChat((prev) => [
      ...prev,
      {
        role: 'user',
        text: userVisibleMsg,
        cites: attachedSnapshot.length ? attachedSnapshot.map((p) => p.id) : undefined,
      },
    ]);
    setActiveHighlights([]);
    setAttachedPapers([]);
    setLoading(true);

    try {
      // 取最近 6 条（3 轮）作为多轮对话历史；过滤掉首条欢迎语与空消息
      const recent = chat
        .filter((c) => c.text && c.role !== 'system')
        .slice(-6)
        .map((c) => ({
          role: (c.role === 'ai' ? 'ai' : 'user') as 'user' | 'ai',
          text: c.text,
        }));
      // 有附加文献时，自动走 rag 模式，确保优先基于这些文献作答
      const effectiveMode = attachedSnapshot.length ? 'rag' : assistantMode;
      const res = await queryWithProvider(backendMsg, llmProvider, recent, effectiveMode);
      
      const citeDetails = (res.cite_details || []) as CiteDetail[];
      const answerRaw = res.answer || 'AI 未返回内容';
      const answer = normalizeAnswerText(answerRaw, citeDetails);
      // 兼容两种引用格式：后端返回的 cites 数组 或 文本中的 [CITE:id] 标记
      const cites = (
        res.cites?.map((c) => (typeof c === 'string' ? c : c?.paper_id || '')) ||
        answer.match(/\[CITE:(\w+)\]/g)?.map((c) => c.replace('[CITE:', '').replace(']', '')) ||
        []
      ).filter(Boolean);
        
      const pu = res.provider_used || '';
      const providerTag =
        pu === 'system'
          ? '（系统）'
          : pu === 'gemini'
          ? '（云端 Gemini）'
          : pu === 'local'
          ? '（本地 Ollama）'
          : pu
          ? `（${pu}）`
          : '';
      setChat((prev) => [
        ...prev,
        {
          role: 'ai',
          text: `${answer}\n\n${providerTag}`,
          cites,
          citeDetails,
        },
      ]);
      setActiveHighlights(cites);
    } catch (err) {
      console.error(err);
      setChat((prev) => [...prev, { role: 'ai', text: '❌ 检索失败：请检查后端连接或 API Key 配置。' }]);
    } finally {
      setLoading(false);
    }
  };

  // search results by filename (case-insensitive)
  const searchResults = (() => {
    const q = searchText.trim().toLowerCase();
    if (!q) return [];
    return papers
      .filter((p) => {
        const title = (p.title || '').toLowerCase();
        const display = (p.displayTitle || '').toLowerCase();
        const filename = (p.filename || '').toLowerCase();
        const kws = (p.keywords || []).map((k) => String(k).toLowerCase());

        return (
          title.includes(q) ||
          display.includes(q) ||
          filename.includes(q) ||
          kws.some((k) => k.includes(q))
        );
      })
      .slice(0, 20);
  })();

  const handleSearchSelect = (p: Paper) => {
    setSelectedPaper(p);
    setSelectedPaperScreenPosition({
      x: Math.max(120, window.innerWidth * 0.42),
      y: Math.max(90, window.innerHeight * 0.25),
    });
    setHighlightedSearchPaperId(p.id);
    // clear search input to hide dropdown
    setSearchText('');
    // focusTarget is selectedPaper which will be passed into GalaxyArea
  };

  const handleDeleteSelectedPaper = async (
    paper: Paper,
    options: { purgeSource?: boolean } = {},
  ) => {
    const purgeSource = options.purgeSource ?? true;
    const ok = window.confirm(
      purgeSource
        ? `确认彻底删除文献「${paper.displayTitle || paper.title}」吗？\n\n这会删除知识库记录 + inbox 源文件，重启后不会回流。`
        : `确认仅从知识库移除「${paper.displayTitle || paper.title}」吗？\n\n将保留 inbox 源文件，之后可再次导入。`,
    );
    if (!ok) return;

    const originalPapers = [...papers];
    const originalEdges = [...edges];

    try {
      // Optimistically update UI first
      setSelectedPaper(null);
      setSelectedPaperScreenPosition(null);
      setActiveHighlights((prev) => prev.filter((id) => id !== paper.id));
      setHighlightedSearchPaperId(null);
      setSearchText('');

      // Remove from local state immediately
      setPapers(prev => prev.filter(p => p.id !== paper.id));
      setEdges(prev => prev.filter(e => e.source !== paper.id && e.target !== paper.id));

      console.log(`[delete] Starting delete of ${paper.id}`);
      await deletePaper(paper.id, { purgeSource });
      console.log(`[delete] Delete request completed`);

      // Wait for backend processing
      console.log(`[delete] Waiting 1 second for backend processing...`);
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Refresh from server with retries
      console.log(`[delete] Refreshing graph`);
      let g = null;
      let lastError = null;
      for (let retries = 3; retries > 0; retries--) {
        try {
          g = await fetchGraph();
          break;
        } catch (err) {
          lastError = err;
          if (retries > 1) {
            console.log(`[delete] Graph refresh failed, retrying... (${retries - 1} left)`);
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
      }
      
      if (!g) {
        throw lastError || new Error('Failed to refresh graph');
      }
      
      console.log(`[delete] Graph refreshed, papers count: ${g.papers.length}`);
      setPapers(g.papers);
      setEdges(g.edges);
      
      setChat((prev) => [
        ...prev,
        {
          role: 'ai',
          text: purgeSource
            ? `🗑️ 已彻底删除文献：${paper.displayTitle || paper.title}（含源文件）`
            : `🗑️ 已从知识库移除：${paper.displayTitle || paper.title}（保留源文件）`,
        },
      ]);
    } catch (error) {
      console.error('Delete failed:', error);
      // Revert optimistic updates on failure
      setPapers(originalPapers);
      setEdges(originalEdges);
      setChat((prev) => [...prev, { role: 'ai', text: `❌ 删除失败: ${error instanceof Error ? error.message : '未知错误'}` }]);
    }
  };

  const openCitation = (hit: CiteDetail) => {
    const cid = hit.paper_id;
    const citedPaper = papers.find((x) => x.id === cid) || null;
    setSelectedPaper(citedPaper);
    setHighlightedSearchPaperId(null);
    setActiveHighlights([cid]);
    if (citedPaper) {
      setSelectedPaperScreenPosition({
        x: Math.max(120, window.innerWidth * 0.42),
        y: Math.max(90, window.innerHeight * 0.25),
      });
      setReaderHighlightText((hit?.snippet || '').trim());
      setReaderHighlightPage((hit?.page && hit.page > 0) ? hit.page : null);
      setReaderPaper(citedPaper);
    } else {
      setSelectedPaperScreenPosition(null);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#020617] text-white overflow-hidden font-sans">
      {/* 左侧功能栏 */}
      <nav className="w-16 border-r border-white/10 flex flex-col items-center py-6 gap-8 glass z-50">
        <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <Zap className="w-5 h-5" />
        </div>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="p-3 hover:bg-white/10 rounded-xl transition-all text-slate-400 hover:text-white group relative"
        >
          <Upload className="w-6 h-6" />
          <span className="absolute left-16 bg-black px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 whitespace-nowrap">
            上传PDF
          </span>
        </button>
        <button
          onClick={() => {
            setSelectedPaper(null);
            setSelectedPaperScreenPosition(null);
            setHighlightedSearchPaperId(null);
            setActiveHighlights([]);
            setSearchText('');
            setChat((prev) => [...prev, { role: 'ai', text: '🧭 已回到全景视图（清空选中与高亮）。' }]);
          }}
          className="p-2 rounded-xl text-slate-500 hover:text-white hover:bg-white/10"
          title="回到全景视图"
        >
          <Compass className="w-6 h-6" />
        </button>
        <button
          onClick={() => refreshGraph(false)}
          className="p-2 rounded-xl text-slate-500 hover:text-white hover:bg-white/10"
          title="刷新星云数据"
        >
          <Activity className="w-6 h-6" />
        </button>
        <input type="file" ref={fileInputRef} onChange={handleUpload} className="hidden" accept=".pdf" />
      </nav>

      {/* 3D 渲染主区（拆分到 GalaxyArea） */}
      <GalaxyArea
        papers={papers}
        edges={edges}
        onSelect={(p, screenPos) => {
          setSelectedPaper(p);
          setSelectedPaperScreenPosition(screenPos);
          setHighlightedSearchPaperId(null);
        }}
        onRequestAddToChat={handleAddPaperToChat}
        highlights={activeHighlights}
        hideLabels={!!readerPaper}
        searchText={searchText}
        setSearchText={setSearchText}
        results={searchResults}
        onResultClick={handleSearchSelect}
        focusTarget={selectedPaper}
        highlightedSearchPaperId={highlightedSearchPaperId}
        htmlPortalTargetRef={htmlPortalTargetRef}
      />

      {/* 悬浮详情浮窗（抽离为 PaperDetail） */}
      {selectedPaper && selectedPaperScreenPosition && (
        <PaperDetail
          selectedPaper={selectedPaper}
          edges={edges}
          onClose={() => {
            setSelectedPaper(null);
            setSelectedPaperScreenPosition(null);
          }}
          onOpenReader={(p) => {
            setReaderHighlightText('');
            setReaderHighlightPage(null);
            setReaderPaper(p);
          }}
          onDelete={handleDeleteSelectedPaper}
          onAddToChat={handleAddPaperToChat}
          onReprocessed={async (paperId) => {
            try {
              const g = await fetchGraph();
              setPapers(g.papers);
              setEdges(g.edges);
              const refreshed = g.papers.find((p) => p.id === paperId) || null;
              if (refreshed) setSelectedPaper(refreshed);
            } catch (e) {
              console.warn('[reprocess] refresh graph failed', e);
            }
          }}
          onSummaryUpdated={(paperId, summary) => {
            setPapers((prev) =>
              prev.map((p) => (p.id === paperId ? { ...p, llmSummary: summary } : p))
            );
            setSelectedPaper((prev) =>
              prev && prev.id === paperId ? { ...prev, llmSummary: summary } : prev
            );
          }}
          screenPosition={selectedPaperScreenPosition}
          aiChatWidth={420}
          llmProvider={llmProvider}
        />
      )}

      <div ref={htmlPortalTargetRef} id="html-portal-target" className="absolute inset-0 pointer-events-none z-0"></div>

      {/* 阅读器弹层（抽离为 ReaderModal） */}
      {readerPaper && (
        <ReaderModal
          readerPaper={readerPaper}
          onClose={() => {
            setReaderPaper(null);
            setReaderHighlightText('');
            setReaderHighlightPage(null);
          }}
          llmProvider={llmProvider}
          onProviderChange={setLlmProvider}
          initialHighlightText={readerHighlightText}
          initialHighlightPage={readerHighlightPage}
        />
      )}

      {/* 右侧 AI 终端 */}
      <aside className="w-[420px] border-l border-white/10 glass flex flex-col z-50">
        <div className="p-6 border-b border-white/10 bg-white/5 flex items-center justify-between">
          <h2 className="flex items-center gap-2 font-bold tracking-tight">
            <Sparkles className="w-4 h-4 text-yellow-400" /> 本地检索终端
          </h2>
          <div className="flex items-center gap-2">
            <IndexAdmin />
            <ModelPicker
              currentProviderId={llmProvider}
              onProviderChange={setLlmProvider}
            />
          </div>
        </div>

        {/* 模式切换条 */}
        <div className="px-6 pt-3 pb-2 border-b border-white/10 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1 p-1 rounded-full bg-white/5 border border-white/10">
            {([
              { id: 'auto', label: '自适应', icon: Sparkles, hint: '根据提问自动判断是否检索文献' },
              { id: 'rag', label: '文献问答', icon: BookOpen, hint: '强制基于你的文献库作答并溯源' },
              { id: 'chat', label: '通用对话', icon: MessageSquare, hint: '纯大模型对话，不检索不引用' },
            ] as const).map((m) => {
              const active = assistantMode === m.id;
              const Icon = m.icon;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setAssistantMode(m.id)}
                  title={m.hint}
                  className={`flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-full transition-all
                    ${active
                      ? 'bg-indigo-500/25 text-indigo-100 ring-1 ring-indigo-400/40 shadow-sm shadow-indigo-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'}`}
                >
                  <Icon className="w-3 h-3" />
                  {m.label}
                </button>
              );
            })}
          </div>
          <div
            className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
              assistantMode === 'chat'
                ? 'bg-slate-500/10 text-slate-400 border-slate-500/30'
                : 'bg-green-500/15 text-green-400 border-green-500/30'
            }`}
          >
            {assistantMode === 'chat' ? 'CHAT ONLY' : 'RAG ACTIVE'}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          {chat.map((m, i) => {
            const details = (m.citeDetails || []).filter((d) => !!d?.paper_id);
            const cards = details.length
              ? details.slice(0, 6)
              : (m.cites || []).slice(0, 6).map((pid) => ({ paper_id: pid } as CiteDetail));
            const sourceSectionId = `chat-source-${i}`;
            const aiBlocks = m.role === 'ai'
              ? (m.text || '')
                  .split(/\n{2,}/)
                  .map((s) => s.trim())
                  .filter(Boolean)
              : [];

            return (
            <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[90%] p-4 rounded-2xl text-sm leading-relaxed break-words ${
                  m.role === 'user' ? 'bg-indigo-600 text-white whitespace-pre-wrap' : 'bg-white/5 border border-white/10'
                }`}
              >
                {m.role === 'user' ? (
                  m.text
                ) : (
                  (m.cites && m.cites.length > 0 && aiBlocks.length > 0) ? (
                    <div className="space-y-3">
                      {aiBlocks.map((block, blockIdx) => {
                        const refIndexes = citationIndexesForBlock(block, cards, blockIdx % Math.max(cards.length, 1));
                        return (
                          <div key={`${i}-block-${blockIdx}`} className="leading-relaxed">
                            <MarkdownMessage text={block} />
                            {refIndexes.length > 0 ? (
                              <span className="mt-1 inline-flex items-center gap-1">
                                {refIndexes.map((refIdx) => {
                                  const hit = cards[refIdx];
                                  const snippet = (hit?.snippet || '').trim();
                                  return (
                                    <button
                                      key={`${i}-block-${blockIdx}-ref-${refIdx}`}
                                      type="button"
                                      onClick={() => openCitation(hit)}
                                      className="inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-slate-500/35 bg-slate-700/35 px-1.5 text-[10px] text-slate-200 hover:bg-slate-600/50 transition-colors"
                                      title={snippet || '点击定位到引用原文'}
                                    >
                                      <span className="font-mono">{refIdx + 1}</span>
                                    </button>
                                  );
                                })}
                                <button
                                  type="button"
                                  onClick={() => {
                                    const el = document.getElementById(sourceSectionId);
                                    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                                  }}
                                  className="text-[10px] text-slate-500 hover:text-slate-300"
                                  title="查看总来源"
                                >
                                  ...
                                </button>
                              </span>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <MarkdownMessage text={m.text} />
                  )
                )}
                {(m.cites && m.cites.length > 0) && (
                  <div id={sourceSectionId} className="mt-4 pt-3 border-t border-white/10">
                    <div className="text-[11px] text-slate-400 mb-2">总来源（点击跳转到原文）</div>
                    <div className="flex flex-col gap-2">
                      {cards.map((hit, idx) => {
                          const cid = hit.paper_id;
                          const citedPaper = papers.find((x) => x.id === cid) || null;
                          const citeColor = citedPaper?.color || '#22d3ee';
                          const title =
                            citedPaper?.displayTitle
                            || citedPaper?.title
                            || citedPaper?.filename
                            || `已删除文献 (${cid.slice(0, 8)})`;
                          const snippet = (hit?.snippet || '').trim();
                          const pageText = hit?.page && hit.page > 0 ? `第 ${hit.page} 页` : '';

                          return (
                            <button
                              key={`${cid}-${hit.chunk_id || idx}`}
                              onClick={() => openCitation(hit)}
                              className="text-left text-[11px] p-2 rounded-lg border transition-all hover:brightness-110"
                              style={{
                                color: citeColor,
                                borderColor: `${citeColor}66`,
                                backgroundColor: `${citeColor}14`,
                                boxShadow: `0 0 8px ${citeColor}22`,
                              }}
                              title={snippet ? `点击跳转：${snippet}` : '点击跳转到文献'}
                            >
                              <div className="flex items-center gap-2">
                                <span className="font-mono opacity-90">[{idx + 1}]</span>
                                <span className="truncate font-semibold">📄 {title}</span>
                                {pageText ? <span className="text-[10px] opacity-70 shrink-0">{pageText}</span> : null}
                              </div>
                              {snippet ? (
                                <div className="mt-1 text-[10px] opacity-85 line-clamp-2">
                                  {snippet}
                                </div>
                              ) : null}
                            </button>
                          );
                        })}
                    </div>
                  </div>
                )}
              </div>
            </div>
            );
          })}
          {loading && (
            <div className="flex gap-2 p-4 bg-white/5 rounded-2xl w-fit animate-pulse">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.2s]" />
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.4s]" />
            </div>
          )}
        </div>

        <div className="p-4 bg-black/40 border-t border-white/10">
          {attachedPapers.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {attachedPapers.map((p) => (
                <span
                  key={p.id}
                  className="group/chip inline-flex items-center gap-1 max-w-full pl-2 pr-1 py-1 rounded-lg text-[11px] border"
                  style={{
                    color: p.color || '#c7d2fe',
                    borderColor: `${p.color || '#6366f1'}55`,
                    backgroundColor: `${p.color || '#6366f1'}18`,
                  }}
                  title={p.displayTitle || p.title || p.filename}
                >
                  <Paperclip className="w-3 h-3 opacity-80" />
                  <span className="truncate max-w-[220px]">{p.displayTitle || p.title || p.filename}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveAttached(p.id)}
                    className="ml-0.5 p-0.5 rounded hover:bg-white/15 transition-colors"
                    title="移除"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
              <button
                type="button"
                onClick={() => setAttachedPapers([])}
                className="text-[10px] text-slate-500 hover:text-slate-300 px-1.5"
              >
                清空
              </button>
            </div>
          )}
          <div className="relative group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
              placeholder={
                attachedPapers.length
                  ? `针对附加的 ${attachedPapers.length} 篇文献提问…`
                  : '提问，AI将检索整个星系的知识...（在星图中右键/Ctrl+Click 节点可添加到对话）'
              }
              className="w-full bg-white/5 border border-white/10 rounded-2xl p-4 pr-14 text-sm outline-none focus:border-indigo-500/50 min-h-[100px] resize-none transition-all"
            />
            <button
              onClick={handleSend}
              disabled={loading}
              className="absolute bottom-4 right-4 p-2 bg-indigo-600 rounded-xl hover:bg-indigo-500 shadow-lg shadow-indigo-600/20 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
          <p className="text-[10px] text-slate-500 mt-3 text-center opacity-50 uppercase tracking-widest">ScholarAI Neural Core v4.0.2</p>
        </div>
      </aside>
    </div>
  );
}