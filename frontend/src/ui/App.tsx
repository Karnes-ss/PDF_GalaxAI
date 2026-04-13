import { useState, useEffect, useRef } from 'react';
import { Activity, BookOpen, Compass, FileText, Search, Send, Sparkles, Upload, X, Zap } from 'lucide-react';
import { deletePaper, fetchGraph, fileUrl, queryWithProvider, uploadPdf } from '../api/client';
import GalaxyArea from './GalaxyArea';
import PaperDetail from './PaperDetail';
import ReaderModal from './ReaderModal';
import type { Edge, Paper } from '../types/scholar';

export function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [readerPaper, setReaderPaper] = useState<Paper | null>(null);
  const [searchText, setSearchText] = useState('');
  const [highlightedSearchPaperId, setHighlightedSearchPaperId] = useState<string | null>(null);
  const [selectedPaperScreenPosition, setSelectedPaperScreenPosition] = useState<{ x: number; y: number } | null>(null);
  const [chat, setChat] = useState<{ role: string; text: string; cites?: string[] }[]>([
    { role: 'ai', text: '欢迎进入本地 Scholar 星系。请上传 PDF 文献以生成你的专属知识星云。' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [llmProvider, setLlmProvider] = useState<'local' | 'gemini'>('local');
  const [activeHighlights, setActiveHighlights] = useState<string[]>([]);
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
    const msg = input;
    setInput('');
    
    // 立即显示用户消息
    setChat((prev) => [...prev, { role: 'user', text: msg }]);
    setActiveHighlights([]);
    setLoading(true);

    try {
      const res = await queryWithProvider(msg, llmProvider);
      
      const answer = res.answer || 'AI 未返回内容';
      // 兼容两种引用格式：后端返回的 cites 数组 或 文本中的 [CITE:id] 标记
      const cites = (
        res.cites?.map((c) => (typeof c === 'string' ? c : c?.paper_id || '')) ||
        answer.match(/\[CITE:(\w+)\]/g)?.map((c) => c.replace('[CITE:', '').replace(']', '')) ||
        []
      ).filter(Boolean);
        
      const providerTag = res.provider_used === 'gemini' ? '（云端 Gemini）' : '（本地 Ollama）';
      setChat((prev) => [...prev, { role: 'ai', text: `${answer}\n\n${providerTag}`, cites }]);
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

  const handleDeleteSelectedPaper = async (paper: Paper) => {
    const ok = window.confirm(`确认删除文献「${paper.displayTitle || paper.title}」吗？此操作不可撤销。`);
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
      await deletePaper(paper.id);
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
      
      setChat((prev) => [...prev, { role: 'ai', text: `🗑️ 已删除文献：${paper.displayTitle || paper.title}` }]);
    } catch (error) {
      console.error('Delete failed:', error);
      // Revert optimistic updates on failure
      setPapers(originalPapers);
      setEdges(originalEdges);
      setChat((prev) => [...prev, { role: 'ai', text: `❌ 删除失败: ${error instanceof Error ? error.message : '未知错误'}` }]);
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
          onOpenReader={(p) => setReaderPaper(p)}
          onDelete={handleDeleteSelectedPaper}
          screenPosition={selectedPaperScreenPosition}
          aiChatWidth={420}
        />
      )}

      <div ref={htmlPortalTargetRef} id="html-portal-target" className="absolute inset-0 pointer-events-none z-0"></div>

      {/* 阅读器弹层（抽离为 ReaderModal） */}
      {readerPaper && <ReaderModal readerPaper={readerPaper} onClose={() => setReaderPaper(null)} />}

      {/* 右侧 AI 终端 */}
      <aside className="w-[420px] border-l border-white/10 glass flex flex-col z-50">
        <div className="p-6 border-b border-white/10 bg-white/5 flex items-center justify-between">
          <h2 className="flex items-center gap-2 font-bold tracking-tight">
            <Sparkles className="w-4 h-4 text-yellow-400" /> 本地检索终端
          </h2>
          <div className="flex items-center gap-2">
            <div className="flex items-center rounded-md border border-white/20 bg-white/5 p-0.5">
              <button
                onClick={() => setLlmProvider('local')}
                className={`text-[11px] px-2 py-1 rounded transition-all ${
                  llmProvider === 'local'
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-300 hover:bg-white/10'
                }`}
              >
                本地 Ollama
              </button>
              <button
                onClick={() => setLlmProvider('gemini')}
                className={`text-[11px] px-2 py-1 rounded transition-all ${
                  llmProvider === 'gemini'
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-300 hover:bg-white/10'
                }`}
              >
                云端 Gemini
              </button>
            </div>
            <div className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full border border-green-500/30">
              RAG ACTIVE
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          {chat.map((m, i) => (
            <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[90%] p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words break-all ${
                  m.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white/5 border border-white/10'
                }`}
              >
                {m.text}
                {m.cites && m.cites.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-white/10 flex flex-wrap gap-2">
                    {m.cites.map((cid) => (
                      <button
                        key={cid}
                        onClick={() => {
                          const p = papers.find((x) => x.id === cid) || null;
                          setSelectedPaper(p);
                          setHighlightedSearchPaperId(null);
                          setActiveHighlights([cid]);
                          if (p) {
                            setSelectedPaperScreenPosition({
                              x: Math.max(120, window.innerWidth * 0.42),
                              y: Math.max(90, window.innerHeight * 0.25),
                            });
                          } else {
                            setSelectedPaperScreenPosition(null);
                          }
                        }}
                        className="text-[10px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/30 px-2 py-0.5 rounded hover:bg-yellow-500/20 transition-all"
                      >
                        证据文献 #{cid.slice(0, 8)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-2 p-4 bg-white/5 rounded-2xl w-fit animate-pulse">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.2s]" />
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.4s]" />
            </div>
          )}
        </div>

        <div className="p-4 bg-black/40 border-t border-white/10">
          <div className="relative group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
              placeholder="提问，AI将检索整个星系的知识..."
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