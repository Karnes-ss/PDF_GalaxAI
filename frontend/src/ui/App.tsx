import React, { useEffect, useRef, useState } from 'react';
import { Activity, BookOpen, Compass, FileText, Search, Send, Sparkles, Upload, X, Zap } from 'lucide-react';
import { fetchGraph, fileUrl, queryLocal, uploadPdf } from '../api/client';
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
  const [chat, setChat] = useState<{ role: string; text: string; cites?: string[] }[]>([
    { role: 'ai', text: '欢迎进入本地 Scholar 星系。请上传 PDF 文献以生成你的专属知识星云。' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const g = await fetchGraph();
        setPapers(g.papers);
        setEdges(g.edges);
      } catch {
        setChat((prev) => [...prev, { role: 'ai', text: '⚠️ 未连接到本地后端：请确保 server.py 正在运行 (端口 8000)。' }]);
      }
    })();
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fileName = f.name;
    setLoading(true);
    try {
      await uploadPdf(f);
      const g = await fetchGraph();
      setPapers(g.papers);
      setEdges(g.edges);
      setChat((prev) => [...prev, { role: 'ai', text: `✅ 成功接入文献: ${fileName}。已完成空间映射与关键词连边。` }]);
    } catch {
      setChat((prev) => [...prev, { role: 'ai', text: '❌ 上传失败：请检查后端日志。' }]);
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
    setLoading(true);

    try {
      // 调用 client.ts 中的 queryLocal (它会请求后端 api.py)
      const res = await queryLocal(msg);
      
      const answer = res.answer || 'AI 未返回内容';
      // 兼容两种引用格式：后端返回的 cites 数组 或 文本中的 [CITE:id] 标记
      const cites =
        res.cites ||
        answer.match(/\[CITE:(\w+)\]/g)?.map((c) => c.replace('[CITE:', '').replace(']', '')) ||
        [];
        
      setChat((prev) => [...prev, { role: 'ai', text: answer, cites }]);
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
    // clear search input to hide dropdown
    setSearchText('');
    // focusTarget is selectedPaper which will be passed into GalaxyArea
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
        <Compass className="w-6 h-6 text-slate-500 hover:text-white cursor-pointer" />
        <Activity className="w-6 h-6 text-slate-500 hover:text-white cursor-pointer" />
        <input type="file" ref={fileInputRef} onChange={handleUpload} className="hidden" accept=".pdf" />
      </nav>

      {/* 3D 渲染主区（拆分到 GalaxyArea） */}
      <GalaxyArea
        papers={papers}
        edges={edges}
        onSelect={(p) => setSelectedPaper(p)}
        highlights={chat[chat.length - 1]?.cites || []}
        hideLabels={!!readerPaper}
        searchText={searchText}
        setSearchText={setSearchText}
        results={searchResults}
        onResultClick={handleSearchSelect}
        focusTarget={selectedPaper}
      />

      {/* 悬浮详情浮窗（抽离为 PaperDetail） */}
      {selectedPaper && (
        <PaperDetail
          selectedPaper={selectedPaper}
          edges={edges}
          onClose={() => setSelectedPaper(null)}
          onOpenReader={(p) => setReaderPaper(p)}
        />
      )}

      {/* 阅读器弹层（抽离为 ReaderModal） */}
      {readerPaper && <ReaderModal readerPaper={readerPaper} onClose={() => setReaderPaper(null)} />}

      {/* 右侧 AI 终端 */}
      <aside className="w-[420px] border-l border-white/10 glass flex flex-col z-50">
        <div className="p-6 border-b border-white/10 bg-white/5 flex items-center justify-between">
          <h2 className="flex items-center gap-2 font-bold tracking-tight">
            <Sparkles className="w-4 h-4 text-yellow-400" /> 本地检索终端
          </h2>
          <div className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full border border-green-500/30">
            RAG ACTIVE
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          {chat.map((m, i) => (
            <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[90%] p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                  m.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white/5 border border-white/10'
                }`}
              >
                {m.text}
                {m.cites && m.cites.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-white/10 flex flex-wrap gap-2">
                    {m.cites.map((cid) => (
                      <button
                        key={cid}
                        onClick={() => setSelectedPaper(papers.find((p) => p.id === cid) || null)}
                        className="text-[10px] bg-yellow-500/10 text-yellow-500 border border-yellow-500/30 px-2 py-0.5 rounded hover:bg-yellow-500/20 transition-all"
                      >
                        证据文献 #{cid.slice(0, 4)}
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