import React, { useEffect, useRef, useState } from 'react';
import { Activity, BookOpen, Compass, FileText, Search, Send, Sparkles, Upload, X, Zap } from 'lucide-react';
import { fetchGraph, fileUrl, queryLocal, uploadPdf } from '../api/client';
import { GalaxyRenderer } from '../rendering/GalaxyRenderer';
import type { Edge, Paper } from '../types/scholar';

export function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [readerPaper, setReaderPaper] = useState<Paper | null>(null);
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
        setChat((prev) => [...prev, { role: 'ai', text: '未连接到本地后端：请先启动 backend（FastAPI/uvicorn）。' }]);
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
      setChat((prev) => [...prev, { role: 'ai', text: `成功接入文献: ${fileName}。已完成空间映射与关键词连边。` }]);
    } catch {
      setChat((prev) => [...prev, { role: 'ai', text: '上传失败：请检查 backend 是否启动，以及是否安装了依赖。' }]);
    } finally {
      setLoading(false);
      e.target.value = '';
    }
  };

  const handleSend = async () => {
    if (!input) return;
    const msg = input;
    setInput('');
    setChat((prev) => [...prev, { role: 'user', text: msg }]);
    setLoading(true);

    try {
      const res = await queryLocal(msg);
      const cites =
        res.cites ||
        res.answer?.match(/\[CITE:(\w+)\]/g)?.map((c) => c.replace('[CITE:', '').replace(']', '')) ||
        [];
      setChat((prev) => [...prev, { role: 'ai', text: res.answer || '', cites }]);
    } catch {
      setChat((prev) => [...prev, { role: 'ai', text: '检索失败：请检查 backend 是否启动。' }]);
    } finally {
      setLoading(false);
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
        <Compass className="w-6 h-6 text-slate-500 hover:text-white cursor-pointer" />
        <Activity className="w-6 h-6 text-slate-500 hover:text-white cursor-pointer" />
        <input type="file" ref={fileInputRef} onChange={handleUpload} className="hidden" accept=".pdf" />
      </nav>

      {/* 3D 渲染主区 */}
      <main className="flex-1 relative">
        <div className="absolute top-6 left-6 z-10 flex gap-4">
          <div className="glass px-4 py-2 rounded-full border border-white/10 flex items-center gap-3">
            <Search className="w-4 h-4 text-slate-400" />
            <input className="bg-transparent border-none outline-none text-sm w-48" placeholder="在星云中搜索..." />
          </div>
        </div>

        <GalaxyRenderer
          papers={papers}
          edges={edges}
          onSelect={(p) => setSelectedPaper(p)}
          highlights={chat[chat.length - 1]?.cites || []}
          hideLabels={!!readerPaper}
        />

        {/* 悬浮预览窗口 */}
        {selectedPaper && (() => {
          const linked = edges.filter((e) => e.source === selectedPaper.id || e.target === selectedPaper.id);
          const linkCount = linked.length;
          const maxW = linked.reduce((m, e) => Math.max(m, Number(e.weight) || 0), 0);
          const relatedness = Math.min(100, Math.round((maxW / 8) * 100));

          return (
            <div className="absolute bottom-10 left-10 w-96 glass rounded-2xl p-6 border-l-4 animate-in fade-in slide-in-from-left-4 z-40" style={{ borderColor: selectedPaper.color }}>
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1 mr-2 overflow-hidden">
                  <h3 className="font-bold text-lg leading-tight truncate" title={selectedPaper.title || selectedPaper.displayTitle}>
                    {selectedPaper.title || selectedPaper.displayTitle}
                  </h3>
                  {selectedPaper.firstSentence && (
                    <p className="text-xs text-slate-400 mt-2 line-clamp-3 italic">
                      "{selectedPaper.firstSentence}"
                    </p>
                  )}
                </div>
                <button onClick={() => setSelectedPaper(null)} className="shrink-0">
                  <X className="w-5 h-5 text-slate-500" />
                </button>
              </div>
              <div className="flex gap-4 mb-6">
                <div className="text-xs text-slate-400 flex items-center gap-1">
                  <BookOpen className="w-3 h-3" /> 连线: {linkCount}
                </div>
                <div className="text-xs text-slate-400 flex items-center gap-1">
                  <Activity className="w-3 h-3" /> 关联度: {relatedness}%
                </div>
              </div>
              <button
                onClick={() => setReaderPaper(selectedPaper)}
                className="w-full bg-blue-600 py-2 rounded-xl text-sm font-bold flex items-center justify-center gap-2 hover:bg-blue-500 transition-colors"
              >
                <FileText className="w-4 h-4" /> 进入上帝视角阅读
              </button>
            </div>
          );
        })()}

        {readerPaper && (
          <div className="absolute inset-0 z-[60] bg-black/70 p-6">
            <div className="w-full h-full glass border border-white/10 rounded-2xl overflow-hidden flex flex-col">
              <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between">
                <div className="font-semibold text-sm truncate">{readerPaper.displayTitle || readerPaper.title}</div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => window.open(fileUrl(readerPaper.id), '_blank', 'noopener,noreferrer')}
                    className="text-xs px-3 py-1 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-all"
                  >
                    新窗口打开
                  </button>
                  <button
                    onClick={() => setReaderPaper(null)}
                    className="text-xs px-3 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition-all"
                  >
                    关闭
                  </button>
                </div>
              </div>
              <iframe title="pdf" src={fileUrl(readerPaper.id)} className="w-full flex-1" />
            </div>
          </div>
        )}
      </main>

      {/* 右侧 AI 终端 */}
      <aside className="w-[420px] border-l border-white/10 glass flex flex-col z-50">
        <div className="p-6 border-b border-white/10 bg-white/5 flex items-center justify-between">
          <h2 className="flex items-center gap-2 font-bold tracking-tight">
            <Sparkles className="w-4 h-4 text-yellow-400" /> 本地检索终端
          </h2>
          <div className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full border border-green-500/30">
          RAG 
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          {chat.map((m, i) => (
            <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[90%] p-4 rounded-2xl text-sm leading-relaxed ${
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
                        证据文献 #{cid}
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
              className="absolute bottom-4 right-4 p-2 bg-indigo-600 rounded-xl hover:bg-indigo-500 shadow-lg shadow-indigo-600/20 transition-all active:scale-95"
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
