import React from 'react';
import { BookOpen, FileText, X, Search } from 'lucide-react';
import type { Edge, Paper } from '../types/scholar';

interface Props {
  selectedPaper: Paper;
  edges: Edge[];
  onClose: () => void;
  onOpenReader: (p: Paper) => void;
}

export default function PaperDetail({ selectedPaper, edges, onClose, onOpenReader }: Props) {
  const linked = edges.filter((e) => e.source === selectedPaper.id || e.target === selectedPaper.id);
  const linkCount = linked.length;

  return (
    <div
      className="absolute bottom-10 left-10 w-[420px] glass rounded-2xl p-6 border-l-4 animate-in fade-in slide-in-from-left-4 z-40 flex flex-col gap-4 shadow-2xl shadow-black/50"
      style={{ borderColor: selectedPaper.color }}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1 mr-4">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider text-black/80"
              style={{ backgroundColor: selectedPaper.color }}
            >
              {selectedPaper.field}
            </span>
            <span className="text-[10px] text-slate-500 font-mono">CONF: {(selectedPaper.confidence * 100).toFixed(0)}%</span>
          </div>
          <h3 className="font-bold text-lg leading-tight text-white/90" title={selectedPaper.title}>
            {selectedPaper.displayTitle}
          </h3>
        </div>
        <button onClick={onClose} className="shrink-0 p-1 hover:bg-white/10 rounded-lg transition-colors">
          <X className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      <div className="bg-black/20 rounded-xl p-3 max-h-[120px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
        <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{selectedPaper.abstract || selectedPaper.firstSentence || '暂无摘要内容...'}</p>
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

        <button
          onClick={() => onOpenReader(selectedPaper)}
          className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-4 py-2 rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-indigo-600/20 active:scale-95"
        >
          <Search className="w-3 h-3" />
          深度阅读
        </button>
      </div>
    </div>
  );
}
