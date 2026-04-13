import React, { useState, useEffect } from 'react';
import { BookOpen, FileText, X, Search } from 'lucide-react';
import type { Edge, Paper } from '../types/scholar';

interface Props {
  selectedPaper: Paper;
  edges: Edge[];
  onClose: () => void;
  onOpenReader: (p: Paper) => void;
  onDelete: (p: Paper) => void;
  screenPosition: { x: number; y: number };
  aiChatWidth: number;
}

export default function PaperDetail({ selectedPaper, edges, onClose, onOpenReader, onDelete, screenPosition, aiChatWidth }: Props) {
  const linked = edges.filter((e) => e.source === selectedPaper.id || e.target === selectedPaper.id);
  const linkCount = linked.length;

  const [panelStyle, setPanelStyle] = useState({});
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [startMousePos, setStartMousePos] = useState({ x: 0, y: 0 });

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

      <div className="pt-2 border-t border-white/10 flex justify-end">
        <button
          onClick={() => onDelete(selectedPaper)}
          className="text-xs px-3 py-1.5 rounded-md border border-red-500/40 text-red-300 hover:bg-red-500/10 transition-colors"
        >
          删除该文献
        </button>
      </div>
    </div>
  );
}
