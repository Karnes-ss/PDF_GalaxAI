import React from 'react';
import { Search } from 'lucide-react';
import { GalaxyRenderer } from '../rendering/GalaxyRenderer';
import type { Edge, Paper } from '../types/scholar';

interface Props {
  papers: Paper[];
  edges: Edge[];
  onSelect: (p: Paper) => void;
  highlights?: string[];
  hideLabels?: boolean;
  searchText: string;
  setSearchText: (s: string) => void;
  results: Paper[];
  onResultClick: (p: Paper) => void;
  focusTarget?: Paper | null;
}

export default function GalaxyArea({ papers, edges, onSelect, highlights, hideLabels, searchText, setSearchText, results, onResultClick, focusTarget }: Props) {
  return (
    <main className="flex-1 relative">
      <div className="absolute top-6 left-6 z-10 flex gap-4">
        <div className="glass px-4 py-2 rounded-full border border-white/10 flex items-center gap-3 relative">
          <Search className="w-4 h-4 text-slate-400" />
          <input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="bg-transparent border-none outline-none text-sm w-48"
            placeholder="在星云中搜索... (按文件名)"
          />

          {/* 简单下拉结果 */}
          {searchText.trim().length > 0 && (
            <div className="absolute top-12 left-0 bg-black/80 rounded-md border border-white/10 w-64 max-h-64 overflow-y-auto z-50 p-2">
              {results.length === 0 && <div className="text-xs text-slate-400 p-2">无匹配项</div>}
              {results.map((r) => (
                <button
                  key={r.id}
                  onClick={() => onResultClick(r)}
                  className="w-full text-left px-2 py-2 hover:bg-white/5 rounded text-sm"
                >
                  <div className="font-semibold truncate">{r.displayTitle || r.title}</div>
                  <div className="text-[11px] text-slate-400 truncate">{r.filename}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <GalaxyRenderer
        papers={papers}
        edges={edges}
        onSelect={(p) => onSelect(p)}
        highlights={highlights || []}
        hideLabels={!!hideLabels}
        focusTarget={focusTarget}
      />
    </main>
  );
}
