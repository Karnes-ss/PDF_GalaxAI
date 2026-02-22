import React from 'react';
import { X } from 'lucide-react';
import { fileUrl } from '../api/client';
import type { Paper } from '../types/scholar';

interface Props {
  readerPaper: Paper;
  onClose: () => void;
}

export default function ReaderModal({ readerPaper, onClose }: Props) {
  return (
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
            <button onClick={onClose} className="text-xs px-3 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition-all">
              关闭
            </button>
          </div>
        </div>
        <iframe title="pdf" src={fileUrl(readerPaper.id)} className="w-full flex-1" />
      </div>
    </div>
  );
}
