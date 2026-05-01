import { useEffect, useState } from 'react';
import { MessageSquare, PanelRightClose, PanelRightOpen, X } from 'lucide-react';
import { fileUrl } from '../api/client';
import type { Paper } from '../types/scholar';
import PdfReaderPane from './PdfReaderPane';
import ReaderChat from './ReaderChat';

interface Props {
  readerPaper: Paper;
  onClose: () => void;
  llmProvider: string;
  onProviderChange: (id: string) => void;
  initialHighlightText?: string;
  initialHighlightPage?: number | null;
}

const CHAT_OPEN_STORAGE_KEY = 'scholar:reader-chat-open';

export default function ReaderModal({
  readerPaper,
  onClose,
  llmProvider,
  onProviderChange,
  initialHighlightText,
  initialHighlightPage,
}: Props) {
  const [chatOpen, setChatOpen] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem(CHAT_OPEN_STORAGE_KEY);
      return v === null ? true : v === '1';
    } catch {
      return true;
    }
  });

  const toggleChat = () => {
    setChatOpen((v) => {
      const next = !v;
      try {
        localStorage.setItem(CHAT_OPEN_STORAGE_KEY, next ? '1' : '0');
      } catch {
        // ignore
      }
      return next;
    });
  };

  const [highlightText, setHighlightText] = useState<string>(initialHighlightText || '');
  const [highlightPage, setHighlightPage] = useState<number | null>(initialHighlightPage || null);
  const [locateNonce, setLocateNonce] = useState<number>(0);
  useEffect(() => {
    setHighlightText(initialHighlightText || '');
    setHighlightPage(initialHighlightPage || null);
    setLocateNonce((n) => n + 1);
  }, [readerPaper.id, initialHighlightText, initialHighlightPage]);

  return (
    <div className="absolute inset-0 z-[60] bg-black/70 p-6">
      <div className="w-full h-full glass border border-white/10 rounded-2xl overflow-hidden flex flex-col">
        {/* 顶栏 */}
        <div className="px-4 py-3 border-b border-white/10 bg-white/5 flex items-center justify-between gap-3">
          <div className="min-w-0 font-semibold text-sm break-words break-all">
            {readerPaper.displayTitle || readerPaper.title}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {(highlightText || highlightPage) && (
              <div className="max-w-[360px] inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-lg border border-yellow-400/35 bg-yellow-300/10 text-yellow-200">
                <span className="font-semibold shrink-0">定位中</span>
                {highlightPage ? (
                  <span className="shrink-0">第 {highlightPage} 页</span>
                ) : null}
                {highlightText ? (
                  <span className="truncate opacity-85">{highlightText}</span>
                ) : null}
                <button
                  className="ml-1 p-0.5 rounded hover:bg-white/10"
                  title="清除定位"
                  onClick={() => {
                    setHighlightText('');
                    setHighlightPage(null);
                    setLocateNonce((n) => n + 1);
                  }}
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}
            <button
              onClick={toggleChat}
              className={`text-xs px-3 py-1 rounded-lg border transition-all flex items-center gap-1.5 ${
                chatOpen
                  ? 'bg-indigo-500/15 border-indigo-400/40 text-indigo-100'
                  : 'bg-white/5 border-white/10 hover:bg-white/10 text-slate-200'
              }`}
              title={chatOpen ? '收起 AI 对话' : '打开 AI 对话'}
            >
              {chatOpen ? (
                <PanelRightClose className="w-3.5 h-3.5" />
              ) : (
                <PanelRightOpen className="w-3.5 h-3.5" />
              )}
              <MessageSquare className="w-3 h-3" />
              AI 对话
            </button>
            <button
              onClick={() =>
                window.open(fileUrl(readerPaper.id), '_blank', 'noopener,noreferrer')
              }
              className="text-xs px-3 py-1 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-all"
            >
              新窗口打开
            </button>
            <button
              onClick={onClose}
              className="text-xs px-3 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 transition-all"
            >
              关闭
            </button>
          </div>
        </div>

        {/* 内容：PDF + 右侧对话 */}
        <div className="flex-1 flex min-h-0">
          <PdfReaderPane
            file={fileUrl(readerPaper.id)}
            locatePage={highlightPage}
            locateText={highlightText}
            locateKey={locateNonce}
          />
          {chatOpen && (
            <div className="w-[380px] shrink-0 border-l border-white/10">
              <ReaderChat
                paper={readerPaper}
                llmProvider={llmProvider}
                onProviderChange={onProviderChange}
                onLocateSnippet={(snippet, page) => {
                  setHighlightText(snippet);
                  setHighlightPage((page && page > 0) ? page : null);
                  setLocateNonce((n) => n + 1); // 同一片段重复点击也强制重新定位
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
