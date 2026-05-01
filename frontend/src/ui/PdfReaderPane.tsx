import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

type Props = {
  file: string;
  locatePage?: number | null;
  locateText?: string;
  locateKey?: number;
};

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function pickHighlightTerms(raw: string): string[] {
  const s = (raw || '').trim();
  if (!s) return [];
  const stopWords = new Set([
    'paper',
    'chunk',
    'reference',
    'according',
    'based',
    'which',
    'with',
    'that',
    'this',
    'from',
    'have',
    'into',
    'about',
    'insert',
    'update',
    'delete',
    'select',
    'where',
    'join',
    'outer',
    'right',
    'left',
    'inner',
    'group',
    'order',
    'limit',
    'table',
    'values',
  ]);
  const zh = s.match(/[\u4e00-\u9fa5]{3,}/g) || [];
  const enWords = s.match(/[A-Za-z][A-Za-z0-9_-]{3,}/g) || [];
  const enLong = s.match(/[A-Za-z][A-Za-z0-9_-]{5,}/g) || [];
  const all = [...zh, ...enLong, ...enWords]
    .map((x) => x.trim())
    .filter((x) => x.length >= 3)
    .filter((x) => !/paper_id|chunk_id/i.test(x))
    .filter((x) => !stopWords.has(x.toLowerCase()));
  return Array.from(new Set(all)).sort((a, b) => b.length - a.length).slice(0, 4);
}

export default function PdfReaderPane({ file, locatePage, locateText, locateKey }: Props) {
  const [numPages, setNumPages] = useState<number>(1);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [zoom, setZoom] = useState<number>(1.15);
  const [loadError, setLoadError] = useState<string>('');
  const [fallbackNative, setFallbackNative] = useState<boolean>(false);
  const [hitCount, setHitCount] = useState<number>(0);
  const viewerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setPageNumber(1);
    setLoadError('');
    setFallbackNative(false);
  }, [file]);

  useEffect(() => {
    if (locatePage && locatePage > 0) {
      setPageNumber(locatePage);
    }
  }, [locatePage, locateKey]);

  const terms = useMemo(() => pickHighlightTerms(locateText || ''), [locateText, locateKey]);
  const primarySearch = terms[0] || '';
  const nativeSrc = locatePage && locatePage > 0
    ? (
      primarySearch
        ? `${file}#page=${locatePage}&search=${encodeURIComponent(primarySearch)}`
        : `${file}#page=${locatePage}`
    )
    : (
      primarySearch
        ? `${file}#search=${encodeURIComponent(primarySearch)}`
        : file
    );
  const shownPage = Math.min(Math.max(1, pageNumber), numPages || 1);
  const locateSignature = `${shownPage}:${terms.join('|')}:${locateKey ?? 0}`;

  useEffect(() => {
    setHitCount(0);
  }, [locateSignature]);

  const afterPageRendered = () => {
    requestAnimationFrame(() => {
      const root = viewerRef.current;
      if (!root) return;
      const marks = root.querySelectorAll('mark.pdf-hit');
      setHitCount(marks.length);
      if (marks.length > 0) {
        (marks[0] as HTMLElement).scrollIntoView({ block: 'center', inline: 'nearest' });
      }
    });
  };

  const textRenderer = (item: { str: string }) => {
    const str = item.str || '';
    if (!terms.length || !str.trim()) return str;
    const matchedIdx = terms.findIndex((t) => {
      const re = new RegExp(`(${escapeRegExp(t)})`, 'gi');
      return re.test(str);
    });
    if (matchedIdx < 0) return str;
    const cls = matchedIdx === 0 ? 'pdf-hit pdf-hit-primary' : 'pdf-hit pdf-hit-secondary';
    return `<mark class="${cls}">${str}</mark>`;
  };

  return (
    <div className="flex-1 min-w-0 h-full relative bg-[#0b1220]">
      <div className="absolute right-3 bottom-3 z-20 flex items-center gap-2 px-2 py-1 rounded-full bg-black/50 border border-white/15 text-xs">
        <button
          onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
          className="p-1 rounded hover:bg-white/10"
          title="上一页"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        <span className="font-mono min-w-[72px] text-center">{shownPage} / {numPages || 1}</span>
        <button
          onClick={() => setPageNumber((p) => Math.min(numPages || 1, p + 1))}
          className="p-1 rounded hover:bg-white/10"
          title="下一页"
        >
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.8, z - 0.1))}
          className="px-1.5 py-0.5 rounded hover:bg-white/10"
          title="缩小"
        >
          -
        </button>
        <span className="w-10 text-center">{Math.round(zoom * 100)}%</span>
        <button
          onClick={() => setZoom((z) => Math.min(2.2, z + 0.1))}
          className="px-1.5 py-0.5 rounded hover:bg-white/10"
          title="放大"
        >
          +
        </button>
      </div>

      {terms.length > 0 && (
        <div className="absolute left-3 top-3 z-20 max-w-[68%] inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-yellow-300/10 border border-yellow-400/30 text-yellow-200 text-[11px]">
          <Search className="w-3 h-3 shrink-0" />
          <span className="truncate">已高亮关键词：{terms.join(' / ')}</span>
          <span className="shrink-0 rounded border border-yellow-300/50 px-1 py-[1px] text-[10px]">
            命中 {hitCount}
          </span>
        </div>
      )}

      {loadError && !fallbackNative && (
        <div className="absolute left-3 top-12 z-20 max-w-[68%] inline-flex items-center gap-2 px-2 py-1 rounded-md bg-red-500/10 border border-red-400/30 text-red-200 text-[11px]">
          <span className="truncate">PDF.js 加载失败：{loadError}</span>
          <button
            className="px-1.5 py-0.5 rounded border border-red-300/40 hover:bg-red-500/20 shrink-0"
            onClick={() => setFallbackNative(true)}
            title="切回浏览器原生 PDF 查看器"
          >
            回退查看器
          </button>
        </div>
      )}

      <div ref={viewerRef} className="w-full h-full overflow-auto p-3">
        {fallbackNative ? (
          <iframe
            title="pdf-native"
            src={nativeSrc}
            className="w-full h-full rounded-md border border-white/10"
          />
        ) : (
          <Document
            file={file}
            onLoadSuccess={({ numPages: n }) => {
              setNumPages(n);
              setPageNumber((p) => Math.min(Math.max(1, p), n));
              setLoadError('');
            }}
            onLoadError={(err) => {
              setLoadError((err as Error)?.message || '未知错误');
            }}
            loading={<div className="text-sm text-slate-300 p-4">PDF 加载中…</div>}
            error={<div className="text-sm text-red-300 p-4">PDF 加载失败，请尝试“回退查看器”。</div>}
          >
            <div className="mx-auto w-fit rounded-md overflow-hidden shadow-2xl">
              <Page
                pageNumber={shownPage}
                scale={zoom}
                renderTextLayer
                renderAnnotationLayer
                customTextRenderer={textRenderer}
                onRenderSuccess={afterPageRendered}
              />
            </div>
          </Document>
        )}
      </div>
    </div>
  );
}

