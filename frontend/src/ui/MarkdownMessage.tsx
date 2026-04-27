import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

type Props = {
  text: string;
  className?: string;
  compact?: boolean;
};

// 把常见的"裸 LaTeX"写法规整成 remark-math 识别的定界符：
// - \( ... \)  ->  $ ... $
// - \[ ... \]  ->  $$ ... $$
// 防止 LLM 输出的公式因为定界符不统一而无法被渲染。
function normalizeMath(src: string): string {
  if (!src) return src;
  let out = src;
  out = out.replace(/\\\[(\s*[\s\S]*?\s*)\\\]/g, (_m, inner) => `\n\n$$${inner}$$\n\n`);
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_m, inner) => `$${inner}$`);
  return out;
}

export function MarkdownMessage({ text, className, compact }: Props) {
  const content = normalizeMath(text ?? '');
  const base = compact
    ? 'prose-xs leading-relaxed'
    : 'prose-sm leading-relaxed';
  return (
    <div className={`markdown-body ${base} ${className || ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          p: ({ children }) => <p className="mb-2 last:mb-0 whitespace-pre-wrap break-words">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          h1: ({ children }) => <h1 className="text-base font-semibold mt-3 mb-2">{children}</h1>,
          h2: ({ children }) => <h2 className="text-[15px] font-semibold mt-3 mb-2">{children}</h2>,
          h3: ({ children }) => <h3 className="text-sm font-semibold mt-2 mb-1.5">{children}</h3>,
          h4: ({ children }) => <h4 className="text-sm font-semibold mt-2 mb-1">{children}</h4>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-white/20 pl-3 text-slate-300 italic my-2">{children}</blockquote>
          ),
          code: ({ inline, className: cls, children, ...rest }: any) => {
            if (inline) {
              return (
                <code className="px-1 py-0.5 rounded bg-white/10 text-[0.92em] text-amber-200" {...rest}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="my-2 p-3 rounded-lg bg-black/40 border border-white/10 overflow-x-auto text-[12px] leading-relaxed">
                <code className={cls} {...rest}>{children}</code>
              </pre>
            );
          },
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-300 underline hover:text-indigo-200">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto">
              <table className="min-w-full text-xs border border-white/10">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="border border-white/10 px-2 py-1 bg-white/5 text-left">{children}</th>,
          td: ({ children }) => <td className="border border-white/10 px-2 py-1 align-top">{children}</td>,
          hr: () => <hr className="my-3 border-white/10" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default MarkdownMessage;
