import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; // Optional: for GitHub Flavored Markdown (tables, etc.)

export const MarkdownMessage = ({ text }) => {
  return (
    // Removed the old background/padding, handled by .chat-message CSS
    <div className="markdown-content">
      {/* Added remarkGfm for better markdown support */}
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
};