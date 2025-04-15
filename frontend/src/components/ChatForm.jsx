import React, { useState } from 'react';

export function ChatForm({ onSubmit }) {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit(query);
    setQuery('');
  };

  return (
    // Class name kept for potential specific styling
    <form onSubmit={handleSubmit} className="chat-form" autoComplete="off">
      <label htmlFor="chat-input" className="sr-only">
        Type your message
      </label>
      <input
        id="chat-input"
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Send a message..." // Updated placeholder
        aria-label="Chat input"
        autoComplete="off" // Explicitly turn off autocomplete
        // Consider adding autoFocus if desired
      />
      <button type="submit" aria-label="Send message">
         {/* Using SVG for send icon - replace with your preferred icon */}
         <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
        </svg>
      </button>
    </form>
  );
}