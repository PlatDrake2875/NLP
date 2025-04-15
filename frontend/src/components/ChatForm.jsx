import React, { useState } from 'react';

export function ChatForm({ onSubmit }) {
  const [query, setQuery] = useState('');

  // Handle Enter for submit, Shift+Enter for newline
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent default newline on Enter
      handleSubmit(e); // Submit the form
    }
    // Allow Shift+Enter to add a newline (default textarea behavior)
  };

  const handleSubmit = (e) => {
    // Can be triggered by button click or Enter key
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit(query);
    setQuery('');
  };

  return (
    // Pass the handleSubmit to the form for the button click
    <form onSubmit={handleSubmit} className="chat-form" autoComplete="off">
      <label htmlFor="chat-input" className="sr-only">
        Type your message (Shift + Enter for new line)
      </label>
      <textarea
        id="chat-input"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown} // Add keydown listener
        placeholder="Send a message... (Shift + Enter for new line)"
        aria-label="Chat input"
        rows="1" // Start with a single row appearance
        className="chat-textarea" // Add a specific class for styling
      />
      <button type="submit" aria-label="Send message" disabled={!query.trim()}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
        </svg>
      </button>
    </form>
  );
}