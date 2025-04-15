import React from 'react';

// Function to format session ID for display in header
const formatSessionName = (sessionId) => {
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};

export function Header({ activeSessionId, clearChatHistory, downloadChatHistory, disabled }) {
  const title = activeSessionId ? formatSessionName(activeSessionId) : "Accessibility Navigator Chat";

  return (
    <header className="chat-header">
      <h1>{title}</h1>
      <div className="header-actions">
        <button
          onClick={clearChatHistory}
          className="header-button clear-history-btn"
          disabled={disabled}
          aria-label="Clear current chat history"
        >
          Clear Chat
        </button>
        <button
          onClick={downloadChatHistory}
          className="header-button download-history-btn"
          disabled={disabled}
          aria-label="Download current chat history"
        >
          Download Chat
        </button>
      </div>
    </header>
  );
}