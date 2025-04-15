// HIA/frontend/src/components/Header.jsx
import React from 'react';

const formatSessionName = (sessionId) => {
    // ... (keep existing function)
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};

export function Header({
    activeSessionId,
    clearChatHistory,
    downloadChatHistory,
    disabled,
    // --- Receive theme props ---
    isDarkMode,
    toggleTheme
 }) {
  const title = activeSessionId ? formatSessionName(activeSessionId) : "Accessibility Navigator Chat";

  return (
    <header className="chat-header">
      <h1>{title}</h1>
      <div className="header-actions">
         {/* --- Add Theme Toggle Button --- */}
         <button
            onClick={toggleTheme}
            className="header-button theme-toggle-btn"
            aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}
         >
            {isDarkMode ? '☀️ Light' : '🌙 Dark'} {/* Example text/icons */}
         </button>
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