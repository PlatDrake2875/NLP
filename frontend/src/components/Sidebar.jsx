import React from 'react';

export function Sidebar({ sessions, activeSessionId, onNewChat, onSelectSession, onDeleteSession }) {
  const sessionIds = Object.keys(sessions);

  // Function to format session ID for display
  const formatSessionName = (sessionId) => {
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
  };

  return (
    <div className="sidebar">
      <button onClick={onNewChat} className="new-chat-button">
        + New Chat
      </button>
      <nav className="conversation-menu">
        <h2>Conversations</h2>
        <ul>
          {sessionIds.length === 0 && <li className="no-sessions">No chats yet.</li>}
          {sessionIds.map((sessionId) => (
            <li
              key={sessionId}
              className={`conversation-item ${sessionId === activeSessionId ? 'active' : ''}`}
            >
                <button
                 className="session-select-button"
                 onClick={() => onSelectSession(sessionId)}
                 aria-current={sessionId === activeSessionId ? 'page' : undefined}
                >
                  {formatSessionName(sessionId)}
                </button>
               <button
                  onClick={(e) => {
                      e.stopPropagation(); // Prevent triggering session selection
                      if (window.confirm(`Are you sure you want to delete "${formatSessionName(sessionId)}"?`)) {
                         onDeleteSession(sessionId);
                      }
                  }}
                  className="delete-session-button"
                  aria-label={`Delete ${formatSessionName(sessionId)}`}
               >
                  🗑️
               </button>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}