// HIA/frontend/src/components/ChatInterface.jsx
import React, { useRef } from 'react';
import { Header } from './Header';
import { ChatHistory } from './ChatHistory';
import { ChatForm } from './ChatForm';
import { ScrollToBottomButton } from './ScrollToBottomButton';

export function ChatInterface({
  activeSessionId,
  chatHistory,
  onSubmit,
  onClearHistory,
  onDownloadHistory,
  // --- Receive theme props ---
  isDarkMode,
  toggleTheme,
}) {
  const chatContainerRef = useRef(null);
  const bottomOfChatRef = useRef(null);

  return (
    <main className="chat-interface">
      <Header
        activeSessionId={activeSessionId}
        clearChatHistory={onClearHistory}
        downloadChatHistory={onDownloadHistory}
        disabled={!activeSessionId}
        // --- Pass theme props down ---
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />
      <div className="chat-area" ref={chatContainerRef}>
          {activeSessionId ? (
              <ChatHistory chatHistory={chatHistory} />
           ) : (
              <div className="no-chat-selected">
                <p>Select a chat or start a new one.</p>
              </div>
           )
          }
        <div ref={bottomOfChatRef} style={{ height: '1px' }} />
      </div>

        {activeSessionId && (
          <div className="chat-input-area">
              <ChatForm onSubmit={onSubmit} />
          </div>
        )}

       <ScrollToBottomButton
           containerRef={chatContainerRef}
           targetRef={bottomOfChatRef}
       />
    </main>
  );
}