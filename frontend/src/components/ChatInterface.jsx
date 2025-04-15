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
}) {
  const chatContainerRef = useRef(null); // Ref for the scrollable chat history container
  const bottomOfChatRef = useRef(null); // Ref for the element ChatForm is visually associated with

  return (
    <main className="chat-interface">
      <Header
        // Pass the session ID to potentially display it
        activeSessionId={activeSessionId}
        clearChatHistory={onClearHistory}
        downloadChatHistory={onDownloadHistory}
        // Disable buttons if no active session
        disabled={!activeSessionId}
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
        {/* An empty div at the bottom for the ScrollToBottomButton to observe */}
        <div ref={bottomOfChatRef} style={{ height: '1px' }} />
      </div>

        {/* Conditionally render form only if a session is active */}
        {activeSessionId && (
          <div className="chat-input-area">
              <ChatForm onSubmit={onSubmit} />
          </div>
        )}

       {/* Scroll button observes the bottom ref relative to the chat container */}
       <ScrollToBottomButton
           containerRef={chatContainerRef} // Pass the scrollable container ref
           targetRef={bottomOfChatRef} // Pass the ref for the bottom marker
       />
    </main>
  );
}