// HIA/frontend/src/components/ChatInterface.jsx
import React, { useRef } from 'react';
import { Header } from './Header';
import { ChatHistory } from './ChatHistory';
import { ChatForm } from './ChatForm';
import { ScrollToBottomButton } from './ScrollToBottomButton';

export function ChatInterface({
  activeSessionId,
  chatHistory, // Receive chatHistory from App.jsx
  onSubmit, // This now expects a function that already knows the model
  onClearHistory,
  onDownloadHistory,
  isSubmitting, // Receive submitting state
  // --- Receive theme props ---
  isDarkMode,
  toggleTheme,
  // --- Receive model selection props ---
  availableModels,
  selectedModel,
  onModelChange,
  modelsLoading,
  modelsError,
}) {
  const chatContainerRef = useRef(null);
  const bottomOfChatRef = useRef(null);

  // Determine if the main chat interaction area should be disabled
  const isDisabled = !activeSessionId || modelsLoading || !!modelsError || isSubmitting;

  return (
    <main className="chat-interface">
      <Header
        activeSessionId={activeSessionId}
        chatHistory={chatHistory} // Pass chatHistory down to Header
        clearChatHistory={onClearHistory}
        downloadChatHistory={onDownloadHistory}
        disabled={isDisabled} // Pass combined disabled state
        // --- Pass theme props down ---
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
        // --- Pass model props down ---
        availableModels={availableModels}
        selectedModel={selectedModel}
        onModelChange={onModelChange}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
      />
      <div className="chat-area" ref={chatContainerRef}>
          {activeSessionId ? (
              // Pass chatHistory to ChatHistory component as well
              <ChatHistory chatHistory={chatHistory} />
           ) : (
              <div className="no-chat-selected">
                <p>Select a chat or start a new one.</p>
              </div>
           )
          }
        {/* Element to help scroll-to-bottom logic */}
        <div ref={bottomOfChatRef} style={{ height: '1px' }} />
      </div>

        {/* Conditionally render input area only if a chat is selected */}
        {activeSessionId && (
          <div className="chat-input-area">
              {/* Pass onSubmit directly - it now includes the model logic */}
              {/* Disable form if no model selected, loading, error, no active session, or submitting */}
              <ChatForm
                onSubmit={onSubmit}
                disabled={!selectedModel || isDisabled}
              />
          </div>
        )}

       {/* Only show scroll button if there's an active chat */}
       {activeSessionId && (
         <ScrollToBottomButton
             containerRef={chatContainerRef}
             targetRef={bottomOfChatRef}
         />
       )}
    </main>
  );
}
