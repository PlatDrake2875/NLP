import React, { useEffect, useRef, useLayoutEffect } from 'react';
import { MarkdownMessage } from './MarkdownMessage';

export const ChatHistory = ({ chatHistory }) => {
  const endOfMessagesRef = useRef(null); // Ref to scroll to the bottom
  const chatHistoryRef = useRef(null); // Ref to the chat history container itself

  // Use useLayoutEffect for scrolling to ensure it happens after render but before paint
  useLayoutEffect(() => {
    const chatContainer = chatHistoryRef.current;
    if (!chatContainer) return;

    // Check if the user is scrolled near the bottom before auto-scrolling
    const scrollThreshold = 100; // Pixels from bottom
    const isScrolledNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < scrollThreshold;

    // Only auto-scroll if near the bottom or if it's the initial load/very few messages
    if (isScrolledNearBottom || chatContainer.scrollHeight < chatContainer.clientHeight + scrollThreshold) {
        endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [chatHistory]); // Dependency on chatHistory ensures scroll on update

  // --- Add console log here ---
  console.log("ChatHistory rendering with history:", chatHistory);
  // --- End console log ---

  return (
    // Add ref to the container
    <div className="chat-history" ref={chatHistoryRef}>
      {/* Ensure chatHistory is an array before mapping */}
      {!Array.isArray(chatHistory) && (
          <div className="message-container system-error"> {/* Optional: Style system errors */}
             <div className="chat-message bot">
                <p>Error: Chat history is unavailable.</p>
             </div>
          </div>
      )}
      {Array.isArray(chatHistory) && chatHistory.map((entry, index) => {
          // --- Add log inside map ---
          // console.log(`Rendering message ${index}:`, entry);
          // --- End log ---
          // Ensure entry is an object before accessing properties
          return (typeof entry === 'object' && entry !== null && (
              <div key={entry.id || `msg-${index}`} className={`message-container ${entry.sender || 'unknown'}`}>
                  <div className={`chat-message ${entry.sender || 'unknown'}`}>
                      {/* Ensure text exists and pass empty string if not */}
                      <MarkdownMessage text={entry.text ?? ''} />
                  </div>
              </div>
          ));
       })}
      {/* Empty div at the end to help scrolling to the bottom */}
      <div ref={endOfMessagesRef} />
    </div>
  );
};
