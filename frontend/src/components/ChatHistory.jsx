import React, { useEffect, useRef } from 'react';
import { MarkdownMessage } from './MarkdownMessage';

export const ChatHistory = ({ chatHistory }) => {
  const endOfMessagesRef = useRef(null); // Ref to scroll to the bottom

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]); // Dependency on chatHistory ensures scroll on update

  return (
    // Removed the old fixed-height container, scrolling is handled by chat-area in ChatInterface
    <div className="chat-history">
      {chatHistory.map((entry, index) => (
        <div key={index} className={`message-container ${entry.sender}`}>
            <div className={`chat-message ${entry.sender}`}>
                 {/* Optional: Add sender label visually if needed */}
                 {/* <span className="sender-label">{entry.sender === 'user' ? 'You' : 'Bot'}</span> */}
                <MarkdownMessage text={entry.text} />
            </div>
        </div>
      ))}
      {/* Empty div at the end to help scrolling to the bottom */}
      <div ref={endOfMessagesRef} />
    </div>
  );
};