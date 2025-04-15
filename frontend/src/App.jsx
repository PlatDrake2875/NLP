import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css'; // Ensure CSS is updated
import { useLocalStorage } from './hooks/useLocalStorage';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';

function App() {
  // Store all chat sessions: { [sessionId]: [{ sender, text }, ...], ... }
  const [sessions, setSessions] = useLocalStorage('chatSessions', {});
  const [activeSessionId, setActiveSessionId] = useState(null);
  // Ref to keep track of the next chat number, persists across renders but not page reloads
  const sessionCounterRef = useRef(1);

  // Initialize session counter based on existing sessions
  useEffect(() => {
    const existingSessionIds = Object.keys(sessions);
    if (existingSessionIds.length > 0) {
      let maxNum = 0;
      existingSessionIds.forEach(id => {
        if (id.startsWith('new-chat-')) {
          const num = parseInt(id.replace('new-chat-', ''), 10);
          if (!isNaN(num) && num > maxNum) {
            maxNum = num;
          }
        }
      });
      sessionCounterRef.current = maxNum + 1;

      // If no active session is set, try to activate the first one
      if (!activeSessionId && existingSessionIds.length > 0) {
        setActiveSessionId(existingSessionIds[0]);
      }
    } else {
        // If no sessions exist, create one automatically on first load
        handleNewChat();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  // Get the chat history for the currently active session
  const activeChatHistory = activeSessionId ? sessions[activeSessionId] || [] : [];

  // Handler to create a new chat session
  const handleNewChat = useCallback(() => {
    const newSessionId = `new-chat-${sessionCounterRef.current}`;
    sessionCounterRef.current++; // Increment for the next new chat
    setSessions((prevSessions) => ({
      ...prevSessions,
      [newSessionId]: [], // Initialize with empty history
    }));
    setActiveSessionId(newSessionId); // Make the new chat active
  }, [setSessions]);

  // Handler to switch to a different chat session
  const handleSelectSession = useCallback((sessionId) => {
    setActiveSessionId(sessionId);
  }, []);

  const handleChatSubmit = useCallback(async (query) => {
    if (!activeSessionId) return;

    const userMessage = { sender: 'user', text: query };

    // Optimistically add user message
    setSessions((prevSessions) => ({
      ...prevSessions,
      [activeSessionId]: [...(prevSessions[activeSessionId] || []), userMessage],
    }));

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      // Handle potential non-OK responses (optional but good practice)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { sender: 'bot', text: data.answer };

      // CORRECTED: Add bot message to the *current* history
      setSessions((prevSessions) => {
        // Get the latest history for the active session
        const currentHistory = prevSessions[activeSessionId] || [];
        // Ensure the user message is present before adding the bot message
        // (This check is defensive, the optimistic update should ensure it's there)
        const historyWithUserMsg = currentHistory.find(msg => msg === userMessage)
          ? currentHistory
          : [...currentHistory, userMessage]; // Fallback if optimistic update somehow failed between renders

        return {
          ...prevSessions,
          [activeSessionId]: [...historyWithUserMsg, botMessage],
        };
      });

    } catch (error) {
      console.error('Error fetching chat response:', error);
      const errorMessage = { sender: 'bot', text: `Sorry, I encountered an error: ${error.message}` }; // Include error message

      // CORRECTED: Add error message to the *current* history
      setSessions((prevSessions) => {
         const currentHistory = prevSessions[activeSessionId] || [];
         const historyWithUserMsg = currentHistory.find(msg => msg === userMessage)
           ? currentHistory
           : [...currentHistory, userMessage]; // Fallback

         return {
           ...prevSessions,
           [activeSessionId]: [...historyWithUserMsg, errorMessage],
         };
       });
    }
  }, [activeSessionId, setSessions]); // Dependencies seem correct

  // Handler to clear the active chat history
  const clearActiveChatHistory = useCallback(() => {
    if (!activeSessionId) return;
    setSessions((prevSessions) => ({
      ...prevSessions,
      [activeSessionId]: [], // Reset history for the active session
    }));
  }, [activeSessionId, setSessions]);

  // Handler to download the active chat history
  const downloadActiveChatHistory = useCallback(() => {
    if (!activeSessionId || !sessions[activeSessionId]) return;

    const historyToDownload = sessions[activeSessionId];
    const json = JSON.stringify(historyToDownload, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    // Use session ID (or a derived name) for the filename
    link.download = `${activeSessionId}_history.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [activeSessionId, sessions]);

    // Handler to delete a chat session
    const handleDeleteSession = useCallback((sessionIdToDelete) => {
        setSessions(prevSessions => {
            const updatedSessions = { ...prevSessions };
            delete updatedSessions[sessionIdToDelete];
            return updatedSessions;
        });
        // If the deleted session was the active one, set active to null or another session
        if (activeSessionId === sessionIdToDelete) {
            const remainingIds = Object.keys(sessions).filter(id => id !== sessionIdToDelete);
            setActiveSessionId(remainingIds.length > 0 ? remainingIds[0] : null);
             // If no sessions left, create a new one
            if (remainingIds.length === 0) {
                 handleNewChat();
            }
        }
    }, [activeSessionId, sessions, setSessions, handleNewChat]);


  return (
    <div className="App">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession} // Pass delete handler
      />
      <ChatInterface
        key={activeSessionId} // Re-mount ChatInterface when session changes to reset scroll etc.
        activeSessionId={activeSessionId}
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmit}
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
      />
    </div>
  );
}

export default App;