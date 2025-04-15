// HIA/frontend/src/hooks/useChatSessions.js
import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocalStorage } from './useLocalStorage';

// Helper function to generate a new session ID
const generateNewSessionId = (counter) => `new-chat-${counter}`;

// Helper function to format history for download
const formatHistoryForDownload = (history) => {
    return JSON.stringify(history, null, 2);
};

export function useChatSessions(apiUrl = 'http://localhost:8000/api/chat') {
  const [sessions, setSessions] = useLocalStorage('chatSessions', {});
  const [activeSessionId, setActiveSessionId] = useState(null);
  const sessionCounterRef = useRef(1);

  // Initialize session counter and set initial active session
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

      // Try to restore last active session ID from local storage or default to first
      // Note: You might want to store/retrieve activeSessionId itself in localStorage
      // if you want it to persist across browser sessions.
      if (!activeSessionId) {
         setActiveSessionId(existingSessionIds[0]); // Default to first if none active
      }
    } else {
      // No sessions exist, create the first one
      const firstSessionId = generateNewSessionId(sessionCounterRef.current);
      sessionCounterRef.current++;
      setSessions({ [firstSessionId]: [] });
      setActiveSessionId(firstSessionId);
    }
    // This effect should run only once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Dependencies: sessions, setSessions - but run only once ideally

  // --- Session Management Handlers ---

  const handleNewChat = useCallback(() => {
    const newSessionId = generateNewSessionId(sessionCounterRef.current);
    sessionCounterRef.current++;
    setSessions(prevSessions => ({
      ...prevSessions,
      [newSessionId]: [],
    }));
    setActiveSessionId(newSessionId);
  }, [setSessions]); // Add setSessions dependency

  const handleSelectSession = useCallback((sessionId) => {
    setActiveSessionId(sessionId);
  }, []); // No dependencies needed if setActiveSessionId is stable

  const handleDeleteSession = useCallback((sessionIdToDelete) => {
    setSessions(prevSessions => {
      const updatedSessions = { ...prevSessions };
      delete updatedSessions[sessionIdToDelete];
      return updatedSessions;
    });

    if (activeSessionId === sessionIdToDelete) {
      const remainingIds = Object.keys(sessions).filter(id => id !== sessionIdToDelete);
      if (remainingIds.length > 0) {
          setActiveSessionId(remainingIds[0]);
      } else {
          // If deleting the last session, create a new one
          handleNewChat(); // Assumes handleNewChat is stable
      }
    }
  }, [activeSessionId, sessions, setSessions, handleNewChat]); // Ensure all dependencies are listed

  const clearActiveChatHistory = useCallback(() => {
    if (!activeSessionId) return;
    setSessions(prevSessions => ({
      ...prevSessions,
      [activeSessionId]: [],
    }));
  }, [activeSessionId, setSessions]);

  const downloadActiveChatHistory = useCallback(() => {
     const currentHistory = activeSessionId ? sessions[activeSessionId] : null;
    if (!activeSessionId || !currentHistory || currentHistory.length === 0) return;

    const json = formatHistoryForDownload(currentHistory);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${activeSessionId}_history.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [activeSessionId, sessions]);

  // --- Chat Submission Handler ---

  const handleChatSubmit = useCallback(async (query) => {
    if (!activeSessionId) return;

    const userMessage = { sender: 'user', text: query };

    // Optimistic update
    setSessions((prevSessions) => ({
      ...prevSessions,
      [activeSessionId]: [...(prevSessions[activeSessionId] || []), userMessage],
    }));

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { sender: 'bot', text: data.answer || 'Received empty answer.' }; // Handle potentially empty answers

      // Add bot response correctly
      setSessions((prevSessions) => {
        const currentHistory = prevSessions[activeSessionId] || [];
         // Use a more robust check if needed (e.g., message ID)
        const historyWithUserMsg = currentHistory.some(msg => msg.sender === 'user' && msg.text === query)
          ? currentHistory
          : [...currentHistory, userMessage]; // Ensure user message is included

        return {
          ...prevSessions,
          [activeSessionId]: [...historyWithUserMsg, botMessage],
        };
      });

    } catch (error) {
      console.error('Error fetching chat response:', error);
      const errorMessage = { sender: 'bot', text: `Sorry, I encountered an error: ${error.message}` };

      // Add error message correctly
      setSessions((prevSessions) => {
        const currentHistory = prevSessions[activeSessionId] || [];
        const historyWithUserMsg = currentHistory.some(msg => msg.sender === 'user' && msg.text === query)
          ? currentHistory
          : [...currentHistory, userMessage]; // Ensure user message is included

        return {
          ...prevSessions,
          [activeSessionId]: [...historyWithUserMsg, errorMessage],
        };
      });
    }
  }, [activeSessionId, setSessions, apiUrl]); // Add apiUrl dependency

  // Derive active chat history
  const activeChatHistory = activeSessionId ? sessions[activeSessionId] || [] : [];

  return {
    sessions,
    activeSessionId,
    activeChatHistory,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit,
  };
}