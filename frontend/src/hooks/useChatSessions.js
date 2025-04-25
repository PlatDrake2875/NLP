// HIA/frontend/src/hooks/useChatSessions.js
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useLocalStorage } from './useLocalStorage'; // Keep for activeSessionId

// Helper function to generate a new session ID
const generateNewSessionId = (counter) => `new-chat-${counter}`;

// Helper function to format history for download
const formatHistoryForDownload = (history) => {
    return JSON.stringify(Array.isArray(history) ? history : [], null, 2);
};

// Default API URL if not provided
const DEFAULT_API_URL = 'http://localhost:8000/api/chat';

export function useChatSessions(apiUrl = DEFAULT_API_URL) {
  // --- Use useState instead of useLocalStorage for sessions ---
  const [sessions, setSessions] = useState({});
  // --- Keep useLocalStorage for activeSessionId ---
  const [activeSessionId, setActiveSessionId] = useLocalStorage('activeSessionId', null);
  const sessionCounterRef = useRef(1);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // --- Load initial sessions from localStorage ONCE on mount ---
  // This avoids re-reading from localStorage on every render
  useEffect(() => {
      try {
          const storedSessions = window.localStorage.getItem('chatSessions');
          if (storedSessions) {
              const parsedSessions = JSON.parse(storedSessions);
              // Basic validation: ensure it's an object
              if (typeof parsedSessions === 'object' && parsedSessions !== null) {
                  setSessions(parsedSessions);
                  console.log("[Init] Loaded sessions from localStorage:", parsedSessions);
              } else {
                  console.warn("[Init] Invalid data found in localStorage for 'chatSessions'. Initializing empty.");
                  setSessions({});
              }
          } else {
               console.log("[Init] No sessions found in localStorage. Initializing empty.");
               setSessions({}); // Initialize empty if nothing in storage
          }
      } catch (error) {
          console.error("[Init] Error reading sessions from localStorage:", error);
          setSessions({}); // Initialize empty on error
      }
  }, []); // Empty dependency array ensures this runs only once on mount

  // --- Effect to save sessions to localStorage whenever sessions state changes ---
  useEffect(() => {
      // Don't save if sessions is empty initially before loading
      if (Object.keys(sessions).length > 0 || localStorage.getItem('chatSessions')) {
          // console.log("[Save] Saving sessions to localStorage:", sessions); // Can be verbose
          try {
             window.localStorage.setItem('chatSessions', JSON.stringify(sessions));
          } catch (error) {
             console.error("[Save] Error saving sessions to localStorage:", error);
          }
      }
  }, [sessions]); // Run this effect whenever the sessions state changes

  // Initialize session counter and ensure an active session exists
  useEffect(() => {
    // Only run initialization logic if sessions have been loaded/initialized
    if (sessions === null || sessions === undefined) return; // Guard against running too early

    const existingSessionIds = Object.keys(sessions);
    let maxNum = 0;
    let sessionCreatedOrSelected = false;

    if (existingSessionIds.length > 0) {
      existingSessionIds.forEach(id => {
        if (id.startsWith('new-chat-')) {
          const num = parseInt(id.replace('new-chat-', ''), 10);
          if (!isNaN(num) && num > maxNum) {
            maxNum = num;
          }
        }
      });
      sessionCounterRef.current = maxNum + 1;

      // Validate activeSessionId
      if (!activeSessionId || !sessions[activeSessionId]) {
        const firstId = existingSessionIds[0];
        console.log(`[Init Check] Invalid/missing activeSessionId ('${activeSessionId}'). Setting to first: ${firstId}`);
        setActiveSessionId(firstId);
        sessionCreatedOrSelected = true;
      } else {
         sessionCreatedOrSelected = true; // Valid session ID exists
      }
    } else {
      // Create initial session if none exist after loading
      console.log("[Init Check] No sessions exist. Creating initial session.");
      const firstSessionId = generateNewSessionId(sessionCounterRef.current);
      sessionCounterRef.current++;
      setSessions({ [firstSessionId]: [] }); // Set initial session directly
      setActiveSessionId(firstSessionId);
      sessionCreatedOrSelected = true;
    }
  }, [sessions, activeSessionId, setActiveSessionId]); // Rerun if sessions load or activeId changes externally


  // --- Session Management Handlers (Remain largely the same, using setSessions) ---

  const handleNewChat = useCallback(() => {
    const newSessionId = generateNewSessionId(sessionCounterRef.current);
    sessionCounterRef.current++;
    setSessions(prevSessions => ({
      ...prevSessions,
      [newSessionId]: [],
    }));
    setActiveSessionId(newSessionId);
  }, [setSessions, setActiveSessionId]);

  const handleSelectSession = useCallback((sessionId) => {
    if (sessions && sessions[sessionId]) { // Check if sessions is loaded
        setActiveSessionId(sessionId);
    } else {
        console.warn(`Attempted to select non-existent session: ${sessionId}`);
        if (sessions) { // Check if sessions is loaded
            const firstSessionId = Object.keys(sessions)[0];
            if (firstSessionId) {
                setActiveSessionId(firstSessionId);
            }
        }
    }
  }, [setActiveSessionId, sessions]);

  const handleDeleteSession = useCallback((sessionIdToDelete) => {
    if (!sessions) return; // Guard against sessions not loaded
    const currentSessionIds = Object.keys(sessions);
    if (currentSessionIds.length <= 1) {
        console.warn("Cannot delete the last chat session.");
        return;
    }

    setSessions(prevSessions => {
        const updatedSessions = { ...prevSessions };
        delete updatedSessions[sessionIdToDelete];
        return updatedSessions;
    });

    if (activeSessionId === sessionIdToDelete) {
      const remainingIds = currentSessionIds.filter(id => id !== sessionIdToDelete);
      setActiveSessionId(remainingIds[0] || null);
    }
  }, [activeSessionId, sessions, setSessions, setActiveSessionId]);

  const clearActiveChatHistory = useCallback(() => {
    if (!activeSessionId || !sessions || !sessions[activeSessionId]) return; // Guard
    if (window.confirm(`Are you sure you want to clear the history for "${formatSessionName(activeSessionId)}"?`)) {
        setSessions(prevSessions => {
            const updatedSessions = { ...prevSessions };
            updatedSessions[activeSessionId] = [];
            return updatedSessions;
        });
    }
  }, [activeSessionId, sessions, setSessions]);

  const downloadActiveChatHistory = useCallback(() => {
     const currentHistory = (sessions && activeSessionId) ? sessions[activeSessionId] : null; // Guard
    if (!activeSessionId || !Array.isArray(currentHistory) || currentHistory.length === 0) return;

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

  const formatSessionName = (sessionId) => {
    if (!sessionId) return "";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
  };

  // --- Chat Submission Handler (Using useState for sessions) ---
  const handleChatSubmit = useCallback(async (query, model) => {
    if (!activeSessionId || !model || isSubmitting) {
        console.warn("Chat submission prevented:", { activeSessionId, model, isSubmitting });
        return;
    }

    console.log(`[${activeSessionId}] SUBMIT: Starting for query: "${query}"`);
    setIsSubmitting(true);

    const userMessage = { sender: 'user', text: query };
    const botMessageId = `bot-${Date.now()}-${Math.random()}`;
    const botMessagePlaceholder = { id: botMessageId, sender: 'bot', text: '...' };

    // --- Add User Message & Placeholder in ONE update ---
    setSessions(prevSessions => {
        console.log(`[${activeSessionId}] SUBMIT: Updating state. Prev history for ${activeSessionId}:`, prevSessions[activeSessionId]);
        const currentHistory = prevSessions[activeSessionId] || [];
        const historyArray = Array.isArray(currentHistory) ? currentHistory : [];
        const newHistory = [...historyArray, userMessage, botMessagePlaceholder];
        console.log(`[${activeSessionId}] SUBMIT: New history state for ${activeSessionId}:`, newHistory);
        const newSessions = {
            ...prevSessions,
            [activeSessionId]: newHistory,
        };
        return newSessions;
    });
    // --- End Initial State Update ---

    let streamError = null;

    try {
      console.log(`[${activeSessionId}] FETCH: Starting fetch to ${apiUrl}`);
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model }),
      });

      if (!response.ok || !response.body) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.text();
            try { const errorJson = JSON.parse(errorData); errorDetail = errorJson.detail || errorJson.error || `Server error: ${response.status}`; }
            catch(jsonError) { errorDetail = `${errorDetail}: ${errorData.substring(0, 150)}`; }
        } catch (e) { errorDetail = `Failed to read error response: ${response.status}`; }
        throw new Error(errorDetail);
      }

      console.log(`[${activeSessionId}] STREAM: Starting to process stream.`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let firstChunk = true;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });

          // --- State Update Logic during stream (using standard useState setter) ---
          setSessions(prevSessions => {
              const latestHistory = prevSessions[activeSessionId] || [];
              const historyArray = Array.isArray(latestHistory) ? latestHistory : [];
              const botMessageIndex = historyArray.findIndex(msg => msg.id === botMessageId);

              if (botMessageIndex === -1) {
                  console.warn(`[${activeSessionId}] STREAM WARN: Bot message ID ${botMessageId} not found in history:`, historyArray);
                  return prevSessions;
              }

              const existingMessage = historyArray[botMessageIndex];
              const updatedMessage = {
                  ...existingMessage,
                  text: firstChunk ? chunk : existingMessage.text + chunk
              };

              const updatedHistoryArray = [
                  ...historyArray.slice(0, botMessageIndex),
                  updatedMessage,
                  ...historyArray.slice(botMessageIndex + 1)
              ];

              const newSessions = {
                  ...prevSessions,
                  [activeSessionId]: updatedHistoryArray,
              };
              return newSessions;
          });
          // --- End State Update Logic ---
          firstChunk = false;
        }
      }
      console.log(`[${activeSessionId}] STREAM: Finished successfully.`);

    } catch (error) {
      console.error(`[${activeSessionId}] ERROR: During fetch/stream:`, error);
      streamError = error;

      // --- Final Error State Update ---
      setSessions(prevSessions => {
            const latestHistory = prevSessions[activeSessionId] || [];
            const historyArray = Array.isArray(latestHistory) ? latestHistory : [];
            const botMessageIndex = historyArray.findIndex(msg => msg.id === botMessageId);

            if (botMessageIndex === -1) {
                 console.warn(`[${activeSessionId}] ERROR update: Bot message placeholder ${botMessageId} not found, adding error as new message.`);
                 const errorMsg = { id: botMessageId, sender: 'bot', text: `⚠️ Error: ${streamError.message}` };
                 return { ...prevSessions, [activeSessionId]: [...historyArray, errorMsg] };
            }

            const updatedMessage = { ...historyArray[botMessageIndex], text: `⚠️ Error: ${streamError.message}` };
            const updatedHistoryArray = [ ...historyArray.slice(0, botMessageIndex), updatedMessage, ...historyArray.slice(botMessageIndex + 1) ];
            console.log(`[${activeSessionId}] Updated message ${botMessageId} with error.`);
            return { ...prevSessions, [activeSessionId]: updatedHistoryArray };
          });
      // --- End Final Error State Update ---

    } finally {
       console.log(`[${activeSessionId}] SUBMIT: Finishing submission.`);
       setIsSubmitting(false);
    }
  }, [activeSessionId, isSubmitting, setSessions, apiUrl]); // Removed setActiveSessionId as it's not used directly here

  // Derive active chat history, ensuring it's always an array
  const activeChatHistory = useMemo(() => {
      // Ensure sessions is loaded before trying to access it
      if (!sessions || !activeSessionId || !sessions[activeSessionId]) {
          return [];
      }
      const history = sessions[activeSessionId];
      return Array.isArray(history) ? history : [];
  }, [activeSessionId, sessions]);

``
  return {
    sessions,
    activeSessionId,
    activeChatHistory,
    isSubmitting,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit,
  };
}
