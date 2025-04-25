// HIA/frontend/src/hooks/useChatSessions.js
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useLocalStorage } from './useLocalStorage'; // Keep for activeSessionId

// Helper function to generate a new session ID
const generateNewSessionId = (counter) => `new-chat-${counter}`;

// Helper function to format history for download
const formatHistoryForDownload = (history) => {
    return JSON.stringify(Array.isArray(history) ? history : [], null, 2);
};

// Helper function to format session names (used as fallback)
const formatSessionIdFallback = (sessionId) => {
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};


// API URLs (adjust if needed)
const CHAT_API_URL_SUFFIX = '/api/chat';
const AUTOMATE_API_URL_SUFFIX = '/api/automate_conversation';

export function useChatSessions(apiBaseUrl) {
  const chatApiUrl = `${apiBaseUrl}${CHAT_API_URL_SUFFIX}`;
  const automateApiUrl = `${apiBaseUrl}${AUTOMATE_API_URL_SUFFIX}`;

  // State for sessions: { [sessionId]: { name: string | null, history: Array } }
  const [sessions, setSessions] = useState({});
  // Persisted active session ID
  const [activeSessionId, setActiveSessionId] = useLocalStorage('activeSessionId', null);
  // Ref for generating unique IDs
  const sessionCounterRef = useRef(1);
  // State for API call status
  const [isSubmitting, setIsSubmitting] = useState(false);
  // State for automation errors
  const [automationError, setAutomationError] = useState(null);
  // State to track if initial load/init is complete
  const [isInitialized, setIsInitialized] = useState(false);

  // --- Effect 1: Load from localStorage and Initialize ONCE on mount ---
  useEffect(() => {
    console.log("[Init] Attempting to load sessions from localStorage...");
    let loadedSessions = {};
    let nextSessionCounter = 1;
    let initialActiveId = activeSessionId; // Get potentially stored active ID

    try {
      const storedSessions = window.localStorage.getItem('chatSessions');
      if (storedSessions) {
        const parsedSessions = JSON.parse(storedSessions);
        // Validate structure
        if (typeof parsedSessions === 'object' && parsedSessions !== null &&
            Object.values(parsedSessions).every(s => s && typeof s === 'object' && s.hasOwnProperty('history') && Array.isArray(s.history))) {

          // Ensure 'name' exists and calculate next counter
          let maxNum = 0;
          loadedSessions = Object.entries(parsedSessions).reduce((acc, [id, sessionData]) => {
            acc[id] = {
              name: sessionData.name !== undefined ? sessionData.name : null,
              history: sessionData.history
            };
            // Update counter based on loaded IDs
            if (id.startsWith('new-chat-')) {
              const num = parseInt(id.replace('new-chat-', ''), 10);
              if (!isNaN(num) && num > maxNum) {
                maxNum = num;
              }
            }
            return acc;
          }, {});
          nextSessionCounter = maxNum + 1;
          console.log("[Init] Successfully loaded sessions:", loadedSessions);
        } else {
          console.warn("[Init] Invalid data format in localStorage. Clearing.");
          window.localStorage.removeItem('chatSessions'); // Clear invalid data
          window.localStorage.removeItem('activeSessionId'); // Clear potentially related invalid ID
          initialActiveId = null; // Reset active ID
        }
      } else {
        console.log("[Init] No sessions found in localStorage.");
        initialActiveId = null; // Reset active ID if storage is empty
      }
    } catch (error) {
      console.error("[Init] Error reading/parsing sessions from localStorage:", error);
      loadedSessions = {}; // Start fresh on error
      window.localStorage.removeItem('chatSessions');
      window.localStorage.removeItem('activeSessionId');
      initialActiveId = null;
    }

    sessionCounterRef.current = nextSessionCounter; // Set the counter

    // --- Ensure a session exists and is active ---
    const loadedSessionIds = Object.keys(loadedSessions);

    if (loadedSessionIds.length === 0) {
      // If no sessions loaded (empty storage or error), create a new one
      console.log("[Init] No valid sessions loaded. Creating initial session.");
      const firstSessionId = generateNewSessionId(sessionCounterRef.current);
      sessionCounterRef.current++;
      loadedSessions[firstSessionId] = { name: null, history: [] };
      initialActiveId = firstSessionId; // Set the new one as active
      setSessions(loadedSessions); // Update state
      setActiveSessionId(initialActiveId); // Persist the new active ID
    } else {
      // If sessions were loaded, validate the activeSessionId
      if (!initialActiveId || !loadedSessions[initialActiveId]) {
        console.log(`[Init] Stored activeSessionId ('${initialActiveId}') is invalid. Selecting first available.`);
        initialActiveId = loadedSessionIds[0]; // Select the first loaded session
        setActiveSessionId(initialActiveId); // Persist the corrected active ID
      }
      // Set the loaded sessions state (active ID is already handled by useLocalStorage or set above)
      setSessions(loadedSessions);
    }

    setIsInitialized(true); // Mark initialization complete
    console.log("[Init] Initialization complete. Active session:", initialActiveId);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only ONCE on mount

  // --- Effect 2: Save to localStorage whenever sessions change (AFTER init) ---
  useEffect(() => {
    // Only save if initialization is complete and there are sessions to save
    if (isInitialized && sessions && Object.keys(sessions).length > 0) {
      console.log("[Save] Saving sessions to localStorage:", sessions);
      try {
        window.localStorage.setItem('chatSessions', JSON.stringify(sessions));
      } catch (error) {
        console.error("[Save] Error saving sessions to localStorage:", error);
      }
    } else if (isInitialized && Object.keys(sessions).length === 0) {
        // If initialized and sessions become empty (e.g., last one deleted), clear storage
        console.log("[Save] Sessions are empty, clearing localStorage.");
        window.localStorage.removeItem('chatSessions');
        window.localStorage.removeItem('activeSessionId'); // Also clear active ID
    }
  }, [sessions, isInitialized]); // Run when sessions change or initialization completes

  // --- Session Management Handlers ---

  const handleNewChat = useCallback(() => {
    const newSessionId = generateNewSessionId(sessionCounterRef.current);
    sessionCounterRef.current++;
    setSessions(prevSessions => ({
      ...prevSessions,
      [newSessionId]: { name: null, history: [] },
    }));
    setActiveSessionId(newSessionId); // Persisted via useLocalStorage hook's setter
    setAutomationError(null);
  }, [setActiveSessionId]); // Removed setSessions dependency as it doesn't rely on prev state directly here

  const handleSelectSession = useCallback((sessionId) => {
    if (sessions && sessions[sessionId]) {
        setActiveSessionId(sessionId);
        setAutomationError(null);
    } else {
        console.warn(`Attempted to select non-existent session: ${sessionId}`);
        // Fallback logic remains the same
        if (sessions) {
            const firstSessionId = Object.keys(sessions)[0];
            setActiveSessionId(firstSessionId || null); // Select first or null if none exist
        }
    }
  }, [setActiveSessionId, sessions]);

  const handleDeleteSession = useCallback((sessionIdToDelete) => {
    if (!sessions || !sessions[sessionIdToDelete]) return;

    const currentSessionIds = Object.keys(sessions);
    const sessionName = sessions[sessionIdToDelete].name || formatSessionIdFallback(sessionIdToDelete);

    if (currentSessionIds.length <= 1) {
        alert("Cannot delete the last chat session.");
        return;
    }
    if (!window.confirm(`Are you sure you want to delete "${sessionName}"?`)) {
        return;
    }

    // Update state first
    let nextActiveId = activeSessionId;
    setSessions(prevSessions => {
        const updatedSessions = { ...prevSessions };
        delete updatedSessions[sessionIdToDelete];

        // Determine the next active ID *before* returning the new state
        if (activeSessionId === sessionIdToDelete) {
            const remainingIds = Object.keys(updatedSessions);
            nextActiveId = remainingIds.length > 0 ? remainingIds[0] : null;
        }
        return updatedSessions;
    });

    // Update activeSessionId *after* setting the new sessions state
    // This ensures the save effect runs with the correct sessions object first
    if (activeSessionId === sessionIdToDelete) {
        setActiveSessionId(nextActiveId); // This will trigger save for activeSessionId
        setAutomationError(null);
    }
  }, [activeSessionId, sessions, setSessions, setActiveSessionId]); // Added setSessions dependency

  const handleRenameSession = useCallback((sessionId, newName) => {
      if (!sessions || !sessions[sessionId]) return;
      const trimmedName = newName.trim();
      if (!trimmedName) {
          alert("Session name cannot be empty.");
          return;
      }
      setSessions(prevSessions => ({
          ...prevSessions,
          [sessionId]: {
              ...prevSessions[sessionId],
              name: trimmedName
          }
      }));
  }, [sessions, setSessions]); // Added setSessions dependency

  const clearActiveChatHistory = useCallback(() => {
    if (!activeSessionId || !sessions || !sessions[activeSessionId]) return;
    const sessionName = sessions[activeSessionId].name || formatSessionIdFallback(activeSessionId);
    if (window.confirm(`Are you sure you want to clear the history for "${sessionName}"?`)) {
        setSessions(prevSessions => ({
            ...prevSessions,
            [activeSessionId]: {
                ...prevSessions[activeSessionId],
                history: []
            }
        }));
        setAutomationError(null);
    }
  }, [activeSessionId, sessions, setSessions]); // Added setSessions dependency

  const downloadActiveChatHistory = useCallback(() => {
     const currentSession = (sessions && activeSessionId) ? sessions[activeSessionId] : null;
    if (!currentSession || !Array.isArray(currentSession.history) || currentSession.history.length === 0) {
        alert("No history to download for the current chat.");
        return;
    }
    const json = formatHistoryForDownload(currentSession.history);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const downloadName = currentSession.name ? currentSession.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() : activeSessionId;
    link.download = `${downloadName}_history.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [activeSessionId, sessions]);


  // --- Interactive Chat Submission Handler ---
  const handleChatSubmit = useCallback(async (query, model) => {
    if (!activeSessionId || !model || isSubmitting || !sessions[activeSessionId]) return;

    console.log(`[${activeSessionId}] SUBMIT: Starting for query: "${query}"`);
    setIsSubmitting(true);
    setAutomationError(null);

    const userMessage = { sender: 'user', text: query, id: `user-${Date.now()}` };
    const botMessageId = `bot-${Date.now()}-${Math.random()}`;
    const botMessagePlaceholder = { id: botMessageId, sender: 'bot', text: '...' };

    setSessions(prevSessions => {
        // Ensure session exists before trying to update history
        if (!prevSessions[activeSessionId]) return prevSessions;
        const currentHistory = prevSessions[activeSessionId].history || [];
        const historyArray = Array.isArray(currentHistory) ? currentHistory : [];
        const newHistory = [...historyArray, userMessage, botMessagePlaceholder];
        return {
            ...prevSessions,
            [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory }
        };
    });

    let streamError = null;
    try {
      const response = await fetch(chatApiUrl, { /* ... fetch options ... */
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model }),
       });
      if (!response.ok || !response.body) { /* ... error handling ... */
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.text();
            try { const errorJson = JSON.parse(errorData); errorDetail = errorJson.detail || errorJson.error || `Server error: ${response.status}`; }
            catch(jsonError) { errorDetail = `${errorDetail}: ${errorData.substring(0, 150)}`; }
        } catch (e) { errorDetail = `Failed to read error response: ${response.status}`; }
        throw new Error(errorDetail);
       }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let firstChunk = true;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          setSessions(prevSessions => {
              if (!prevSessions[activeSessionId]) return prevSessions; // Check again inside setter
              const latestHistory = prevSessions[activeSessionId].history || [];
              const historyArray = Array.isArray(latestHistory) ? latestHistory : [];
              const botMessageIndex = historyArray.findIndex(msg => msg.id === botMessageId);
              if (botMessageIndex === -1) return prevSessions;

              const existingMessage = historyArray[botMessageIndex];
              const currentText = firstChunk ? '' : (existingMessage.text || '');
              const updatedMessage = { ...existingMessage, text: currentText + chunk };
              const updatedHistoryArray = [
                  ...historyArray.slice(0, botMessageIndex),
                  updatedMessage,
                  ...historyArray.slice(botMessageIndex + 1)
              ];
              return {
                  ...prevSessions,
                  [activeSessionId]: { ...prevSessions[activeSessionId], history: updatedHistoryArray }
              };
          });
          firstChunk = false;
        }
      }
    } catch (error) {
      console.error(`[${activeSessionId}] ERROR: During fetch/stream:`, error);
      streamError = error;
      setSessions(prevSessions => {
            if (!prevSessions[activeSessionId]) return prevSessions;
            const latestHistory = prevSessions[activeSessionId].history || [];
            const historyArray = Array.isArray(latestHistory) ? latestHistory : [];
            const botMessageIndex = historyArray.findIndex(msg => msg.id === botMessageId);
            const errorText = `⚠️ Error: ${streamError.message}`;
            let updatedHistoryArray;
            if (botMessageIndex === -1) {
                 const errorMsg = { id: botMessageId, sender: 'bot', text: errorText };
                 updatedHistoryArray = [...historyArray, errorMsg];
            } else {
                const updatedMessage = { ...historyArray[botMessageIndex], text: errorText };
                updatedHistoryArray = [ ...historyArray.slice(0, botMessageIndex), updatedMessage, ...historyArray.slice(botMessageIndex + 1) ];
            }
            return {
                ...prevSessions,
                [activeSessionId]: { ...prevSessions[activeSessionId], history: updatedHistoryArray }
            };
          });
    } finally {
       setIsSubmitting(false);
    }
  }, [activeSessionId, isSubmitting, sessions, chatApiUrl, setSessions]); // Include sessions and setSessions


  // --- Automated Conversation Handler ---
  const handleAutomateConversation = useCallback(async (jsonInputString, model) => {
    if (!activeSessionId || !model || isSubmitting || !sessions[activeSessionId]) {
        setAutomationError("Automation cannot start: Another process is running, or no session/model selected.");
        return;
    }
    // ... JSON parsing/validation ...
    let parsedInputs;
    try {
      const jsonData = JSON.parse(jsonInputString);
      if (!jsonData || !Array.isArray(jsonData.inputs) || !jsonData.inputs.every(i => typeof i === 'string')) { throw new Error('Invalid JSON format. Expected: { "inputs": [...] }'); }
      parsedInputs = jsonData.inputs;
      if (parsedInputs.length === 0) { throw new Error('JSON "inputs" array cannot be empty.'); }
    } catch (error) {
      setAutomationError(`Automation failed: Invalid JSON input. ${error.message}`);
      return;
    }

    setIsSubmitting(true);
    setAutomationError(null);

    // Clear history before starting
    setSessions(prevSessions => ({
        ...prevSessions,
        [activeSessionId]: { ...prevSessions[activeSessionId], history: [] }
    }));

    try {
      const response = await fetch(automateApiUrl, { /* ... fetch options ... */
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: parsedInputs, model }),
       });
      if (!response.ok) { /* ... error handling ... */
        let errorDetail = `HTTP error! status: ${response.status}`;
         try {
            const errorData = await response.text();
            try { const errorJson = JSON.parse(errorData); errorDetail = errorJson.detail || errorJson.error || `Server error: ${response.status}`; }
            catch(jsonError) { errorDetail = `${errorDetail}: ${errorData.substring(0, 200)}`; }
        } catch (e) { errorDetail = `Failed to read error response: ${response.status}`; }
        throw new Error(errorDetail);
       }
      const conversationResult = await response.json();
      if (!Array.isArray(conversationResult) || !conversationResult.every(turn => turn && typeof turn.sender === 'string' && typeof turn.text === 'string')) {
          throw new Error("Received invalid conversation structure from backend.");
      }
       const historyWithIds = conversationResult.map((turn, index) => ({ ...turn, id: `${turn.sender}-${Date.now()}-${index}` }));
      setSessions(prevSessions => ({
          ...prevSessions,
          [activeSessionId]: { ...prevSessions[activeSessionId], history: historyWithIds }
      }));
    } catch (error) {
        console.error(`[${activeSessionId}] AUTOMATE ERROR:`, error);
        setAutomationError(`Automation failed: ${error.message}`);
        setSessions(prevSessions => {
            if (!prevSessions[activeSessionId]) return prevSessions;
            const currentHistory = prevSessions[activeSessionId].history || [];
            const errorMsg = { id: `error-${Date.now()}`, sender: 'bot', text: `⚠️ Automation Error: ${error.message}` };
            return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: [...currentHistory, errorMsg] } };
        });
    } finally {
      setIsSubmitting(false);
    }
  }, [activeSessionId, isSubmitting, sessions, automateApiUrl, setSessions]); // Include sessions and setSessions


  // Derive active chat history
  const activeChatHistory = useMemo(() => {
      if (!isInitialized || !sessions || !activeSessionId || !sessions[activeSessionId]) {
          return []; // Return empty array if not initialized or session doesn't exist
      }
      const history = sessions[activeSessionId].history;
      return Array.isArray(history) ? history : [];
  }, [activeSessionId, sessions, isInitialized]); // Depend on initialization status


  return {
    sessions,
    activeSessionId,
    activeChatHistory,
    isSubmitting,
    automationError,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    handleRenameSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit,
    handleAutomateConversation,
    isInitialized, // Expose initialization status if needed by parent
  };
}
