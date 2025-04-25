// HIA/frontend/src/hooks/useChatSessions.js
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useLocalStorage } from './useLocalStorage'; // Keep for activeSessionId

// Helper function to generate a new session ID
const generateNewSessionId = (counter) => `new-chat-${counter}`;

// Helper function to format history for download
const formatHistoryForDownload = (history) => {
    return JSON.stringify(Array.isArray(history) ? history : [], null, 2);
};

// API URLs (adjust if needed)
const CHAT_API_URL_SUFFIX = '/api/chat';
const AUTOMATE_API_URL_SUFFIX = '/api/automate_conversation';

export function useChatSessions(apiBaseUrl) {
  // Construct full API URLs
  const chatApiUrl = `${apiBaseUrl}${CHAT_API_URL_SUFFIX}`;
  const automateApiUrl = `${apiBaseUrl}${AUTOMATE_API_URL_SUFFIX}`;

  // --- Use useState instead of useLocalStorage for sessions ---
  const [sessions, setSessions] = useState({});
  // --- Keep useLocalStorage for activeSessionId ---
  const [activeSessionId, setActiveSessionId] = useLocalStorage('activeSessionId', null);
  const sessionCounterRef = useRef(1);
  const [isSubmitting, setIsSubmitting] = useState(false); // Used for both interactive and automated submits
  const [automationError, setAutomationError] = useState(null); // State for automation errors

  // --- Load initial sessions from localStorage ONCE on mount ---
  useEffect(() => {
      try {
          const storedSessions = window.localStorage.getItem('chatSessions');
          if (storedSessions) {
              const parsedSessions = JSON.parse(storedSessions);
              if (typeof parsedSessions === 'object' && parsedSessions !== null) {
                  setSessions(parsedSessions);
                  console.log("[Init] Loaded sessions from localStorage:", parsedSessions);
              } else {
                  console.warn("[Init] Invalid data found in localStorage for 'chatSessions'. Initializing empty.");
                  setSessions({});
              }
          } else {
               console.log("[Init] No sessions found in localStorage. Initializing empty.");
               setSessions({});
          }
      } catch (error) {
          console.error("[Init] Error reading sessions from localStorage:", error);
          setSessions({});
      }
  }, []); // Empty dependency array

  // --- Effect to save sessions to localStorage whenever sessions state changes ---
  useEffect(() => {
      if (Object.keys(sessions).length > 0 || localStorage.getItem('chatSessions')) {
          try {
             window.localStorage.setItem('chatSessions', JSON.stringify(sessions));
          } catch (error) {
             console.error("[Save] Error saving sessions to localStorage:", error);
          }
      }
  }, [sessions]);

  // Initialize session counter and ensure an active session exists
  useEffect(() => {
    if (sessions === null || sessions === undefined) return;

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

      if (!activeSessionId || !sessions[activeSessionId]) {
        const firstId = existingSessionIds[0];
        console.log(`[Init Check] Invalid/missing activeSessionId ('${activeSessionId}'). Setting to first: ${firstId}`);
        setActiveSessionId(firstId);
        sessionCreatedOrSelected = true;
      } else {
         sessionCreatedOrSelected = true;
      }
    } else {
      console.log("[Init Check] No sessions exist. Creating initial session.");
      const firstSessionId = generateNewSessionId(sessionCounterRef.current);
      sessionCounterRef.current++;
      setSessions({ [firstSessionId]: [] });
      setActiveSessionId(firstSessionId);
      sessionCreatedOrSelected = true;
    }
  }, [sessions, activeSessionId, setActiveSessionId]);


  // --- Session Management Handlers ---

  const handleNewChat = useCallback(() => {
    const newSessionId = generateNewSessionId(sessionCounterRef.current);
    sessionCounterRef.current++;
    setSessions(prevSessions => ({
      ...prevSessions,
      [newSessionId]: [],
    }));
    setActiveSessionId(newSessionId);
    setAutomationError(null); // Clear errors on new chat
  }, [setSessions, setActiveSessionId]);

  const handleSelectSession = useCallback((sessionId) => {
    if (sessions && sessions[sessionId]) {
        setActiveSessionId(sessionId);
        setAutomationError(null); // Clear errors on selecting chat
    } else {
        console.warn(`Attempted to select non-existent session: ${sessionId}`);
        if (sessions) {
            const firstSessionId = Object.keys(sessions)[0];
            if (firstSessionId) {
                setActiveSessionId(firstSessionId);
            }
        }
    }
  }, [setActiveSessionId, sessions]);

  const handleDeleteSession = useCallback((sessionIdToDelete) => {
    if (!sessions) return;
    const currentSessionIds = Object.keys(sessions);
    if (currentSessionIds.length <= 1) {
        console.warn("Cannot delete the last chat session.");
        // Optionally, clear the history instead of preventing deletion
        // if (window.confirm(`Clear history for "${formatSessionName(sessionIdToDelete)}"? This is the last session.`)) {
        //     setSessions(prev => ({ ...prev, [sessionIdToDelete]: [] }));
        // }
        alert("Cannot delete the last chat session."); // Use alert or a more integrated message
        return;
    }

    // Confirm deletion
    if (!window.confirm(`Are you sure you want to delete "${formatSessionName(sessionIdToDelete)}"?`)) {
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
      setAutomationError(null); // Clear errors if active session is deleted
    }
  }, [activeSessionId, sessions, setSessions, setActiveSessionId]);

  const clearActiveChatHistory = useCallback(() => {
    if (!activeSessionId || !sessions || !sessions[activeSessionId]) return;
    if (window.confirm(`Are you sure you want to clear the history for "${formatSessionName(activeSessionId)}"?`)) {
        setSessions(prevSessions => {
            const updatedSessions = { ...prevSessions };
            updatedSessions[activeSessionId] = [];
            return updatedSessions;
        });
        setAutomationError(null); // Clear errors on clear history
    }
  }, [activeSessionId, sessions, setSessions]);

  const downloadActiveChatHistory = useCallback(() => {
     const currentHistory = (sessions && activeSessionId) ? sessions[activeSessionId] : null;
    if (!activeSessionId || !Array.isArray(currentHistory) || currentHistory.length === 0) {
        alert("No history to download for the current chat.");
        return;
    }

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

  // --- Interactive Chat Submission Handler ---
  const handleChatSubmit = useCallback(async (query, model) => {
    if (!activeSessionId || !model || isSubmitting) {
        console.warn("Chat submission prevented:", { activeSessionId, model, isSubmitting });
        return;
    }

    console.log(`[${activeSessionId}] SUBMIT: Starting for query: "${query}"`);
    setIsSubmitting(true);
    setAutomationError(null); // Clear previous errors

    const userMessage = { sender: 'user', text: query, id: `user-${Date.now()}` }; // Add unique ID
    const botMessageId = `bot-${Date.now()}-${Math.random()}`;
    const botMessagePlaceholder = { id: botMessageId, sender: 'bot', text: '...' };

    // Add User Message & Placeholder
    setSessions(prevSessions => {
        const currentHistory = prevSessions[activeSessionId] || [];
        const historyArray = Array.isArray(currentHistory) ? currentHistory : [];
        const newHistory = [...historyArray, userMessage, botMessagePlaceholder];
        return { ...prevSessions, [activeSessionId]: newHistory };
    });

    let streamError = null;

    try {
      console.log(`[${activeSessionId}] FETCH: Starting fetch to ${chatApiUrl}`);
      const response = await fetch(chatApiUrl, {
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

          setSessions(prevSessions => {
              const latestHistory = prevSessions[activeSessionId] || [];
              const historyArray = Array.isArray(latestHistory) ? latestHistory : [];
              const botMessageIndex = historyArray.findIndex(msg => msg.id === botMessageId);

              if (botMessageIndex === -1) {
                  console.warn(`[${activeSessionId}] STREAM WARN: Bot message ID ${botMessageId} not found.`);
                  return prevSessions; // Skip update if placeholder is gone
              }

              const existingMessage = historyArray[botMessageIndex];
              // Ensure text is treated as string, handle potential initial undefined/null
              const currentText = existingMessage.text === '...' ? '' : (existingMessage.text || '');
              const updatedMessage = {
                  ...existingMessage,
                  text: firstChunk ? chunk : currentText + chunk
              };

              const updatedHistoryArray = [
                  ...historyArray.slice(0, botMessageIndex),
                  updatedMessage,
                  ...historyArray.slice(botMessageIndex + 1)
              ];

              return { ...prevSessions, [activeSessionId]: updatedHistoryArray };
          });
          firstChunk = false;
        }
      }
      console.log(`[${activeSessionId}] STREAM: Finished successfully.`);

    } catch (error) {
      console.error(`[${activeSessionId}] ERROR: During fetch/stream:`, error);
      streamError = error;

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
            return { ...prevSessions, [activeSessionId]: updatedHistoryArray };
          });

    } finally {
       console.log(`[${activeSessionId}] SUBMIT: Finishing submission.`);
       setIsSubmitting(false);
    }
  }, [activeSessionId, isSubmitting, setSessions, chatApiUrl]);

  // --- NEW: Automated Conversation Handler ---
  const handleAutomateConversation = useCallback(async (jsonInputString, model) => {
    if (!activeSessionId || !model || isSubmitting) {
      console.warn("Automation prevented:", { activeSessionId, model, isSubmitting });
      setAutomationError("Automation cannot start: Another process is running, or no session/model selected.");
      return;
    }
    if (!jsonInputString) {
        setAutomationError("Automation cannot start: JSON input is empty.");
        return;
    }

    let parsedInputs;
    try {
      const jsonData = JSON.parse(jsonInputString);
      // Validate the structure { "inputs": [...] }
      if (!jsonData || !Array.isArray(jsonData.inputs) || !jsonData.inputs.every(i => typeof i === 'string')) {
          throw new Error('Invalid JSON format. Expected: { "inputs": ["message1", "message2", ...] }');
      }
      parsedInputs = jsonData.inputs;
      if (parsedInputs.length === 0) {
          throw new Error('JSON "inputs" array cannot be empty.');
      }
    } catch (error) {
      console.error("Error parsing automation JSON:", error);
      setAutomationError(`Automation failed: Invalid JSON input. ${error.message}`);
      return;
    }

    console.log(`[${activeSessionId}] AUTOMATE: Starting for ${parsedInputs.length} inputs with model ${model}.`);
    setIsSubmitting(true);
    setAutomationError(null); // Clear previous errors

    // Clear current history before starting automation
    setSessions(prevSessions => ({
        ...prevSessions,
        [activeSessionId]: []
    }));

    try {
      console.log(`[${activeSessionId}] AUTOMATE: Sending request to ${automateApiUrl}`);
      const response = await fetch(automateApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: parsedInputs, model }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP error! status: ${response.status}`;
         try {
            const errorData = await response.text(); // Use text() first
            try { const errorJson = JSON.parse(errorData); errorDetail = errorJson.detail || errorJson.error || `Server error: ${response.status}`; }
            catch(jsonError) { errorDetail = `${errorDetail}: ${errorData.substring(0, 200)}`; } // Show text snippet if not JSON
        } catch (e) { errorDetail = `Failed to read error response: ${response.status}`; }
        throw new Error(errorDetail);
      }

      const conversationResult = await response.json(); // Backend returns List[ConversationTurn]
      console.log(`[${activeSessionId}] AUTOMATE: Received ${conversationResult.length} turns from backend.`);

      // Validate backend response structure (basic check)
      if (!Array.isArray(conversationResult) || !conversationResult.every(turn => turn && typeof turn.sender === 'string' && typeof turn.text === 'string')) {
          throw new Error("Received invalid conversation structure from backend.");
      }

      // Add unique IDs to each message for React keys
       const historyWithIds = conversationResult.map((turn, index) => ({
            ...turn,
            id: `${turn.sender}-${Date.now()}-${index}` // Simple unique ID generation
       }));


      // Update the session state with the full conversation
      setSessions(prevSessions => ({
          ...prevSessions,
          [activeSessionId]: historyWithIds // Replace history with the automated result
      }));

      console.log(`[${activeSessionId}] AUTOMATE: Successfully updated history.`);

    } catch (error) {
        console.error(`[${activeSessionId}] AUTOMATE ERROR:`, error);
        setAutomationError(`Automation failed: ${error.message}`);
        // Optionally add error message to chat history as well
        setSessions(prevSessions => {
            const currentHistory = prevSessions[activeSessionId] || [];
            const errorMsg = { id: `error-${Date.now()}`, sender: 'bot', text: `⚠️ Automation Error: ${error.message}` };
            return { ...prevSessions, [activeSessionId]: [...currentHistory, errorMsg] };
        });
    } finally {
      console.log(`[${activeSessionId}] AUTOMATE: Finishing automation process.`);
      setIsSubmitting(false);
    }

  }, [activeSessionId, isSubmitting, setSessions, automateApiUrl]); // Add dependencies


  // Derive active chat history, ensuring it's always an array
  const activeChatHistory = useMemo(() => {
      if (!sessions || !activeSessionId || !sessions[activeSessionId]) {
          return [];
      }
      const history = sessions[activeSessionId];
      return Array.isArray(history) ? history : [];
  }, [activeSessionId, sessions]);


  return {
    sessions,
    activeSessionId,
    activeChatHistory,
    isSubmitting,
    automationError, // Expose automation error state
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit, // Interactive chat submit
    handleAutomateConversation, // Automated chat submit
  };
}
