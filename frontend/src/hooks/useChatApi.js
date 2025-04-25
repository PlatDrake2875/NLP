// HIA/frontend/src/hooks/useChatApi.js
import { useState, useCallback } from 'react';

const CHAT_API_URL_SUFFIX = '/api/chat';
const AUTOMATE_API_URL_SUFFIX = '/api/automate_conversation';

/**
 * Handles API interactions for chat and automation.
 * @param {string} apiBaseUrl - The base URL for the API.
 * @param {string | null} activeSessionId - The currently active session ID.
 * @param {React.Dispatch<React.SetStateAction<Record<string, { name: string | null, history: Array<any> }>>>} setSessions - State setter for the sessions object.
 * @returns {{
 * isSubmitting: boolean,
 * automationError: string | null,
 * handleChatSubmit: (query: string, model: string) => Promise<void>,
 * handleAutomateConversation: (jsonInputString: string, model: string) => Promise<void>,
 * setAutomationError: React.Dispatch<React.SetStateAction<string | null>> // Expose setter
 * }}
 */
export function useChatApi(apiBaseUrl, activeSessionId, setSessions) {
  const chatApiUrl = `${apiBaseUrl}${CHAT_API_URL_SUFFIX}`;
  const automateApiUrl = `${apiBaseUrl}${AUTOMATE_API_URL_SUFFIX}`;

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [automationError, setAutomationError] = useState(null);

  // --- Interactive Chat Submission ---
  const handleChatSubmit = useCallback(async (query, model) => {
    if (!activeSessionId || !model || isSubmitting) {
      console.warn("API Hook: Chat submission prevented.", { activeSessionId, model, isSubmitting });
      return;
    }
    console.log(`API Hook: [${activeSessionId}] SUBMIT starting.`);
    setIsSubmitting(true);
    setAutomationError(null); // Clear any previous automation errors

    const userMessage = { sender: 'user', text: query, id: `user-${Date.now()}` };
    const botMessageId = `bot-${Date.now()}-${Math.random()}`;
    const botMessagePlaceholder = { id: botMessageId, sender: 'bot', text: '...' };

    // Update state optimistically
    setSessions(prevSessions => {
      if (!prevSessions[activeSessionId]) return prevSessions; // Session might have been deleted
      const currentHistory = prevSessions[activeSessionId].history || [];
      const newHistory = [...currentHistory, userMessage, botMessagePlaceholder];
      return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
    });

    let streamError = null;
    try {
      const response = await fetch(chatApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model }),
      });

      if (!response.ok || !response.body) {
        // Simplified error extraction for brevity
        let errorDetail = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.text(); errorDetail = `${errorDetail}: ${errorData.substring(0,150)}`;} catch(e){}
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
            if (!prevSessions[activeSessionId]) return prevSessions;
            const history = prevSessions[activeSessionId].history || [];
            const botIndex = history.findIndex(msg => msg.id === botMessageId);
            if (botIndex === -1) return prevSessions;
            const currentText = firstChunk ? '' : (history[botIndex].text || '');
            const updatedMsg = { ...history[botIndex], text: currentText + chunk };
            const newHistory = [...history.slice(0, botIndex), updatedMsg, ...history.slice(botIndex + 1)];
            return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
          });
          firstChunk = false;
        }
      }
    } catch (error) {
      console.error(`API Hook: [${activeSessionId}] Chat stream error:`, error);
      streamError = error;
      // Update placeholder with error message
      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions;
        const history = prevSessions[activeSessionId].history || [];
        const botIndex = history.findIndex(msg => msg.id === botMessageId);
        const errorText = `⚠️ Error: ${streamError?.message || 'Unknown stream error'}`;
        let newHistory;
        if (botIndex === -1) {
          newHistory = [...history, { id: botMessageId, sender: 'bot', text: errorText }];
        } else {
          const updatedMsg = { ...history[botIndex], text: errorText };
          newHistory = [...history.slice(0, botIndex), updatedMsg, ...history.slice(botIndex + 1)];
        }
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
      });
    } finally {
      setIsSubmitting(false);
      console.log(`API Hook: [${activeSessionId}] SUBMIT finished.`);
    }
  }, [activeSessionId, isSubmitting, setSessions, chatApiUrl]); // Dependencies

  // --- Automated Conversation Submission ---
  const handleAutomateConversation = useCallback(async (jsonInputString, model) => {
    if (!activeSessionId || !model || isSubmitting) {
      setAutomationError("Automation cannot start: Another process is running, or no session/model selected.");
      return;
    }
    let parsedInputs;
    try {
      const jsonData = JSON.parse(jsonInputString);
      if (!jsonData || !Array.isArray(jsonData.inputs) || !jsonData.inputs.every(i => typeof i === 'string')) throw new Error('Invalid JSON format.');
      parsedInputs = jsonData.inputs;
      if (parsedInputs.length === 0) throw new Error('JSON "inputs" array cannot be empty.');
    } catch (error) {
      setAutomationError(`Automation failed: Invalid JSON input. ${error.message}`);
      return;
    }

    console.log(`API Hook: [${activeSessionId}] AUTOMATE starting.`);
    setIsSubmitting(true);
    setAutomationError(null);

    // Clear history optimistically before API call
    setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions;
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: [] } };
    });

    try {
      const response = await fetch(automateApiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: parsedInputs, model }),
      });

      if (!response.ok) {
        // Simplified error extraction
        let errorDetail = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.text(); errorDetail = `${errorDetail}: ${errorData.substring(0,200)}`;} catch(e){}
        throw new Error(errorDetail);
      }

      const conversationResult = await response.json();
      if (!Array.isArray(conversationResult)) throw new Error("Invalid response structure from backend.");

      const historyWithIds = conversationResult.map((turn, index) => ({ ...turn, id: `${turn.sender}-${Date.now()}-${index}` }));

      // Update state with the full history from backend
      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions; // Check if session still exists
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: historyWithIds } };
      });
      console.log(`API Hook: [${activeSessionId}] AUTOMATE success.`);

    } catch (error) {
      console.error(`API Hook: [${activeSessionId}] AUTOMATE error:`, error);
      setAutomationError(`Automation failed: ${error.message}`);
      // Add error message to history
      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions;
        const currentHistory = prevSessions[activeSessionId].history || []; // Should be empty from optimistic update
        const errorMsg = { id: `error-${Date.now()}`, sender: 'bot', text: `⚠️ Automation Error: ${error.message}` };
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: [...currentHistory, errorMsg] } };
      });
    } finally {
      setIsSubmitting(false);
      console.log(`API Hook: [${activeSessionId}] AUTOMATE finished.`);
    }
  }, [activeSessionId, isSubmitting, setSessions, automateApiUrl]); // Dependencies

  return {
    isSubmitting,
    automationError,
    handleChatSubmit,
    handleAutomateConversation,
    setAutomationError, // Expose setter if needed externally
  };
}
