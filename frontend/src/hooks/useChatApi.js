// HIA/frontend/src/hooks/useChatApi.js
import { useState, useCallback } from 'react';

// Assuming API_BASE_URL is defined elsewhere or passed in correctly
// const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
 * setAutomationError: React.Dispatch<React.SetStateAction<string | null>>
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
    const reqId = `req_${Date.now()}`; // Simple request ID
    console.log(`API Hook: [${activeSessionId} - ${reqId}] SUBMIT starting.`);
    setIsSubmitting(true);
    setAutomationError(null); 

    const userMessage = { sender: 'user', text: query, id: `user-${Date.now()}` };
    const botMessageId = `bot-${Date.now()}-${Math.random()}`;
    // Add placeholder with isLoading flag
    const botMessagePlaceholder = { id: botMessageId, sender: 'bot', text: '...', isLoading: true }; 

    // Update state optimistically
    setSessions(prevSessions => {
      if (!prevSessions[activeSessionId]) return prevSessions; 
      const currentHistory = prevSessions[activeSessionId].history || [];
      const newHistory = [...currentHistory, userMessage, botMessagePlaceholder];
      return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
    });

    let accumulatedBotResponse = ''; // Accumulate parsed tokens here
    let streamError = null;

    try {
      const response = await fetch(chatApiUrl, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' // Explicitly accept SSE
        },
        body: JSON.stringify({ query, model }),
      });

      if (!response.ok || !response.body) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.text(); errorDetail = `${errorDetail}: ${errorData.substring(0,150)}`;} catch(e){}
        throw new Error(errorDetail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = ''; // Buffer for incomplete SSE messages

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { value, done: readerDone } = await reader.read();
        if (readerDone) {
            console.log(`API Hook: [${activeSessionId} - ${reqId}] Stream finished.`);
            break; // Exit the loop
        }

        // Add the new chunk to the buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process buffer line by line for SSE messages (separated by \n\n)
        let eolIndex;
        while ((eolIndex = buffer.indexOf('\n\n')) >= 0) {
            const message = buffer.substring(0, eolIndex).trim();
            buffer = buffer.substring(eolIndex + 2); // Remove message and \n\n from buffer

            if (message.startsWith('data:')) {
                const jsonString = message.substring(5).trim(); // Remove 'data:' prefix
                if (jsonString) {
                    try {
                        const parsedData = JSON.parse(jsonString);
                        
                        if (parsedData.token) {
                            // *** CORRECTED PART ***
                            // Accumulate only the token value
                            accumulatedBotResponse += parsedData.token; 
                            
                            // Update the specific bot message with accumulated text
                            setSessions(prevSessions => {
                                if (!prevSessions[activeSessionId]) return prevSessions;
                                const history = prevSessions[activeSessionId].history || [];
                                const botIndex = history.findIndex(msg => msg.id === botMessageId);
                                if (botIndex === -1) return prevSessions; 
                                // Update existing message with accumulated text, keep isLoading true
                                const updatedMsg = { ...history[botIndex], text: accumulatedBotResponse, isLoading: true }; 
                                const newHistory = [...history.slice(0, botIndex), updatedMsg, ...history.slice(botIndex + 1)];
                                return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
                            });
                        } else if (parsedData.status === 'done') {
                            console.log(`API Hook: [${activeSessionId} - ${reqId}] Received 'done' status via SSE.`);
                            // Final update happens in 'finally' block
                        } else if (parsedData.error) {
                            console.error(`API Hook: [${activeSessionId} - ${reqId}] Error received via SSE:`, parsedData.error);
                            throw new Error(`Stream error: ${parsedData.error}`); 
                        }
                    } catch (e) {
                        console.error(`API Hook: [${activeSessionId} - ${reqId}] Error parsing JSON from SSE line:`, jsonString, e);
                    }
                }
            } else if (message) {
                 console.log(`API Hook: [${activeSessionId} - ${reqId}] Received non-data SSE line: ${message}`);
            }
        } // End while(eolIndex)
      } // End while(true)

    } catch (error) {
      console.error(`API Hook: [${activeSessionId} - ${reqId}] Chat stream error:`, error);
      streamError = error;
    } finally {
      // Final update to the bot message, setting isLoading to false
      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions;
        const history = prevSessions[activeSessionId].history || [];
        const botIndex = history.findIndex(msg => msg.id === botMessageId);
        if (botIndex === -1) return prevSessions; // Message no longer exists

        const finalMsg = { 
            ...history[botIndex], 
            // Use accumulated text if no error, otherwise show error
            text: streamError ? `⚠️ Error: ${streamError?.message || 'Unknown stream error'}` : (accumulatedBotResponse || "..."), 
            isLoading: false // Mark as not loading
        };
        const newHistory = [...history.slice(0, botIndex), finalMsg, ...history.slice(botIndex + 1)];
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: newHistory } };
      });

      setIsSubmitting(false);
      console.log(`API Hook: [${activeSessionId} - ${reqId}] SUBMIT finished.`);
    }
  }, [activeSessionId, isSubmitting, setSessions, chatApiUrl]); // Dependencies

  // --- Automated Conversation Submission ---
  // (Keep handleAutomateConversation as it was)
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
        let errorDetail = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.text(); errorDetail = `${errorDetail}: ${errorData.substring(0,200)}`;} catch(e){}
        throw new Error(errorDetail);
      }

      const conversationResult = await response.json();
      if (!Array.isArray(conversationResult)) throw new Error("Invalid response structure from backend.");

      const historyWithIds = conversationResult.map((turn, index) => ({ ...turn, id: `${turn.sender}-${Date.now()}-${index}` }));

      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions; 
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: historyWithIds } };
      });
      console.log(`API Hook: [${activeSessionId}] AUTOMATE success.`);

    } catch (error) {
      console.error(`API Hook: [${activeSessionId}] AUTOMATE error:`, error);
      setAutomationError(`Automation failed: ${error.message}`);
      setSessions(prevSessions => {
        if (!prevSessions[activeSessionId]) return prevSessions;
        const currentHistory = prevSessions[activeSessionId].history || []; 
        const errorMsg = { id: `error-${Date.now()}`, sender: 'bot', text: `⚠️ Automation Error: ${error.message}` };
        return { ...prevSessions, [activeSessionId]: { ...prevSessions[activeSessionId], history: [...currentHistory, errorMsg] } };
      });
    } finally {
      setIsSubmitting(false);
      console.log(`API Hook: [${activeSessionId}] AUTOMATE finished.`);
    }
  }, [activeSessionId, isSubmitting, setSessions, automateApiUrl]); 

  return {
    isSubmitting,
    automationError,
    handleChatSubmit,
    handleAutomateConversation,
    setAutomationError, 
  };
}
