// HIA/frontend/src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
// Import custom hooks
import { useTheme } from './hooks/useTheme';
import { useChatSessions } from './hooks/useChatSessions';
// Import components
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';

// Define the backend API URL (consider moving to an environment variable)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'; // Use Vite env var or default

function App() {
  // Use custom hooks to manage state and logic
  const { isDarkMode, toggleTheme } = useTheme(); // Manages theme state and body class
  const {
    sessions,
    activeSessionId,
    activeChatHistory,
    isSubmitting,
    automationError, // Get automation error state
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit: originalHandleChatSubmit, // Interactive submit
    handleAutomateConversation, // Automated submit
  } = useChatSessions(API_BASE_URL); // Pass the base API URL

  // --- State for Model Selection ---
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState(null);

  // --- Fetch Available Models ---
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail;
        } catch(e) { /* Ignore */ }
        throw new Error(errorDetail);
      }
      const modelsData = await response.json();
      const modelNames = modelsData.map(m => m.name).sort();
      setAvailableModels(modelNames);

      // --- Logic to set selectedModel ---
      // 1. Try to persist selection if it's still valid
      const storedModel = localStorage.getItem('selectedModel');
      if (storedModel && modelNames.includes(storedModel)) {
          setSelectedModel(storedModel);
      }
      // 2. If no valid stored model, or current selection is invalid, pick the first available
      else if (modelNames.length > 0 && (!selectedModel || !modelNames.includes(selectedModel))) {
        setSelectedModel(modelNames[0]);
        localStorage.setItem('selectedModel', modelNames[0]); // Store the default
      }
      // 3. If no models are available
      else if (modelNames.length === 0) {
        setSelectedModel('');
        localStorage.removeItem('selectedModel');
      }
      // 4. If a model was already selected and is still valid, keep it (no change needed)

    } catch (error) {
      console.error('Error fetching models:', error);
      setModelsError(`Failed to load models: ${error.message}`);
      setAvailableModels([]);
      setSelectedModel('');
      localStorage.removeItem('selectedModel');
    } finally {
      setModelsLoading(false);
    }
  }, [API_BASE_URL, selectedModel]); // Keep selectedModel dependency here to re-validate on model list change

  // Fetch models on initial mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]); // fetchModels is memoized by useCallback

  // --- Handle Model Selection Change ---
  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
    // Persist selection in localStorage
    if (newModel) {
        localStorage.setItem('selectedModel', newModel);
    } else {
        localStorage.removeItem('selectedModel');
    }
  };

  // --- Modified Chat Submit Handler (for interactive chat) ---
  const handleChatSubmitWithModel = useCallback(async (query) => {
    if (!selectedModel) {
      console.error("No model selected for chat submission.");
      // Optionally display an error to the user in the chat window
      // This might require modifying useChatSessions hook
      return;
    }
    await originalHandleChatSubmit(query, selectedModel);
  }, [selectedModel, originalHandleChatSubmit]);

  return (
    <div className="App">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        selectedModel={selectedModel} // Pass selected model
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onAutomateConversation={handleAutomateConversation} // Pass automation handler
        isSubmitting={isSubmitting} // Pass submitting state
        automationError={automationError} // Pass automation error
      />
      <ChatInterface
        // Use a key that changes when the session OR the history length changes
        // This helps ensure ChatHistory scrolls correctly after automation replaces history
        key={`${activeSessionId || 'no-session'}-${activeChatHistory.length}`}
        activeSessionId={activeSessionId}
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmitWithModel} // Interactive submit
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
        isSubmitting={isSubmitting}
        // Pass theme state and toggle function
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
        // --- Pass model selection props ---
        availableModels={availableModels}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
      />
    </div>
  );
}

export default App;
