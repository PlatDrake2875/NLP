// HIA/frontend/src/App.jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react'; // Added useMemo here
import './App.css';
// Import custom hooks
import { useTheme } from './hooks/useTheme';
import { useChatSessions } from './hooks/useChatSessions';
// Import components
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface'; // Corrected import path if needed

// Define the backend API URL (consider moving to an environment variable)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'; // Use Vite env var or default

function App() {
  // Use custom hooks to manage state and logic
  const { isDarkMode, toggleTheme } = useTheme(); // Manages theme state and body class
  const {
    sessions, // Now contains { name, history }
    activeSessionId,
    activeChatHistory,
    isSubmitting,
    automationError,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    handleRenameSession, // Get rename handler
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

      const storedModel = localStorage.getItem('selectedModel');
      if (storedModel && modelNames.includes(storedModel)) {
          setSelectedModel(storedModel);
      }
      else if (modelNames.length > 0 && (!selectedModel || !modelNames.includes(selectedModel))) {
        setSelectedModel(modelNames[0]);
        localStorage.setItem('selectedModel', modelNames[0]);
      }
      else if (modelNames.length === 0) {
        setSelectedModel('');
        localStorage.removeItem('selectedModel');
      }

    } catch (error) {
      console.error('Error fetching models:', error);
      setModelsError(`Failed to load models: ${error.message}`);
      setAvailableModels([]);
      setSelectedModel('');
      localStorage.removeItem('selectedModel');
    } finally {
      setModelsLoading(false);
    }
  }, [API_BASE_URL, selectedModel]);

  // Fetch models on initial mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // --- Handle Model Selection Change ---
  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
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
      return;
    }
    await originalHandleChatSubmit(query, selectedModel);
  }, [selectedModel, originalHandleChatSubmit]);

  // --- Get current session name for Header ---
  // This is line 107 where the error occurred
  const activeSessionName = useMemo(() => {
      if (!activeSessionId || !sessions || !sessions[activeSessionId]) {
          return null; // Or a default like "Chat"
      }
      // Use the custom name if it exists, otherwise fallback to formatting the ID
      return sessions[activeSessionId].name || `Chat ${activeSessionId.split('-').pop()}`; // Example fallback
  }, [activeSessionId, sessions]);


  return (
    <div className="App">
      <Sidebar
        sessions={sessions} // Pass full sessions object
        activeSessionId={activeSessionId}
        selectedModel={selectedModel}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession} // Pass rename handler
        onAutomateConversation={handleAutomateConversation}
        isSubmitting={isSubmitting}
        automationError={automationError}
      />
      <ChatInterface
        key={`${activeSessionId || 'no-session'}-${activeChatHistory.length}`}
        activeSessionId={activeSessionId}
        // Pass the derived name to the ChatInterface/Header
        activeSessionName={activeSessionName}
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmitWithModel} // Interactive submit
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
        isSubmitting={isSubmitting}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
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
