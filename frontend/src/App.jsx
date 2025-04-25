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
    activeChatHistory, // This is the history we need to pass
    isSubmitting, // Get submitting state
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit: originalHandleChatSubmit, // Rename original submit handler
  } = useChatSessions(`${API_BASE_URL}/api/chat`); // Pass the specific chat API URL

  // --- State for Model Selection ---
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(''); // Store the name of the selected model
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState(null);

  // --- Fetch Available Models ---
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) {
        // Attempt to parse error detail from backend
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail;
        } catch(e) { /* Ignore if response isn't JSON */ }
        throw new Error(errorDetail);
      }
      const modelsData = await response.json();
      // Extract just the names, sort them, and update state
      const modelNames = modelsData.map(m => m.name).sort();
      setAvailableModels(modelNames);
      // Set the first model as default if none is selected or current selection is invalid
      if (modelNames.length > 0 && (!selectedModel || !modelNames.includes(selectedModel))) {
        setSelectedModel(modelNames[0]);
      } else if (modelNames.length === 0) {
        setSelectedModel(''); // No models available
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      setModelsError(`Failed to load models: ${error.message}`);
      setAvailableModels([]);
      setSelectedModel('');
    } finally {
      setModelsLoading(false);
    }
  // Only re-run fetchModels if the base URL changes (which it shouldn't typically)
  // selectedModel dependency removed to avoid re-fetching when selecting a model
  }, [API_BASE_URL]);

  // Fetch models on initial mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]); // fetchModels is memoized by useCallback

  // --- Handle Model Selection Change ---
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  // --- Modified Chat Submit Handler ---
  // This wrapper function includes the selected model when calling the original handler
  const handleChatSubmitWithModel = useCallback(async (query) => {
    if (!selectedModel) {
      // Handle case where no model is selected (e.g., show an error)
      console.error("No model selected for chat submission.");
      // Optionally, update chat history with an error message
      // This might require modifying useChatSessions to accept direct message additions
      return;
    }
    // Call the original submit handler from the hook, passing the model
    await originalHandleChatSubmit(query, selectedModel);
  }, [selectedModel, originalHandleChatSubmit]);

  return (
    <div className="App">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />
      <ChatInterface
        key={activeSessionId || 'no-session'}
        activeSessionId={activeSessionId}
        chatHistory={activeChatHistory} // Pass activeChatHistory here
        // Use the new submit handler that includes the model
        onSubmit={handleChatSubmitWithModel}
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
        isSubmitting={isSubmitting} // Pass submitting state
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
