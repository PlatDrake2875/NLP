// HIA/frontend/src/App.jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styles from './App.module.css'; // Import App CSS Module
import './index.css'; // Keep importing global styles/variables

// Import custom hooks
import { useTheme } from './hooks/useTheme';
import { useChatSessions } from './hooks/useChatSessions'; // Ensure path is correct

// Import components
import { Sidebar } from './components/Sidebar'; // Ensure path is correct
import { ChatInterface } from './components/ChatInterface'; // Ensure path is correct

// Define the backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const { isDarkMode, toggleTheme } = useTheme();
  const {
    sessions,
    activeSessionId,
    activeChatHistory,
    isInitialized, // Get initialization status
    isSubmitting,
    automationError,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    handleRenameSession,
    clearActiveChatHistory,
    // downloadActiveChatHistory, // Rebuild or move this if needed
    handleChatSubmit: originalHandleChatSubmit,
    handleAutomateConversation,
  } = useChatSessions(API_BASE_URL);

  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState(null);

  const fetchModels = useCallback(async () => {
    // ... (fetchModels logic remains the same) ...
     setModelsLoading(true);
    setModelsError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch(e) { /* Ignore */ }
        throw new Error(errorDetail);
      }
      const modelsData = await response.json();
      const modelNames = modelsData.map(m => m.name).sort();
      setAvailableModels(modelNames);

      const storedModel = localStorage.getItem('selectedModel');
      if (storedModel && modelNames.includes(storedModel)) {
          setSelectedModel(storedModel);
      } else if (modelNames.length > 0 && (!selectedModel || !modelNames.includes(selectedModel))) {
        setSelectedModel(modelNames[0]);
        localStorage.setItem('selectedModel', modelNames[0]);
      } else if (modelNames.length === 0) {
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
  }, [API_BASE_URL, selectedModel]); // Keep selectedModel dependency

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
    if (newModel) { localStorage.setItem('selectedModel', newModel); }
    else { localStorage.removeItem('selectedModel'); }
  };

  const handleChatSubmitWithModel = useCallback(async (query) => {
    if (!selectedModel) { console.error("No model selected."); return; }
    await originalHandleChatSubmit(query, selectedModel);
  }, [selectedModel, originalHandleChatSubmit]);

  // Helper function to format session names (used as fallback in Header)
  const formatSessionIdFallback = (sessionId) => {
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
  };

  const activeSessionName = useMemo(() => {
    if (!isInitialized || !activeSessionId || !sessions || !sessions[activeSessionId]) {
      return "Chat"; // Default title when loading or no session
    }
    // Use custom name or format the ID as fallback
    return sessions[activeSessionId].name || formatSessionIdFallback(activeSessionId);
  }, [activeSessionId, sessions, isInitialized]);

  // Rebuild download handler here if needed, accessing sessions/activeSessionId
   const downloadActiveChatHistory = useCallback(() => {
     const currentSession = (sessions && activeSessionId) ? sessions[activeSessionId] : null;
    if (!currentSession || !Array.isArray(currentSession.history) || currentSession.history.length === 0) {
        alert("No history to download for the current chat.");
        return;
    }
    const historyToDownload = currentSession.history; // Get the history array
    const json = JSON.stringify(historyToDownload, null, 2); // Format it
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


  return (
    // Apply the main layout class from the CSS module
    <div className={styles.App}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        selectedModel={selectedModel}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        onAutomateConversation={handleAutomateConversation}
        isSubmitting={isSubmitting}
        automationError={automationError}
        isInitialized={isInitialized} // Pass init status if Sidebar needs it
      />
      <ChatInterface
        key={`${activeSessionId || 'no-session'}-${activeChatHistory.length}`}
        activeSessionId={activeSessionId}
        activeSessionName={activeSessionName} // Pass calculated name
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmitWithModel}
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory} // Pass rebuilt handler
        isSubmitting={isSubmitting}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
        availableModels={availableModels}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
        isInitialized={isInitialized} // Pass init status
      />
    </div>
  );
}

export default App;
