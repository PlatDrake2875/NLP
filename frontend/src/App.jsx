// HIA/frontend/src/App.jsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import styles from './App.module.css'; // Import App CSS Module
import './index.css'; // Keep importing global styles/variables

// Import custom hooks
import { useTheme } from './hooks/useTheme';
import { useChatSessions } from './hooks/useChatSessions';

// Import components
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';

// Define the backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'; // Ensure this is correct for your setup

function App() {
  const { isDarkMode, toggleTheme } = useTheme();
  const {
    sessions,
    activeSessionId,
    activeChatHistory,
    isInitialized,
    isSubmitting: isChatSubmitting, // Renamed to avoid conflict
    automationError,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    handleRenameSession,
    clearActiveChatHistory,
    handleChatSubmit: originalHandleChatSubmit,
    handleAutomateConversation,
  } = useChatSessions(API_BASE_URL);

  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState(null);

  // --- State for PDF Upload ---
  const [isUploadingPdf, setIsUploadingPdf] = useState(false);
  const [pdfUploadStatus, setPdfUploadStatus] = useState(null); // { success: boolean, message: string } | null


  const fetchModels = useCallback(async () => {
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
  }, [API_BASE_URL, selectedModel]);

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
    if (!selectedModel) {
        // Consider showing an alert or message to the user in the UI
        console.error("No model selected. Cannot submit chat.");
        // You could update a state here to show an error in ChatInterface
        return;
    }
    await originalHandleChatSubmit(query, selectedModel);
  }, [selectedModel, originalHandleChatSubmit]);

  const formatSessionIdFallback = (sessionId) => {
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
  };

  const activeSessionName = useMemo(() => {
    if (!isInitialized || !activeSessionId || !sessions || !sessions[activeSessionId]) {
      return "Chat";
    }
    return sessions[activeSessionId].name || formatSessionIdFallback(activeSessionId);
  }, [activeSessionId, sessions, isInitialized]);

   const downloadActiveChatHistory = useCallback(() => {
     const currentSession = (sessions && activeSessionId) ? sessions[activeSessionId] : null;
    if (!currentSession || !Array.isArray(currentSession.history) || currentSession.history.length === 0) {
        alert("No history to download for the current chat.");
        return;
    }
    const historyToDownload = currentSession.history;
    const json = JSON.stringify(historyToDownload, null, 2);
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

  // --- PDF Upload Handler ---
  const handlePdfUpload = useCallback(async (file) => {
    if (!file) return;

    setIsUploadingPdf(true);
    setPdfUploadStatus(null); // Clear previous status

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        // Note: 'Content-Type' header is automatically set by the browser for FormData
      });

      const result = await response.json(); // Assuming backend sends JSON response

      if (!response.ok) {
        // Use detail from backend if available, otherwise use a generic message
        const errorMessage = result.detail || `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }
      
      // Example success message from your backend schema:
      // { message: "Document processed...", filename: "...", chunks_added: ... }
      setPdfUploadStatus({ success: true, message: result.message || 'PDF uploaded successfully!' });
      // Clear message after a few seconds
      setTimeout(() => setPdfUploadStatus(null), 5000);

    } catch (error) {
      console.error('Error uploading PDF:', error);
      setPdfUploadStatus({ success: false, message: error.message || 'PDF upload failed.' });
      // Optionally, clear error message after some time, or let user dismiss it
      // setTimeout(() => setPdfUploadStatus(null), 7000);
    } finally {
      setIsUploadingPdf(false);
    }
  }, [API_BASE_URL]);


  return (
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
        isSubmitting={isChatSubmitting} // Pass chat submission status
        automationError={automationError}
        isInitialized={isInitialized}
        // PDF Upload props
        onUploadPdf={handlePdfUpload}
        isUploadingPdf={isUploadingPdf}
        pdfUploadStatus={pdfUploadStatus}
      />
      <ChatInterface
        key={`${activeSessionId || 'no-session'}-${activeChatHistory.length}`}
        activeSessionId={activeSessionId}
        activeSessionName={activeSessionName}
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmitWithModel}
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
        isSubmitting={isChatSubmitting} // Pass chat submission status
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
        availableModels={availableModels}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
        isInitialized={isInitialized}
      />
    </div>
  );
}

export default App;
