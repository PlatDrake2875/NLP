// HIA/frontend/src/components/Header.jsx
import React from 'react';

// Helper function to format session names (can be moved to a utils file)
const formatSessionName = (sessionId) => {
    if (!sessionId) return "Chat";
    // Replace hyphens, capitalize first letter
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};

export function Header({
    activeSessionId,
    chatHistory, // <-- Accept chatHistory prop
    clearChatHistory,
    downloadChatHistory,
    disabled, // General disabled state for actions (includes loading/error/submitting)
    // --- Receive theme props ---
    isDarkMode,
    toggleTheme,
    // --- Receive model props ---
    availableModels,
    selectedModel,
    onModelChange,
    modelsLoading,
    modelsError,
 }) {
  const title = activeSessionId ? formatSessionName(activeSessionId) : "Accessibility Navigator Chat";
  // Determine if action buttons should be disabled based on history length
  // Ensure chatHistory is an array before checking length
  const isHistoryEmpty = !Array.isArray(chatHistory) || chatHistory.length === 0;

  return (
    <header className="chat-header">
      {/* Left side: Title */}
      <div className="header-title">
        <h1>{title}</h1>
      </div>

      {/* Right side: Actions and Model Selector */}
      <div className="header-controls">
        {/* --- Model Selector --- */}
        <div className="model-selector-container">
          <label htmlFor="model-select" className="sr-only">Select Model:</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={onModelChange}
            disabled={modelsLoading || availableModels.length === 0 || !!modelsError}
            className="model-select"
            aria-label="Select AI model"
          >
            {modelsLoading && <option value="">Loading models...</option>}
            {modelsError && <option value="">Error loading models</option>}
            {!modelsLoading && !modelsError && availableModels.length === 0 && (
              <option value="">No models found</option>
            )}
            {!modelsLoading && !modelsError && availableModels.map(modelName => (
              <option key={modelName} value={modelName}>
                {/* Display only the base model name before the colon if present */}
                {modelName.split(':')[0]}
              </option>
            ))}
          </select>
          {/* Optional: Display error message near selector */}
          {modelsError && <span className="model-error-indicator" title={modelsError}>⚠️</span>}
        </div>

        {/* --- Action Buttons --- */}
        <div className="header-actions">
          <button
              onClick={toggleTheme}
              className="header-button theme-toggle-btn"
              aria-label={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}
              title={`Switch to ${isDarkMode ? 'Light' : 'Dark'} Mode`} // Add title
          >
              {isDarkMode ? '☀️' : '🌙'} {/* Use icons for brevity */}
          </button>
          <button
            onClick={clearChatHistory}
            className="header-button clear-history-btn"
            // Disable if generally disabled OR if history is empty
            disabled={disabled || isHistoryEmpty}
            aria-label="Clear current chat history"
            title="Clear Chat" // Add title for clarity
          >
            {/* Use an icon (example using text, replace with SVG/icon font if desired) */}
            🗑️
          </button>
          <button
            onClick={downloadChatHistory}
            className="header-button download-history-btn"
             // Disable if generally disabled OR if history is empty
            disabled={disabled || isHistoryEmpty}
            aria-label="Download current chat history"
            title="Download Chat" // Add title
          >
            {/* Use an icon */}
            💾
          </button>
        </div>
      </div>
    </header>
  );
}
