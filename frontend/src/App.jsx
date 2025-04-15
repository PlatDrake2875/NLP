// HIA/frontend/src/App.jsx
import React from 'react';
import './App.css';
// Import custom hooks
import { useTheme } from './hooks/useTheme';
import { useChatSessions } from './hooks/useChatSessions';
// Import components
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';

function App() {
  // Use custom hooks to manage state and logic
  const { isDarkMode, toggleTheme } = useTheme(); // Manages theme state and body class
  const {
    sessions,
    activeSessionId,
    activeChatHistory,
    handleNewChat,
    handleSelectSession,
    handleDeleteSession,
    clearActiveChatHistory,
    downloadActiveChatHistory,
    handleChatSubmit,
  } = useChatSessions(); // Manages all session and chat logic

  return (
    // The useEffect in useTheme handles the body class now
    <div className="App">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />
      <ChatInterface
        // Use a key derived from activeSessionId to ensure ChatInterface
        // remounts or resets state appropriately when the session changes.
        // Using just activeSessionId is fine if it's null initially.
        key={activeSessionId || 'no-session'}
        activeSessionId={activeSessionId}
        chatHistory={activeChatHistory}
        onSubmit={handleChatSubmit}
        onClearHistory={clearActiveChatHistory}
        onDownloadHistory={downloadActiveChatHistory}
        // Pass theme state and toggle function
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />
    </div>
  );
}

export default App;