// HIA/frontend/src/components/Sidebar.jsx
import React, { useState, useRef, useEffect } from 'react';
import styles from './Sidebar.module.css'; // Import CSS Module

// Helper function (keep local or move to utils)
const formatSessionIdFallback = (sessionId) => {
    if (!sessionId) return "Chat";
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};

export function Sidebar({
    sessions,
    activeSessionId,
    selectedModel,
    onNewChat,
    onSelectSession,
    onDeleteSession,
    onRenameSession,
    onAutomateConversation,
    isSubmitting,
    automationError,
    isInitialized // Receive initialization status
}) {
  // Only get sessionIds once initialized and sessions object is available
  const sessionIds = isInitialized ? Object.keys(sessions) : [];
  const [automationJson, setAutomationJson] = useState('{\n  "inputs": [\n    "Hello!",\n    "How are you?"\n  ]\n}');
  const fileInputRef = useRef(null);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingValue, setEditingValue] = useState('');
  const editInputRef = useRef(null);

  const handleEditClick = (e, sessionId) => {
    e.stopPropagation();
    const currentName = sessions[sessionId]?.name || formatSessionIdFallback(sessionId);
    setEditingSessionId(sessionId);
    setEditingValue(currentName);
  };

  const handleSaveEdit = () => {
    if (editingSessionId && editingValue.trim()) {
      onRenameSession(editingSessionId, editingValue.trim());
    }
    setEditingSessionId(null);
    setEditingValue('');
  };

  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingValue('');
  };

  const handleInputChange = (e) => {
    setEditingValue(e.target.value);
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter') handleSaveEdit();
    else if (e.key === 'Escape') handleCancelEdit();
  };

  useEffect(() => {
    if (editingSessionId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingSessionId]);

  const handleAutomationSubmit = () => {
      if (!selectedModel) { alert("Please select a model first."); return; }
      if (onAutomateConversation) onAutomateConversation(automationJson, selectedModel);
  };

  const handleUploadClick = () => { fileInputRef.current?.click(); };

  const handleFileChange = (event) => {
    // ... (file reading logic remains the same) ...
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.type !== 'application/json') {
        alert('Please select a valid JSON file (.json)');
        if(fileInputRef.current) fileInputRef.current.value = '';
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result;
        if (typeof text === 'string') {
          JSON.parse(text); // Validate
          setAutomationJson(text);
        } else { throw new Error('Failed to read file content as text.'); }
      } catch (error) {
        alert(`Error reading file: ${error.message}. Please ensure it's valid JSON.`);
      } finally { if(fileInputRef.current) fileInputRef.current.value = ''; }
    };
    reader.onerror = () => { alert('Error reading file.'); if(fileInputRef.current) fileInputRef.current.value = ''; };
    reader.readAsText(file);
  };

  return (
    // Apply styles using the imported object
    <div className={styles.sidebar}>
      <button onClick={onNewChat} className={styles.newChatButton}>
        + New Chat
      </button>
      <nav className={styles.conversationMenu}>
        <h2>Conversations</h2>
        <ul>
          {/* Display loading or no chats message */}
          {!isInitialized && <li className={styles.noSessions}>Loading...</li>}
          {isInitialized && sessionIds.length === 0 && <li className={styles.noSessions}>No chats yet.</li>}

          {isInitialized && sessionIds.map((sessionId) => {
            const session = sessions[sessionId];
            const displayName = session?.name || formatSessionIdFallback(sessionId);
            const isEditing = editingSessionId === sessionId;
            const isActive = sessionId === activeSessionId;

            // Combine classes conditionally
            const itemClasses = `${styles.conversationItem} ${isActive ? styles.active : ''} ${isEditing ? styles.editing : ''}`;

            return (
              <li key={sessionId} className={itemClasses}>
                {isEditing ? (
                  <input
                    ref={editInputRef}
                    type="text"
                    value={editingValue}
                    onChange={handleInputChange}
                    onKeyDown={handleInputKeyDown}
                    onBlur={handleSaveEdit}
                    className={styles.sessionEditInput} // Use module style
                    aria-label={`Rename chat ${displayName}`}
                  />
                ) : (
                  <>
                    <button
                      className={styles.sessionSelectButton} // Use module style
                      onClick={() => onSelectSession(sessionId)}
                      aria-current={isActive ? 'page' : undefined}
                      title={displayName}
                    >
                      {displayName}
                    </button>
                    <button
                        onClick={(e) => handleEditClick(e, sessionId)}
                        className={styles.editSessionButton} // Use module style
                        aria-label={`Rename ${displayName}`}
                        title={`Rename ${displayName}`}
                    >
                        ✏️
                    </button>
                  </>
                )}
                {!isEditing && (
                     <button
                        onClick={(e) => { e.stopPropagation(); onDeleteSession(sessionId); }}
                        className={styles.deleteSessionButton} // Use module style
                        aria-label={`Delete ${displayName}`}
                        title={`Delete ${displayName}`}
                    >
                        🗑️
                    </button>
                )}
              </li>
            );
          })}
        </ul>
      </nav>

      {/* --- Automation Section --- */}
      <div className={styles.automationSection}>
        <div className={styles.automationHeader}>
            <h2>Automate</h2>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".json,application/json"
                style={{ display: 'none' }}
                aria-hidden="true"
            />
            <button
                onClick={handleUploadClick}
                className={styles.uploadJsonButton} // Use module style
                title="Upload JSON file"
                aria-label="Upload JSON file"
                disabled={isSubmitting}
            >
                 ⬆️ Upload
            </button>
        </div>
        <label htmlFor="automation-json-input" className={styles.automationLabel}>
          Paste JSON or upload file: ({`{ "inputs": ["msg1", ...] }`})
        </label>
        <textarea
          id="automation-json-input"
          className={styles.automationTextarea} // Use module style
          value={automationJson}
          onChange={(e) => setAutomationJson(e.target.value)}
          rows={6}
          placeholder='{ "inputs": ["Hello!", "Tell me a joke."] }'
          aria-label="JSON input for automated conversation"
          disabled={isSubmitting}
        />
        {automationError && (
            <p className={styles.automationErrorMessage} role="alert"> {/* Use module style */}
                Error: {automationError}
            </p>
        )}
        <button
          onClick={handleAutomationSubmit}
          className={styles.automationButton} // Use module style
          disabled={isSubmitting || !activeSessionId || !selectedModel || !isInitialized} // Also disable if not initialized
          title={!isInitialized ? "Loading..." : !activeSessionId ? "Select or create a chat first" : !selectedModel ? "Select a model first" : "Run automated conversation"}
        >
          {isSubmitting ? 'Running...' : 'Run Automation'}
        </button>
         {!selectedModel && activeSessionId && isInitialized && <p className={styles.automationWarning}>Select a model above.</p>}
         {!activeSessionId && isInitialized && <p className={styles.automationWarning}>Create or select a chat.</p>}
      </div>
    </div>
  );
}

