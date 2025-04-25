import React, { useState, useRef, useEffect } from 'react'; // Import useEffect

// Helper function to format session ID as a fallback name
const formatSessionIdFallback = (sessionId) => {
    if (!sessionId) return "Chat";
    // Replace hyphens, capitalize first letter
    return sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase());
};

export function Sidebar({
    sessions, // Sessions now contain { name, history }
    activeSessionId,
    selectedModel,
    onNewChat,
    onSelectSession,
    onDeleteSession,
    onRenameSession, // New handler prop
    onAutomateConversation,
    isSubmitting,
    automationError
}) {
  const sessionIds = Object.keys(sessions);
  const [automationJson, setAutomationJson] = useState('{\n  "inputs": [\n    "Hello!",\n    "How are you?"\n  ]\n}');
  const fileInputRef = useRef(null);

  // --- State for Renaming ---
  const [editingSessionId, setEditingSessionId] = useState(null); // Track which session is being edited
  const [editingValue, setEditingValue] = useState(''); // Track the input value during edit
  const editInputRef = useRef(null); // Ref to focus the input field

  // --- Handle Starting Edit ---
  const handleEditClick = (e, sessionId) => {
    e.stopPropagation(); // Prevent session selection when clicking edit
    const currentName = sessions[sessionId]?.name || formatSessionIdFallback(sessionId);
    setEditingSessionId(sessionId);
    setEditingValue(currentName);
  };

  // --- Handle Saving Edit ---
  const handleSaveEdit = () => {
    if (editingSessionId && editingValue.trim()) {
      onRenameSession(editingSessionId, editingValue.trim());
    }
    // Reset editing state regardless of save success
    setEditingSessionId(null);
    setEditingValue('');
  };

  // --- Handle Cancelling Edit ---
  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingValue('');
  };

  // --- Handle Input Change ---
  const handleInputChange = (e) => {
    setEditingValue(e.target.value);
  };

  // --- Handle Keyboard Events (Enter/Escape) ---
  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  // --- Focus input when editing starts ---
  useEffect(() => {
    if (editingSessionId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select(); // Select text for easy replacement
    }
  }, [editingSessionId]);


  // --- Automation Handlers ---
  const handleAutomationSubmit = () => {
      if (!selectedModel) {
          alert("Please select a model before starting automation.");
          return;
      }
      if (onAutomateConversation) {
          onAutomateConversation(automationJson, selectedModel);
      }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
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
          JSON.parse(text);
          setAutomationJson(text);
        } else { throw new Error('Failed to read file content as text.'); }
      } catch (error) {
        console.error("Error reading or parsing JSON file:", error);
        alert(`Error reading file: ${error.message}. Please ensure it's valid JSON.`);
      } finally { if(fileInputRef.current) fileInputRef.current.value = ''; }
    };
    reader.onerror = (e) => {
        console.error("Error reading file:", e);
        alert('Error reading file.');
        if(fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.readAsText(file);
  };


  return (
    <div className="sidebar">
      <button onClick={onNewChat} className="new-chat-button">
        + New Chat
      </button>
      <nav className="conversation-menu">
        <h2>Conversations</h2>
        <ul>
          {sessionIds.length === 0 && <li className="no-sessions">No chats yet.</li>}
          {sessionIds.map((sessionId) => {
            // Get the session object
            const session = sessions[sessionId];
            // Determine the display name
            const displayName = session?.name || formatSessionIdFallback(sessionId);
            const isEditing = editingSessionId === sessionId;

            return (
              <li
                key={sessionId}
                className={`conversation-item ${sessionId === activeSessionId ? 'active' : ''} ${isEditing ? 'editing' : ''}`}
              >
                {isEditing ? (
                  // --- Render Input when Editing ---
                  <input
                    ref={editInputRef}
                    type="text"
                    value={editingValue}
                    onChange={handleInputChange}
                    onKeyDown={handleInputKeyDown}
                    onBlur={handleSaveEdit} // Save when focus is lost
                    className="session-edit-input"
                    aria-label={`Rename chat ${displayName}`}
                  />
                ) : (
                  // --- Render Name and Buttons ---
                  <>
                    <button
                      className="session-select-button"
                      onClick={() => onSelectSession(sessionId)}
                      aria-current={sessionId === activeSessionId ? 'page' : undefined}
                      title={displayName}
                    >
                      {displayName}
                    </button>
                    {/* Edit Button */}
                    <button
                        onClick={(e) => handleEditClick(e, sessionId)}
                        className="edit-session-button"
                        aria-label={`Rename ${displayName}`}
                        title={`Rename ${displayName}`}
                    >
                        ✏️
                    </button>
                  </>
                )}
                {/* Delete Button (always visible unless editing) */}
                {!isEditing && (
                     <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(sessionId);
                        }}
                        className="delete-session-button"
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

      {/* --- Automation Section (Unchanged) --- */}
      <div className="automation-section">
        <div className="automation-header">
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
                className="upload-json-button"
                title="Upload JSON file"
                aria-label="Upload JSON file"
                disabled={isSubmitting}
            >
                 ⬆️ Upload
            </button>
        </div>
        <label htmlFor="automation-json-input" className="automation-label">
          Paste JSON or upload file: ({`{ "inputs": ["msg1", ...] }`})
        </label>
        <textarea
          id="automation-json-input"
          className="automation-textarea"
          value={automationJson}
          onChange={(e) => setAutomationJson(e.target.value)}
          rows={6}
          placeholder='{ "inputs": ["Hello!", "Tell me a joke."] }'
          aria-label="JSON input for automated conversation"
          disabled={isSubmitting}
        />
        {automationError && (
            <p className="automation-error-message" role="alert">
                Error: {automationError}
            </p>
        )}
        <button
          onClick={handleAutomationSubmit}
          className="automation-button"
          disabled={isSubmitting || !activeSessionId || !selectedModel}
          title={!activeSessionId ? "Select or create a chat first" : !selectedModel ? "Select a model first" : "Run automated conversation"}
        >
          {isSubmitting ? 'Running...' : 'Run Automation'}
        </button>
         {!selectedModel && activeSessionId && <p className="automation-warning">Select a model above.</p>}
         {!activeSessionId && <p className="automation-warning">Create or select a chat.</p>}
      </div>
      {/* --- End Automation Section --- */}

    </div>
  );
}
