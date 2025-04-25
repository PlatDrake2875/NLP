import React, { useState, useRef } from 'react'; // Import useRef

export function Sidebar({
    sessions,
    activeSessionId,
    selectedModel, // Need the selected model
    onNewChat,
    onSelectSession,
    onDeleteSession,
    onAutomateConversation, // New handler prop
    isSubmitting, // To disable button while processing
    automationError // To display errors
}) {
  const sessionIds = Object.keys(sessions);
  const [automationJson, setAutomationJson] = useState('{\n  "inputs": [\n    "Hello!",\n    "How are you?"\n  ]\n}'); // Default example JSON
  const fileInputRef = useRef(null); // Create a ref for the file input

  // Function to format session ID for display
  const formatSessionName = (sessionId) => {
    return sessionId ? sessionId.replace(/-/g, ' ').replace(/^./, str => str.toUpperCase()) : "Chat";
  };

  const handleAutomationSubmit = () => {
      if (!selectedModel) {
          alert("Please select a model before starting automation.");
          return;
      }
      if (onAutomateConversation) {
          onAutomateConversation(automationJson, selectedModel);
      }
  };

  // --- File Upload Handlers ---
  const handleUploadClick = () => {
    // Trigger the hidden file input when the button is clicked
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return; // User cancelled or no file selected
    }

    if (file.type !== 'application/json') {
        alert('Please select a valid JSON file (.json)');
        // Reset file input value to allow re-selecting the same file if needed
        if(fileInputRef.current) fileInputRef.current.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result;
        if (typeof text === 'string') {
          // Basic validation: Try parsing to ensure it's valid JSON before setting
          JSON.parse(text);
          setAutomationJson(text); // Update textarea content
        } else {
            throw new Error('Failed to read file content as text.');
        }
      } catch (error) {
        console.error("Error reading or parsing JSON file:", error);
        alert(`Error reading file: ${error.message}. Please ensure it's valid JSON.`);
      } finally {
         // Reset file input value to allow re-selecting the same file if needed
         if(fileInputRef.current) fileInputRef.current.value = '';
      }
    };
    reader.onerror = (e) => {
        console.error("Error reading file:", e);
        alert('Error reading file.');
        // Reset file input value
        if(fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.readAsText(file);
  };
  // --- End File Upload Handlers ---


  return (
    <div className="sidebar">
      <button onClick={onNewChat} className="new-chat-button">
        + New Chat
      </button>
      <nav className="conversation-menu">
        <h2>Conversations</h2>
        <ul>
          {sessionIds.length === 0 && <li className="no-sessions">No chats yet.</li>}
          {sessionIds.map((sessionId) => (
            <li
              key={sessionId}
              className={`conversation-item ${sessionId === activeSessionId ? 'active' : ''}`}
            >
                <button
                 className="session-select-button"
                 onClick={() => onSelectSession(sessionId)}
                 aria-current={sessionId === activeSessionId ? 'page' : undefined}
                 title={formatSessionName(sessionId)} // Add title for long names
                >
                  {formatSessionName(sessionId)}
                </button>
               <button
                  onClick={(e) => {
                      e.stopPropagation(); // Prevent triggering session selection
                      onDeleteSession(sessionId); // Confirmation is now inside the hook
                  }}
                  className="delete-session-button"
                  aria-label={`Delete ${formatSessionName(sessionId)}`}
                  title={`Delete ${formatSessionName(sessionId)}`}
               >
                  🗑️
               </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* --- Automation Section --- */}
      <div className="automation-section">
        <div className="automation-header"> {/* Added header for title and upload button */}
            <h2>Automate</h2>
            {/* Hidden File Input */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".json,application/json" // Accept only JSON files
                style={{ display: 'none' }} // Hide the default input
                aria-hidden="true" // Hide from accessibility tree
            />
             {/* Upload Button */}
            <button
                onClick={handleUploadClick}
                className="upload-json-button"
                title="Upload JSON file"
                aria-label="Upload JSON file"
                disabled={isSubmitting}
            >
                 {/* Simple Upload Icon (replace with SVG if preferred) */}
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
          disabled={isSubmitting} // Disable while submitting
        />
        {automationError && (
            <p className="automation-error-message" role="alert">
                Error: {automationError}
            </p>
        )}
        <button
          onClick={handleAutomationSubmit}
          className="automation-button"
          disabled={isSubmitting || !activeSessionId || !selectedModel} // Disable if submitting, no active session, or no model
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
