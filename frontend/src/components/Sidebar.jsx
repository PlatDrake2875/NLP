// HIA/frontend/src/components/Sidebar.jsx

import PropTypes from "prop-types"; // Import PropTypes
import React, { useEffect, useRef, useState } from "react";
import styles from "./Sidebar.module.css"; // Import CSS Module

// Helper function (keep local or move to utils)
const formatSessionIdFallback = (sessionId) => {
	if (!sessionId) return "Chat";
	return sessionId.replace(/-/g, " ").replace(/^./, (str) => str.toUpperCase());
};

export function Sidebar({
	sessions,
	activeSessionId,
	selectedModel, // This prop is crucial for automation
	onNewChat,
	onSelectSession,
	onDeleteSession,
	onRenameSession,
	onAutomateConversation, // This is handleAutomateConversation from useChatApi
	isSubmitting, // General submission state for automation/chat
	automationError,
	isInitialized, // Receive initialization status
	// PDF upload props
	onUploadPdf,
	isUploadingPdf,
	pdfUploadStatus,
	// New prop for switching view
	onViewDocuments,
}) {
	const sessionIds = isInitialized ? Object.keys(sessions) : [];
	const [automationJson, setAutomationJson] = useState(
		'{\n  "inputs": [\n    "Hello!",\n    "How are you?"\n  ]\n}',
	);
	const automationFileInputRef = useRef(null);
	const pdfFileInputRef = useRef(null);
	const [selectedPdfFile, setSelectedPdfFile] = useState(null);
	const [editingSessionId, setEditingSessionId] = useState(null);
	const [editingValue, setEditingValue] = useState("");
	const editInputRef = useRef(null);

	const handleEditClick = (e, sessionId) => {
		e.stopPropagation();
		const currentName =
			sessions[sessionId]?.name || formatSessionIdFallback(sessionId);
		setEditingSessionId(sessionId);
		setEditingValue(currentName);
	};

	const handleSaveEdit = () => {
		if (editingSessionId && editingValue.trim()) {
			onRenameSession(editingSessionId, editingValue.trim());
		}
		setEditingSessionId(null);
		setEditingValue("");
	};

	const handleCancelEdit = () => {
		setEditingSessionId(null);
		setEditingValue("");
	};

	const handleInputChange = (e) => {
		setEditingValue(e.target.value);
	};

	const handleInputKeyDown = (e) => {
		if (e.key === "Enter") handleSaveEdit();
		else if (e.key === "Escape") handleCancelEdit();
	};

	useEffect(() => {
		if (editingSessionId && editInputRef.current) {
			editInputRef.current.focus();
			editInputRef.current.select();
		}
	}, [editingSessionId]);

	const handleAutomationSubmit = () => {
		if (!selectedModel) {
			alert(
				"Please select a model first (usually selected in the ChatInterface or App level and passed to Sidebar).",
			);
			return;
		}
		if (!activeSessionId) {
			alert("Please select an active chat session to automate.");
			return;
		}
		// Let's hardcode a task for now. You can make this dynamic later (e.g., from a dropdown).
		const taskToPerform = "summarize_conversation"; // Or "suggest_next_reply" or null

		console.log(
			`[Sidebar] handleAutomationSubmit called. JSON: ${automationJson}, Model: ${selectedModel}, Task: ${taskToPerform}`,
		);
		if (onAutomateConversation) {
			onAutomateConversation(automationJson, selectedModel, taskToPerform);
		}
	};

	const handleUploadJsonClick = () => {
		automationFileInputRef.current?.click();
	};

	const handleJsonFileChange = (event) => {
		const file = event.target.files?.[0];
		if (!file) return;
		if (file.type !== "application/json") {
			alert("Please select a valid JSON file (.json)");
			if (automationFileInputRef.current)
				automationFileInputRef.current.value = "";
			return;
		}
		const reader = new FileReader();
		reader.onload = (e) => {
			try {
				const text = e.target?.result;
				if (typeof text === "string") {
					JSON.parse(text); // Validate
					setAutomationJson(text);
				} else {
					throw new Error("Failed to read file content as text.");
				}
			} catch (error) {
				alert(
					`Error reading file: ${error.message}. Please ensure it's valid JSON.`,
				);
			} finally {
				if (automationFileInputRef.current)
					automationFileInputRef.current.value = "";
			}
		};
		reader.onerror = () => {
			alert("Error reading file.");
			if (automationFileInputRef.current)
				automationFileInputRef.current.value = "";
		};
		reader.readAsText(file);
	};

	const handlePdfFileChange = (event) => {
		const file = event.target.files?.[0];
		if (file && file.type === "application/pdf") {
			setSelectedPdfFile(file);
			// Clear upload status when a new file is selected
			if (pdfUploadStatus && onUploadPdf) onUploadPdf(null, true); // Pass true to indicate clearing status
		} else {
			setSelectedPdfFile(null);
			if (file) alert("Please select a PDF file.");
			if (pdfFileInputRef.current) pdfFileInputRef.current.value = "";
		}
	};

	const handlePdfUploadClick = () => {
		if (selectedPdfFile && onUploadPdf) {
			onUploadPdf(selectedPdfFile, false); // Pass false to indicate actual upload
			// Clear selection after upload attempt
			setSelectedPdfFile(null);
			if (pdfFileInputRef.current) pdfFileInputRef.current.value = "";
		} else if (!selectedPdfFile) {
			alert("Please select a PDF file to upload.");
		}
	};

	return (
		<div className={styles.sidebar}>
			{/* Top Buttons Section */}
			<div className={styles.sidebarTopActions}>
				<button onClick={onNewChat} className={styles.newChatButton}>
					+ New Chat
				</button>
				{/* Add the View Documents Button */}
				<button
					onClick={onViewDocuments}
					className={styles.viewDocumentsButton}
				>
					View Documents
				</button>
			</div>

			<nav className={styles.conversationMenu}>
				<h2>Conversations</h2>
				<ul>
					{!isInitialized && <li className={styles.noSessions}>Loading...</li>}
					{isInitialized && sessionIds.length === 0 && (
						<li className={styles.noSessions}>No chats yet.</li>
					)}

					{isInitialized &&
						sessionIds.map((sessionId) => {
							const session = sessions[sessionId];
							const displayName =
								session?.name || formatSessionIdFallback(sessionId);
							const isEditing = editingSessionId === sessionId;
							const isActive = sessionId === activeSessionId;
							const itemClasses = `${styles.conversationItem} ${isActive ? styles.active : ""} ${isEditing ? styles.editing : ""}`;

							return (
								<li key={sessionId} className={itemClasses}>
									{isEditing ? (
										<input
											ref={editInputRef}
											type="text"
											value={editingValue}
											onChange={handleInputChange}
											onKeyDown={handleInputKeyDown}
											onBlur={handleSaveEdit} // Consider changing to a save button for better UX
											className={styles.sessionEditInput}
											aria-label={`Rename chat ${displayName}`}
										/>
									) : (
										<>
											<button
												className={styles.sessionSelectButton}
												onClick={() => onSelectSession(sessionId)}
												aria-current={isActive ? "page" : undefined}
												title={displayName}
											>
												{displayName}
											</button>
											<button
												onClick={(e) => handleEditClick(e, sessionId)}
												className={styles.editSessionButton}
												aria-label={`Rename ${displayName}`}
												title={`Rename ${displayName}`}
											>
												✏️
											</button>
										</>
									)}
									{!isEditing && (
										<button
											onClick={(e) => {
												e.stopPropagation();
												onDeleteSession(sessionId);
											}}
											className={styles.deleteSessionButton}
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

			{/* --- PDF Upload Section --- */}
			<div className={styles.pdfUploadSection}>
				<h2>Upload Document</h2>
				<input
					type="file"
					ref={pdfFileInputRef}
					onChange={handlePdfFileChange}
					accept=".pdf,application/pdf"
					style={{ display: "none" }}
					id="pdf-upload-input"
					aria-labelledby="pdf-upload-button"
				/>
				<label
					htmlFor="pdf-upload-input"
					className={styles.pdfUploadLabelButton}
				>
					{selectedPdfFile
						? `Selected: ${selectedPdfFile.name.substring(0, 25)}${selectedPdfFile.name.length > 25 ? "..." : ""}`
						: "Choose PDF File"}
				</label>
				<button
					id="pdf-upload-button"
					onClick={handlePdfUploadClick}
					className={styles.pdfUploadButton}
					disabled={isUploadingPdf || !selectedPdfFile}
					title={
						!selectedPdfFile ? "Select a PDF file first" : "Upload selected PDF"
					}
				>
					{isUploadingPdf ? "Uploading..." : "Upload PDF"}
				</button>
				{pdfUploadStatus && (
					<p
						className={`${styles.uploadMessage} ${pdfUploadStatus.success ? styles.success : styles.error}`}
					>
						{pdfUploadStatus.message}
					</p>
				)}
			</div>

			{/* --- Automation Section --- */}
			<div className={styles.automationSection}>
				<div className={styles.automationHeader}>
					<h2>Automate</h2>
					<input
						type="file"
						ref={automationFileInputRef}
						onChange={handleJsonFileChange}
						accept=".json,application/json"
						style={{ display: "none" }}
						aria-hidden="true"
					/>
					<button
						onClick={handleUploadJsonClick}
						className={styles.uploadJsonButton}
						title="Upload JSON file for automation"
						aria-label="Upload JSON file for automation"
						disabled={isSubmitting} // General submitting state
					>
						⬆️ Upload JSON
					</button>
				</div>
				<label
					htmlFor="automation-json-input"
					className={styles.automationLabel}
				>
					Paste JSON or upload file: ({`{ "inputs": ["msg1", ...] }`})
				</label>
				<textarea
					id="automation-json-input"
					className={styles.automationTextarea}
					value={automationJson}
					onChange={(e) => setAutomationJson(e.target.value)}
					rows={4}
					placeholder='{ "inputs": ["Hello!", "Tell me a joke."] }'
					aria-label="JSON input for automated conversation"
					disabled={isSubmitting}
				/>
				{automationError && (
					<p className={styles.automationErrorMessage} role="alert">
						Error: {automationError}
					</p>
				)}
				<button
					onClick={handleAutomationSubmit}
					className={styles.automationButton}
					disabled={
						isSubmitting || !activeSessionId || !selectedModel || !isInitialized
					}
					title={
						!isInitialized
							? "Loading..."
							: !activeSessionId
								? "Select or create a chat first"
								: !selectedModel
									? "Select a model first"
									: "Run automated conversation"
					}
				>
					{isSubmitting ? "Running..." : "Run Automation"}
				</button>
				{!selectedModel && activeSessionId && isInitialized && (
					<p className={styles.automationWarning}>Select a model above.</p>
				)}
				{!activeSessionId && isInitialized && (
					<p className={styles.automationWarning}>Create or select a chat.</p>
				)}
			</div>
		</div>
	);
}

// PropTypes (optional, but good practice)
// Add other props as needed
Sidebar.propTypes = {
	sessions: PropTypes.object.isRequired,
	activeSessionId: PropTypes.string,
	selectedModel: PropTypes.string, // Ensure this is passed from App.jsx
	onNewChat: PropTypes.func.isRequired,
	onSelectSession: PropTypes.func.isRequired,
	onDeleteSession: PropTypes.func.isRequired,
	onRenameSession: PropTypes.func.isRequired,
	onAutomateConversation: PropTypes.func.isRequired,
	isSubmitting: PropTypes.bool,
	automationError: PropTypes.string,
	isInitialized: PropTypes.bool.isRequired,
	onUploadPdf: PropTypes.func,
	isUploadingPdf: PropTypes.bool,
	pdfUploadStatus: PropTypes.object,
	onViewDocuments: PropTypes.func,
};

export default Sidebar; // If you prefer default export
