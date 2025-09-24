// HIA/frontend/src/components/Header.jsx
import React from "react";
import styles from "./Header.module.css"; // Import CSS Module

export function Header({
	activeSessionName, // Receive calculated name from parent
	chatHistory,
	clearChatHistory,
	downloadChatHistory,
	disabled, // General disabled state
	isDarkMode,
	toggleTheme,
	availableModels,
	selectedModel,
	onModelChange,
	modelsLoading,
	modelsError,
}) {
	// Use the passed name directly, provide a fallback if needed
	const title = activeSessionName || "Chat";
	const isHistoryEmpty =
		!Array.isArray(chatHistory) || chatHistory.length === 0;

	return (
		// Apply styles using the imported object
		<header className={styles.chatHeader}>
			<div className={styles.headerTitle}>
				<h1>{title}</h1>
			</div>

			<div className={styles.headerControls}>
				<div className={styles.modelSelectorContainer}>
					<label htmlFor="model-select" className="sr-only">
						Select Model:
					</label>{" "}
					{/* Keep sr-only global */}
					<select
						id="model-select"
						value={selectedModel}
						onChange={onModelChange}
						disabled={
							modelsLoading ||
							availableModels.length === 0 ||
							!!modelsError ||
							disabled
						} // Also check general disabled
						className={styles.modelSelect} // Use module style
						aria-label="Select AI model"
					>
						{modelsLoading && <option value="">Loading models...</option>}
						{modelsError && <option value="">Error loading models</option>}
						{!modelsLoading && !modelsError && availableModels.length === 0 && (
							<option value="">No models found</option>
						)}
						{!modelsLoading &&
							!modelsError &&
							availableModels.map((modelName) => (
								<option key={modelName} value={modelName}>
									{modelName.split(":")[0]}
								</option>
							))}
					</select>
					{modelsError && (
						<span className={styles.modelErrorIndicator} title={modelsError}>
							{" "}
							{/* Use module style */}
							⚠️
						</span>
					)}
				</div>

				{/* Use headerActions style for the container */}
				<div className={styles.headerActions}>
					<button
						onClick={toggleTheme}
						// className={styles.themeToggleBtn} // Can target via parent or add specific class
						aria-label={`Switch to ${isDarkMode ? "light" : "dark"} mode`}
						title={`Switch to ${isDarkMode ? "Light" : "Dark"} Mode`}
						disabled={disabled} // Use general disabled state
					>
						{isDarkMode ? "☀️" : "🌙"}
					</button>
					<button
						onClick={clearChatHistory}
						// className={styles.clearHistoryBtn}
						disabled={disabled || isHistoryEmpty}
						aria-label="Clear current chat history"
						title="Clear Chat"
					>
						🗑️
					</button>
					<button
						onClick={downloadChatHistory}
						// className={styles.downloadHistoryBtn}
						disabled={disabled || isHistoryEmpty}
						aria-label="Download current chat history"
						title="Download Chat"
					>
						💾
					</button>
				</div>
			</div>
		</header>
	);
}
