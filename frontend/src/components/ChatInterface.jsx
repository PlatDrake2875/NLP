// HIA/frontend/src/components/ChatInterface.jsx
import React, { useRef } from "react";
import { ChatForm } from "./ChatForm";
import { ChatHistory } from "./ChatHistory";
import styles from "./ChatInterface.module.css"; // Import CSS Module
import { Header } from "./Header";
import { ScrollToBottomButton } from "./ScrollToBottomButton";

export function ChatInterface({
	activeSessionId,
	activeSessionName, // Receive calculated name from App
	chatHistory,
	onSubmit,
	onClearHistory,
	onDownloadHistory,
	isSubmitting,
	isDarkMode,
	toggleTheme,
	availableModels,
	selectedModel,
	onModelChange,
	modelsLoading,
	modelsError,
	isInitialized, // Receive initialization status
	showAgentSelector, // New prop for agent selector visibility
	sessionAgents, // New prop mapping sessionId to selected agent
}) {
	const chatContainerRef = useRef(null);
	const bottomOfChatRef = useRef(null); // Ref for scroll-to-bottom button logic

	// Determine general disabled state
	const isDisabled =
		!isInitialized ||
		!activeSessionId ||
		modelsLoading ||
		!!modelsError ||
		isSubmitting;
	// Determine if form specifically should be disabled
	const isFormDisabled = isDisabled || !selectedModel;

	return (
		// Apply styles using the imported object
		<main className={styles.chatInterface}>
			<Header
				// Pass activeSessionName instead of activeSessionId for title display
				activeSessionName={activeSessionName}
				chatHistory={chatHistory}
				clearChatHistory={onClearHistory}
				downloadChatHistory={onDownloadHistory}
				disabled={isDisabled} // General disabled state for header actions
				isDarkMode={isDarkMode}
				toggleTheme={toggleTheme}
				availableModels={availableModels}
				selectedModel={selectedModel}
				onModelChange={onModelChange}
				modelsLoading={modelsLoading}
				modelsError={modelsError}
			/>
			{/* Use chatArea style */}
			<div className={styles.chatArea} ref={chatContainerRef}>
				{!isInitialized ? (
					<div className={styles.noChatSelected}>
						{" "}
						{/* Use module style */}
						<p>Loading sessions...</p>
					</div>
				) : activeSessionId ? (
					// Check if this session has an agent selected
					sessionAgents[activeSessionId] ? (
						<ChatHistory chatHistory={chatHistory} />
					) : (
						<div className={styles.noChatSelected}>
							{" "}
							{/* Use module style */}
							<p>Please select an AI assistant to start chatting...</p>
						</div>
					)
				) : (
					<div className={styles.noChatSelected}>
						{" "}
						{/* Use module style */}
						<p>Select a chat or start a new one.</p>
					</div>
				)}
				{/* Element to help scroll-to-bottom logic */}
				<div ref={bottomOfChatRef} style={{ height: "1px" }} />
			</div>

			{/* Conditionally render input area only if initialized, chat is selected, and agent is selected */}
			{isInitialized && activeSessionId && sessionAgents[activeSessionId] && (
				// Use chatInputArea style
				<div className={styles.chatInputArea}>
					<ChatForm
						onSubmit={onSubmit}
						disabled={isFormDisabled} // Pass specific disabled state for form
					/>
				</div>
			)}

			{/* Only show scroll button if initialized, there's an active chat, and agent is selected */}
			{isInitialized && activeSessionId && sessionAgents[activeSessionId] && (
				<ScrollToBottomButton
					containerRef={chatContainerRef}
					targetRef={bottomOfChatRef}
				/>
			)}
		</main>
	);
}
