// HIA/frontend/src/components/ChatHistory.jsx
import React, { useEffect, useLayoutEffect, useRef } from "react";
import styles from "./ChatHistory.module.css"; // Import CSS Module
import { MarkdownMessage } from "./MarkdownMessage"; // Keep MarkdownMessage separate

export const ChatHistory = ({ chatHistory }) => {
	const endOfMessagesRef = useRef(null);
	const chatHistoryContainerRef = useRef(null); // Ref for the scroll container

	useLayoutEffect(() => {
		const chatContainer = chatHistoryContainerRef.current;
		if (!chatContainer || !endOfMessagesRef.current) return;

		// Determine if user is near the bottom before auto-scrolling
		const scrollThreshold = 150; // Pixels from bottom tolerance
		const isNearBottom =
			chatContainer.scrollHeight -
				chatContainer.scrollTop -
				chatContainer.clientHeight <
			scrollThreshold;

		// Scroll to bottom if near bottom or initial load/few messages
		if (isNearBottom) {
			endOfMessagesRef.current.scrollIntoView({
				behavior: "smooth",
				block: "end",
			});
		}
		// If not near bottom, only scroll if it's essentially the first message load
		else if (
			chatContainer.scrollHeight <=
			chatContainer.clientHeight + scrollThreshold
		) {
			endOfMessagesRef.current.scrollIntoView({
				behavior: "auto",
				block: "end",
			}); // Use auto for initial load
		}
	}, [chatHistory]); // Scroll when history changes

	return (
		// Apply styles using the imported object
		// Note: The parent (.chatArea) handles the scrolling, this is just the content list
		<div className={styles.chatHistory} ref={chatHistoryContainerRef}>
			{!Array.isArray(chatHistory) || chatHistory.length === 0
				? // Optional: Add a message for empty history if needed
					// <div className={styles.emptyHistory}>Start chatting...</div>
					null // Or render nothing
				: chatHistory.map((entry, index) => {
						// Basic check for valid entry structure
						if (
							typeof entry !== "object" ||
							entry === null ||
							!entry.sender ||
							!entry.text
						) {
							console.warn("Skipping invalid chat history entry:", entry);
							return null; // Skip rendering invalid entries
						}

						// Combine base and modifier classes
						const messageContainerClasses = `${styles.messageContainer} ${entry.sender === "user" ? styles.user : styles.bot}`;
						const chatMessageClasses = `${styles.chatMessage} ${entry.sender === "user" ? styles.user : styles.bot} ${entry.text.startsWith("⚠️ Error:") ? styles.error : ""}`; // Add error class conditionally

						return (
							<div
								key={entry.id || `msg-${index}`}
								className={messageContainerClasses}
							>
								<div className={chatMessageClasses}>
									{/* Apply markdownContent class to the wrapper inside MarkdownMessage */}
									<MarkdownMessage
										text={entry.text}
										markdownClassName={styles.markdownContent}
									/>
								</div>
							</div>
						);
					})}
			{/* Empty div at the end to help scrolling to the bottom */}
			<div ref={endOfMessagesRef} />
		</div>
	);
};
