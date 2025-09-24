// HIA/frontend/src/components/ChatForm.jsx
import { useEffect, useRef, useState } from "react";
import styles from "./ChatForm.module.css"; // Import CSS Module

export function ChatForm({ onSubmit, disabled }) {
	const [query, setQuery] = useState("");
	const textareaRef = useRef(null);

	// Auto-resize textarea
	useEffect(() => {
		const textarea = textareaRef.current;
		if (textarea) {
			textarea.style.height = "auto";
			const scrollHeight = textarea.scrollHeight;
			// Consider max-height defined in CSS module
			textarea.style.height = `${scrollHeight}px`;
		}
	}, []);

	const handleKeyDown = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSubmit(e);
		}
	};

	const handleSubmit = (e) => {
		e.preventDefault();
		if (!query.trim() || disabled) return;
		onSubmit(query);
		setQuery("");
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"; // Reset height after submit
		}
	};

	return (
		// Apply styles using the imported object
		<form
			onSubmit={handleSubmit}
			className={styles.chatForm}
			autoComplete="off"
		>
			<label htmlFor="chat-input" className="sr-only">
				{" "}
				{/* Keep sr-only global */}
				Type your message (Shift + Enter for new line)
			</label>
			<textarea
				id="chat-input"
				ref={textareaRef}
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				onKeyDown={handleKeyDown}
				placeholder={
					disabled
						? "Select a model to chat..."
						: "Send a message... (Shift + Enter for new line)"
				}
				aria-label="Chat input"
				rows="1"
				className={styles.chatTextarea} // Use module style
				disabled={disabled}
			/>
			<button
				type="submit"
				aria-label="Send message"
				className={styles.submitButton} // Use module style
				disabled={!query.trim() || disabled}
			>
				<svg
					width="24"
					height="24"
					viewBox="0 0 24 24"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
					aria-label="Send message"
				>
					<title>Send message</title>
					<path
						d="M7 11L12 6L17 11M12 18V7"
						stroke="currentColor"
						strokeWidth="2"
						strokeLinecap="round"
						strokeLinejoin="round"
					></path>
				</svg>
			</button>
		</form>
	);
}
