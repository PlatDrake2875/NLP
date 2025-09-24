// HIA/frontend/src/components/MarkdownMessage.jsx
import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Accept markdownClassName prop from parent (ChatHistory)
export const MarkdownMessage = ({ text, markdownClassName }) => {
	return (
		// Apply the passed className (e.g., styles.markdownContent)
		<div className={markdownClassName || ""}>
			<ReactMarkdown remarkPlugins={[remarkGfm]}>{text || ""}</ReactMarkdown>{" "}
			{/* Handle null/undefined text */}
		</div>
	);
};
