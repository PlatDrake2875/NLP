// HIA/frontend/src/components/ScrollToBottomButton.jsx
import React, { useEffect, useState } from "react";
import styles from "./ScrollToBottomButton.module.css"; // Import CSS Module

export const ScrollToBottomButton = ({ containerRef, targetRef }) => {
	const [isVisible, setIsVisible] = useState(false);

	useEffect(() => {
		const containerElement = containerRef.current;
		const targetElement = targetRef.current;

		if (!containerElement || !targetElement) return;

		const observer = new IntersectionObserver(
			([entry]) => {
				setIsVisible(!entry.isIntersecting);
			},
			{
				root: containerElement,
				rootMargin: "0px",
				threshold: 0.1, // Show when less than 10% of the target is visible
			},
		);

		observer.observe(targetElement);

		return () => {
			if (targetElement) {
				observer.unobserve(targetElement);
			}
		};
	}, [containerRef, targetRef]);

	const scrollToBottom = () => {
		containerRef.current?.scrollTo({
			top: containerRef.current.scrollHeight,
			behavior: "smooth",
		});
	};

	if (!isVisible) return null;

	return (
		// Apply styles using the imported object
		<div className={styles.scrollToBottomContainer}>
			<button
				onClick={scrollToBottom}
				className={styles.scrollToBottomButton} // Use module style
				aria-label="Scroll to bottom"
			>
				<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
					{" "}
					{/* Changed fill */}
					<path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
				</svg>
			</button>
		</div>
	);
};
