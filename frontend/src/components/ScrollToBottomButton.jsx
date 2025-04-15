import React, { useState, useEffect } from 'react';

export const ScrollToBottomButton = ({ containerRef, targetRef }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const containerElement = containerRef.current;
    const targetElement = targetRef.current; // The element at the bottom to observe

    if (!containerElement || !targetElement) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        // Show button if the target element is NOT visible within the container
        setIsVisible(!entry.isIntersecting);
      },
      {
        root: containerElement, // Observe within the chat container
        rootMargin: '0px',      // No margin
        threshold: 0.1,        // Trigger when 10% visible/invisible
      }
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
      behavior: 'smooth',
    });
  };

  if (!isVisible) return null;

  return (
    // Positioned relative to the chat-interface container
    <div className="scroll-to-bottom-container">
      <button
        onClick={scrollToBottom}
        className="scroll-to-bottom-button"
        aria-label="Scroll to bottom"
      >
        {/* Simple down arrow SVG */}
        <svg viewBox="0 0 24 24" width="24" height="24" fill="white">
          <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
        </svg>
      </button>
    </div>
  );
};