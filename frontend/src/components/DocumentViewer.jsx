// src/components/DocumentViewer.jsx
import React, { useState, useEffect } from 'react';
import styles from './DocumentViewer.module.css'; // Import CSS module

// Define the backend API URL (ensure this matches your setup)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DOCUMENTS_API_URL = `${API_BASE_URL}/api/documents`;

export function DocumentViewer({ onBackToChat }) {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(DOCUMENTS_API_URL);
        if (!response.ok) {
          let errorDetail = `HTTP error! status: ${response.status}`;
          try {
            const errorData = await response.json();
            errorDetail = errorData.detail || errorDetail;
          } catch (e) { /* Ignore if response body is not JSON */ }
          throw new Error(errorDetail);
        }
        const data = await response.json(); // Expects { count: number, documents: [...] }
        if (data && Array.isArray(data.documents)) {
          setDocuments(data.documents);
        } else {
          throw new Error("Invalid data format received from server.");
        }
      } catch (err) {
        console.error("Error fetching documents:", err);
        setError(err.message || "Failed to fetch documents.");
        setDocuments([]); // Clear documents on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchDocuments();
  }, []); // Fetch only once when the component mounts

  return (
    <div className={styles.documentViewer}>
      <header className={styles.viewerHeader}>
        <h1>Uploaded Document Chunks</h1>
        <button onClick={onBackToChat} className={styles.backButton}>
          &larr; Back to Chat
        </button>
      </header>

      <div className={styles.viewerContent}>
        {isLoading && <p className={styles.loadingMessage}>Loading documents...</p>}
        {error && <p className={styles.errorMessage}>Error: {error}</p>}
        {!isLoading && !error && documents.length === 0 && (
          <p className={styles.noDocumentsMessage}>No documents found in the vector store.</p>
        )}
        {!isLoading && !error && documents.length > 0 && (
          <ul className={styles.documentList}>
            {documents.map((doc) => (
              <li key={doc.id} className={styles.documentItem}>
                <div className={styles.documentMetadata}>
                  {/* Display original filename if available, otherwise fallback */}
                  <strong>Source:</strong> {doc.metadata?.original_filename || doc.metadata?.source || 'N/A'}
                  {doc.metadata?.page !== undefined && (
                    <span> | <strong>Page:</strong> {doc.metadata.page + 1} </span>
                  )}
                   <span> | <strong>ID:</strong> {doc.id}</span>
                </div>
                <pre className={styles.documentContent}>{doc.content}</pre>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
