// HIA/frontend/src/hooks/useActiveSession.js
import { useLocalStorage } from "./useLocalStorage";

/**
 * Manages the active session ID using localStorage.
 * @returns {{
 * activeSessionId: string | null,
 * setActiveSessionId: (id: string | null) => void
 * }}
 */
export function useActiveSession() {
	const [activeSessionId, setActiveSessionId] = useLocalStorage(
		"activeSessionId",
		null,
	);

	// Basic validation or side effects related to activeSessionId changes
	// could potentially go here in a useEffect, but the core validation
	// against existing sessions happens during initialization in usePersistentSessions.

	return {
		activeSessionId,
		setActiveSessionId,
	};
}
