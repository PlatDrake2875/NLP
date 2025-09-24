import { useCallback, useEffect, useState } from "react";

export function useLocalStorage(key, initialValue) {
	const [storedValue, setStoredValue] = useState(() => {
		try {
			const item = window.localStorage.getItem(key);
			return item ? JSON.parse(item) : initialValue;
		} catch (error) {
			console.error(`Error reading localStorage key "${key}":`, error);
			return initialValue;
		}
	});

	const setValue = useCallback((value) => {
		try {
			setStoredValue((prevValue) => {
				const valueToStore = value instanceof Function ? value(prevValue) : value;
				window.localStorage.setItem(key, JSON.stringify(valueToStore));
				return valueToStore;
			});
		} catch (error) {
			console.error(`Error setting localStorage key "${key}":`, error);
		}
	}, [key]);

	// Optional: Listen for storage changes from other tabs/windows
	useEffect(() => {
		const handleStorageChange = (e) => {
			if (e.key === key && e.newValue) {
				try {
					setStoredValue(JSON.parse(e.newValue));
				} catch (error) {
					console.error("Error parsing storage event value:", error);
				}
			}
		};

		window.addEventListener("storage", handleStorageChange);

		return () => {
			window.removeEventListener("storage", handleStorageChange);
		};
	}, [key]);

	return [storedValue, setValue];
}