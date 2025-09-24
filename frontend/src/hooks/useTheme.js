// HIA/frontend/src/hooks/useTheme.js
import { useCallback, useEffect } from "react";
import { useLocalStorage } from "./useLocalStorage";

export function useTheme(defaultMode = false) {
	const [isDarkMode, setIsDarkMode] = useLocalStorage("darkMode", defaultMode);

	const toggleTheme = useCallback(() => {
		setIsDarkMode((prevMode) => !prevMode);
	}, [setIsDarkMode]);

	useEffect(() => {
		const bodyClassList = document.body.classList;
		if (isDarkMode) {
			bodyClassList.add("dark-mode");
		} else {
			bodyClassList.remove("dark-mode");
		}
		// Cleanup function to remove the class when the component unmounts
		// or before the effect runs again if isDarkMode changes.
		return () => {
			bodyClassList.remove("dark-mode");
		};
	}, [isDarkMode]);

	return { isDarkMode, toggleTheme };
}
