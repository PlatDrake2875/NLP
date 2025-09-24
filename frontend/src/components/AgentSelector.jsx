// AgentSelector.jsx
import { useEffect, useState } from "react";
import styles from "./AgentSelector.module.css";

export function AgentSelector({ onAgentSelect, onCancel }) {
	const [agents, setAgents] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	useEffect(() => {
		// Fetch agent metadata from backend
		const fetchAgents = async () => {
			try {
				const response = await fetch("/api/agents/metadata");
				if (!response.ok) {
					throw new Error("Failed to fetch agent metadata");
				}
				const data = await response.json();
				setAgents(data.agents || []);
			} catch (err) {
				console.error("Error fetching agents:", err);
				setError("Failed to load available agents");
				// Fallback to default agents if fetch fails
				setAgents([
					{
						name: "Math Assistant",
						directory: "math_assistant",
						description:
							"Specialized in mathematics, equations, and mathematical concepts",
						icon: "🧮",
						persona: "Martin Scorsese-inspired math specialist",
					},
					{
						name: "Bank Assistant",
						directory: "bank_assistant",
						description:
							"Expert in banking, financial services, and account management",
						icon: "🏦",
						persona: "Professional banking advisor",
					},
					{
						name: "Aviation Assistant",
						directory: "aviation_assistant",
						description:
							"Specialist in flight operations, aircraft systems, and aviation",
						icon: "✈️",
						persona: "Aviation operations expert",
					},
				]);
			} finally {
				setLoading(false);
			}
		};

		fetchAgents();
	}, []);

	const handleAgentSelect = (agent) => {
		onAgentSelect(agent.directory);
	};

	if (loading) {
		return (
			<div className={styles.agentSelector}>
				<div className={styles.overlay}>
					<div className={styles.modal}>
						<h2 className={styles.title}>Loading Agents...</h2>
						<div className={styles.spinner}></div>
					</div>
				</div>
			</div>
		);
	}

	if (error && agents.length === 0) {
		return (
			<div className={styles.agentSelector}>
				<div className={styles.overlay}>
					<div className={styles.modal}>
						<h2 className={styles.title}>Error</h2>
						<p className={styles.error}>{error}</p>
						<button
							type="button"
							className={styles.cancelButton}
							onClick={onCancel}
						>
							Close
						</button>
					</div>
				</div>
			</div>
		);
	}

	return (
		<div className={styles.agentSelector}>
			<div className={styles.overlay}>
				<div className={styles.modal}>
					<div className={styles.header}>
						<h2 className={styles.title}>Choose Your AI Assistant</h2>
						<p className={styles.subtitle}>
							Select the specialized assistant for your conversation
						</p>
						{error && (
							<p className={styles.warningText}>{error} (using defaults)</p>
						)}
					</div>

					<div className={styles.agentList}>
						{agents.map((agent, index) => (
							<button
								key={agent.directory || index}
								type="button"
								className={styles.agentCard}
								onClick={() => handleAgentSelect(agent)}
							>
								<div className={styles.agentIcon}>{agent.icon}</div>
								<div className={styles.agentInfo}>
									<h3 className={styles.agentName}>{agent.name}</h3>
									<p className={styles.agentDescription}>{agent.description}</p>
									<p className={styles.agentPersona}>{agent.persona}</p>
								</div>
								<div className={styles.selectIndicator}>→</div>
							</button>
						))}
					</div>

					<div className={styles.footer}>
						<button
							type="button"
							className={styles.cancelButton}
							onClick={onCancel}
						>
							Cancel
						</button>
					</div>
				</div>
			</div>
		</div>
	);
}
