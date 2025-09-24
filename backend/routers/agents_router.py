"""
Agent router for NeMo Guardrails agent management.
Handles agent metadata and configuration.
"""

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def load_agent_metadata() -> dict:
    """Load agent metadata from YAML file."""
    try:
        metadata_path = (
            Path(__file__).parent.parent / "guardrails_config" / "metadata.yaml"
        )

        if not metadata_path.exists():
            logger.warning("Metadata file not found, using defaults")
            return {
                "agents": [
                    {
                        "name": "Math Assistant",
                        "directory": "math_assistant",
                        "description": "Specialized in mathematics, equations, and mathematical concepts",
                        "icon": "🧮",
                        "persona": "Martin Scorsese-inspired math specialist",
                    },
                    {
                        "name": "Bank Assistant",
                        "directory": "bank_assistant",
                        "description": "Expert in banking, financial services, and account management",
                        "icon": "🏦",
                        "persona": "Professional banking advisor",
                    },
                    {
                        "name": "Aviation Assistant",
                        "directory": "aviation_assistant",
                        "description": "Specialist in flight operations, aircraft systems, and aviation",
                        "icon": "✈️",
                        "persona": "Aviation operations expert",
                    },
                ]
            }

        with open(metadata_path, encoding="utf-8") as f:
            metadata = yaml.safe_load(f)

        # Validate that required fields exist
        if not metadata or "agents" not in metadata:
            raise ValueError("Invalid metadata structure")

        for agent in metadata["agents"]:
            required_fields = ["name", "directory", "description"]
            for field in required_fields:
                if field not in agent:
                    raise ValueError(
                        f"Missing required field '{field}' in agent metadata"
                    )

        return metadata

    except Exception as e:
        logger.error("Error loading agent metadata: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to load agent metadata: {str(e)}"
        ) from e


@router.get("/metadata")
async def get_agent_metadata():
    """
    Get metadata for all available NeMo Guardrails agents.

    Returns:
        Dict containing agent information including names, directories, descriptions, and icons.
    """
    try:
        metadata = load_agent_metadata()
        logger.info(
            "Successfully loaded metadata for %d agents",
            len(metadata.get("agents", [])),
        )
        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in get_agent_metadata: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/available")
async def get_available_agents():
    """
    Get list of available agent directories.

    Returns:
        List of agent directory names that actually exist on the filesystem.
    """
    try:
        config_dir = Path(__file__).parent.parent / "guardrails_config"

        if not config_dir.exists():
            raise HTTPException(
                status_code=404, detail="Guardrails configuration directory not found"
            )

        # Find all subdirectories that contain both config.yml and config.co
        available_agents = []

        for item in config_dir.iterdir():
            if item.is_dir() and item.name not in ["__pycache__"]:
                config_yml = item / "config.yml"
                config_co = item / "config.co"

                if config_yml.exists() and config_co.exists():
                    available_agents.append(item.name)

        logger.info(
            "Found %d available agents: %s", len(available_agents), available_agents
        )

        return {"agents": available_agents}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting available agents: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Failed to get available agents: {str(e)}"
        ) from e


@router.get("/validate/{agent_name}")
async def validate_agent(agent_name: str):
    """
    Validate that an agent configuration exists and is properly configured.

    Args:
        agent_name: Name of the agent directory to validate

    Returns:
        Dict with validation status and details.
    """
    try:
        config_dir = Path(__file__).parent.parent / "guardrails_config" / agent_name

        validation_result = {
            "agent_name": agent_name,
            "exists": False,
            "config_yml_exists": False,
            "config_co_exists": False,
            "valid": False,
            "errors": [],
        }

        if not config_dir.exists():
            validation_result["errors"].append(
                f"Agent directory '{agent_name}' does not exist"
            )
            return validation_result

        validation_result["exists"] = True

        config_yml = config_dir / "config.yml"
        config_co = config_dir / "config.co"

        validation_result["config_yml_exists"] = config_yml.exists()
        validation_result["config_co_exists"] = config_co.exists()

        if not config_yml.exists():
            validation_result["errors"].append("config.yml file missing")

        if not config_co.exists():
            validation_result["errors"].append("config.co file missing")

        # If both files exist, try to load and validate the YAML
        if config_yml.exists():
            try:
                with open(config_yml, encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)

                # Basic validation of YAML structure
                if not isinstance(yaml_content, dict):
                    validation_result["errors"].append(
                        "config.yml is not a valid YAML object"
                    )
                elif "models" not in yaml_content:
                    validation_result["errors"].append(
                        "config.yml missing required 'models' section"
                    )

            except yaml.YAMLError as e:
                validation_result["errors"].append(f"Invalid YAML syntax: {str(e)}")
            except Exception as e:
                validation_result["errors"].append(
                    f"Error reading config.yml: {str(e)}"
                )

        validation_result["valid"] = len(validation_result["errors"]) == 0

        logger.info(
            "Agent validation for '%s': %s",
            agent_name,
            "valid"
            if validation_result["valid"]
            else f"invalid - {validation_result['errors']}",
        )

        return validation_result

    except Exception as e:
        logger.error("Error validating agent '%s': %s", agent_name, e)
        raise HTTPException(
            status_code=500, detail=f"Validation error: {str(e)}"
        ) from e
