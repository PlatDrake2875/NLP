# nemo_guardrails_build/Dockerfile
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install git and other build dependencies if necessary
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Clone the NeMo Guardrails repository
# Pinning to a specific release tag is recommended for stability.
# Using v0.9.0 as it was the version you were trying to pull.
RUN pip install --no-cache-dir "nemoguardrails[server]==0.13.0"

# Expose the port the server runs on
EXPOSE 8001

# Command to run the NeMo Guardrails server.
# Point --config to the DIRECTORY /app/config.
# The server will then look for config.yml (and other .yml/.co files) inside it.
CMD ["nemoguardrails", "server", "--config", "/app/config/mybot/config.yml", "--port", "8001"]
