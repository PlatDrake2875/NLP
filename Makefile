# Simple development commands

# Start minimal dependencies (nothing by default)
up:
	docker compose up -d

# Start with UI (frontend)
up-ui:
	docker compose --profile ui up -d

# Start with LLM support (Ollama)
up-llm:
	docker compose --profile llm up -d

# Start with RAG support (ChromaDB + Ollama)
up-rag:
	docker compose --profile rag --profile llm up -d

# Start everything
up-all:
	docker compose --profile ui --profile llm --profile rag up -d

# Start full stack (frontend + backend + all dependencies)
up-fullstack:
	@echo "Starting full stack (frontend + backend + all dependencies)..."
	docker compose --profile ui --profile llm --profile rag up -d
	@echo "Waiting for services to start..."
	@sleep 3
	@echo "Starting FastAPI locally in new terminal..."
	@gnome-terminal -- bash -c "cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000; exec bash" 2>/dev/null || \
	 xterm -e "cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000" 2>/dev/null || \
	 echo "Please run 'make api' in another terminal to start FastAPI"

# Stop all services
down:
	docker compose down

# Run FastAPI locally (development mode)
api:
	cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run FastAPI tests
test:
	cd backend && uv run pytest

# Show running services
ps:
	docker compose ps

# View logs
logs:
	docker compose logs -f

# Clean up (remove volumes too)
clean:
	docker compose down -v
	docker compose down --rmi all

# Help
help:
	@echo "Available commands:"
	@echo "  make up          - Start no services (minimal)"
	@echo "  make up-ui       - Start frontend only"
	@echo "  make up-llm      - Start Ollama for LLM"
	@echo "  make up-rag      - Start ChromaDB + Ollama for RAG"
	@echo "  make up-all      - Start everything (containers only)"
	@echo "  make up-fullstack - Start frontend + backend + all deps"
	@echo "  make api         - Run FastAPI locally (recommended for dev)"
	@echo "  make test        - Run backend tests"
	@echo "  make down        - Stop all services"
	@echo "  make clean       - Stop and remove everything"
	@echo "  make ps          - Show running services"
	@echo "  make logs        - Show logs"
	@echo ""
	@echo "Example workflows:"
	@echo "  make up-fullstack  # Everything in one command!"
	@echo "  make up-rag        # Start dependencies"
	@echo "  make api           # Run FastAPI locally"

.PHONY: up up-ui up-llm up-rag up-all up-fullstack down api test ps logs clean help