# Simplified Development Workflow

This setup follows **Option A**: Run FastAPI locally + minimal dependencies in Docker for the best development experience.

## Quick Start

1. **Start only what you need:**
   ```bash
   # Nothing (minimal - good for basic API testing)
   make up
   
   # Add LLM support (Ollama)
   make up-llm
   
   # Add RAG support (ChromaDB + Ollama)  
   make up-rag
   
   # Add frontend UI
   make up-ui
   
   # Everything
   make up-all
   ```

2. **Run FastAPI locally** (recommended for development):
   ```bash
   make api
   # or directly: cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access your services:**
   - FastAPI: http://localhost:8000
   - FastAPI Docs: http://localhost:8000/docs
   - Frontend: http://localhost:5173 (if running)
   - Ollama: http://localhost:11434 (if running)
   - ChromaDB: http://localhost:8001 (if running)

## Why This is Better

### ✅ **Faster Development**
- FastAPI runs locally with instant reload
- No container rebuild when you change code
- Full debugger support in your IDE

### ✅ **Minimal Resource Usage**
- Only run services you actually need
- Use profiles to toggle optional components
- No heavyweight containers for simple API changes

### ✅ **Better Debugging**
- Set breakpoints directly in your IDE
- Full stack traces and error messages
- No container logs to parse

## Common Workflows

### Basic API Development
```bash
make api  # Just run FastAPI locally
# Edit code, see changes instantly
```

### Testing with LLM
```bash
make up-llm  # Start Ollama
make api     # Run FastAPI locally
# Now you can test chat endpoints
```

### Full RAG Development
```bash
make up-rag  # Start ChromaDB + Ollama
make api     # Run FastAPI locally
# Now you can test RAG functionality
```

### Frontend + Backend Development
```bash
make up-rag  # Start dependencies
make up-ui   # Start frontend
make api     # Run FastAPI locally
# Full stack development
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `make up` | Start minimal dependencies |
| `make up-llm` | Start with Ollama |
| `make up-rag` | Start with ChromaDB + Ollama |
| `make up-ui` | Start with frontend |
| `make up-all` | Start everything |
| `make api` | Run FastAPI locally (recommended) |
| `make test` | Run backend tests |
| `make down` | Stop all services |
| `make clean` | Stop and remove everything |
| `make ps` | Show running services |
| `make logs` | Show logs |

## Environment Configuration

Edit `backend/.env` to customize:
- API ports and hosts
- LLM models and URLs
- RAG and vector store settings
- Feature flags (RAG_ENABLED, USE_GUARDRAILS)

## When to Use Docker for API

Only use `docker compose up backend` when you need to:
- Test the containerized API specifically
- Run in an environment similar to production
- Test with exact Python/dependency versions

For day-to-day development, `make api` is much faster and more convenient.