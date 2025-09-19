# Simple development commands for Windows
# Usage: .\dev.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

switch ($Command) {
    "up" {
        Write-Host "Starting minimal dependencies..." -ForegroundColor Green
        docker compose up -d
    }
    "up-ui" {
        Write-Host "Starting with frontend..." -ForegroundColor Green
        docker compose --profile ui up -d
    }
    "up-rag" {
        Write-Host "Starting with RAG (ChromaDB only - using local Ollama)..." -ForegroundColor Green
        docker compose --profile rag up -d
    }
    "up-all" {
        Write-Host "Starting everything (frontend + ChromaDB - using local Ollama)..." -ForegroundColor Green
        docker compose --profile ui --profile rag up -d
    }
    "up-fullstack" {
        Write-Host "Starting full stack (frontend + ChromaDB + local FastAPI - using local Ollama)..." -ForegroundColor Green
        docker compose --profile ui --profile rag up -d
        Start-Sleep -Seconds 3
        Write-Host "Starting FastAPI locally..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000"
    }
    "down" {
        Write-Host "Stopping all services..." -ForegroundColor Yellow
        docker compose down
    }
    "api" {
        Write-Host "Running FastAPI locally (development mode)..." -ForegroundColor Green
        Set-Location backend
        uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
    }
    "test" {
        Write-Host "Running backend tests..." -ForegroundColor Green
        Set-Location backend
        uv run pytest
    }
    "ps" {
        Write-Host "Showing running services..." -ForegroundColor Blue
        docker compose ps
    }
    "logs" {
        Write-Host "Showing logs..." -ForegroundColor Blue
        docker compose logs -f
    }
    "clean" {
        Write-Host "Cleaning up (removing volumes)..." -ForegroundColor Red
        docker compose down -v
        docker compose down --rmi all
    }
    "help" {
        Write-Host "Available commands:" -ForegroundColor Cyan
        Write-Host "  .\dev.ps1 up       - Start no services (minimal)" -ForegroundColor White
        Write-Host "  .\dev.ps1 up-ui    - Start frontend only" -ForegroundColor White
        Write-Host "  .\dev.ps1 up-llm   - Start Ollama for LLM" -ForegroundColor White
        Write-Host "  .\dev.ps1 up-rag   - Start ChromaDB + Ollama for RAG" -ForegroundColor White
        Write-Host "  .\dev.ps1 up-all   - Start everything (containers only)" -ForegroundColor White
        Write-Host "  .\dev.ps1 up-fullstack - Start frontend + backend + all deps" -ForegroundColor Green
        Write-Host "  .\dev.ps1 api      - Run FastAPI locally (recommended for dev)" -ForegroundColor Green
        Write-Host "  .\dev.ps1 test     - Run backend tests" -ForegroundColor White
        Write-Host "  .\dev.ps1 down     - Stop all services" -ForegroundColor White
        Write-Host "  .\dev.ps1 clean    - Stop and remove everything" -ForegroundColor White
        Write-Host "  .\dev.ps1 ps       - Show running services" -ForegroundColor White
        Write-Host "  .\dev.ps1 logs     - Show logs" -ForegroundColor White
        Write-Host ""
        Write-Host "Example workflows:" -ForegroundColor Yellow
        Write-Host "  .\dev.ps1 up-fullstack  # Everything in one command!" -ForegroundColor Gray
        Write-Host "  .\dev.ps1 up-rag        # Start dependencies" -ForegroundColor Gray
        Write-Host "  .\dev.ps1 api           # Run FastAPI locally" -ForegroundColor Gray
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Use '.\dev.ps1 help' to see available commands" -ForegroundColor Yellow
    }
}