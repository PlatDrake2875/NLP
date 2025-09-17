FROM python:3.9-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy pyproject.toml first for better Docker layer caching
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-editable --no-cache

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]