FROM python:3.12-slim

WORKDIR /app

# System deps for ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir chromadb aiosqlite pydantic pydantic-settings \
    typer rich pluggy numpy scipy xxhash tenacity structlog tomli

# Source
COPY src/ src/
COPY config/ config/
COPY migrations/ migrations/
COPY README.md .

# Data directory
RUN mkdir -p /app/data/chroma

ENV EVOMIND_DATABASE__PATH=/app/data/evo_mind.db
ENV EVOMIND_CHROMA__PATH=/app/data/chroma

CMD ["python", "-c", "from evo_mind.daily_evolve import DailyEvolutionEngine; import asyncio; asyncio.run(DailyEvolutionEngine().evolve())"]
