#!/usr/bin/env python3
"""
Standalone MCP Memory Server for Docker deployment.

This version runs the FastAPI server directly without importing
the full bot package structure.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryStore:
    """Simple in-memory storage for experiences."""

    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}
        self.storage_path = Path("/app/data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted memories from disk."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with file_path.open() as f:
                    memory = json.load(f)
                    self.memories[memory["id"]] = memory
            except Exception as e:
                logger.exception("Failed to load %s: %s", file_path, e)

    def save(self, memory_id: str, data: dict[str, Any]) -> None:
        """Save a memory to storage."""
        data["id"] = memory_id
        data["timestamp"] = datetime.now(UTC).isoformat()
        self.memories[memory_id] = data

        # Persist to disk
        try:
            file_path = self.storage_path / f"{memory_id}.json"
            with file_path.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.exception("Failed to persist memory %s: %s", memory_id, e)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all memories."""
        return list(self.memories.values())

    def query(self, _criteria: dict[str, Any]) -> list[dict[str, Any]]:
        """Query memories based on criteria."""
        # Simple implementation - return all for now
        # In production, implement similarity search
        return list(self.memories.values())


# Create FastAPI app
app = FastAPI(
    title="MCP Memory Server",
    description="Standalone memory server for AI trading bot",
    version="1.0.0",
)

# Initialize memory store
memory_store = MemoryStore()


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory_count": len(memory_store.memories),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/experience")
async def store_experience(
    market_state: dict[str, Any],
    trade_action: dict[str, Any],
    additional_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Store a new trading experience."""
    try:
        import uuid

        experience_id = str(uuid.uuid4())

        experience_data = {
            "market_state": market_state,
            "trade_action": trade_action,
            "additional_context": additional_context or {},
        }

        memory_store.save(experience_id, experience_data)

        logger.info("Stored experience %s", experience_id)
        return {"experience_id": experience_id}

    except Exception as e:
        logger.exception("Failed to store experience: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/experience/{experience_id}")
async def get_experience(experience_id: str) -> dict[str, Any]:
    """Retrieve a specific experience."""
    experience = memory_store.get(experience_id)
    if not experience:
        raise HTTPException(status_code=404, detail="Experience not found")
    return experience


@app.post("/query")
async def query_experiences(
    _market_state: dict[str, Any], query_params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Query similar experiences."""
    try:
        experiences = memory_store.query(query_params or {})
        return {"count": len(experiences), "experiences": experiences}
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memories")
async def list_memories() -> dict[str, Any]:
    """List all stored memories."""
    memories = memory_store.list_all()
    return {"count": len(memories), "memories": memories}


async def main() -> None:
    """Main entry point for standalone server."""
    logger.info("Starting MCP Memory Server...")

    # Get port from environment
    port = int(os.getenv("MCP_SERVER_PORT", "8765"))

    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    logger.info("MCP Memory Server starting on port %s", port)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
