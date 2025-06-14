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

import uvicorn
from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryStore:
    """Simple in-memory storage for experiences."""

    def __init__(self):
        self.memories = {}
        self.storage_path = Path("/app/data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _load_from_disk(self):
        """Load persisted memories from disk."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    memory = json.load(f)
                    self.memories[memory["id"]] = memory
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

    def save(self, memory_id: str, data: dict):
        """Save a memory to storage."""
        data["id"] = memory_id
        data["timestamp"] = datetime.now(UTC).isoformat()
        self.memories[memory_id] = data

        # Persist to disk
        try:
            file_path = self.storage_path / f"{memory_id}.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist memory {memory_id}: {e}")

    def get(self, memory_id: str) -> dict | None:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def list_all(self) -> list[dict]:
        """List all memories."""
        return list(self.memories.values())

    def query(self, criteria: dict) -> list[dict]:
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
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory_count": len(memory_store.memories),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/experience")
async def store_experience(
    market_state: dict, trade_action: dict, additional_context: dict | None = None
):
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

        logger.info(f"Stored experience {experience_id}")
        return {"experience_id": experience_id}

    except Exception as e:
        logger.error(f"Failed to store experience: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/experience/{experience_id}")
async def get_experience(experience_id: str):
    """Retrieve a specific experience."""
    experience = memory_store.get(experience_id)
    if not experience:
        raise HTTPException(status_code=404, detail="Experience not found")
    return experience


@app.post("/query")
async def query_experiences(market_state: dict, query_params: dict | None = None):
    """Query similar experiences."""
    try:
        experiences = memory_store.query(query_params or {})
        return {"count": len(experiences), "experiences": experiences}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memories")
async def list_memories():
    """List all stored memories."""
    memories = memory_store.list_all()
    return {"count": len(memories), "memories": memories}


async def main():
    """Main entry point for standalone server."""
    logger.info("Starting MCP Memory Server...")

    # Get port from environment
    port = int(os.getenv("MCP_SERVER_PORT", "8765"))

    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    logger.info(f"MCP Memory Server starting on port {port}")
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
