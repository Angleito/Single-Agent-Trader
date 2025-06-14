"""
MCP Memory Server entry point.

Run the memory server directly with: python -m bot.mcp
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from bot.mcp.memory_server import main

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server
    asyncio.run(main())
