"""
MCP Memory Server entry point.

Run the memory server directly with: python -m bot.mcp
"""

import asyncio
import logging
import os
import sys


def main():
    """Main entry point for module execution."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        # Try to import the full bot package version first
        from bot.mcp.memory_server import main as server_main

        logger.info("Running full MCP memory server from bot package")
        asyncio.run(server_main())
    except ImportError:
        # Fall back to standalone version
        logger.info("Bot package not available, trying standalone server")
        try:
            # Import and run standalone version
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from server_standalone import main as standalone_main

            asyncio.run(standalone_main())
        except ImportError as e:
            logger.exception(f"Could not import any MCP server version: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
