#!/usr/bin/env python3
"""
Entry point for Rabi MCP Server with real quantum physics calculations
Deployable on Smithery and works locally
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rabi-server-entry")

def main():
    """Main entry point for Rabi MCP Server"""
    try:
        logger.info("Starting Rabi MCP Server (Quantum Physics)...")
        logger.info(f"Python version: {sys.version}")
        
        from src.http_server import main as server_main
        server_main()
        
    except Exception as e:
        logger.error(f"Rabi MCP Server failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()