#!/usr/bin/env python3
"""
Entry point for ultra-minimal Rabi MCP Server
Optimized for instant Smithery deployment with robust error handling
"""

import sys
import logging
import traceback

# Set up logging for debugging deployment issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rabi-entry")

def main():
    """Main entry point with comprehensive error handling."""
    try:
        logger.info("Starting Rabi MCP Server (minimal)...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {sys.path}")
        
        # Import and run the minimal server
        logger.info("Importing minimal server module...")
        from src.http_server_minimal import main as server_main
        
        logger.info("Starting minimal server...")
        server_main()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"Python path: {sys.path}")
        traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()