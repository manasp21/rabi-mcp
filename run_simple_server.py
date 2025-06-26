#!/usr/bin/env python3
"""
Entry point for ultra-simple HTTP server using only standard library
For testing Smithery deployment issues
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-entry")

def main():
    """Main entry point for simple server"""
    try:
        logger.info("Starting simple test server...")
        logger.info(f"Python version: {sys.version}")
        
        from src.http_server_simple import main as server_main
        server_main()
        
    except Exception as e:
        logger.error(f"Simple server failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()