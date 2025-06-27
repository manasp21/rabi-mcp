#!/usr/bin/env python3
"""
Entry point for Advanced Rabi MCP Server 
Progressive Enhancement Architecture with Research-Grade Capabilities
"""

import sys
import logging
import time

# Configure logging for detailed debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced-rabi-entry")

def main():
    """Main entry point for Advanced AMO Physics Server"""
    try:
        start_time = time.time()
        
        logger.info("🚀 Starting Advanced Rabi MCP Server...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {sys.path[0]}")
        
        # Import and run the advanced server
        logger.info("Importing advanced server module...")
        from src.advanced_server import main as server_main
        
        startup_time = time.time() - start_time
        logger.info(f"Server module loaded in {startup_time:.3f}s")
        
        logger.info("🔬 Starting advanced AMO physics server...")
        server_main()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Falling back to basic server...")
        try:
            from src.http_server import main as basic_server_main
            basic_server_main()
        except Exception as fallback_error:
            logger.error(f"Fallback server also failed: {fallback_error}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Advanced server startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()