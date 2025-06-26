#!/usr/bin/env python3
"""
Test script to verify minimal server can start up correctly
"""

import asyncio
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("startup-test")

async def test_minimal_server():
    """Test that the minimal server can be imported and app created."""
    try:
        logger.info("Testing minimal server startup...")
        
        # Test imports
        logger.info("Testing imports...")
        from src.http_server_minimal import create_app, DISCOVERY_TOOL, get_tools_minimal
        
        # Test static tool creation
        logger.info("Testing static tool creation...")
        tools = get_tools_minimal()
        logger.info(f"Static tools: {len(tools)} tools loaded")
        assert len(tools) > 0, "No tools returned"
        assert tools[0]["name"] == "simulate_two_level_atom", "Wrong tool name"
        
        # Test app creation
        logger.info("Testing FastAPI app creation...")
        app = create_app()
        logger.info("FastAPI app created successfully")
        
        # Test basic app properties
        assert app.title == "Rabi MCP Server (Minimal)", "Wrong app title"
        logger.info("App properties verified")
        
        logger.info("✅ All startup tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Startup test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal_server())
    sys.exit(0 if success else 1)