"""
Main entry point for the Rabi MCP Server module.
"""

if __name__ == "__main__":
    from .mcp_server import main
    import asyncio
    
    asyncio.run(main())