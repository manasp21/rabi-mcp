#!/usr/bin/env python3
"""
HTTP-compatible MCP Server for Smithery deployment
Rabi MCP Server - Advanced Atomic, Molecular and Optical Physics
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)

# Import our MCP server class
from .mcp_server import RabiMCPServer
from .config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rabi-http-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Rabi MCP HTTP Server...")
    yield
    logger.info("Shutting down Rabi MCP HTTP Server...")


# Create FastAPI app
app = FastAPI(
    title="Rabi MCP Server",
    description="Advanced MCP server specialized in Atomic, Molecular and Optical Physics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP server instance
mcp_server = RabiMCPServer()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Rabi MCP Server",
        "version": "1.0.0",
        "description": "Advanced MCP server specialized in Atomic, Molecular and Optical Physics",
        "status": "running",
        "protocol": "http",
        "tools_available": len(await mcp_server.list_tools())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        tools = await mcp_server.list_tools()
        return {
            "status": "healthy",
            "server": "rabi-mcp-server",
            "version": "1.0.0",
            "tools_count": len(tools),
            "computational_backend": settings.computational_backend,
            "max_hilbert_dim": settings.max_hilbert_dim,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/mcp/tools")
async def list_tools():
    """List all available MCP tools."""
    try:
        tools = await mcp_server.list_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools
            ]
        }
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@app.post("/mcp/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """Call a specific MCP tool."""
    try:
        # Get arguments from request body
        body = await request.json()
        arguments = body.get("arguments", {})
        
        logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
        
        # Call the tool using our MCP server
        result = await mcp_server.call_tool(tool_name, arguments)
        
        # Extract text content from MCP result
        if result and len(result) > 0 and hasattr(result[0], 'text'):
            result_data = json.loads(result[0].text)
        else:
            result_data = {"success": False, "error": "No result returned"}
        
        return {
            "success": True,
            "tool": tool_name,
            "result": result_data
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e)
        }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint for JSON-RPC style requests."""
    try:
        body = await request.json()
        
        # Handle MCP JSON-RPC requests
        if "method" in body:
            method = body["method"]
            params = body.get("params", {})
            request_id = body.get("id", 0)
            
            if method == "tools/list":
                tools = await mcp_server.list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in tools
                        ]
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if not tool_name:
                    raise HTTPException(status_code=400, detail="Tool name is required")
                
                result = await mcp_server.call_tool(tool_name, arguments)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": content.text}
                            for content in result
                        ]
                    }
                }
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
        
        # Handle direct tool calls
        elif "tool" in body:
            tool_name = body["tool"]
            arguments = body.get("arguments", {})
            
            result = await mcp_server.call_tool(tool_name, arguments)
            
            if result and len(result) > 0:
                return {"result": json.loads(result[0].text)}
            else:
                return {"error": "No result returned"}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp")
async def mcp_get_info():
    """GET endpoint for MCP server information."""
    try:
        tools = await mcp_server.list_tools()
        return {
            "server": {
                "name": "rabi-mcp-server",
                "version": "1.0.0",
                "protocol": "http-mcp",
            },
            "capabilities": {
                "tools": len(tools),
                "max_hilbert_dim": settings.max_hilbert_dim,
                "backends": ["numpy", "jax", "numba"],
                "current_backend": settings.computational_backend,
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools
            ]
        }
    except Exception as e:
        logger.error(f"Error getting MCP info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for HTTP server."""
    import os
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Rabi MCP HTTP Server on {host}:{port}")
    
    uvicorn.run(
        "src.http_server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()