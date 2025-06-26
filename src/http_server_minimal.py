#!/usr/bin/env python3
"""
Ultra-minimal HTTP MCP Server for Smithery deployment
Optimized for instant tool discovery with zero heavy imports at module level
"""

# ONLY essential imports at module level
import json
import logging
import time

# Set up comprehensive logging for debugging timeouts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("rabi-minimal-server")

# Track request timing for debugging
REQUEST_TIMES = []

def log_timing(operation, start_time):
    """Log operation timing for debugging."""
    elapsed = time.time() - start_time
    REQUEST_TIMES.append((operation, elapsed))
    logger.info(f"TIMING: {operation} took {elapsed:.4f}s")
    if elapsed > 1.0:
        logger.warning(f"SLOW OPERATION: {operation} took {elapsed:.4f}s")
    return elapsed

# Ultra-minimal tool for instant discovery (single tool, minimal schema)
DISCOVERY_TOOL = {
    "name": "simulate_two_level_atom",
    "description": "Quantum physics simulation tool",
    "inputSchema": {
        "type": "object",
        "properties": {
            "frequency": {"type": "number"}
        },
        "required": ["frequency"]
    }
}

def create_app():
    """Create FastAPI app with conditional imports to avoid startup delays."""
    logger.info("Creating minimal FastAPI app...")
    
    # Import FastAPI only when actually creating the app
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Rabi MCP Server (Minimal)", version="1.0.0")
    
    # Minimal CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Ultra-fast root endpoint."""
        return {
            "name": "Rabi MCP Server",
            "version": "1.0.0",
            "status": "running",
            "tools": 1
        }
    
    @app.get("/health")
    async def health():
        """Ultra-fast health check."""
        return {"status": "healthy", "ready": True}
    
    @app.get("/mcp")
    @app.post("/mcp") 
    @app.delete("/mcp")
    async def mcp_endpoint(request: Request):
        """Ultra-minimal MCP endpoint for instant Smithery discovery."""
        request_start = time.time()
        request_id = f"req_{int(request_start * 1000)}"
        
        try:
            # Log comprehensive request details
            method = request.method
            headers = dict(request.headers)
            is_smithery = "smithery" in headers.get("user-agent", "").lower()
            client_ip = headers.get("x-forwarded-for", "unknown")
            
            logger.info(f"[{request_id}] MCP {method} from {client_ip} - Smithery: {is_smithery}")
            logger.info(f"[{request_id}] Headers: {headers}")
            
            parse_start = time.time()
            
            # Handle DELETE for connection cleanup
            if method == "DELETE":
                log_timing(f"[{request_id}] DELETE response", request_start)
                return {"status": "connection_closed"}
            
            # Handle GET for server info
            if method == "GET":
                log_timing(f"[{request_id}] GET response", request_start)
                return {
                    "server": {"name": "rabi-mcp-server", "version": "1.0.0"},
                    "capabilities": {"tools": True}
                }
            
            # Handle POST with JSON-RPC
            try:
                body = await request.json()
                log_timing(f"[{request_id}] JSON parse", parse_start)
            except Exception as e:
                logger.error(f"[{request_id}] JSON parse error: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "error": {"code": -32700, "message": "Parse error"}
                }
            
            # Extract method and params
            rpc_method = body.get("method", "")
            params = body.get("params", {})
            rpc_id = body.get("id", 0)
            
            method_start = time.time()
            logger.info(f"[{request_id}] MCP method: {rpc_method} (Smithery: {is_smithery})")
            
            # Handle each MCP method with minimal processing and timing
            if rpc_method == "initialize":
                logger.info(f"[{request_id}] MCP initialize - instant response")
                result = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "rabi-mcp-server", "version": "1.0.0"}
                    }
                }
                log_timing(f"[{request_id}] initialize complete", method_start)
                return result
            
            elif rpc_method == "notifications/initialized":
                logger.info(f"[{request_id}] MCP initialized notification")
                result = {"jsonrpc": "2.0", "id": rpc_id, "result": {}}
                log_timing(f"[{request_id}] notifications/initialized", method_start)
                return result
            
            elif rpc_method == "ping":
                logger.info(f"[{request_id}] MCP ping - instant pong")
                result = {"jsonrpc": "2.0", "id": rpc_id, "result": {}}
                log_timing(f"[{request_id}] ping complete", method_start)
                return result
            
            elif rpc_method == "resources/list":
                logger.info(f"[{request_id}] MCP resources/list - empty")
                result = {"jsonrpc": "2.0", "id": rpc_id, "result": {"resources": []}}
                log_timing(f"[{request_id}] resources/list complete", method_start)
                return result
            
            elif rpc_method == "prompts/list":
                logger.info(f"[{request_id}] MCP prompts/list - empty")
                result = {"jsonrpc": "2.0", "id": rpc_id, "result": {"prompts": []}}
                log_timing(f"[{request_id}] prompts/list complete", method_start)
                return result
            
            elif rpc_method == "tools/list":
                # CRITICAL: Ultra-fast tool discovery for Smithery
                tools_start = time.time()
                
                if is_smithery:
                    logger.info(f"[{request_id}] SMITHERY DISCOVERY: Returning single minimal tool")
                    tools_data = [DISCOVERY_TOOL]  # Static data, no computation
                    log_timing(f"[{request_id}] Smithery tool discovery", tools_start)
                else:
                    logger.info(f"[{request_id}] Non-Smithery request: Loading full tools lazily")
                    tools_data = await load_full_tools()
                    log_timing(f"[{request_id}] Full tools loading", tools_start)
                
                result = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {"tools": tools_data}
                }
                
                total_elapsed = log_timing(f"[{request_id}] tools/list COMPLETE", method_start)
                logger.info(f"[{request_id}] FULL REQUEST COMPLETED in {total_elapsed:.4f}s")
                
                return result
            
            elif rpc_method == "tools/call":
                # Tool execution - load full server only when actually needed
                logger.info(f"Tool call: {params.get('name')} - loading full server")
                server = await get_full_mcp_server()
                result = await server.call_tool(params.get("name"), params.get("arguments", {}))
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": content.text} for content in result]
                    }
                }
            
            else:
                logger.warning(f"Unknown MCP method: {rpc_method}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}
                }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"MCP endpoint error after {elapsed:.3f}s: {e}")
            return {
                "jsonrpc": "2.0",
                "id": 0,
                "error": {"code": -32603, "message": "Internal error", "data": str(e)}
            }
    
    logger.info("Minimal FastAPI app created successfully")
    return app

# Global variables for lazy loading
_full_mcp_server = None
_full_tools_cache = None

async def get_full_mcp_server():
    """Load full MCP server only when actually needed."""
    global _full_mcp_server
    if _full_mcp_server is None:
        logger.info("Loading full MCP server (heavy imports)...")
        from .mcp_server import RabiMCPServer
        _full_mcp_server = RabiMCPServer()
        logger.info("Full MCP server loaded")
    return _full_mcp_server

async def load_full_tools():
    """Load full tool list only when actually needed."""
    global _full_tools_cache
    if _full_tools_cache is None:
        logger.info("Loading full tools list (heavy operation)...")
        server = await get_full_mcp_server()
        tools = await server.list_tools()
        _full_tools_cache = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in tools
        ]
        logger.info(f"Loaded {len(_full_tools_cache)} full tools")
    return _full_tools_cache

def main():
    """Main entry point with robust error handling and logging."""
    import os
    
    try:
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        logger.info(f"Configuring server: {host}:{port}")
        logger.info(f"Environment: PORT={os.getenv('PORT')}, HOST={os.getenv('HOST')}")
        
        # Import uvicorn only when actually starting server
        logger.info("Importing uvicorn...")
        import uvicorn
        
        # Create the app with error handling
        logger.info("Creating FastAPI app...")
        app = create_app()
        logger.info("FastAPI app created successfully")
        
        # Start server with comprehensive configuration
        logger.info(f"Starting uvicorn server on {host}:{port}...")
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            workers=1,
            log_level="info",
            access_log=True,
            server_header=False,
            timeout_keep_alive=30,
        )
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()