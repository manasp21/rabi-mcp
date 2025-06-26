#!/usr/bin/env python3
"""
Ultra-simple HTTP server for testing Smithery deployment
Uses only Python standard library to isolate deployment issues
"""

import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-server")

# Minimal tool for testing
SIMPLE_TOOL = {
    "name": "test_tool",
    "description": "Simple test tool",
    "inputSchema": {"type": "object", "properties": {"test": {"type": "string"}}}
}

class SimpleRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to use proper logging"""
        logger.info(f"Request: {format % args}")
    
    def do_GET(self):
        """Handle GET requests with comprehensive logging"""
        logger.info(f"GET {self.path}")
        logger.info(f"Headers: {dict(self.headers)}")
        
        # Handle all possible paths that Smithery might try
        if self.path == "/":
            self.send_json_response({"status": "ok", "server": "simple-test"})
        elif self.path == "/health":
            self.send_json_response({"status": "healthy"})
        elif self.path == "/mcp":
            self.send_json_response({
                "server": {"name": "simple-test", "version": "1.0.0"},
                "capabilities": {"tools": True}
            })
        elif self.path.startswith("/mcp"):
            # Handle any /mcp/* paths that Smithery might try
            logger.info(f"Redirecting {self.path} to /mcp")
            self.send_json_response({
                "server": {"name": "simple-test", "version": "1.0.0"},
                "capabilities": {"tools": True}
            })
        else:
            logger.warning(f"404 GET request to unknown path: {self.path}")
            # Return helpful debug info instead of generic 404
            self.send_json_response({
                "error": "Path not found",
                "requested_path": self.path,
                "available_paths": ["/", "/health", "/mcp"],
                "server": "simple-test"
            }, status=404)
    
    def do_POST(self):
        """Handle POST requests with comprehensive logging"""
        logger.info(f"POST {self.path}")
        logger.info(f"Headers: {dict(self.headers)}")
        
        # Log all POST requests to debug Smithery's endpoint usage
        try:
            content_length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            logger.info(f"POST body: {body}")
        except Exception as e:
            logger.error(f"Failed to read POST body: {e}")
            body = ""
        
        if self.path == "/mcp" or self.path.startswith("/mcp"):
            try:
                data = json.loads(body) if body else {}
                
                method = data.get("method", "")
                request_id = data.get("id", 0)
                
                logger.info(f"MCP method: {method}, id: {request_id}")
                
                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "simple-test", "version": "1.0.0"}
                        }
                    }
                elif method == "ping":
                    response = {"jsonrpc": "2.0", "id": request_id, "result": {}}
                elif method == "tools/list":
                    logger.info("Returning simple tool list for Smithery")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": [SIMPLE_TOOL]}
                    }
                elif method == "resources/list":
                    response = {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
                elif method == "prompts/list":
                    response = {"jsonrpc": "2.0", "id": request_id, "result": {"prompts": []}}
                else:
                    logger.warning(f"Unknown MCP method: {method}")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"}
                    }
                
                logger.info(f"Sending response: {json.dumps(response)}")
                self.send_json_response(response)
                
            except Exception as e:
                logger.error(f"POST /mcp error: {e}")
                import traceback
                traceback.print_exc()
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": 0,
                    "error": {"code": -32603, "message": "Internal error"}
                })
        else:
            logger.warning(f"404 POST request to unknown path: {self.path}")
            # Send a more detailed 404 response with debugging info
            error_response = {
                "error": "Not Found",
                "path": self.path,
                "method": "POST",
                "available_endpoints": ["/", "/health", "/mcp"],
                "note": "This server only handles MCP requests at /mcp"
            }
            self.send_json_response(error_response, status=404)
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        logger.info(f"DELETE {self.path}")
        logger.info(f"Headers: {dict(self.headers)}")
        
        if self.path == "/mcp" or self.path.startswith("/mcp"):
            self.send_json_response({"status": "connection_closed"})
        else:
            logger.warning(f"404 DELETE request to unknown path: {self.path}")
            self.send_json_response({
                "error": "Path not found",
                "requested_path": self.path,
                "method": "DELETE",
                "available_paths": ["/mcp"]
            }, status=404)
    
    def send_json_response(self, data, status=200):
        """Send JSON response with configurable status"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        logger.info(f"OPTIONS {self.path}")
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Content-Length', '0')
        self.end_headers()

def main():
    """Start simple HTTP server"""
    import os
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting simple HTTP server on {host}:{port}")
    
    server = HTTPServer((host, port), SimpleRequestHandler)
    
    try:
        logger.info("Server ready - listening for requests")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopping...")
        server.shutdown()

if __name__ == "__main__":
    main()