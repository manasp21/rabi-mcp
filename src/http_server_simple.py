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
        """Handle GET requests"""
        logger.info(f"GET {self.path}")
        
        if self.path == "/":
            self.send_json_response({"status": "ok", "server": "simple-test"})
        elif self.path == "/health":
            self.send_json_response({"status": "healthy"})
        elif self.path == "/mcp":
            self.send_json_response({
                "server": {"name": "simple-test", "version": "1.0.0"},
                "capabilities": {"tools": True}
            })
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        logger.info(f"POST {self.path}")
        
        if self.path == "/mcp":
            try:
                content_length = int(self.headers.get('content-length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(body) if body else {}
                
                method = data.get("method", "")
                request_id = data.get("id", 0)
                
                logger.info(f"MCP method: {method}")
                
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
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"}
                    }
                
                self.send_json_response(response)
                
            except Exception as e:
                logger.error(f"POST error: {e}")
                self.send_json_response({
                    "jsonrpc": "2.0",
                    "id": 0,
                    "error": {"code": -32603, "message": "Internal error"}
                })
        else:
            self.send_error(404)
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        logger.info(f"DELETE {self.path}")
        if self.path == "/mcp":
            self.send_json_response({"status": "connection_closed"})
        else:
            self.send_error(404)
    
    def send_json_response(self, data):
        """Send JSON response"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(response)

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