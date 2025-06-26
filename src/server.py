"""
Rabi MCP Server - Advanced Atomic, Molecular and Optical Physics MCP Server
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
)

from .config.settings import settings
from .tools import (
    quantum_systems,
    spectroscopy,
    visualization,
    cold_atoms,
    quantum_optics,
    utilities,
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp_server = Server("rabi-mcp-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Rabi MCP Server...")
    settings.setup_environment()
    
    # Initialize computational backends
    await initialize_backends()
    
    logger.info(f"Server initialized with backend: {settings.computational_backend}")
    logger.info(f"Maximum Hilbert dimension: {settings.max_hilbert_dim}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Rabi MCP Server...")


# Create FastAPI app
app = FastAPI(
    title="Rabi MCP Server",
    description="Advanced MCP server specialized in Atomic, Molecular and Optical Physics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def initialize_backends():
    """Initialize computational backends."""
    try:
        if settings.computational_backend == "jax" and settings.enable_gpu:
            import jax
            logger.info(f"JAX devices: {jax.devices()}")
        
        if settings.computational_backend == "numba" and settings.enable_jit_compilation:
            import numba
            logger.info(f"Numba version: {numba.__version__}")
        
        logger.info("Computational backends initialized successfully")
        
    except ImportError as e:
        logger.warning(f"Could not initialize backend {settings.computational_backend}: {e}")
        logger.info("Falling back to NumPy backend")
        settings.computational_backend = "numpy"


# MCP Tool Registration
@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools."""
    tools = []
    
    # Quantum Systems Tools
    tools.extend([
        Tool(
            name="simulate_two_level_atom",
            description="Simulate dynamics of a two-level atom in an electromagnetic field",
            inputSchema={
                "type": "object",
                "properties": {
                    "rabi_frequency": {"type": "number", "description": "Rabi frequency (rad/s)"},
                    "detuning": {"type": "number", "description": "Laser detuning (rad/s)"},
                    "evolution_time": {"type": "number", "description": "Evolution time (s)"},
                    "initial_state": {"type": "string", "enum": ["ground", "excited", "superposition"], "default": "ground"},
                    "decay_rate": {"type": "number", "default": 0, "description": "Spontaneous emission rate (rad/s)"},
                },
                "required": ["rabi_frequency", "detuning", "evolution_time"]
            }
        ),
        Tool(
            name="rabi_oscillations",
            description="Calculate Rabi oscillations for a two-level system",
            inputSchema={
                "type": "object",
                "properties": {
                    "rabi_frequency": {"type": "number", "description": "Rabi frequency (rad/s)"},
                    "time_points": {"type": "integer", "default": 1000, "description": "Number of time points"},
                    "max_time": {"type": "number", "description": "Maximum evolution time (s)"},
                    "include_decay": {"type": "boolean", "default": False, "description": "Include spontaneous emission"},
                    "decay_rate": {"type": "number", "default": 0, "description": "Decay rate if included (rad/s)"},
                },
                "required": ["rabi_frequency", "max_time"]
            }
        ),
        Tool(
            name="multi_level_atom",
            description="Simulate multi-level atomic system with arbitrary level structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "energy_levels": {"type": "array", "items": {"type": "number"}, "description": "Energy levels (rad/s)"},
                    "transition_dipoles": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Transition dipole matrix"},
                    "laser_frequencies": {"type": "array", "items": {"type": "number"}, "description": "Laser frequencies (rad/s)"},
                    "laser_intensities": {"type": "array", "items": {"type": "number"}, "description": "Laser intensities (W/m²)"},
                    "evolution_time": {"type": "number", "description": "Evolution time (s)"},
                    "initial_populations": {"type": "array", "items": {"type": "number"}, "description": "Initial level populations"},
                },
                "required": ["energy_levels", "transition_dipoles", "laser_frequencies", "laser_intensities", "evolution_time"]
            }
        ),
    ])
    
    # Spectroscopy Tools
    tools.extend([
        Tool(
            name="absorption_spectrum",
            description="Calculate absorption spectrum for atomic transitions",
            inputSchema={
                "type": "object",
                "properties": {
                    "transition_frequency": {"type": "number", "description": "Transition frequency (rad/s)"},
                    "linewidth": {"type": "number", "description": "Natural linewidth (rad/s)"},
                    "frequency_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[min_freq, max_freq] (rad/s)"},
                    "broadening_type": {"type": "string", "enum": ["natural", "doppler", "collisional"], "default": "natural"},
                    "temperature": {"type": "number", "default": 300, "description": "Temperature for Doppler broadening (K)"},
                    "atomic_mass": {"type": "number", "default": 23, "description": "Atomic mass (amu)"},
                },
                "required": ["transition_frequency", "linewidth", "frequency_range"]
            }
        ),
        Tool(
            name="laser_atom_interaction",
            description="Simulate strong-field laser-atom interactions",
            inputSchema={
                "type": "object",
                "properties": {
                    "laser_intensity": {"type": "number", "description": "Laser intensity (W/cm²)"},
                    "laser_wavelength": {"type": "number", "description": "Laser wavelength (nm)"},
                    "pulse_duration": {"type": "number", "description": "Pulse duration (fs)"},
                    "ionization_potential": {"type": "number", "description": "Ionization potential (eV)"},
                    "interaction_type": {"type": "string", "enum": ["tunneling", "multiphoton", "over_barrier"], "default": "tunneling"},
                },
                "required": ["laser_intensity", "laser_wavelength", "pulse_duration", "ionization_potential"]
            }
        ),
    ])
    
    # Visualization Tools
    tools.extend([
        Tool(
            name="plot_bloch_sphere",
            description="Create 3D visualization of quantum state on Bloch sphere",
            inputSchema={
                "type": "object",
                "properties": {
                    "state_vector": {"type": "array", "items": {"type": "number"}, "description": "Complex state vector [c0, c1]"},
                    "show_trajectory": {"type": "boolean", "default": False, "description": "Show time evolution trajectory"},
                    "trajectory_data": {"type": "array", "items": {"type": "array"}, "description": "Time series of state vectors"},
                    "title": {"type": "string", "default": "Quantum State", "description": "Plot title"},
                },
                "required": ["state_vector"]
            }
        ),
        Tool(
            name="plot_population_dynamics",
            description="Visualize population dynamics of multi-level system",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_data": {"type": "array", "items": {"type": "number"}, "description": "Time points"},
                    "population_data": {"type": "array", "items": {"type": "array"}, "description": "Population data [level0, level1, ...]"},
                    "level_labels": {"type": "array", "items": {"type": "string"}, "description": "Labels for energy levels"},
                    "title": {"type": "string", "default": "Population Dynamics", "description": "Plot title"},
                },
                "required": ["time_data", "population_data"]
            }
        ),
    ])
    
    # Cold Atoms Tools
    tools.extend([
        Tool(
            name="bec_simulation",
            description="Simulate Bose-Einstein condensate using Gross-Pitaevskii equation",
            inputSchema={
                "type": "object",
                "properties": {
                    "grid_size": {"type": "integer", "default": 256, "description": "Spatial grid size"},
                    "box_length": {"type": "number", "description": "Box length (μm)"},
                    "particle_number": {"type": "integer", "description": "Number of particles"},
                    "scattering_length": {"type": "number", "description": "s-wave scattering length (nm)"},
                    "trap_frequency": {"type": "number", "description": "Harmonic trap frequency (Hz)"},
                    "evolution_time": {"type": "number", "description": "Evolution time (ms)"},
                },
                "required": ["box_length", "particle_number", "scattering_length", "trap_frequency", "evolution_time"]
            }
        ),
    ])
    
    # Quantum Optics Tools
    tools.extend([
        Tool(
            name="cavity_qed",
            description="Simulate cavity quantum electrodynamics (Jaynes-Cummings model)",
            inputSchema={
                "type": "object",
                "properties": {
                    "coupling_strength": {"type": "number", "description": "Atom-cavity coupling strength (rad/s)"},
                    "cavity_frequency": {"type": "number", "description": "Cavity mode frequency (rad/s)"},
                    "atomic_frequency": {"type": "number", "description": "Atomic transition frequency (rad/s)"},
                    "max_photons": {"type": "integer", "default": 10, "description": "Maximum photon number to consider"},
                    "evolution_time": {"type": "number", "description": "Evolution time (μs)"},
                    "initial_state": {"type": "string", "enum": ["vacuum", "coherent", "fock"], "default": "vacuum"},
                },
                "required": ["coupling_strength", "cavity_frequency", "atomic_frequency", "evolution_time"]
            }
        ),
    ])
    
    return tools


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        # Route to appropriate tool handler
        if name.startswith("simulate_") or name in ["rabi_oscillations", "multi_level_atom"]:
            result = await quantum_systems.handle_tool(name, arguments)
        elif name.endswith("_spectrum") or name == "laser_atom_interaction":
            result = await spectroscopy.handle_tool(name, arguments)
        elif name.startswith("plot_"):
            result = await visualization.handle_tool(name, arguments)
        elif name.endswith("_simulation") or name == "bec_simulation":
            result = await cold_atoms.handle_tool(name, arguments)
        elif name in ["cavity_qed"]:
            result = await quantum_optics.handle_tool(name, arguments)
        else:
            result = await utilities.handle_tool(name, arguments)
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error calling tool {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


# HTTP Endpoints for Smithery compatibility
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Rabi MCP Server",
        "version": "1.0.0",
        "description": "Advanced MCP server specialized in Atomic, Molecular and Optical Physics",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server": settings.server_name,
        "version": settings.server_version,
        "computational_backend": settings.computational_backend,
        "max_hilbert_dim": settings.max_hilbert_dim,
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint for Smithery integration."""
    try:
        body = await request.json()
        
        # Handle MCP JSON-RPC messages
        if isinstance(body, dict) and "method" in body:
            method = body["method"]
            params = body.get("params", {})
            
            if method == "tools/list":
                tools = await handle_list_tools()
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {"tools": [tool.dict() for tool in tools]}
                })
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = await handle_call_tool(tool_name, arguments)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {"content": [content.dict() for content in result]}
                })
        
        # Handle direct tool calls
        elif isinstance(body, dict) and "tool" in body:
            tool_name = body["tool"]
            arguments = body.get("arguments", {})
            
            result = await handle_call_tool(tool_name, arguments)
            return JSONResponse({"result": result[0].text if result else "No result"})
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp")
async def mcp_get_endpoint():
    """GET endpoint for MCP server info."""
    tools = await handle_list_tools()
    return {
        "server": {
            "name": settings.server_name,
            "version": settings.server_version,
        },
        "capabilities": {
            "tools": len(tools),
            "max_hilbert_dim": settings.max_hilbert_dim,
            "backends": ["numpy", "jax", "numba"],
            "current_backend": settings.computational_backend,
        },
        "tools": [tool.dict() for tool in tools]
    }


def main():
    """Main entry point."""
    logger.info("Starting Rabi MCP Server...")
    
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()