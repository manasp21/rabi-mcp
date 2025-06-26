#!/usr/bin/env python3
"""
Pure MCP Server Implementation for Smithery
Rabi MCP Server - Advanced Atomic, Molecular and Optical Physics
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Import our physics tools
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rabi-mcp-server")


class RabiMCPServer:
    """Rabi MCP Server for AMO Physics simulations."""
    
    def __init__(self):
        """Initialize the Rabi MCP server."""
        self.server_info = {
            "name": "rabi-mcp-server",
            "version": "1.0.0",
            "description": "Advanced MCP server specialized in Atomic, Molecular and Optical Physics",
        }
        
        # Initialize computational backend
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the computational environment."""
        try:
            settings.setup_environment()
            logger.info(f"Initialized with backend: {settings.computational_backend}")
            logger.info(f"Max Hilbert dimension: {settings.max_hilbert_dim}")
        except Exception as e:
            logger.warning(f"Environment setup warning: {e}")
    
    async def list_tools(self) -> List[Tool]:
        """List all available physics tools."""
        tools = []
        
        # Quantum Systems Tools
        tools.extend([
            Tool(
                name="simulate_two_level_atom",
                description="Simulate dynamics of a two-level atom in an electromagnetic field with Rabi oscillations and spontaneous emission",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "rabi_frequency": {
                            "type": "number", 
                            "description": "Rabi frequency (rad/s) - determines oscillation rate between ground and excited states"
                        },
                        "detuning": {
                            "type": "number", 
                            "description": "Laser detuning from resonance (rad/s) - frequency difference between laser and atomic transition"
                        },
                        "evolution_time": {
                            "type": "number", 
                            "description": "Total evolution time (s) - how long to simulate the system dynamics"
                        },
                        "initial_state": {
                            "type": "string", 
                            "enum": ["ground", "excited", "superposition"], 
                            "default": "ground",
                            "description": "Initial atomic state"
                        },
                        "decay_rate": {
                            "type": "number", 
                            "default": 0, 
                            "description": "Spontaneous emission rate (rad/s) - natural linewidth of the transition"
                        },
                    },
                    "required": ["rabi_frequency", "detuning", "evolution_time"]
                }
            ),
            Tool(
                name="rabi_oscillations",
                description="Calculate and analyze Rabi oscillations for a two-level system, including frequency analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "rabi_frequency": {
                            "type": "number", 
                            "description": "Rabi frequency (rad/s)"
                        },
                        "time_points": {
                            "type": "integer", 
                            "default": 1000, 
                            "description": "Number of time points for simulation"
                        },
                        "max_time": {
                            "type": "number", 
                            "description": "Maximum evolution time (s)"
                        },
                        "include_decay": {
                            "type": "boolean", 
                            "default": False, 
                            "description": "Include spontaneous emission effects"
                        },
                        "decay_rate": {
                            "type": "number", 
                            "default": 0, 
                            "description": "Decay rate if included (rad/s)"
                        },
                    },
                    "required": ["rabi_frequency", "max_time"]
                }
            ),
            Tool(
                name="multi_level_atom",
                description="Simulate multi-level atomic system with arbitrary energy structure and laser couplings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "energy_levels": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Energy levels (rad/s) - array of energy eigenvalues"
                        },
                        "transition_dipoles": {
                            "type": "array", 
                            "items": {"type": "array", "items": {"type": "number"}}, 
                            "description": "Transition dipole moment matrix (a.u.) - matrix elements for allowed transitions"
                        },
                        "laser_frequencies": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Laser frequencies (rad/s)"
                        },
                        "laser_intensities": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Laser intensities (W/m²)"
                        },
                        "evolution_time": {
                            "type": "number", 
                            "description": "Evolution time (s)"
                        },
                        "initial_populations": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Initial level populations (normalized)"
                        },
                    },
                    "required": ["energy_levels", "transition_dipoles", "laser_frequencies", "laser_intensities", "evolution_time"]
                }
            ),
        ])
        
        # Spectroscopy Tools
        tools.extend([
            Tool(
                name="absorption_spectrum",
                description="Calculate absorption spectrum with various broadening mechanisms (natural, Doppler, collisional)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transition_frequency": {
                            "type": "number", 
                            "description": "Central transition frequency (rad/s)"
                        },
                        "linewidth": {
                            "type": "number", 
                            "description": "Natural linewidth (FWHM, rad/s)"
                        },
                        "frequency_range": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "minItems": 2, 
                            "maxItems": 2, 
                            "description": "[min_freq, max_freq] frequency range (rad/s)"
                        },
                        "broadening_type": {
                            "type": "string", 
                            "enum": ["natural", "doppler", "collisional"], 
                            "default": "natural",
                            "description": "Type of line broadening mechanism"
                        },
                        "temperature": {
                            "type": "number", 
                            "default": 300, 
                            "description": "Temperature for Doppler broadening (K)"
                        },
                        "atomic_mass": {
                            "type": "number", 
                            "default": 87, 
                            "description": "Atomic mass (amu) - used for Doppler width calculation"
                        },
                    },
                    "required": ["transition_frequency", "linewidth", "frequency_range"]
                }
            ),
            Tool(
                name="laser_atom_interaction",
                description="Analyze strong-field laser-atom interactions including tunneling ionization and multiphoton processes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "laser_intensity": {
                            "type": "number", 
                            "description": "Laser intensity (W/cm²)"
                        },
                        "laser_wavelength": {
                            "type": "number", 
                            "description": "Laser wavelength (nm)"
                        },
                        "pulse_duration": {
                            "type": "number", 
                            "description": "Pulse duration (fs)"
                        },
                        "ionization_potential": {
                            "type": "number", 
                            "description": "Atomic ionization potential (eV)"
                        },
                        "interaction_type": {
                            "type": "string", 
                            "enum": ["tunneling", "multiphoton", "over_barrier"], 
                            "default": "tunneling",
                            "description": "Type of ionization mechanism"
                        },
                    },
                    "required": ["laser_intensity", "laser_wavelength", "pulse_duration", "ionization_potential"]
                }
            ),
        ])
        
        # Visualization Tools
        tools.extend([
            Tool(
                name="plot_bloch_sphere",
                description="Create interactive 3D Bloch sphere visualization of quantum states with optional trajectory plotting",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "state_vector": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Complex state vector [Re(c0), Im(c0), Re(c1), Im(c1)] for |ψ⟩ = c0|0⟩ + c1|1⟩"
                        },
                        "show_trajectory": {
                            "type": "boolean", 
                            "default": False, 
                            "description": "Show time evolution trajectory on sphere"
                        },
                        "trajectory_data": {
                            "type": "array", 
                            "items": {"type": "array"}, 
                            "description": "Time series of state vectors for trajectory"
                        },
                        "title": {
                            "type": "string", 
                            "default": "Quantum State", 
                            "description": "Plot title"
                        },
                    },
                    "required": ["state_vector"]
                }
            ),
            Tool(
                name="plot_population_dynamics",
                description="Visualize population dynamics of multi-level quantum systems over time",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "time_data": {
                            "type": "array", 
                            "items": {"type": "number"}, 
                            "description": "Time points (s)"
                        },
                        "population_data": {
                            "type": "array", 
                            "items": {"type": "array"}, 
                            "description": "Population data [time][level] - 2D array"
                        },
                        "level_labels": {
                            "type": "array", 
                            "items": {"type": "string"}, 
                            "description": "Labels for energy levels"
                        },
                        "title": {
                            "type": "string", 
                            "default": "Population Dynamics", 
                            "description": "Plot title"
                        },
                    },
                    "required": ["time_data", "population_data"]
                }
            ),
        ])
        
        # Cold Atoms Tools
        tools.extend([
            Tool(
                name="bec_simulation",
                description="Simulate Bose-Einstein condensate using Gross-Pitaevskii equation with realistic physical parameters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "grid_size": {
                            "type": "integer", 
                            "default": 128, 
                            "description": "Spatial grid size (power of 2 recommended)"
                        },
                        "box_length": {
                            "type": "number", 
                            "description": "Simulation box length (μm)"
                        },
                        "particle_number": {
                            "type": "integer", 
                            "description": "Number of particles in the condensate"
                        },
                        "scattering_length": {
                            "type": "number", 
                            "description": "s-wave scattering length (nm) - determines interaction strength"
                        },
                        "trap_frequency": {
                            "type": "number", 
                            "description": "Harmonic trap frequency (Hz)"
                        },
                        "evolution_time": {
                            "type": "number", 
                            "description": "Evolution time (ms)"
                        },
                    },
                    "required": ["box_length", "particle_number", "scattering_length", "trap_frequency", "evolution_time"]
                }
            ),
        ])
        
        # Quantum Optics Tools
        tools.extend([
            Tool(
                name="cavity_qed",
                description="Simulate cavity quantum electrodynamics using Jaynes-Cummings model with atom-photon interactions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "coupling_strength": {
                            "type": "number", 
                            "description": "Atom-cavity coupling strength g (rad/s)"
                        },
                        "cavity_frequency": {
                            "type": "number", 
                            "description": "Cavity mode frequency (rad/s)"
                        },
                        "atomic_frequency": {
                            "type": "number", 
                            "description": "Atomic transition frequency (rad/s)"
                        },
                        "max_photons": {
                            "type": "integer", 
                            "default": 10, 
                            "description": "Maximum photon number to consider in Hilbert space"
                        },
                        "evolution_time": {
                            "type": "number", 
                            "description": "Evolution time (μs)"
                        },
                        "initial_state": {
                            "type": "string", 
                            "enum": ["vacuum", "coherent", "fock"], 
                            "default": "vacuum",
                            "description": "Initial cavity state"
                        },
                    },
                    "required": ["coupling_strength", "cavity_frequency", "atomic_frequency", "evolution_time"]
                }
            ),
        ])
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
        """Call a physics tool and return results."""
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
            
            # Format result as text content
            result_text = json.dumps(result, indent=2)
            return [TextContent(type="text", text=result_text)]
            
        except Exception as e:
            logger.error(f"Error calling tool {name}: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Main entry point for the MCP server."""
    server = RabiMCPServer()
    
    # Create server session with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        session = ServerSession(read_stream, write_stream)
        
        @session.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Handle list tools request."""
            return await server.list_tools()
        
        @session.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
            """Handle tool call request."""
            return await server.call_tool(name, arguments)
        
        # Initialize and run the session
        await session.initialize()
        logger.info("Rabi MCP Server initialized and ready")
        
        # Keep the server running
        await session.run()


if __name__ == "__main__":
    asyncio.run(main())