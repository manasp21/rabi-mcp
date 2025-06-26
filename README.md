# 🔬 Rabi MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/)
[![Smithery](https://img.shields.io/badge/smithery-deployable-green.svg)](https://smithery.ai/)

**Advanced MCP server specialized in Atomic, Molecular and Optical (AMO) Physics**

Rabi MCP Server is a comprehensive Model Context Protocol (MCP) server that brings cutting-edge quantum physics simulations and analysis tools to Claude and other AI assistants. Named after the Rabi oscillations fundamental to quantum optics, this server provides an extensive suite of tools for simulating quantum systems, analyzing spectroscopic data, and visualizing complex AMO physics phenomena.

## ✨ Features

### 🔬 Quantum Systems Simulation
- **Two-level atoms**: Rabi oscillations, Bloch vector dynamics, spontaneous emission
- **Multi-level systems**: Arbitrary energy level structures, complex transition networks
- **Time evolution**: Schrödinger equation solvers, master equation dynamics
- **Open quantum systems**: Lindblad equations, quantum trajectories

### 📊 Spectroscopy & Analysis
- **Absorption spectra**: Natural, Doppler, and collisional broadening
- **Line shapes**: Lorentzian, Gaussian, and Voigt profiles
- **Strong-field physics**: Tunneling ionization, multiphoton processes
- **High harmonic generation**: Laser-atom interaction analysis

### ❄️ Cold Atoms & Quantum Gases
- **Bose-Einstein condensates**: Gross-Pitaevskii equation solver
- **Optical lattices**: Band structure calculations, tight-binding models
- **Magnetic trapping**: Harmonic and anharmonic potentials
- **Quantum phase transitions**: Critical phenomena in ultracold gases

### 🌈 Quantum Optics
- **Cavity QED**: Jaynes-Cummings model, strong coupling regime
- **Photon statistics**: g²(τ) correlations, antibunching detection
- **Quantum entanglement**: Bell states, entanglement measures
- **Quantum metrology**: Squeezed states, precision measurements

### 📈 Advanced Visualizations
- **3D Bloch sphere**: Interactive quantum state visualization
- **Population dynamics**: Real-time evolution plots
- **Spectrograms**: Time-frequency analysis
- **Phase space plots**: Wigner functions, Husimi Q-functions

## 🚀 Quick Start

### One-Line Installation
```bash
curl -sSL https://raw.githubusercontent.com/manasp21/rabi-mcp/main/scripts/install.sh | bash
```

### Manual Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp
```

#### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

#### 3. Start the Server
```bash
python -m src.mcp_server
```

## 🐳 Docker Deployment

### Quick Start with Docker
```bash
docker run -p 8000:8000 manasp21/rabi-mcp-server:latest
```

### Development with Docker Compose
```bash
docker-compose up --build
```

## ☁️ Smithery Deployment

Deploy directly to Smithery for seamless cloud integration:

```bash
npx @smithery/cli deploy
```

Or use the Smithery dashboard with the provided `smithery.yaml` configuration.

## 🔧 Configuration

### Environment Variables
```bash
# Computational backend
COMPUTATIONAL_BACKEND=numpy  # Options: numpy, jax, numba
MAX_HILBERT_DIM=1000
ENABLE_GPU=false
PRECISION=double

# Performance settings
NUM_THREADS=4
ENABLE_PARALLEL=true
CACHE_RESULTS=true

# Server configuration
PORT=8000
LOG_LEVEL=INFO
```

### Claude Desktop Integration
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rabi-mcp": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "path/to/rabi-mcp"
      }
    }
  }
}
```

## 📚 Usage Examples

### 1. Two-Level Atom Simulation
```python
# Simulate Rabi oscillations
result = await mcp_client.call_tool("rabi_oscillations", {
    "rabi_frequency": 1e6,    # 1 MHz
    "max_time": 1e-5,         # 10 μs
    "time_points": 1000
})
```

### 2. BEC Simulation
```python
# Gross-Pitaevskii equation solver
result = await mcp_client.call_tool("bec_simulation", {
    "grid_size": 128,
    "box_length": 20,         # μm
    "particle_number": 10000,
    "scattering_length": 100, # nm
    "trap_frequency": 100,    # Hz
    "evolution_time": 10      # ms
})
```

### 3. Cavity QED Dynamics
```python
# Jaynes-Cummings model
result = await mcp_client.call_tool("cavity_qed", {
    "coupling_strength": 1e6,     # rad/s
    "cavity_frequency": 2e14,     # rad/s
    "atomic_frequency": 2e14,     # rad/s
    "max_photons": 10,
    "evolution_time": 50,         # μs
    "initial_state": "vacuum"
})
```

### 4. Absorption Spectrum
```python
# Calculate Doppler-broadened spectrum
result = await mcp_client.call_tool("absorption_spectrum", {
    "transition_frequency": 2.4e15,  # rad/s (D2 line)
    "linewidth": 6e6,                # rad/s
    "frequency_range": [2.39e15, 2.41e15],
    "broadening_type": "doppler",
    "temperature": 300,              # K
    "atomic_mass": 87                # Rb-87
})
```

## 🛠️ Available Tools

### Quantum Systems
| Tool | Description |
|------|-------------|
| `simulate_two_level_atom` | Two-level atom dynamics in EM field |
| `rabi_oscillations` | Calculate Rabi oscillations |
| `multi_level_atom` | Multi-level atomic system simulation |

### Spectroscopy
| Tool | Description |
|------|-------------|
| `absorption_spectrum` | Calculate absorption spectra |
| `laser_atom_interaction` | Strong-field laser-atom interactions |

### Visualization
| Tool | Description |
|------|-------------|
| `plot_bloch_sphere` | 3D Bloch sphere visualization |
| `plot_population_dynamics` | Population evolution plots |

### Cold Atoms
| Tool | Description |
|------|-------------|
| `bec_simulation` | Bose-Einstein condensate simulation |

### Quantum Optics
| Tool | Description |
|------|-------------|
| `cavity_qed` | Cavity quantum electrodynamics |

## 🧪 Physics Background

### Rabi Oscillations
The server's namesake, Rabi oscillations describe the coherent cycling between atomic energy levels under resonant electromagnetic radiation:

```
P_excited(t) = sin²(Ωt/2)
```

Where Ω is the Rabi frequency, fundamental to understanding atom-light interactions.

### Gross-Pitaevskii Equation
For BEC simulations, we solve the time-dependent Gross-Pitaevskii equation:

```
iℏ ∂ψ/∂t = [-ℏ²∇²/2m + V(r) + g|ψ|²]ψ
```

### Jaynes-Cummings Model
Cavity QED dynamics are governed by the Jaynes-Cummings Hamiltonian:

```
H = ℏωc a†a + ℏωa σz/2 + ℏg(a†σ- + aσ+)
```

## 🔬 Scientific Applications

- **Quantum computing**: Qubit dynamics and gate operations
- **Atomic clocks**: Frequency stability and coherence analysis
- **Laser cooling**: Doppler and sub-Doppler cooling mechanisms
- **Quantum sensing**: Magnetometry and gravimetry
- **Quantum simulation**: Many-body physics with ultracold atoms
- **Precision spectroscopy**: High-resolution measurements

## 📖 Documentation

- [API Reference](docs/api.md) - Complete tool documentation
- [Physics Guide](docs/physics_guide.md) - Theoretical background
- [Examples](docs/examples/) - Worked examples and tutorials
- [Configuration](docs/configuration.md) - Advanced configuration options

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

For specific physics tests:
```bash
pytest tests/test_quantum_systems.py -v
pytest tests/test_spectroscopy.py -v
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp
pip install -e ".[dev]"
pre-commit install
```

### Adding New Physics Tools
1. Implement tool in appropriate module (e.g., `src/tools/quantum_systems.py`)
2. Add tool registration in `src/server.py`
3. Write comprehensive tests
4. Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **QuTiP**: Quantum Toolbox in Python for quantum dynamics
- **NumPy/SciPy**: Fundamental scientific computing libraries
- **FastAPI**: Modern web framework for the server
- **Plotly**: Interactive visualization capabilities
- **MCP Protocol**: Model Context Protocol specification

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/manasp21/rabi-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manasp21/rabi-mcp/discussions)
- **Email**: [manasp21@example.com](mailto:manasp21@example.com)

## 🗺️ Roadmap

- [ ] **Machine Learning Integration**: AI-assisted parameter optimization
- [ ] **Experimental Interface**: Direct laboratory instrument control
- [ ] **Cloud Computing**: Distributed calculations for large systems
- [ ] **Real-time Analysis**: Live data streaming and analysis
- [ ] **Educational Tools**: Interactive tutorials and coursework integration

---

<div align="center">

**Built with ❤️ for the AMO physics community**

[⚛️ Quantum Systems](https://github.com/manasp21/rabi-mcp) • [🔬 AMO Physics](https://en.wikipedia.org/wiki/Atomic,_molecular,_and_optical_physics) • [🤖 MCP Protocol](https://modelcontextprotocol.io/)

</div>