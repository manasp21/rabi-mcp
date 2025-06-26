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

## 📦 Installation Methods

Choose your preferred installation method below. All methods give you access to the same powerful AMO physics tools.

### 🚀 Method 1: One-Line Auto-Install (Recommended)

**The fastest way to get started:**

```bash
curl -sSL https://raw.githubusercontent.com/manasp21/rabi-mcp/main/scripts/install.sh | bash
```

**What this does:**
- ✅ Checks system requirements (Python 3.8+, Git)
- ✅ Installs system dependencies (gcc, gfortran, BLAS/LAPACK)
- ✅ Sets up isolated Python virtual environment
- ✅ Installs all physics dependencies (NumPy, SciPy, QuTiP, etc.)
- ✅ Configures Claude Desktop integration automatically
- ✅ Creates convenient startup commands
- ✅ Runs comprehensive tests

**Test the installation:**
```bash
# Check server functionality
~/.rabi-mcp/start-server.sh --test

# Or use the command if ~/.local/bin is in PATH
rabi-mcp-server --test
```

---

### ☁️ Method 2: Smithery Cloud (Zero Setup)

**Deploy directly to the cloud with one command:**

```bash
# Install Smithery CLI
npm install -g @smithery/cli

# Deploy to Smithery cloud (REQUIRED FIRST STEP)
npx @smithery/cli deploy https://github.com/manasp21/rabi-mcp.git
```

**⚠️ Important: Deploy First!**
The server URL `https://server.smithery.ai/@manasp21/rabi-mcp` will only become accessible **after** successful deployment. If you see "401 Unauthorized" or connection timeouts, it means the server hasn't been deployed yet.

**What you get:**
- ✅ Instant cloud deployment (no local setup needed)
- ✅ Automatic scaling and resource management
- ✅ Built-in configuration management
- ✅ Web-based tool inspection and testing
- ✅ Integration with Claude and other AI assistants

**Deployment Process:**
1. **Deploy**: Run the deploy command above
2. **Wait**: Deployment typically takes 2-5 minutes
3. **Verify**: Check deployment status in Smithery dashboard
4. **Test**: Server will be accessible once deployment completes

**Test on Smithery:**
1. Go to [Smithery Dashboard](https://smithery.ai/)
2. Wait for deployment status to show "Running"
3. Find your deployed `rabi-mcp-server`
4. Click "Connect" to test tools
5. Try running: `simulate_two_level_atom` with sample parameters

**Troubleshooting:**
- **"401 Unauthorized"**: Server not deployed yet - run deploy command first
- **"Connection timeout"**: Server still building - wait a few more minutes
- **"Please configure server"**: Deployment failed - check build logs in dashboard

---

### 🐳 Method 3: Docker (Containerized)

**Run in an isolated container environment:**

#### Quick Start:
```bash
# Pull and run the pre-built image
docker run --name rabi-mcp -it manasp21/rabi-mcp-server:latest
```

#### Build from Source:
```bash
# Clone repository
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp

# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t rabi-mcp-server .
docker run -it rabi-mcp-server
```

#### For Development:
```bash
# Run with volume mounting for live code changes
docker run -it \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  manasp21/rabi-mcp-server:latest
```

**Test Docker deployment:**
```bash
# Test container is working
docker exec -it rabi-mcp python test_mcp_server.py

# Run physics calculations
docker exec -it rabi-mcp python -c "
from src.tools.quantum_systems import simulate_two_level_atom
import asyncio
result = asyncio.run(simulate_two_level_atom(1e6, 0, 1e-6))
print('Docker test passed!' if result['success'] else 'Test failed')
"
```

---

### 🛠️ Method 4: Manual Installation (Full Control)

**For developers and advanced users:**

#### Prerequisites:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
  python3.8+ python3-pip python3-venv git \
  gcc g++ gfortran libopenblas-dev liblapack-dev libfftw3-dev

# macOS (with Homebrew)
brew install python@3.11 gcc gfortran openblas fftw git

# CentOS/RHEL
sudo yum install -y python3 python3-pip git gcc gcc-gfortran \
  openblas-devel lapack-devel fftw-devel
```

#### Installation Steps:
```bash
# 1. Clone the repository
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# 4. Test installation
python test_mcp_server.py

# 5. Start the server
python -m src.mcp_server
```

**Test manual installation:**
```bash
# Activate environment
source venv/bin/activate

# Run comprehensive tests
pytest tests/ -v

# Test a quantum simulation
python -c "
import asyncio
from src.tools.quantum_systems import rabi_oscillations

async def test():
    result = await rabi_oscillations(
        rabi_frequency=2*3.14159*1e6,  # 1 MHz
        max_time=1e-6,                 # 1 μs
        time_points=100
    )
    print(f'Rabi oscillations test: {\"PASSED\" if result[\"success\"] else \"FAILED\"}')
    return result['success']

success = asyncio.run(test())
print('Manual installation verified!' if success else 'Test failed!')
"
```

---

### 🖥️ Method 5: Claude Desktop Integration

**Add directly to Claude Desktop for seamless use:**

#### Automatic Setup (after auto-install):
If you used Method 1, Claude Desktop is already configured! Just restart Claude Desktop.

#### Manual Setup:
1. **Find your Claude config directory:**
   - **Windows**: `%APPDATA%\Claude\`
   - **macOS**: `~/Library/Application Support/Claude/`
   - **Linux**: `~/.config/claude/`

2. **Edit `claude_desktop_config.json`:**
```json
{
  "mcpServers": {
    "rabi-mcp": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/rabi-mcp"
      }
    }
  }
}
```

3. **Or use installed version:**
```json
{
  "mcpServers": {
    "rabi-mcp": {
      "command": "/home/user/.rabi-mcp/venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "/home/user/.rabi-mcp/source"
      }
    }
  }
}
```

**Test Claude Desktop integration:**
1. Restart Claude Desktop
2. Start a new conversation
3. Type: "Can you simulate Rabi oscillations for a two-level atom?"
4. Claude should automatically use the physics tools

---

### 📱 Method 6: VS Code Extension

**For development with full IDE support:**

```bash
# Install the MCP VS Code extension
code --install-extension mcp.mcp-tools

# Configure workspace settings
echo '{
  "mcp.servers": {
    "rabi-mcp": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "'$(pwd)'"
    }
  }
}' > .vscode/settings.json
```

---

### 🐍 Method 7: Python Package Installation

**Install as a Python package:**

```bash
# From PyPI (when published)
pip install rabi-mcp-server

# Or install from GitHub
pip install git+https://github.com/manasp21/rabi-mcp.git

# Run as module
python -m rabi_mcp_server

# Or use entry point
rabi-mcp-server
```

**Test package installation:**
```python
import rabi_mcp_server
from rabi_mcp_server.tools import quantum_systems
print("Package installation successful!")
```

---

## 🧪 Testing Your Installation

### Quick Test Script

Run this after any installation method:

```python
#!/usr/bin/env python3
"""Quick test of Rabi MCP Server installation."""

import asyncio
import sys
from pathlib import Path

# Add to path if needed
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def quick_test():
    """Test basic functionality."""
    try:
        from src.tools.quantum_systems import simulate_two_level_atom
        
        print("🔬 Testing Rabi MCP Server...")
        
        # Test Rabi oscillations
        result = await simulate_two_level_atom(
            rabi_frequency=2*3.14159*1e6,  # 1 MHz
            detuning=0,                     # On resonance
            evolution_time=1e-6,            # 1 μs
            initial_state="ground"
        )
        
        if result["success"]:
            print("✅ Two-level atom simulation: PASSED")
            print(f"   Max excited population: {result['summary']['max_excited_population']:.3f}")
            
            # Test BEC simulation
            from src.tools.cold_atoms import bec_simulation
            bec_result = await bec_simulation(
                grid_size=64,
                box_length=10,      # μm
                particle_number=1000,
                scattering_length=100,  # nm
                trap_frequency=100,     # Hz
                evolution_time=1        # ms
            )
            
            if bec_result["success"]:
                print("✅ BEC simulation: PASSED")
                print(f"   Healing length: {bec_result['characteristic_scales']['healing_length_um']:.2f} μm")
            else:
                print("❌ BEC simulation: FAILED")
                
            # Test spectroscopy
            from src.tools.spectroscopy import absorption_spectrum
            spec_result = await absorption_spectrum(
                transition_frequency=2.4e15,  # rad/s (D2 line)
                linewidth=6e6,                # rad/s
                frequency_range=[2.39e15, 2.41e15]
            )
            
            if spec_result["success"]:
                print("✅ Spectroscopy analysis: PASSED")
                print(f"   Peak wavelength: {spec_result['analysis']['peak_wavelength_nm']:.1f} nm")
            else:
                print("❌ Spectroscopy analysis: FAILED")
                
            print("\n🎉 Installation test completed successfully!")
            print("🚀 Rabi MCP Server is ready for advanced AMO physics!")
            return True
            
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
```

Save as `test_installation.py` and run:
```bash
python test_installation.py
```

### Comprehensive Test Suite

For thorough testing:

```bash
# Run all physics tests
pytest tests/ -v

# Test specific modules
pytest tests/test_quantum_systems.py -v
pytest tests/test_spectroscopy.py -v

# Test with coverage
pytest --cov=src tests/

# Performance benchmarks
python -m pytest tests/ --benchmark-only
```

### Tool-by-Tool Testing

Test individual physics tools:

```bash
# Test quantum systems
python -c "
import asyncio
from src.tools import quantum_systems
asyncio.run(quantum_systems.simulate_two_level_atom(1e6, 0, 1e-6))
"

# Test spectroscopy
python -c "
import asyncio
from src.tools import spectroscopy
asyncio.run(spectroscopy.absorption_spectrum(2e15, 1e6, [1.9e15, 2.1e15]))
"

# Test cold atoms
python -c "
import asyncio
from src.tools import cold_atoms
asyncio.run(cold_atoms.bec_simulation(64, 10, 1000, 100, 100, 1))
"

# Test quantum optics
python -c "
import asyncio
from src.tools import quantum_optics
asyncio.run(quantum_optics.cavity_qed(1e6, 2e14, 2e14, 10))
"
```

## 🔧 Configuration

### Environment Variables
```bash
# Computational settings
export COMPUTATIONAL_BACKEND=numpy  # Options: numpy, jax, numba
export MAX_HILBERT_DIM=1000
export ENABLE_GPU=false
export PRECISION=double
export ENABLE_PARALLEL=true

# Performance tuning
export NUM_THREADS=4
export MEMORY_LIMIT_GB=8
export CACHE_RESULTS=true

# Logging
export LOG_LEVEL=INFO
export ENABLE_PERFORMANCE_LOGGING=true
```

### Configuration File
Create `.env` file in your installation directory:
```env
# Rabi MCP Server Configuration
COMPUTATIONAL_BACKEND=numpy
MAX_HILBERT_DIM=1000
ENABLE_GPU=false
PRECISION=double
ENABLE_PARALLEL=true
CACHE_RESULTS=true
NUM_THREADS=4
LOG_LEVEL=INFO
```

## 🚀 Usage Examples

### Basic Quantum Simulation
```python
# Rabi oscillations
await mcp_client.call_tool("rabi_oscillations", {
    "rabi_frequency": 1e6,    # 1 MHz
    "max_time": 1e-5,         # 10 μs
    "time_points": 1000
})
```

### Advanced BEC Simulation
```python
# Gross-Pitaevskii equation
await mcp_client.call_tool("bec_simulation", {
    "grid_size": 256,
    "box_length": 50,         # μm
    "particle_number": 50000,
    "scattering_length": 100, # nm
    "trap_frequency": 50,     # Hz
    "evolution_time": 100     # ms
})
```

### Spectroscopy Analysis
```python
# Doppler-broadened absorption
await mcp_client.call_tool("absorption_spectrum", {
    "transition_frequency": 2.4e15,
    "linewidth": 6e6,
    "frequency_range": [2.39e15, 2.41e15],
    "broadening_type": "doppler",
    "temperature": 300,
    "atomic_mass": 87
})
```

### Interactive Visualization
```python
# 3D Bloch sphere
await mcp_client.call_tool("plot_bloch_sphere", {
    "state_vector": [0.707, 0, 0.707, 0],  # |+⟩ state
    "show_trajectory": true,
    "title": "Superposition State"
})
```

## 🛠️ Available Tools

### Quantum Systems (8 tools)
- `simulate_two_level_atom` - Two-level atom dynamics
- `rabi_oscillations` - Rabi frequency analysis
- `multi_level_atom` - Complex atomic systems
- `bloch_dynamics` - Bloch vector evolution
- `master_equation` - Open quantum systems
- `quantum_trajectories` - Stochastic evolution
- `coherent_control` - Optimal control sequences
- `adiabatic_evolution` - Slow parameter changes

### Spectroscopy (6 tools)
- `absorption_spectrum` - Absorption line analysis
- `emission_spectrum` - Fluorescence and emission
- `laser_atom_interaction` - Strong-field physics
- `doppler_broadening` - Temperature effects
- `stark_shift` - Electric field effects
- `zeeman_effect` - Magnetic field splitting

### Cold Atoms (5 tools)
- `bec_simulation` - Bose-Einstein condensates
- `optical_lattice` - Band structure calculations
- `magnetic_trapping` - Trap design and analysis
- `evaporative_cooling` - Cooling dynamics
- `quantum_gas_transport` - Non-equilibrium dynamics

### Quantum Optics (6 tools)
- `cavity_qed` - Jaynes-Cummings dynamics
- `photon_statistics` - g²(τ) correlations
- `squeezed_states` - Non-classical light
- `entanglement_analysis` - Bell states and measures
- `quantum_interferometry` - Precision measurements
- `parametric_amplification` - Nonlinear optics

### Visualization (8+ tools)
- `plot_bloch_sphere` - 3D quantum state visualization
- `plot_population_dynamics` - Time evolution plots
- `plot_spectrum` - Spectroscopic data
- `plot_wigner_function` - Phase space distributions
- `plot_correlation_functions` - Quantum correlations
- `animate_evolution` - Time-dependent animations
- `interactive_dashboard` - Real-time parameter control

### Utilities (5+ tools)
- `unit_conversion` - Physics unit conversions
- `physical_constants` - Fundamental constants lookup
- `data_analysis` - Fitting and statistics
- `experimental_parameters` - Lab calculation helpers
- `literature_lookup` - Physics reference data

## 📚 Documentation

- [**API Reference**](docs/api.md) - Complete tool documentation
- [**Physics Guide**](docs/physics_guide.md) - Theoretical background
- [**Examples**](docs/examples/) - Worked physics examples
- [**Configuration**](docs/configuration.md) - Advanced setup options
- [**Contributing**](CONTRIBUTING.md) - Development guidelines
- [**FAQ**](docs/faq.md) - Common questions and troubleshooting

## 🤝 Contributing

We welcome contributions from the AMO physics community!

```bash
# Development setup
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v

# Submit a pull request
git checkout -b feature/new-physics-tool
# ... make changes ...
git commit -m "Add new quantum simulation tool"
git push origin feature/new-physics-tool
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **QuTiP Team**: Quantum Toolbox in Python
- **NumPy/SciPy**: Scientific computing foundation
- **MCP Protocol**: Model Context Protocol standard
- **AMO Physics Community**: Theoretical foundations

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/manasp21/rabi-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manasp21/rabi-mcp/discussions)
- **Discord**: [AMO Physics MCP Community](https://discord.gg/amo-physics-mcp)
- **Email**: [manasp21@example.com](mailto:manasp21@example.com)

---

<div align="center">

**🔬 Built with ❤️ for the AMO physics community ⚛️**

[🚀 Get Started](https://github.com/manasp21/rabi-mcp) • [📖 Documentation](docs/) • [🔬 Examples](docs/examples/) • [💬 Community](https://discord.gg/amo-physics-mcp)

</div>