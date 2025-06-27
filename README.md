# 🔬 Rabi MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/)
[![Smithery](https://img.shields.io/badge/smithery-deployable-green.svg)](https://smithery.ai/)

**Atomic, Molecular and Optical (AMO) Physics MCP Server**

Rabi MCP Server is a Model Context Protocol (MCP) server that provides essential quantum physics simulation tools for Claude and other AI assistants. Named after the Rabi oscillations fundamental to quantum optics, this server offers 5 core tools for simulating basic quantum systems and analyzing AMO physics phenomena.

## ✨ Core Features

### 🔬 Available Physics Tools (5)

**1. Two-Level Atom Simulation**
- Simulates quantum dynamics of two-level atomic systems
- Real-time population dynamics and coherence effects
- Rabi frequency and detuning parameter control

**2. Rabi Oscillations Analysis**
- Calculates coherent oscillations between atomic energy levels
- On-resonance and off-resonance behavior
- Time-resolved population transfer analysis

**3. Bose-Einstein Condensate (BEC) Simulation**
- Basic BEC dynamics using simplified Gross-Pitaevskii equation
- Particle interactions and quantum statistics
- Characteristic length scales and energy analysis

**4. Absorption Spectrum Calculation**
- Spectral line analysis with natural and Doppler broadening
- Temperature-dependent linewidth effects
- Lorentzian and Gaussian profile modeling

**5. Cavity QED Simulation**
- Basic cavity quantum electrodynamics using Jaynes-Cummings model
- Atom-photon coupling dynamics
- Strong and weak coupling regime analysis

## 📦 Installation Methods

Choose your preferred installation method below. All methods give you access to the same 5 core AMO physics tools.

### 🚀 Method 1: Smithery Cloud (Recommended)

**Deploy directly to the cloud with zero local setup:**

```bash
# Deploy to Smithery cloud
npx @smithery/cli deploy https://github.com/manasp21/rabi-mcp.git
```

**What you get:**
- ✅ Instant cloud deployment (no local setup needed)
- ✅ Automatic scaling and resource management
- ✅ Built-in configuration management
- ✅ 5 core physics tools ready to use
- ✅ Integration with Claude and other AI assistants

**Deployment Process:**
1. **Deploy**: Run the deploy command above
2. **Wait**: Deployment typically takes 2-5 minutes
3. **Verify**: Check deployment status in Smithery dashboard
4. **Test**: Server will be accessible once deployment completes

### 🛠️ Method 2: Manual Installation

**For local development and testing:**

```bash
# Clone repository
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test the server
python run_simple_server.py
```

---

### 🐳 Method 3: Docker

**Build and run with Docker:**

```bash
# Clone and build
git clone https://github.com/manasp21/rabi-mcp.git
cd rabi-mcp
docker build -t rabi-mcp-server .
docker run -p 8000:8000 rabi-mcp-server
```

---

## 🧪 Testing Your Installation

### Quick Test

After installation, test with:

```bash
# Test basic server functionality
curl http://localhost:8000/health

# Or test a physics calculation
python -c "
import sys
sys.path.append('src')
from http_server import execute_physics_tool
result = execute_physics_tool('simulate_two_level_atom', {
    'rabi_frequency': 1e6, 
    'detuning': 0, 
    'evolution_time': 1e-6
})
print('✅ Test passed!' if result['success'] else '❌ Test failed!')
"
```

## 🔧 Configuration

Basic configuration via environment variables:
```bash
export PORT=8000
export HOST=0.0.0.0
export LOG_LEVEL=INFO
```

## 🚀 Usage Examples

### Tool 1: Two-Level Atom Simulation
```python
{
    "rabi_frequency": 1000000,  # 1 MHz in rad/s
    "detuning": 0,              # On resonance
    "evolution_time": 0.000001  # 1 μs
}
```

### Tool 2: Rabi Oscillations
```python
{
    "rabi_frequency": 2000000,  # 2 MHz in rad/s
    "max_time": 0.00001,        # 10 μs
    "time_points": 1000
}
```

### Tool 3: BEC Simulation
```python
{
    "particle_number": 1000,
    "scattering_length": 5.29,  # Bohr radii
    "trap_frequency": 100       # Hz
}
```

### Tool 4: Absorption Spectrum
```python
{
    "transition_frequency": 3.8e15,  # rad/s
    "linewidth": 6.07e6,             # rad/s
    "temperature": 300               # Kelvin
}
```

### Tool 5: Cavity QED
```python
{
    "coupling_strength": 1000000,   # 1 MHz in rad/s
    "cavity_frequency": 3.8e15,     # rad/s
    "atom_frequency": 3.8e15        # rad/s
}
```

## 🛠️ Available Tools (5)

1. **`simulate_two_level_atom`** - Two-level quantum system dynamics
2. **`rabi_oscillations`** - Coherent population oscillations
3. **`bec_simulation`** - Bose-Einstein condensate physics
4. **`absorption_spectrum`** - Spectral line analysis
5. **`cavity_qed`** - Atom-cavity coupling dynamics

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NumPy/SciPy**: Scientific computing foundation
- **MCP Protocol**: Model Context Protocol standard
- **AMO Physics Community**: Physics knowledge base

---

<div align="center">

**🔬 Rabi MCP Server - AMO Physics Tools ⚛️**

[🚀 Get Started](https://github.com/manasp21/rabi-mcp) • [🛠️ Issues](https://github.com/manasp21/rabi-mcp/issues)

</div>