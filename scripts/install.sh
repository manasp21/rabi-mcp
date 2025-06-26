#!/bin/bash
# Rabi MCP Server Installation Script
# Advanced MCP server specialized in Atomic, Molecular and Optical Physics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/manasp21/rabi-mcp.git"
INSTALL_DIR="$HOME/.rabi-mcp"
PYTHON_MIN_VERSION="3.8"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Rabi MCP Server Installation${NC}"
    echo -e "${BLUE}  Advanced AMO Physics MCP Server${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        REQUIRED_VERSION=$PYTHON_MIN_VERSION
        
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python $PYTHON_VERSION detected (✓)"
            return 0
        else
            print_error "Python $PYTHON_VERSION detected, but Python $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if command_exists apt-get; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y git python3 python3-pip python3-venv \
            gcc g++ gfortran libopenblas-dev liblapack-dev libfftw3-dev pkg-config \
            curl wget
    elif command_exists brew; then
        # macOS
        brew install git python@3.11 gcc gfortran openblas fftw pkg-config
    elif command_exists yum; then
        # CentOS/RHEL
        sudo yum install -y git python3 python3-pip gcc gcc-c++ gcc-gfortran \
            openblas-devel lapack-devel fftw-devel pkgconfig curl wget
    elif command_exists pacman; then
        # Arch Linux
        sudo pacman -S --noconfirm git python python-pip gcc gfortran \
            openblas lapack fftw pkg-config curl wget
    else
        print_warning "Unable to detect package manager. Please install dependencies manually:"
        print_warning "- git, python3, python3-pip, python3-venv"
        print_warning "- gcc, g++, gfortran"
        print_warning "- libopenblas-dev, liblapack-dev, libfftw3-dev"
    fi
}

# Function to create virtual environment
setup_virtual_env() {
    print_status "Setting up Python virtual environment..."
    
    # Remove existing installation
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Removing existing installation at $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
    fi
    
    # Create virtual environment
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
}

# Function to clone repository
clone_repository() {
    print_status "Cloning Rabi MCP repository..."
    
    mkdir -p "$INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR/source"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    source "$INSTALL_DIR/venv/bin/activate"
    cd "$INSTALL_DIR/source"
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install in development mode
    pip install -e .
}

# Function to create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > "$INSTALL_DIR/start-server.sh" << 'EOF'
#!/bin/bash
# Start Rabi MCP Server

INSTALL_DIR="$HOME/.rabi-mcp"
source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR/source"

echo "Starting Rabi MCP Server..."
python src/server.py "$@"
EOF
    
    chmod +x "$INSTALL_DIR/start-server.sh"
    
    # Create symlink in PATH if possible
    if [ -d "$HOME/.local/bin" ]; then
        ln -sf "$INSTALL_DIR/start-server.sh" "$HOME/.local/bin/rabi-mcp-server"
        print_status "Created symlink: rabi-mcp-server command available"
    fi
}

# Function to create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    cp "$INSTALL_DIR/source/.env.example" "$INSTALL_DIR/config.env"
    
    cat >> "$INSTALL_DIR/config.env" << EOF

# Installation-specific settings
INSTALLATION_PATH=$INSTALL_DIR
INSTALLATION_DATE=$(date)
EOF
    
    print_status "Configuration file created at: $INSTALL_DIR/config.env"
}

# Function to run tests
run_tests() {
    print_status "Running basic tests..."
    
    source "$INSTALL_DIR/venv/bin/activate"
    cd "$INSTALL_DIR/source"
    
    # Test import
    python -c "
import sys
sys.path.insert(0, 'src')
try:
    from src.server import app
    from src.tools import quantum_systems, spectroscopy, visualization
    print('✓ All modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    # Test server startup (without actually starting)
    python -c "
import sys
sys.path.insert(0, 'src')
try:
    from src.config.settings import settings
    print(f'✓ Configuration loaded: backend={settings.computational_backend}')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    sys.exit(1)
"
}

# Function to setup Claude Desktop integration
setup_claude_integration() {
    print_status "Setting up Claude Desktop integration..."
    
    CLAUDE_CONFIG_DIR="$HOME/.claude"
    CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
    
    # Create Claude config directory if it doesn't exist
    mkdir -p "$CLAUDE_CONFIG_DIR"
    
    # Backup existing config
    if [ -f "$CLAUDE_CONFIG_FILE" ]; then
        cp "$CLAUDE_CONFIG_FILE" "$CLAUDE_CONFIG_FILE.backup.$(date +%s)"
        print_status "Backed up existing Claude config"
    fi
    
    # Create or update config
    python3 << EOF
import json
import os

config_file = "$CLAUDE_CONFIG_FILE"
install_dir = "$INSTALL_DIR"

# Load existing config or create new
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Ensure mcpServers section exists
if 'mcpServers' not in config:
    config['mcpServers'] = {}

# Add Rabi MCP server
config['mcpServers']['rabi-mcp'] = {
    "command": f"{install_dir}/venv/bin/python",
    "args": [f"{install_dir}/source/src/server.py"],
    "env": {
        "PYTHONPATH": f"{install_dir}/source"
    }
}

# Write updated config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Updated Claude Desktop config: {config_file}")
EOF
}

# Function to display completion message
show_completion() {
    print_header
    print_status "Installation completed successfully!"
    echo ""
    echo -e "${GREEN}🚀 Rabi MCP Server is now installed!${NC}"
    echo ""
    echo "📂 Installation directory: $INSTALL_DIR"
    echo "⚙️  Configuration file: $INSTALL_DIR/config.env"
    echo ""
    echo "🔧 Usage options:"
    echo "   1. Direct execution:"
    echo "      $INSTALL_DIR/start-server.sh"
    echo ""
    if [ -f "$HOME/.local/bin/rabi-mcp-server" ]; then
        echo "   2. Command line (if ~/.local/bin is in PATH):"
        echo "      rabi-mcp-server"
        echo ""
    fi
    echo "   3. Claude Desktop:"
    echo "      Restart Claude Desktop to activate the MCP server"
    echo ""
    echo "   4. Docker:"
    echo "      cd $INSTALL_DIR/source && docker-compose up"
    echo ""
    echo "   5. Smithery deployment:"
    echo "      Deploy to Smithery using the provided smithery.yaml"
    echo ""
    echo "📚 Documentation:"
    echo "   • README: $INSTALL_DIR/source/README.md"
    echo "   • API Docs: $INSTALL_DIR/source/docs/"
    echo ""
    echo "🔬 Features available:"
    echo "   • Quantum system simulations (2-level atoms, Rabi oscillations)"
    echo "   • Spectroscopy analysis (absorption spectra, line shapes)"
    echo "   • Cold atom simulations (BEC, Gross-Pitaevskii equation)"
    echo "   • Quantum optics (cavity QED, photon statistics)"
    echo "   • Advanced visualizations (Bloch sphere, population dynamics)"
    echo ""
    echo -e "${BLUE}Happy physics computing! 🔬⚛️${NC}"
}

# Main installation function
main() {
    print_header
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists git; then
        print_error "Git is required but not installed"
        exit 1
    fi
    
    if ! check_python_version; then
        print_error "Python requirements not met"
        exit 1
    fi
    
    # Install dependencies
    if [ "$1" != "--skip-system-deps" ]; then
        install_system_deps
    fi
    
    # Main installation steps
    setup_virtual_env
    clone_repository
    install_python_deps
    create_startup_script
    create_config
    
    # Run tests
    if [ "$1" != "--skip-tests" ]; then
        run_tests
    fi
    
    # Setup integrations
    if [ "$1" != "--skip-claude" ]; then
        setup_claude_integration
    fi
    
    # Show completion
    show_completion
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Rabi MCP Server Installation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --help, -h           Show this help message"
            echo "  --skip-system-deps   Skip system dependency installation"
            echo "  --skip-tests         Skip running tests after installation"
            echo "  --skip-claude        Skip Claude Desktop integration"
            echo ""
            exit 0
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        --skip-claude)
            SKIP_CLAUDE=1
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main installation
main "$@"