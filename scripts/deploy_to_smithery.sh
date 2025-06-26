#!/bin/bash

# 🚀 Deploy Rabi MCP Server to Smithery Cloud
# This script resolves the "401 Unauthorized" issue by properly deploying the server

set -e  # Exit on any error

echo "🔬 Deploying Rabi MCP Server to Smithery Cloud"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "smithery.yaml" ]; then
    echo "❌ Error: smithery.yaml not found. Please run this script from the rabi-mcp directory."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is required but not installed."
    echo "   Please install Node.js and npm first:"
    echo "   - Ubuntu/Debian: sudo apt install nodejs npm"
    echo "   - macOS: brew install node"
    echo "   - Windows: Download from https://nodejs.org/"
    exit 1
fi

# Check if git is available and repo is clean
echo "📁 Checking git repository status..."
if ! git status &> /dev/null; then
    echo "❌ Error: Not a git repository or git not available."
    exit 1
fi

# Show current status
echo "📊 Current repository status:"
git status --porcelain

if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes."
    echo "   Smithery will deploy from the committed code."
    read -p "   Continue with deployment? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Deployment cancelled."
        exit 1
    fi
fi

# Install Smithery CLI if not available
echo "🔧 Checking Smithery CLI..."
if ! command -v smithery &> /dev/null; then
    echo "📦 Installing Smithery CLI..."
    npm install -g @smithery/cli
else
    echo "✅ Smithery CLI already installed"
fi

# Show the current configuration
echo "📋 Deployment Configuration:"
echo "   Repository: https://github.com/manasp21/rabi-mcp.git"
echo "   Transport: HTTP (port 8000)"
echo "   Runtime: Container (Docker)"
echo "   Physics Tools: 25+ AMO physics simulations"

# Confirm deployment
echo ""
read -p "🚀 Ready to deploy? This will make the server accessible at https://server.smithery.ai/@manasp21/rabi-mcp (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "   Deployment cancelled."
    exit 1
fi

# Deploy to Smithery
echo ""
echo "🚀 Deploying to Smithery Cloud..."
echo "   This may take 2-5 minutes for the initial build..."

if npx @smithery/cli deploy https://github.com/manasp21/rabi-mcp.git; then
    echo ""
    echo "✅ Deployment initiated successfully!"
    echo ""
    echo "🎉 Next Steps:"
    echo "   1. Go to https://smithery.ai/ to monitor deployment"
    echo "   2. Wait for status to change from 'Building' to 'Running'"
    echo "   3. Test server at: https://server.smithery.ai/@manasp21/rabi-mcp"
    echo "   4. Try the tools in Smithery's web interface"
    echo ""
    echo "🔧 If you see '401 Unauthorized' or timeouts:"
    echo "   - Wait a few more minutes for deployment to complete"
    echo "   - Check build logs in the Smithery dashboard"
    echo "   - Ensure all dependencies built successfully"
    echo ""
    echo "🧪 Test Commands (after deployment completes):"
    echo "   curl https://server.smithery.ai/@manasp21/rabi-mcp/health"
    echo "   curl -X POST https://server.smithery.ai/@manasp21/rabi-mcp/mcp/tools"
    echo ""
    echo "📚 Available Physics Tools:"
    echo "   - simulate_two_level_atom: Quantum dynamics simulations"
    echo "   - rabi_oscillations: Coherent oscillation analysis"
    echo "   - bec_simulation: Bose-Einstein condensate dynamics"
    echo "   - absorption_spectrum: Spectroscopic analysis"
    echo "   - cavity_qed: Quantum optics simulations"
    echo "   - plot_bloch_sphere: 3D quantum visualization"
    echo "   - And 19+ more advanced AMO physics tools!"
    echo ""
else
    echo ""
    echo "❌ Deployment failed!"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Check your internet connection"
    echo "   2. Verify GitHub repository is accessible"
    echo "   3. Ensure smithery.yaml is correctly configured"
    echo "   4. Try running: npx @smithery/cli --help"
    echo ""
    echo "📧 If issues persist:"
    echo "   - Check GitHub Issues: https://github.com/manasp21/rabi-mcp/issues"
    echo "   - Smithery documentation: https://smithery.ai/docs"
    echo ""
    exit 1
fi