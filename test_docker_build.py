#!/usr/bin/env python3
"""
Test Docker build and basic functionality for Rabi MCP Server.
This test verifies Docker deployment without needing local Python dependencies.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_command(cmd, timeout=60):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_docker_build():
    """Test Docker build process."""
    print("🐳 Testing Docker build...")
    
    # Check if Docker is available
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("  ❌ Docker not available - skipping Docker tests")
        print(f"     Error: {stderr}")
        return False
    
    print(f"  ✅ Docker version: {stdout.strip()}")
    
    # Build the Docker image
    print("  🔨 Building Docker image...")
    build_cmd = "docker build -t rabi-mcp-test:latest ."
    success, stdout, stderr = run_command(build_cmd, timeout=300)  # 5 minute timeout
    
    if not success:
        print("  ❌ Docker build failed")
        print(f"     stdout: {stdout}")
        print(f"     stderr: {stderr}")
        return False
    
    print("  ✅ Docker image built successfully")
    return True

def test_docker_run():
    """Test running the Docker container."""
    print("\n🚀 Testing Docker container...")
    
    # Test container can start (quick test)
    test_cmd = "docker run --rm rabi-mcp-test:latest python -c \"print('Container test successful')\""
    success, stdout, stderr = run_command(test_cmd, timeout=30)
    
    if not success:
        print("  ❌ Container failed to run")
        print(f"     stdout: {stdout}")
        print(f"     stderr: {stderr}")
        return False
    
    if "Container test successful" in stdout:
        print("  ✅ Container runs successfully")
    else:
        print("  ❌ Container output unexpected")
        return False
    
    # Test MCP server structure in container
    print("  🔍 Testing MCP server structure in container...")
    
    structure_cmd = """docker run --rm rabi-mcp-test:latest python -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.mcp_server import RabiMCPServer
    server = RabiMCPServer()
    print('MCP server class instantiated successfully')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Other error: {e}')
    sys.exit(1)
"
"""
    
    success, stdout, stderr = run_command(structure_cmd, timeout=30)
    
    if not success:
        print("  ❌ MCP server structure test failed")
        print(f"     stdout: {stdout}")
        print(f"     stderr: {stderr}")
        return False
    
    if "MCP server class instantiated successfully" in stdout:
        print("  ✅ MCP server structure verified in container")
    else:
        print("  ❌ MCP server structure test failed")
        print(f"     Output: {stdout}")
        return False
    
    return True

def test_smithery_config():
    """Test Smithery configuration validity."""
    print("\n☁️ Testing Smithery configuration...")
    
    # Read smithery.yaml
    smithery_file = Path("smithery.yaml")
    if not smithery_file.exists():
        print("  ❌ smithery.yaml not found")
        return False
    
    with open(smithery_file, 'r') as f:
        content = f.read()
    
    # Check required fields
    required_fields = [
        'version: 1',
        'runtime: "container"',
        'type: "mcp"',
        'src.mcp_server',
        'configSchema:'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in content:
            missing_fields.append(field)
        else:
            print(f"  ✅ Found: {field}")
    
    if missing_fields:
        print(f"  ❌ Missing fields: {missing_fields}")
        return False
    
    print("  ✅ Smithery configuration is valid")
    return True

def test_installation_methods():
    """Test various installation method configurations."""
    print("\n📦 Testing installation method configurations...")
    
    # Test package.json MCP configuration
    package_file = Path("package.json")
    if package_file.exists():
        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            if "mcp" in package_data and "server" in package_data["mcp"]:
                mcp_config = package_data["mcp"]["server"]
                if "src.mcp_server" in str(mcp_config.get("args", [])):
                    print("  ✅ package.json MCP configuration correct")
                else:
                    print("  ❌ package.json MCP configuration incorrect")
                    return False
            else:
                print("  ❌ package.json missing MCP configuration")
                return False
        except json.JSONDecodeError:
            print("  ❌ package.json is not valid JSON")
            return False
    
    # Test Dockerfile configuration
    dockerfile = Path("Dockerfile")
    if dockerfile.exists():
        with open(dockerfile, 'r') as f:
            dockerfile_content = f.read()
        
        if "src.mcp_server" in dockerfile_content:
            print("  ✅ Dockerfile configured for MCP server")
        else:
            print("  ❌ Dockerfile not configured for MCP server")
            return False
    
    # Test install script
    install_script = Path("scripts/install.sh")
    if install_script.exists():
        with open(install_script, 'r') as f:
            script_content = f.read()
        
        if "src.mcp_server" in script_content:
            print("  ✅ Install script configured for MCP server")
        else:
            print("  ❌ Install script not configured for MCP server")
            return False
    
    print("  ✅ All installation methods properly configured")
    return True

def test_readme_examples():
    """Test that README examples are consistent."""
    print("\n📖 Testing README examples...")
    
    readme_file = Path("README.md")
    if not readme_file.exists():
        print("  ❌ README.md not found")
        return False
    
    with open(readme_file, 'r') as f:
        readme_content = f.read()
    
    # Check for consistent command usage
    if "src.mcp_server" in readme_content:
        print("  ✅ README uses correct MCP server command")
    else:
        print("  ❌ README doesn't use correct MCP server command")
        return False
    
    # Check for multiple installation methods
    installation_methods = [
        "Method 1: One-Line Auto-Install",
        "Method 2: Smithery Cloud", 
        "Method 3: Docker",
        "Method 4: Manual Installation",
        "Method 5: Claude Desktop Integration"
    ]
    
    missing_methods = []
    for method in installation_methods:
        if method not in readme_content:
            missing_methods.append(method)
        else:
            print(f"  ✅ Found: {method}")
    
    if missing_methods:
        print(f"  ❌ Missing installation methods: {missing_methods}")
        return False
    
    print("  ✅ README has comprehensive installation guide")
    return True

def main():
    """Run all tests."""
    print("🧪 Rabi MCP Server - Installation & Deployment Test")
    print("=" * 70)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    tests = [
        ("Smithery Configuration", test_smithery_config),
        ("Installation Methods", test_installation_methods), 
        ("README Examples", test_readme_examples),
        ("Docker Build", test_docker_build),
        ("Docker Run", test_docker_run),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔧 Running: {test_name}")
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All deployment tests passed!")
        print("✅ Rabi MCP Server is ready for production deployment")
        print("\n🚀 Deployment Options:")
        print("   1. Smithery Cloud: npx @smithery/cli deploy")
        print("   2. Docker Hub: docker push manasp21/rabi-mcp-server")
        print("   3. Local Install: curl -sSL install.sh | bash")
        print("   4. Manual Install: git clone && pip install -r requirements.txt")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("Please fix the issues above before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)