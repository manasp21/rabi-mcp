#!/usr/bin/env python3
"""
Test the basic structure and imports of Rabi MCP Server
This test verifies the project structure without requiring heavy dependencies.
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_project_structure():
    """Test that all required files and directories exist."""
    
    print("🔍 Testing project structure...")
    
    base_dir = Path(__file__).parent
    
    # Required files
    required_files = [
        "README.md",
        "smithery.yaml", 
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "scripts/install.sh",
        "src/__init__.py",
        "src/mcp_server.py",
        "src/__main__.py",
        "src/config/__init__.py",
        "src/config/settings.py",
        "src/tools/__init__.py",
        "src/tools/quantum_systems.py",
        "src/tools/spectroscopy.py",
        "src/tools/visualization.py",
        "src/tools/cold_atoms.py",
        "src/tools/quantum_optics.py",
        "src/tools/utilities.py",
        "src/models/__init__.py",
        "src/models/hamiltonians.py",
        "src/resources/__init__.py",
        "src/resources/constants.py",
        "tests/__init__.py",
        "tests/test_quantum_systems.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True


def test_python_syntax():
    """Test that Python files have valid syntax."""
    
    print("\n🐍 Testing Python syntax...")
    
    base_dir = Path(__file__).parent
    python_files = [
        "src/mcp_server.py",
        "src/config/settings.py", 
        "src/tools/quantum_systems.py",
        "src/tools/spectroscopy.py",
        "src/tools/visualization.py",
        "src/tools/cold_atoms.py",
        "src/tools/quantum_optics.py",
        "src/tools/utilities.py",
        "src/models/hamiltonians.py",
        "src/resources/constants.py",
    ]
    
    syntax_errors = []
    for file_path in python_files:
        full_path = base_dir / file_path
        try:
            with open(full_path, 'r') as f:
                source = f.read()
            compile(source, file_path, 'exec')
            print(f"  ✅ {file_path}")
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"  ❌ {file_path}: {e}")
        except Exception as e:
            print(f"  ⚠️  {file_path}: Could not read file - {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found: {len(syntax_errors)}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True


def test_configuration_files():
    """Test configuration files are valid."""
    
    print("\n⚙️ Testing configuration files...")
    
    base_dir = Path(__file__).parent
    
    # Test JSON files
    json_files = ["package.json"]
    
    for json_file in json_files:
        try:
            import json
            with open(base_dir / json_file, 'r') as f:
                json.load(f)
            print(f"  ✅ {json_file} is valid JSON")
        except Exception as e:
            print(f"  ❌ {json_file}: {e}")
            return False
    
    # Test YAML files  
    yaml_files = ["smithery.yaml"]
    
    for yaml_file in yaml_files:
        try:
            # Basic YAML structure check (without pyyaml dependency)
            with open(base_dir / yaml_file, 'r') as f:
                content = f.read()
                # Basic checks
                if "version:" in content and "runtime:" in content:
                    print(f"  ✅ {yaml_file} has required fields")
                else:
                    print(f"  ❌ {yaml_file} missing required fields")
                    return False
        except Exception as e:
            print(f"  ❌ {yaml_file}: {e}")
            return False
    
    print("✅ Configuration files are valid")
    return True


def test_deployment_readiness():
    """Test deployment configuration."""
    
    print("\n🚀 Testing deployment readiness...")
    
    base_dir = Path(__file__).parent
    
    # Check Dockerfile
    dockerfile_path = base_dir / "Dockerfile"
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
            
        if "FROM python" in dockerfile_content:
            print("  ✅ Dockerfile has Python base image")
        else:
            print("  ❌ Dockerfile missing Python base image")
            return False
            
        if "src.mcp_server" in dockerfile_content:
            print("  ✅ Dockerfile runs MCP server")
        else:
            print("  ❌ Dockerfile not configured for MCP server")
            return False
    
    # Check smithery.yaml
    smithery_path = base_dir / "smithery.yaml"
    if smithery_path.exists():
        with open(smithery_path, 'r') as f:
            smithery_content = f.read()
            
        if 'type: "mcp"' in smithery_content:
            print("  ✅ Smithery configured for MCP protocol")
        else:
            print("  ❌ Smithery not configured for MCP protocol")
            return False
    
    # Check install script
    install_script = base_dir / "scripts" / "install.sh"
    if install_script.exists() and os.access(install_script, os.X_OK):
        print("  ✅ Install script is executable")
    else:
        print("  ❌ Install script not executable")
        return False
    
    print("✅ Deployment configuration ready")
    return True


def test_mcp_server_structure():
    """Test MCP server file structure."""
    
    print("\n🔌 Testing MCP server structure...")
    
    base_dir = Path(__file__).parent
    mcp_server_path = base_dir / "src" / "mcp_server.py"
    
    if not mcp_server_path.exists():
        print("  ❌ MCP server file missing")
        return False
    
    with open(mcp_server_path, 'r') as f:
        server_content = f.read()
    
    # Check for essential MCP components
    required_imports = [
        "from mcp.server",
        "from mcp.types",
        "asyncio",
    ]
    
    for import_stmt in required_imports:
        if import_stmt in server_content:
            print(f"  ✅ Has {import_stmt}")
        else:
            print(f"  ❌ Missing {import_stmt}")
            return False
    
    # Check for essential functions
    required_functions = [
        "list_tools",
        "call_tool",
        "async def main",
    ]
    
    for func in required_functions:
        if func in server_content:
            print(f"  ✅ Has {func}")
        else:
            print(f"  ❌ Missing {func}")
            return False
    
    print("✅ MCP server structure is correct")
    return True


def main():
    """Run all tests."""
    
    print("🔬 Rabi MCP Server - Structure & Configuration Test")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_python_syntax, 
        test_configuration_files,
        test_deployment_readiness,
        test_mcp_server_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("✅ Project is ready for deployment")
        print("\n📋 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Test functionality: python test_mcp_server.py")  
        print("   3. Deploy to Smithery: npx @smithery/cli deploy")
        print("   4. Or run locally: python -m src.mcp_server")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("Please fix the issues above before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)