#!/usr/bin/env python3
"""
Diagnostic entry point to identify deployment issues
Shows exactly why advanced server might fall back to basic
"""

import sys
import logging
import time
import os

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diagnostic-server")

def main():
    """Diagnostic main with detailed failure analysis"""
    try:
        logger.info("🔍 DIAGNOSTIC MODE: Analyzing deployment environment...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path[:3]}...")
        
        # Test 1: Check file structure
        logger.info("📁 Checking file structure...")
        required_files = [
            "src/advanced_server.py",
            "src/http_server.py", 
            "smithery.yaml"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                logger.info(f"   ✅ {file_path} exists")
            else:
                logger.error(f"   ❌ {file_path} MISSING")
        
        # Test 2: Check imports
        logger.info("📦 Testing imports...")
        try:
            import numpy as np
            logger.info(f"   ✅ numpy {np.__version__} available")
        except ImportError:
            logger.error("   ❌ numpy not available")
        
        try:
            import scipy
            logger.info(f"   ✅ scipy {scipy.__version__} available")
        except ImportError:
            logger.warning("   ⚠️  scipy not available (optional)")
        
        # Test 3: Try advanced server import
        logger.info("🔬 Testing advanced server import...")
        try:
            sys.path.append('src')
            from advanced_server import AdvancedAMOServer
            
            server = AdvancedAMOServer()
            tools = server.get_all_tool_specs()
            
            logger.info(f"   ✅ Advanced server imported successfully!")
            logger.info(f"   ✅ {len(tools)} tools available")
            
            # Test tool breakdown
            core_tools = [t for t in tools if '[CORE]' in t['description']]
            advanced_tools = [t for t in tools if '[ADVANCED]' in t['description']]
            research_tools = [t for t in tools if '[RESEARCH]' in t['description']]
            
            logger.info(f"   📊 Tool breakdown: {len(core_tools)} core, {len(advanced_tools)} advanced, {len(research_tools)} research")
            
            # Start the advanced server
            logger.info("🚀 Starting advanced server...")
            from advanced_server import main as server_main
            server_main()
            
        except ImportError as e:
            logger.error(f"   ❌ Advanced server import failed: {e}")
            logger.info("🔄 Falling back to basic server...")
            
            try:
                from http_server import main as basic_server_main
                logger.info("   ✅ Basic server loaded - 5 tools available")
                basic_server_main()
            except Exception as fallback_error:
                logger.error(f"   ❌ Basic server also failed: {fallback_error}")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"   ❌ Advanced server failed: {e}")
            import traceback
            traceback.print_exc()
            
            logger.info("🔄 Falling back to basic server...")
            try:
                from http_server import main as basic_server_main
                logger.info("   ✅ Basic server loaded - 5 tools available")
                basic_server_main()
            except Exception as fallback_error:
                logger.error(f"   ❌ Basic server also failed: {fallback_error}")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()