#!/usr/bin/env python3
"""
Simple test to validate OmniSearch integration in Docker environment.
"""

import subprocess
import json
import time

def test_mcp_server_direct():
    """Test the MCP server directly using Node.js."""
    print("🔍 Testing MCP-OmniSearch Server Directly...")
    
    # Test message for MCP server
    init_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    
    try:
        # Start the MCP server process
        proc = subprocess.Popen(
            ["node", "/app/bot/mcp/omnisearch-server/dist/index.js"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                'TAVILY_API_KEY': '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',
                'PERPLEXITY_API_KEY': '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',
                'JINA_AI_API_KEY': 'jina_12fef9cf92a844d4a7d5ad0d984a2baczkxPW3o2Svq6DyX8GDvr6AziM-D5',
                'FIRECRAWL_API_KEY': 'fc-5ae56c3903cc47a19217e66a28d2f32d'
            }
        )
        
        # Send initialization message
        proc.stdin.write(json.dumps(init_message) + "\n")
        proc.stdin.flush()
        
        # Wait for response
        time.sleep(2)
        
        # Try to read response
        response = proc.stdout.readline()
        if response:
            print(f"✅ MCP Server Response: {response.strip()}")
            result = json.loads(response.strip())
            if result.get("result"):
                print("✅ MCP Server initialized successfully!")
                return True
        else:
            print("❌ No response from MCP server")
            
        # Check stderr for any errors
        stderr_output = proc.stderr.read()
        if stderr_output:
            print(f"⚠️ Server stderr: {stderr_output}")
            
        proc.terminate()
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False
    
    return False

def test_tavily_search():
    """Test a simple search using the MCP server."""
    print("\n📰 Testing Tavily Search...")
    
    # Search message
    search_message = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "tavily_search",
            "arguments": {
                "query": "Bitcoin price today"
            }
        }
    }
    
    try:
        proc = subprocess.Popen(
            ["node", "/app/bot/mcp/omnisearch-server/dist/index.js"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                'TAVILY_API_KEY': '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',
                'PERPLEXITY_API_KEY': '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',
                'JINA_AI_API_KEY': 'jina_12fef9cf92a844d4a7d5ad0d984a2baczkxPW3o2Svq6DyX8GDvr6AziM-D5',
                'FIRECRAWL_API_KEY': 'fc-5ae56c3903cc47a19217e66a28d2f32d'
            }
        )
        
        # Initialize first
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        proc.stdin.write(json.dumps(init_message) + "\n")
        proc.stdin.flush()
        time.sleep(1)
        
        # Read initialization response
        init_response = proc.stdout.readline()
        print(f"Init response: {init_response.strip()}")
        
        # Send search request
        proc.stdin.write(json.dumps(search_message) + "\n")
        proc.stdin.flush()
        time.sleep(3)  # Give more time for API call
        
        # Read search response
        search_response = proc.stdout.readline()
        if search_response:
            print(f"✅ Search Response: {search_response.strip()}")
            result = json.loads(search_response.strip())
            if result.get("result"):
                print("✅ Search completed successfully!")
                return True
        
        proc.terminate()
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False
    
    return False

if __name__ == "__main__":
    print("🚀 Testing MCP-OmniSearch in Docker Environment\n")
    
    # Test 1: Direct MCP server
    mcp_test = test_mcp_server_direct()
    
    # Test 2: Search functionality
    search_test = test_tavily_search()
    
    print(f"\n📋 Test Results:")
    print(f"  MCP Server: {'✅ PASS' if mcp_test else '❌ FAIL'}")
    print(f"  Search Test: {'✅ PASS' if search_test else '❌ FAIL'}")
    
    if mcp_test or search_test:
        print("\n🎉 MCP-OmniSearch is working!")
    else:
        print("\n⚠️ Tests failed. MCP server may need debugging.")