#!/usr/bin/env node

/**
 * WebSocket Connection Validation Script
 * This script validates the WebSocket setup between frontend and backend
 */

const { execSync } = require('child_process');
const net = require('net');

console.log('🔍 WebSocket Setup Validation');
console.log('===============================\n');

// Check if backend is running
function checkBackend() {
    console.log('📡 Checking backend server...');
    
    return new Promise((resolve) => {
        const socket = new net.Socket();
        
        socket.setTimeout(3000);
        
        socket.on('connect', () => {
            console.log('✅ Backend server is running on port 8000');
            socket.destroy();
            resolve(true);
        });
        
        socket.on('timeout', () => {
            console.log('❌ Backend server connection timeout on port 8000');
            socket.destroy();
            resolve(false);
        });
        
        socket.on('error', () => {
            console.log('❌ Backend server is not running on port 8000');
            resolve(false);
        });
        
        socket.connect(8000, 'localhost');
    });
}

// Check if frontend dev server is running
function checkFrontend(port) {
    console.log(`🌐 Checking frontend server on port ${port}...`);
    
    return new Promise((resolve) => {
        const socket = new net.Socket();
        
        socket.setTimeout(3000);
        
        socket.on('connect', () => {
            console.log(`✅ Frontend server is running on port ${port}`);
            socket.destroy();
            resolve(true);
        });
        
        socket.on('timeout', () => {
            console.log(`❌ Frontend server connection timeout on port ${port}`);
            socket.destroy();
            resolve(false);
        });
        
        socket.on('error', () => {
            console.log(`❌ Frontend server is not running on port ${port}`);
            resolve(false);
        });
        
        socket.connect(port, 'localhost');
    });
}

// Check WebSocket endpoint
async function checkWebSocketEndpoint() {
    console.log('🔌 Checking WebSocket endpoint...');
    
    try {
        const response = execSync('curl -s -I -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost:8000/ws', {
            encoding: 'utf8',
            timeout: 5000
        });
        
        if (response.includes('400 Bad Request')) {
            console.log('✅ WebSocket endpoint exists (returns 400 as expected for invalid key)');
            return true;
        } else if (response.includes('404')) {
            console.log('❌ WebSocket endpoint not found (404)');
            return false;
        } else {
            console.log('⚠️ Unexpected response from WebSocket endpoint:', response.split('\n')[0]);
            return false;
        }
    } catch (error) {
        console.log('❌ Failed to check WebSocket endpoint:', error.message);
        return false;
    }
}

// Check proxy configuration
async function checkProxy() {
    console.log('🔀 Checking proxy configuration...');
    
    // Check common frontend ports
    const frontendPorts = [3000, 3001, 3002, 5173, 5174];
    
    for (const port of frontendPorts) {
        const isRunning = await checkFrontend(port);
        
        if (isRunning) {
            try {
                const response = execSync(`curl -s -H "Upgrade: websocket" -H "Connection: Upgrade" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" -H "Sec-WebSocket-Version: 13" http://localhost:${port}/ws`, {
                    encoding: 'utf8',
                    timeout: 5000
                });
                
                if (response.includes('Failed to open a WebSocket connection') || response.includes('Switching Protocols')) {
                    console.log(`✅ WebSocket proxy working on port ${port}`);
                    return port;
                } else {
                    console.log(`❌ WebSocket proxy not working on port ${port}`);
                }
            } catch (error) {
                console.log(`❌ Failed to test proxy on port ${port}`);
            }
        }
    }
    
    return null;
}

// Main validation function
async function validateSetup() {
    console.log('Starting validation...\n');
    
    const backendRunning = await checkBackend();
    const wsEndpointWorking = await checkWebSocketEndpoint();
    const workingPort = await checkProxy();
    
    console.log('\n📊 Validation Summary:');
    console.log('====================');
    console.log(`Backend (port 8000): ${backendRunning ? '✅ Running' : '❌ Not running'}`);
    console.log(`WebSocket endpoint: ${wsEndpointWorking ? '✅ Available' : '❌ Not available'}`);
    console.log(`Proxy configuration: ${workingPort ? `✅ Working on port ${workingPort}` : '❌ Not working'}`);
    
    if (backendRunning && wsEndpointWorking && workingPort) {
        console.log('\n🎉 All checks passed!');
        console.log(`👉 Frontend should connect to: ws://localhost:${workingPort}/ws`);
        return { success: true, frontendPort: workingPort };
    } else {
        console.log('\n❌ Some checks failed. Please fix the issues above.');
        
        if (!backendRunning) {
            console.log('💡 Start the backend with: cd ../backend && python main.py');
        }
        
        if (!workingPort) {
            console.log('💡 Start the frontend with: npm run dev');
        }
        
        return { success: false };
    }
}

// Run validation
validateSetup().then((result) => {
    process.exit(result.success ? 0 : 1);
}).catch((error) => {
    console.error('❌ Validation failed:', error);
    process.exit(1);
});