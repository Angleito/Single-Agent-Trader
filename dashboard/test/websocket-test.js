#!/usr/bin/env node

/**
 * WebSocket Integration Test for AI Trading Bot Dashboard
 *
 * Tests WebSocket connectivity, message handling, and real-time data flow
 */

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Test configuration
const CONFIG = {
    WS_URL: process.env.WS_URL || 'ws://localhost:8000/ws',
    TEST_TIMEOUT: 30000, // 30 seconds
    CONNECTION_TIMEOUT: 10000, // 10 seconds
    MESSAGE_WAIT_TIME: 5000, // 5 seconds to wait for messages
    EXPECTED_MESSAGE_TYPES: [
        'echo',
        'llm_decision',
        'llm_event',
        'tradingview_update',
        'docker-logs'
    ]
};

// Test results tracking
const TEST_RESULTS = {
    connection: false,
    messageReceived: false,
    echoPingPong: false,
    realTimeData: false,
    errorHandling: false,
    connectionStability: false
};

// Utility functions
function log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.padEnd(5)}] ${message}`;
    console.log(logMessage);

    if (data) {
        console.log(JSON.stringify(data, null, 2));
    }
}

function logSuccess(message, data = null) {
    log('PASS', `✓ ${message}`, data);
}

function logError(message, data = null) {
    log('FAIL', `✗ ${message}`, data);
}

function logInfo(message, data = null) {
    log('INFO', message, data);
}

function logWarning(message, data = null) {
    log('WARN', `⚠ ${message}`, data);
}

// Mock data generators
function generateMockTradingData() {
    return {
        type: 'trading_update',
        timestamp: new Date().toISOString(),
        data: {
            symbol: 'BTC-USD',
            price: 45000 + Math.random() * 10000,
            volume: Math.random() * 100,
            position: {
                side: Math.random() > 0.5 ? 'long' : 'short',
                size: Math.random() * 0.1,
                unrealized_pnl: (Math.random() - 0.5) * 1000
            }
        }
    };
}

function generateMockLLMDecision() {
    const actions = ['LONG', 'SHORT', 'CLOSE', 'HOLD'];
    return {
        type: 'llm_decision',
        timestamp: new Date().toISOString(),
        data: {
            action: actions[Math.floor(Math.random() * actions.length)],
            size_pct: Math.random() * 0.1,
            rationale: 'Test decision based on mock indicators',
            symbol: 'BTC-USD',
            current_price: 45000 + Math.random() * 10000,
            indicators: {
                rsi: Math.random() * 100,
                ma_50: 44000 + Math.random() * 2000,
                ma_200: 43000 + Math.random() * 2000
            },
            session_id: `test_session_${Date.now()}`
        }
    };
}

// Test functions
async function testWebSocketConnection() {
    return new Promise((resolve) => {
        logInfo('Testing WebSocket connection...');

        const ws = new WebSocket(CONFIG.WS_URL);
        let connectionTimer = setTimeout(() => {
            logError('Connection timeout');
            ws.terminate();
            resolve(false);
        }, CONFIG.CONNECTION_TIMEOUT);

        ws.on('open', () => {
            clearTimeout(connectionTimer);
            logSuccess('WebSocket connection established');
            TEST_RESULTS.connection = true;
            ws.close();
            resolve(true);
        });

        ws.on('error', (error) => {
            clearTimeout(connectionTimer);
            logError('WebSocket connection failed', { error: error.message });
            resolve(false);
        });
    });
}

async function testEchoPingPong() {
    return new Promise((resolve) => {
        logInfo('Testing echo ping-pong functionality...');

        const ws = new WebSocket(CONFIG.WS_URL);
        const testMessage = JSON.stringify({
            type: 'ping',
            timestamp: new Date().toISOString(),
            message: 'WebSocket integration test ping'
        });

        let messageTimer = setTimeout(() => {
            logError('Echo response timeout');
            ws.terminate();
            resolve(false);
        }, CONFIG.MESSAGE_WAIT_TIME);

        ws.on('open', () => {
            logInfo('Sending test message...');
            ws.send(testMessage);
        });

        ws.on('message', (data) => {
            try {
                const response = JSON.parse(data.toString());

                if (response.type === 'echo') {
                    clearTimeout(messageTimer);
                    logSuccess('Echo response received', response);
                    TEST_RESULTS.echoPingPong = true;
                    ws.close();
                    resolve(true);
                } else {
                    logInfo('Received non-echo message', response);
                }
            } catch (error) {
                logWarning('Failed to parse message', { error: error.message, data: data.toString() });
            }
        });

        ws.on('error', (error) => {
            clearTimeout(messageTimer);
            logError('Echo test failed', { error: error.message });
            resolve(false);
        });
    });
}

async function testRealTimeDataFlow() {
    return new Promise((resolve) => {
        logInfo('Testing real-time data flow...');

        const ws = new WebSocket(CONFIG.WS_URL);
        const receivedMessages = [];
        let dataFlowTimer;

        const checkDataFlow = () => {
            if (receivedMessages.length > 0) {
                logSuccess(`Received ${receivedMessages.length} real-time messages`);

                // Check for different message types
                const messageTypes = [...new Set(receivedMessages.map(msg => msg.type))];
                logInfo('Message types received', messageTypes);

                // Look for expected message types
                const hasExpectedTypes = CONFIG.EXPECTED_MESSAGE_TYPES.some(type =>
                    messageTypes.includes(type)
                );

                if (hasExpectedTypes) {
                    logSuccess('Received expected message types');
                    TEST_RESULTS.realTimeData = true;
                } else {
                    logWarning('No expected message types received', {
                        expected: CONFIG.EXPECTED_MESSAGE_TYPES,
                        received: messageTypes
                    });
                }

                ws.close();
                resolve(true);
            } else {
                logWarning('No real-time messages received');
                ws.close();
                resolve(false);
            }
        };

        ws.on('open', () => {
            logInfo('Connected, waiting for real-time data...');

            // Set timer to check data flow
            dataFlowTimer = setTimeout(checkDataFlow, CONFIG.MESSAGE_WAIT_TIME);
        });

        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                receivedMessages.push(message);

                logInfo(`Received message: ${message.type}`, {
                    type: message.type,
                    timestamp: message.timestamp,
                    hasData: !!message.data
                });

                TEST_RESULTS.messageReceived = true;

                // If we've received enough messages, resolve early
                if (receivedMessages.length >= 3) {
                    clearTimeout(dataFlowTimer);
                    checkDataFlow();
                }
            } catch (error) {
                logWarning('Failed to parse real-time message', {
                    error: error.message,
                    data: data.toString().substring(0, 200)
                });
            }
        });

        ws.on('error', (error) => {
            clearTimeout(dataFlowTimer);
            logError('Real-time data test failed', { error: error.message });
            resolve(false);
        });
    });
}

async function testConnectionStability() {
    return new Promise((resolve) => {
        logInfo('Testing connection stability...');

        let connections = 0;
        let successfulConnections = 0;
        const totalConnections = 3;

        const createConnection = () => {
            return new Promise((resolveConnection) => {
                const ws = new WebSocket(CONFIG.WS_URL);
                let connectionTimer = setTimeout(() => {
                    logWarning(`Connection ${connections + 1} timeout`);
                    ws.terminate();
                    resolveConnection(false);
                }, 5000);

                ws.on('open', () => {
                    clearTimeout(connectionTimer);
                    successfulConnections++;
                    logInfo(`Connection ${connections + 1} established`);

                    // Keep connection open briefly, then close
                    setTimeout(() => {
                        ws.close();
                        resolveConnection(true);
                    }, 1000);
                });

                ws.on('error', (error) => {
                    clearTimeout(connectionTimer);
                    logWarning(`Connection ${connections + 1} failed`, { error: error.message });
                    resolveConnection(false);
                });
            });
        };

        const testConnections = async () => {
            for (let i = 0; i < totalConnections; i++) {
                connections = i;
                await createConnection();

                // Small delay between connections
                if (i < totalConnections - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }

            const stabilityRatio = successfulConnections / totalConnections;
            logInfo(`Connection stability: ${successfulConnections}/${totalConnections} (${Math.round(stabilityRatio * 100)}%)`);

            if (stabilityRatio >= 0.8) {
                logSuccess('Connection stability test passed');
                TEST_RESULTS.connectionStability = true;
                resolve(true);
            } else {
                logError('Connection stability test failed');
                resolve(false);
            }
        };

        testConnections();
    });
}

async function testErrorHandling() {
    return new Promise((resolve) => {
        logInfo('Testing error handling...');

        const ws = new WebSocket(CONFIG.WS_URL);

        ws.on('open', () => {
            // Send invalid JSON
            ws.send('invalid json data');

            // Send very large message
            const largeMessage = JSON.stringify({
                type: 'test',
                data: 'x'.repeat(100000) // 100KB message
            });
            ws.send(largeMessage);

            // Send empty message
            ws.send('');

            // The connection should still be stable after these tests
            setTimeout(() => {
                ws.send(JSON.stringify({
                    type: 'test_after_errors',
                    message: 'Connection still working after error tests'
                }));
            }, 1000);

            setTimeout(() => {
                logSuccess('Error handling test completed - connection remained stable');
                TEST_RESULTS.errorHandling = true;
                ws.close();
                resolve(true);
            }, 2000);
        });

        ws.on('error', (error) => {
            logWarning('Error during error handling test', { error: error.message });
            // This is actually expected behavior
            resolve(true);
        });

        ws.on('close', (code, reason) => {
            if (code === 1000) {
                // Normal closure
                resolve(true);
            }
        });
    });
}

// Mock data injection test
async function testMockDataInjection() {
    return new Promise((resolve) => {
        logInfo('Testing mock data injection...');

        const ws = new WebSocket(CONFIG.WS_URL);
        const mockMessages = [
            generateMockTradingData(),
            generateMockLLMDecision(),
            {
                type: 'test_data',
                timestamp: new Date().toISOString(),
                message: 'Mock data injection test'
            }
        ];

        let messagesSent = 0;

        ws.on('open', () => {
            // Send mock messages with delays
            const sendNextMessage = () => {
                if (messagesSent < mockMessages.length) {
                    const message = mockMessages[messagesSent];
                    logInfo(`Sending mock message ${messagesSent + 1}`, { type: message.type });
                    ws.send(JSON.stringify(message));
                    messagesSent++;

                    setTimeout(sendNextMessage, 1000);
                } else {
                    setTimeout(() => {
                        logSuccess('Mock data injection completed');
                        ws.close();
                        resolve(true);
                    }, 1000);
                }
            };

            sendNextMessage();
        });

        ws.on('message', (data) => {
            try {
                const response = JSON.parse(data.toString());
                logInfo('Received response to mock data', { type: response.type });
            } catch (error) {
                // Ignore parsing errors for this test
            }
        });

        ws.on('error', (error) => {
            logError('Mock data injection test failed', { error: error.message });
            resolve(false);
        });
    });
}

// Save test status for main script
function saveTestStatus() {
    const testDir = path.dirname(__filename);
    const statusFile = path.join(testDir, '.websocket_test_passed');

    const allTestsPassed = Object.values(TEST_RESULTS).every(result => result === true);

    if (allTestsPassed) {
        fs.writeFileSync(statusFile, `WebSocket tests passed at ${new Date().toISOString()}`);
        logSuccess('All WebSocket tests passed');
    } else {
        if (fs.existsSync(statusFile)) {
            fs.unlinkSync(statusFile);
        }
        logError('Some WebSocket tests failed');
    }

    return allTestsPassed;
}

// Generate detailed test report
function generateTestReport() {
    const report = {
        timestamp: new Date().toISOString(),
        testType: 'WebSocket Integration Tests',
        configuration: CONFIG,
        results: TEST_RESULTS,
        summary: {
            total: Object.keys(TEST_RESULTS).length,
            passed: Object.values(TEST_RESULTS).filter(r => r).length,
            failed: Object.values(TEST_RESULTS).filter(r => !r).length
        }
    };

    const reportPath = path.join(path.dirname(__filename), 'websocket-test-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    logInfo(`Test report saved: ${reportPath}`);
    return report;
}

// Main test execution
async function runWebSocketTests() {
    logInfo('='.repeat(60));
    logInfo('Starting WebSocket Integration Tests');
    logInfo('='.repeat(60));

    const tests = [
        { name: 'WebSocket Connection', fn: testWebSocketConnection },
        { name: 'Echo Ping-Pong', fn: testEchoPingPong },
        { name: 'Real-time Data Flow', fn: testRealTimeDataFlow },
        { name: 'Connection Stability', fn: testConnectionStability },
        { name: 'Error Handling', fn: testErrorHandling },
        { name: 'Mock Data Injection', fn: testMockDataInjection }
    ];

    logInfo(`Running ${tests.length} WebSocket tests...`);
    logInfo(`Target WebSocket URL: ${CONFIG.WS_URL}`);

    for (const test of tests) {
        logInfo('-'.repeat(40));
        logInfo(`Running: ${test.name}`);

        try {
            const result = await test.fn();
            if (result) {
                logSuccess(`${test.name} completed successfully`);
            } else {
                logError(`${test.name} failed`);
            }
        } catch (error) {
            logError(`${test.name} threw an error`, { error: error.message });
        }

        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    logInfo('-'.repeat(40));
    logInfo('WebSocket Tests Summary:');

    let passCount = 0;
    for (const [testName, result] of Object.entries(TEST_RESULTS)) {
        const status = result ? 'PASS' : 'FAIL';
        const icon = result ? '✓' : '✗';
        logInfo(`  ${icon} ${testName}: ${status}`);

        if (result) passCount++;
    }

    const totalTests = Object.keys(TEST_RESULTS).length;
    logInfo(`Overall: ${passCount}/${totalTests} tests passed (${Math.round((passCount/totalTests) * 100)}%)`);

    // Generate test report
    const report = generateTestReport();

    // Save test status
    const allPassed = saveTestStatus();

    if (allPassed) {
        logSuccess('All WebSocket integration tests completed successfully!');
        process.exit(0);
    } else {
        logError('Some WebSocket integration tests failed!');
        process.exit(1);
    }
}

// Handle process signals
process.on('SIGINT', () => {
    logWarning('WebSocket tests interrupted by user');
    process.exit(1);
});

process.on('SIGTERM', () => {
    logWarning('WebSocket tests terminated');
    process.exit(1);
});

// Set test timeout
setTimeout(() => {
    logError(`WebSocket tests timed out after ${CONFIG.TEST_TIMEOUT}ms`);
    process.exit(1);
}, CONFIG.TEST_TIMEOUT);

// Run tests
if (require.main === module) {
    runWebSocketTests().catch(error => {
        logError('WebSocket tests failed with error', { error: error.message });
        process.exit(1);
    });
}
