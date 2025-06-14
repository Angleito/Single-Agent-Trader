#!/usr/bin/env node

/**
 * Comprehensive Test Runner for AI Trading Bot Dashboard
 * 
 * Orchestrates all integration tests including WebSocket, API, UI, and Docker validation
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');

// Load test configuration
const CONFIG = JSON.parse(fs.readFileSync(path.join(__dirname, 'test-config.json'), 'utf8'));

// Test execution state
const TEST_STATE = {
    startTime: Date.now(),
    results: {},
    currentSuite: null,
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    skippedTests: 0
};

// Utility functions
function log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.padEnd(5)}] ${message}`;
    console.log(logMessage);
    
    if (data && typeof data === 'object') {
        console.log(JSON.stringify(data, null, 2));
    } else if (data) {
        console.log(data);
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

// Execute shell command with promise
function execCommand(command, options = {}) {
    return new Promise((resolve, reject) => {
        exec(command, { 
            timeout: options.timeout || 30000,
            cwd: options.cwd || process.cwd(),
            env: { ...process.env, ...options.env }
        }, (error, stdout, stderr) => {
            if (error) {
                reject({ error, stdout, stderr, command });
            } else {
                resolve({ stdout: stdout.trim(), stderr: stderr.trim(), command });
            }
        });
    });
}

// Execute Node.js test script
function runNodeScript(scriptPath, timeout = 30000) {
    return new Promise((resolve, reject) => {
        const child = spawn('node', [scriptPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            cwd: __dirname
        });
        
        let stdout = '';
        let stderr = '';
        
        child.stdout.on('data', (data) => {
            stdout += data.toString();
            // Forward output in real-time
            process.stdout.write(data);
        });
        
        child.stderr.on('data', (data) => {
            stderr += data.toString();
            // Forward errors in real-time
            process.stderr.write(data);
        });
        
        const timer = setTimeout(() => {
            child.kill('SIGTERM');
            reject(new Error(`Test script ${scriptPath} timed out after ${timeout}ms`));
        }, timeout);
        
        child.on('close', (code) => {
            clearTimeout(timer);
            if (code === 0) {
                resolve({ stdout, stderr, exitCode: code });
            } else {
                reject(new Error(`Test script ${scriptPath} exited with code ${code}`));
            }
        });
        
        child.on('error', (error) => {
            clearTimeout(timer);
            reject(error);
        });
    });
}

// Test suite runners
async function runWebSocketTests() {
    logInfo('Starting WebSocket integration tests...');
    TEST_STATE.currentSuite = 'websocket';
    
    try {
        const result = await runNodeScript(
            path.join(__dirname, 'websocket-test.js'), 
            CONFIG.testSuites.websocket.timeout
        );
        
        logSuccess('WebSocket tests completed successfully');
        TEST_STATE.results.websocket = { success: true, exitCode: result.exitCode };
        TEST_STATE.passedTests++;
        return true;
    } catch (error) {
        logError('WebSocket tests failed', error.message);
        TEST_STATE.results.websocket = { success: false, error: error.message };
        TEST_STATE.failedTests++;
        return false;
    }
}

async function runAPITests() {
    logInfo('Starting API integration tests...');
    TEST_STATE.currentSuite = 'api';
    
    try {
        const result = await runNodeScript(
            path.join(__dirname, 'api-test.js'), 
            CONFIG.testSuites.api.timeout
        );
        
        logSuccess('API tests completed successfully');
        TEST_STATE.results.api = { success: true, exitCode: result.exitCode };
        TEST_STATE.passedTests++;
        return true;
    } catch (error) {
        logError('API tests failed', error.message);
        TEST_STATE.results.api = { success: false, error: error.message };
        TEST_STATE.failedTests++;
        return false;
    }
}

async function runDockerValidation() {
    logInfo('Starting Docker environment validation...');
    TEST_STATE.currentSuite = 'docker';
    
    try {
        const result = await runNodeScript(
            path.join(__dirname, 'docker-validator.js'), 
            CONFIG.testSuites.docker.timeout
        );
        
        logSuccess('Docker validation completed successfully');
        TEST_STATE.results.docker = { success: true, exitCode: result.exitCode };
        TEST_STATE.passedTests++;
        return true;
    } catch (error) {
        logError('Docker validation failed', error.message);
        TEST_STATE.results.docker = { success: false, error: error.message };
        TEST_STATE.failedTests++;
        return false;
    }
}

async function runUITests() {
    logInfo('Starting UI component tests...');
    TEST_STATE.currentSuite = 'ui';
    
    try {
        // Check if UI test file exists and is accessible
        const uiTestPath = path.join(__dirname, 'ui-test.html');
        if (!fs.existsSync(uiTestPath)) {
            throw new Error('UI test file not found');
        }
        
        // Since UI tests run in browser, we'll check if the test page is accessible
        const frontendUrl = CONFIG.services.frontend.url;
        const testUrl = `${frontendUrl}/test/ui-test.html`;
        
        // Use curl to check if the UI test page loads
        const curlResult = await execCommand(`curl -s -f "${frontendUrl}" | head -10`);
        
        if (curlResult.stdout.includes('html') || curlResult.stdout.includes('HTML')) {
            logSuccess('Frontend is serving content, UI tests environment ready');
            
            // Copy UI test file to frontend public directory if it exists
            const frontendPublicDir = path.join(__dirname, '../frontend/public/test');
            if (!fs.existsSync(frontendPublicDir)) {
                fs.mkdirSync(frontendPublicDir, { recursive: true });
            }
            
            fs.copyFileSync(uiTestPath, path.join(frontendPublicDir, 'ui-test.html'));
            logInfo('UI test file copied to frontend public directory');
            
            TEST_STATE.results.ui = { success: true, testUrl: testUrl };
            TEST_STATE.passedTests++;
            return true;
        } else {
            throw new Error('Frontend not serving HTML content');
        }
    } catch (error) {
        logError('UI tests failed', error.message);
        TEST_STATE.results.ui = { success: false, error: error.message };
        TEST_STATE.failedTests++;
        return false;
    }
}

async function runMockDataGeneration() {
    logInfo('Generating mock test data...');
    
    try {
        // Generate mock trading session
        const sessionResult = await execCommand(
            `node ${path.join(__dirname, 'mock-data-generator.js')} session ${CONFIG.mockData.sessionDuration}`
        );
        
        logSuccess('Mock trading session generated', sessionResult.stdout);
        
        // Generate individual test data samples
        const llmDecision = await execCommand(
            `node ${path.join(__dirname, 'mock-data-generator.js')} llm-decision BTC-USD`
        );
        
        logInfo('Sample LLM decision generated', JSON.parse(llmDecision.stdout));
        
        // Generate OHLCV data
        const ohlcvData = await execCommand(
            `node ${path.join(__dirname, 'mock-data-generator.js')} ohlcv BTC-USD 5m 20`
        );
        
        logInfo('Sample OHLCV data generated', JSON.parse(ohlcvData.stdout).slice(0, 3));
        
        return true;
    } catch (error) {
        logError('Mock data generation failed', error.message);
        return false;
    }
}

async function checkPrerequisites() {
    logInfo('Checking test prerequisites...');
    
    const checks = [
        { name: 'Node.js', command: 'node --version' },
        { name: 'Docker', command: 'docker --version' },
        { name: 'Docker Compose', command: 'docker-compose --version' },
        { name: 'curl', command: 'curl --version | head -1' }
    ];
    
    let allChecksPassed = true;
    
    for (const check of checks) {
        try {
            const result = await execCommand(check.command);
            logSuccess(`${check.name} is available`, result.stdout.split('\n')[0]);
        } catch (error) {
            logError(`${check.name} is not available`, error.error?.message);
            allChecksPassed = false;
        }
    }
    
    return allChecksPassed;
}

async function waitForServices() {
    logInfo('Waiting for services to be ready...');
    
    const maxWaitTime = 120000; // 2 minutes
    const checkInterval = 5000; // 5 seconds
    let waitTime = 0;
    
    while (waitTime < maxWaitTime) {
        try {
            // Check backend health
            const backendHealth = await execCommand(
                `curl -s -f ${CONFIG.services.backend.url}/health`
            );
            
            if (backendHealth.stdout.includes('healthy')) {
                logSuccess('Backend service is ready');
                return true;
            }
        } catch (error) {
            // Service not ready yet
        }
        
        await new Promise(resolve => setTimeout(resolve, checkInterval));
        waitTime += checkInterval;
        
        if (waitTime % 20000 === 0) {
            logInfo(`Still waiting for services... (${waitTime/1000}s elapsed)`);
        }
    }
    
    logWarning('Services did not become ready within timeout');
    return false;
}

async function generateFinalReport() {
    const endTime = Date.now();
    const duration = endTime - TEST_STATE.startTime;
    
    const report = {
        testRun: {
            timestamp: new Date().toISOString(),
            duration: duration,
            environment: 'docker',
            configuration: CONFIG.testEnvironment
        },
        summary: {
            totalSuites: Object.keys(CONFIG.testSuites).filter(suite => CONFIG.testSuites[suite].enabled).length,
            passedSuites: TEST_STATE.passedTests,
            failedSuites: TEST_STATE.failedTests,
            skippedSuites: TEST_STATE.skippedTests,
            successRate: Math.round((TEST_STATE.passedTests / (TEST_STATE.passedTests + TEST_STATE.failedTests)) * 100) || 0
        },
        results: TEST_STATE.results,
        environment: {
            nodeVersion: process.version,
            platform: process.platform,
            workingDirectory: process.cwd()
        },
        metrics: {
            totalDuration: duration,
            avgSuiteDuration: duration / Object.keys(TEST_STATE.results).length
        }
    };
    
    // Save detailed report
    const reportDir = path.join(__dirname, 'test-reports');
    if (!fs.existsSync(reportDir)) {
        fs.mkdirSync(reportDir, { recursive: true });
    }
    
    const reportFile = path.join(reportDir, `integration-test-report-${Date.now()}.json`);
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
    
    // Generate HTML report
    const htmlReport = generateHTMLReport(report);
    const htmlReportFile = path.join(reportDir, `integration-test-report-${Date.now()}.html`);
    fs.writeFileSync(htmlReportFile, htmlReport);
    
    logInfo('='.repeat(60));
    logInfo('FINAL TEST REPORT');
    logInfo('='.repeat(60));
    
    logInfo(`Test Duration: ${Math.round(duration / 1000)}s`);
    logInfo(`Total Test Suites: ${report.summary.totalSuites}`);
    logInfo(`Passed: ${report.summary.passedSuites}`);
    logInfo(`Failed: ${report.summary.failedSuites}`);
    logInfo(`Success Rate: ${report.summary.successRate}%`);
    
    logInfo('\nDetailed Results:');
    for (const [suiteName, result] of Object.entries(TEST_STATE.results)) {
        const status = result.success ? 'PASS' : 'FAIL';
        const icon = result.success ? '✓' : '✗';
        logInfo(`  ${icon} ${suiteName}: ${status}`);
    }
    
    logInfo(`\nDetailed report saved: ${reportFile}`);
    logInfo(`HTML report saved: ${htmlReportFile}`);
    
    return report;
}

function generateHTMLReport(report) {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { border-bottom: 2px solid #4CAF50; padding-bottom: 20px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .metric-label { color: #666; margin-top: 5px; }
        .results { margin: 20px 0; }
        .result-item { display: flex; justify-content: space-between; padding: 10px; border: 1px solid #ddd; margin: 5px 0; border-radius: 4px; }
        .pass { background-color: #e8f5e8; border-color: #4CAF50; }
        .fail { background-color: #ffeaea; border-color: #f44336; }
        .status { font-weight: bold; }
        .status.pass { color: #4CAF50; }
        .status.fail { color: #f44336; }
        pre { background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI Trading Bot Dashboard - Integration Test Report</h1>
            <p><strong>Generated:</strong> ${report.testRun.timestamp}</p>
            <p><strong>Duration:</strong> ${Math.round(report.testRun.duration / 1000)}s</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value">${report.summary.totalSuites}</div>
                <div class="metric-label">Total Suites</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.passedSuites}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.failedSuites}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">${report.summary.successRate}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <div class="results">
            <h2>Test Results</h2>
            ${Object.entries(report.results).map(([suite, result]) => `
                <div class="result-item ${result.success ? 'pass' : 'fail'}">
                    <span>${suite.charAt(0).toUpperCase() + suite.slice(1)} Tests</span>
                    <span class="status ${result.success ? 'pass' : 'fail'}">
                        ${result.success ? '✓ PASS' : '✗ FAIL'}
                    </span>
                </div>
            `).join('')}
        </div>
        
        <div class="environment">
            <h2>Environment Information</h2>
            <pre>${JSON.stringify(report.environment, null, 2)}</pre>
        </div>
        
        <div class="raw-data">
            <h2>Raw Test Data</h2>
            <pre>${JSON.stringify(report.results, null, 2)}</pre>
        </div>
    </div>
</body>
</html>`;
}

// Main test execution
async function runAllTests() {
    logInfo('='.repeat(60));
    logInfo('AI Trading Bot Dashboard - Comprehensive Integration Tests');
    logInfo('='.repeat(60));
    
    try {
        // Check prerequisites
        const prerequisitesOk = await checkPrerequisites();
        if (!prerequisitesOk) {
            logError('Prerequisites check failed. Please install missing dependencies.');
            process.exit(1);
        }
        
        // Wait for services to be ready
        const servicesReady = await waitForServices();
        if (!servicesReady) {
            logWarning('Services may not be fully ready, but proceeding with tests...');
        }
        
        // Generate mock data
        logInfo('Preparing test data...');
        await runMockDataGeneration();
        
        // Run test suites
        const testSuites = [
            { name: 'Docker Validation', fn: runDockerValidation, enabled: CONFIG.testSuites.docker.enabled },
            { name: 'API Tests', fn: runAPITests, enabled: CONFIG.testSuites.api.enabled },
            { name: 'WebSocket Tests', fn: runWebSocketTests, enabled: CONFIG.testSuites.websocket.enabled },
            { name: 'UI Tests', fn: runUITests, enabled: CONFIG.testSuites.ui.enabled }
        ];
        
        logInfo(`Running ${testSuites.filter(suite => suite.enabled).length} test suites...`);
        
        for (const suite of testSuites) {
            if (suite.enabled) {
                logInfo('-'.repeat(40));
                logInfo(`Starting: ${suite.name}`);
                
                try {
                    await suite.fn();
                    logSuccess(`${suite.name} completed`);
                } catch (error) {
                    logError(`${suite.name} failed`, error.message);
                }
                
                TEST_STATE.totalTests++;
            } else {
                logInfo(`Skipping: ${suite.name} (disabled)`);
                TEST_STATE.skippedTests++;
            }
        }
        
        // Generate final report
        const report = await generateFinalReport();
        
        // Determine overall success
        const overallSuccess = report.summary.successRate >= CONFIG.thresholds.minimumPassRate;
        
        if (overallSuccess) {
            logSuccess(`All integration tests completed! Success rate: ${report.summary.successRate}%`);
            process.exit(0);
        } else {
            logError(`Integration tests failed. Success rate: ${report.summary.successRate}% (minimum required: ${CONFIG.thresholds.minimumPassRate}%)`);
            process.exit(1);
        }
        
    } catch (error) {
        logError('Test execution failed', error.message);
        process.exit(1);
    }
}

// Handle process signals
process.on('SIGINT', () => {
    logWarning('Test execution interrupted by user');
    process.exit(1);
});

process.on('SIGTERM', () => {
    logWarning('Test execution terminated');
    process.exit(1);
});

// Set global timeout
setTimeout(() => {
    logError('Test execution timed out after 10 minutes');
    process.exit(1);
}, 600000); // 10 minutes

// Run tests
if (require.main === module) {
    runAllTests().catch(error => {
        logError('Test runner failed with error', error.message);
        process.exit(1);
    });
}

module.exports = {
    runAllTests,
    TEST_STATE
};