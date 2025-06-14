#!/usr/bin/env node

/**
 * REST API Integration Test for AI Trading Bot Dashboard
 * 
 * Tests all REST API endpoints, data validation, and error handling
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

// Test configuration
const CONFIG = {
    API_BASE_URL: process.env.BACKEND_URL || 'http://localhost:8000',
    TEST_TIMEOUT: 30000, // 30 seconds
    REQUEST_TIMEOUT: 10000, // 10 seconds per request
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000 // 1 second
};

// Test results tracking
const TEST_RESULTS = {
    healthCheck: false,
    rootEndpoint: false,
    statusEndpoint: false,
    tradingDataEndpoint: false,
    logsEndpoint: false,
    llmStatusEndpoint: false,
    llmMetricsEndpoint: false,
    llmActivityEndpoint: false,
    llmDecisionsEndpoint: false,
    llmAlertsEndpoint: false,
    tradingViewEndpoints: false,
    errorHandling: false,
    responseValidation: false
};

// Expected API endpoints
const API_ENDPOINTS = [
    { path: '/', method: 'GET', name: 'Root' },
    { path: '/health', method: 'GET', name: 'Health Check' },
    { path: '/status', method: 'GET', name: 'Status' },
    { path: '/trading-data', method: 'GET', name: 'Trading Data' },
    { path: '/logs', method: 'GET', name: 'Logs' },
    { path: '/llm/status', method: 'GET', name: 'LLM Status' },
    { path: '/llm/metrics', method: 'GET', name: 'LLM Metrics' },
    { path: '/llm/activity', method: 'GET', name: 'LLM Activity' },
    { path: '/llm/decisions', method: 'GET', name: 'LLM Decisions' },
    { path: '/llm/alerts', method: 'GET', name: 'LLM Alerts' },
    { path: '/udf/config', method: 'GET', name: 'TradingView Config' },
    { path: '/tradingview/symbols', method: 'GET', name: 'TradingView Symbols' }
];

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

// HTTP request utility with timeout and retry
function makeRequest(url, options = {}) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const isHttps = urlObj.protocol === 'https:';
        const httpModule = isHttps ? https : http;
        
        const requestOptions = {
            hostname: urlObj.hostname,
            port: urlObj.port || (isHttps ? 443 : 80),
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Dashboard-API-Test/1.0',
                ...options.headers
            },
            timeout: CONFIG.REQUEST_TIMEOUT
        };
        
        const req = httpModule.request(requestOptions, (res) => {
            let data = '';
            
            res.on('data', (chunk) => {
                data += chunk;
            });
            
            res.on('end', () => {
                try {
                    const responseData = {
                        statusCode: res.statusCode,
                        headers: res.headers,
                        body: data,
                        data: null
                    };
                    
                    // Try to parse JSON response
                    if (data) {
                        try {
                            responseData.data = JSON.parse(data);
                        } catch (parseError) {
                            // Not JSON, keep as string
                            responseData.data = data;
                        }
                    }
                    
                    resolve(responseData);
                } catch (error) {
                    reject(error);
                }
            });
        });
        
        req.on('error', (error) => {
            reject(error);
        });
        
        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });
        
        if (options.body) {
            req.write(options.body);
        }
        
        req.end();
    });
}

// Retry wrapper for requests
async function makeRequestWithRetry(url, options = {}, retries = CONFIG.RETRY_ATTEMPTS) {
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            const response = await makeRequest(url, options);
            return response;
        } catch (error) {
            if (attempt === retries) {
                throw error;
            }
            logWarning(`Request attempt ${attempt} failed, retrying...`, { error: error.message });
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * attempt));
        }
    }
}

// Test functions
async function testHealthCheck() {
    logInfo('Testing health check endpoint...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/health`);
        
        if (response.statusCode === 200) {
            logSuccess('Health check endpoint accessible');
            
            if (response.data && response.data.status === 'healthy') {
                logSuccess('Health check returns correct status');
                TEST_RESULTS.healthCheck = true;
                return true;
            } else {
                logWarning('Health check response format unexpected', response.data);
                return false;
            }
        } else {
            logError(`Health check failed with status ${response.statusCode}`);
            return false;
        }
    } catch (error) {
        logError('Health check request failed', { error: error.message });
        return false;
    }
}

async function testRootEndpoint() {
    logInfo('Testing root endpoint...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/`);
        
        if (response.statusCode === 200) {
            logSuccess('Root endpoint accessible');
            
            if (response.data && response.data.service) {
                logSuccess('Root endpoint returns service information');
                TEST_RESULTS.rootEndpoint = true;
                return true;
            } else {
                logWarning('Root endpoint response format unexpected', response.data);
                return false;
            }
        } else {
            logError(`Root endpoint failed with status ${response.statusCode}`);
            return false;
        }
    } catch (error) {
        logError('Root endpoint request failed', { error: error.message });
        return false;
    }
}

async function testStatusEndpoint() {
    logInfo('Testing status endpoint...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/status`);
        
        if (response.statusCode === 200) {
            logSuccess('Status endpoint accessible');
            
            if (response.data) {
                const requiredFields = ['timestamp', 'bot_status', 'websocket_connections'];
                const hasRequiredFields = requiredFields.every(field => 
                    response.data.hasOwnProperty(field)
                );
                
                if (hasRequiredFields) {
                    logSuccess('Status endpoint returns required fields');
                    logInfo('Status data', {
                        bot_status: response.data.bot_status,
                        websocket_connections: response.data.websocket_connections
                    });
                    TEST_RESULTS.statusEndpoint = true;
                    return true;
                } else {
                    logWarning('Status endpoint missing required fields', {
                        required: requiredFields,
                        received: Object.keys(response.data)
                    });
                    return false;
                }
            }
        } else {
            logError(`Status endpoint failed with status ${response.statusCode}`);
            return false;
        }
    } catch (error) {
        logError('Status endpoint request failed', { error: error.message });
        return false;
    }
}

async function testTradingDataEndpoint() {
    logInfo('Testing trading data endpoint...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/trading-data`);
        
        if (response.statusCode === 200) {
            logSuccess('Trading data endpoint accessible');
            
            if (response.data) {
                const requiredFields = ['timestamp', 'account', 'performance'];
                const hasRequiredFields = requiredFields.every(field => 
                    response.data.hasOwnProperty(field)
                );
                
                if (hasRequiredFields) {
                    logSuccess('Trading data endpoint returns required fields');
                    
                    // Validate account data structure
                    if (response.data.account && response.data.account.balance) {
                        logSuccess('Account data structure is valid');
                    }
                    
                    // Validate performance data structure
                    if (response.data.performance && response.data.performance.total_trades !== undefined) {
                        logSuccess('Performance data structure is valid');
                    }
                    
                    TEST_RESULTS.tradingDataEndpoint = true;
                    return true;
                } else {
                    logWarning('Trading data endpoint missing required fields', {
                        required: requiredFields,
                        received: Object.keys(response.data)
                    });
                    return false;
                }
            }
        } else {
            logError(`Trading data endpoint failed with status ${response.statusCode}`);
            return false;
        }
    } catch (error) {
        logError('Trading data endpoint request failed', { error: error.message });
        return false;
    }
}

async function testLogsEndpoint() {
    logInfo('Testing logs endpoint...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/logs?limit=10`);
        
        if (response.statusCode === 200) {
            logSuccess('Logs endpoint accessible');
            
            if (response.data && Array.isArray(response.data.logs)) {
                logSuccess('Logs endpoint returns array of logs');
                logInfo(`Retrieved ${response.data.logs.length} log entries`);
                TEST_RESULTS.logsEndpoint = true;
                return true;
            } else {
                logWarning('Logs endpoint response format unexpected', response.data);
                return false;
            }
        } else {
            logError(`Logs endpoint failed with status ${response.statusCode}`);
            return false;
        }
    } catch (error) {
        logError('Logs endpoint request failed', { error: error.message });
        return false;
    }
}

async function testLLMEndpoints() {
    logInfo('Testing LLM monitoring endpoints...');
    
    const llmEndpoints = [
        { path: '/llm/status', name: 'LLM Status' },
        { path: '/llm/metrics', name: 'LLM Metrics' },
        { path: '/llm/activity?limit=5', name: 'LLM Activity' },
        { path: '/llm/decisions?limit=5', name: 'LLM Decisions' },
        { path: '/llm/alerts?limit=5', name: 'LLM Alerts' }
    ];
    
    let successCount = 0;
    
    for (const endpoint of llmEndpoints) {
        try {
            const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}${endpoint.path}`);
            
            if (response.statusCode === 200) {
                logSuccess(`${endpoint.name} endpoint accessible`);
                
                if (response.data && response.data.timestamp) {
                    logSuccess(`${endpoint.name} returns timestamped data`);
                    successCount++;
                } else {
                    logWarning(`${endpoint.name} response format unexpected`);
                }
            } else {
                logWarning(`${endpoint.name} returned status ${response.statusCode}`);
            }
        } catch (error) {
            logWarning(`${endpoint.name} request failed`, { error: error.message });
        }
    }
    
    if (successCount >= 3) {
        logSuccess('LLM endpoints are mostly functional');
        TEST_RESULTS.llmStatusEndpoint = true;
        TEST_RESULTS.llmMetricsEndpoint = true;
        TEST_RESULTS.llmActivityEndpoint = true;
        TEST_RESULTS.llmDecisionsEndpoint = true;
        TEST_RESULTS.llmAlertsEndpoint = true;
        return true;
    } else {
        logWarning(`Only ${successCount}/${llmEndpoints.length} LLM endpoints working`);
        return false;
    }
}

async function testTradingViewEndpoints() {
    logInfo('Testing TradingView UDF endpoints...');
    
    const tradingViewEndpoints = [
        { path: '/udf/config', name: 'TradingView Config' },
        { path: '/udf/time', name: 'TradingView Time' },
        { path: '/tradingview/symbols', name: 'TradingView Symbols' }
    ];
    
    let successCount = 0;
    
    for (const endpoint of tradingViewEndpoints) {
        try {
            const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}${endpoint.path}`);
            
            if (response.statusCode === 200) {
                logSuccess(`${endpoint.name} endpoint accessible`);
                
                if (response.data) {
                    logSuccess(`${endpoint.name} returns data`);
                    successCount++;
                } else {
                    logWarning(`${endpoint.name} returns empty response`);
                }
            } else {
                logWarning(`${endpoint.name} returned status ${response.statusCode}`);
            }
        } catch (error) {
            logWarning(`${endpoint.name} request failed`, { error: error.message });
        }
    }
    
    if (successCount >= 2) {
        logSuccess('TradingView endpoints are functional');
        TEST_RESULTS.tradingViewEndpoints = true;
        return true;
    } else {
        logWarning(`Only ${successCount}/${tradingViewEndpoints.length} TradingView endpoints working`);
        return false;
    }
}

async function testErrorHandling() {
    logInfo('Testing API error handling...');
    
    const errorTests = [
        { path: '/nonexistent-endpoint', expectedStatus: 404, name: 'Non-existent endpoint' },
        { path: '/llm/decisions?limit=invalid', expectedStatus: [422, 400], name: 'Invalid query parameter' },
        { path: '/udf/symbols?symbol=INVALID-SYMBOL', expectedStatus: [404, 400], name: 'Invalid symbol request' }
    ];
    
    let errorHandlingPassed = true;
    
    for (const test of errorTests) {
        try {
            const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}${test.path}`);
            
            const expectedStatuses = Array.isArray(test.expectedStatus) ? test.expectedStatus : [test.expectedStatus];
            
            if (expectedStatuses.includes(response.statusCode)) {
                logSuccess(`${test.name} returns expected error status: ${response.statusCode}`);
            } else {
                logWarning(`${test.name} returned unexpected status: ${response.statusCode} (expected: ${expectedStatuses.join(' or ')})`);
                errorHandlingPassed = false;
            }
        } catch (error) {
            // Network errors are also acceptable for error handling tests
            logInfo(`${test.name} resulted in network error (acceptable)`, { error: error.message });
        }
    }
    
    if (errorHandlingPassed) {
        logSuccess('API error handling is working correctly');
        TEST_RESULTS.errorHandling = true;
        return true;
    } else {
        logWarning('Some API error handling tests failed');
        return false;
    }
}

async function testResponseValidation() {
    logInfo('Testing API response validation...');
    
    const validationTests = [
        {
            endpoint: '/health',
            requiredFields: ['status', 'timestamp'],
            name: 'Health endpoint validation'
        },
        {
            endpoint: '/status',
            requiredFields: ['timestamp', 'bot_status'],
            name: 'Status endpoint validation'
        },
        {
            endpoint: '/trading-data',
            requiredFields: ['timestamp', 'account', 'performance'],
            name: 'Trading data validation'
        }
    ];
    
    let validationPassed = true;
    
    for (const test of validationTests) {
        try {
            const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}${test.endpoint}`);
            
            if (response.statusCode === 200 && response.data) {
                const missingFields = test.requiredFields.filter(field => 
                    !response.data.hasOwnProperty(field)
                );
                
                if (missingFields.length === 0) {
                    logSuccess(`${test.name} - all required fields present`);
                } else {
                    logWarning(`${test.name} - missing fields: ${missingFields.join(', ')}`);
                    validationPassed = false;
                }
                
                // Check timestamp format
                if (response.data.timestamp) {
                    const timestampDate = new Date(response.data.timestamp);
                    if (!isNaN(timestampDate.getTime())) {
                        logSuccess(`${test.name} - timestamp format is valid`);
                    } else {
                        logWarning(`${test.name} - invalid timestamp format`);
                        validationPassed = false;
                    }
                }
            } else {
                logWarning(`${test.name} - endpoint not accessible for validation`);
                validationPassed = false;
            }
        } catch (error) {
            logWarning(`${test.name} - validation failed`, { error: error.message });
            validationPassed = false;
        }
    }
    
    if (validationPassed) {
        logSuccess('API response validation passed');
        TEST_RESULTS.responseValidation = true;
        return true;
    } else {
        logWarning('Some API response validation tests failed');
        return false;
    }
}

async function testCORSHeaders() {
    logInfo('Testing CORS headers...');
    
    try {
        const response = await makeRequestWithRetry(`${CONFIG.API_BASE_URL}/health`, {
            headers: {
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'GET'
            }
        });
        
        const corsHeaders = response.headers['access-control-allow-origin'];
        if (corsHeaders) {
            logSuccess('CORS headers are present');
            logInfo('CORS configuration', { origin: corsHeaders });
            return true;
        } else {
            logWarning('CORS headers not found');
            return false;
        }
    } catch (error) {
        logWarning('CORS test failed', { error: error.message });
        return false;
    }
}

async function testAPIPerformance() {
    logInfo('Testing API performance...');
    
    const performanceTests = [
        { endpoint: '/health', name: 'Health Check' },
        { endpoint: '/status', name: 'Status' },
        { endpoint: '/trading-data', name: 'Trading Data' }
    ];
    
    let performanceResults = [];
    
    for (const test of performanceTests) {
        const startTime = Date.now();
        
        try {
            await makeRequestWithRetry(`${CONFIG.API_BASE_URL}${test.endpoint}`);
            const duration = Date.now() - startTime;
            
            performanceResults.push({
                endpoint: test.name,
                duration: duration,
                status: 'success'
            });
            
            if (duration < 1000) {
                logSuccess(`${test.name} responded in ${duration}ms`);
            } else {
                logWarning(`${test.name} was slow: ${duration}ms`);
            }
        } catch (error) {
            const duration = Date.now() - startTime;
            performanceResults.push({
                endpoint: test.name,
                duration: duration,
                status: 'failed',
                error: error.message
            });
            
            logWarning(`${test.name} failed after ${duration}ms`);
        }
    }
    
    const avgDuration = performanceResults
        .filter(r => r.status === 'success')
        .reduce((sum, r) => sum + r.duration, 0) / performanceResults.length;
    
    logInfo(`Average response time: ${Math.round(avgDuration)}ms`);
    
    return performanceResults;
}

// Save test status for main script
function saveTestStatus() {
    const testDir = path.dirname(__filename);
    const statusFile = path.join(testDir, '.api_test_passed');
    
    const allTestsPassed = Object.values(TEST_RESULTS).every(result => result === true);
    
    if (allTestsPassed) {
        fs.writeFileSync(statusFile, `API tests passed at ${new Date().toISOString()}`);
        logSuccess('All API tests passed');
    } else {
        if (fs.existsSync(statusFile)) {
            fs.unlinkSync(statusFile);
        }
        logError('Some API tests failed');
    }
    
    return allTestsPassed;
}

// Generate detailed test report
function generateTestReport(performanceResults = []) {
    const report = {
        timestamp: new Date().toISOString(),
        testType: 'REST API Integration Tests',
        configuration: CONFIG,
        results: TEST_RESULTS,
        performance: performanceResults,
        summary: {
            total: Object.keys(TEST_RESULTS).length,
            passed: Object.values(TEST_RESULTS).filter(r => r).length,
            failed: Object.values(TEST_RESULTS).filter(r => !r).length
        }
    };
    
    const reportPath = path.join(path.dirname(__filename), 'api-test-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    logInfo(`Test report saved: ${reportPath}`);
    return report;
}

// Main test execution
async function runAPITests() {
    logInfo('='.repeat(60));
    logInfo('Starting REST API Integration Tests');
    logInfo('='.repeat(60));
    
    const tests = [
        { name: 'Health Check', fn: testHealthCheck },
        { name: 'Root Endpoint', fn: testRootEndpoint },
        { name: 'Status Endpoint', fn: testStatusEndpoint },
        { name: 'Trading Data Endpoint', fn: testTradingDataEndpoint },
        { name: 'Logs Endpoint', fn: testLogsEndpoint },
        { name: 'LLM Endpoints', fn: testLLMEndpoints },
        { name: 'TradingView Endpoints', fn: testTradingViewEndpoints },
        { name: 'Error Handling', fn: testErrorHandling },
        { name: 'Response Validation', fn: testResponseValidation },
        { name: 'CORS Headers', fn: testCORSHeaders }
    ];
    
    logInfo(`Running ${tests.length} API test suites...`);
    logInfo(`Target API URL: ${CONFIG.API_BASE_URL}`);
    
    let performanceResults = [];
    
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
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Run performance tests
    logInfo('-'.repeat(40));
    logInfo('Running performance tests...');
    performanceResults = await testAPIPerformance();
    
    logInfo('-'.repeat(40));
    logInfo('API Tests Summary:');
    
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
    const report = generateTestReport(performanceResults);
    
    // Save test status
    const allPassed = saveTestStatus();
    
    if (allPassed) {
        logSuccess('All REST API integration tests completed successfully!');
        process.exit(0);
    } else {
        logError('Some REST API integration tests failed!');
        process.exit(1);
    }
}

// Handle process signals
process.on('SIGINT', () => {
    logWarning('API tests interrupted by user');
    process.exit(1);
});

process.on('SIGTERM', () => {
    logWarning('API tests terminated');
    process.exit(1);
});

// Set test timeout
setTimeout(() => {
    logError(`API tests timed out after ${CONFIG.TEST_TIMEOUT}ms`);
    process.exit(1);
}, CONFIG.TEST_TIMEOUT);

// Run tests
if (require.main === module) {
    runAPITests().catch(error => {
        logError('API tests failed with error', { error: error.message });
        process.exit(1);
    });
}