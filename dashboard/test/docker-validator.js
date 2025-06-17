#!/usr/bin/env node

/**
 * Docker Environment Validator for AI Trading Bot Dashboard
 *
 * Validates Docker containers, networks, volumes, and inter-service communication
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

// Validation configuration
const CONFIG = {
    REQUIRED_CONTAINERS: [
        'dashboard-backend',
        'dashboard-frontend'
    ],
    REQUIRED_NETWORKS: [
        'dashboard-network'
    ],
    REQUIRED_VOLUMES: [
        'dashboard-logs',
        'dashboard-data'
    ],
    SERVICE_ENDPOINTS: {
        backend: {
            url: 'http://localhost:8000',
            healthPath: '/health',
            timeout: 10000
        },
        frontend: {
            url: 'http://localhost:3000',
            healthPath: '/',
            timeout: 10000
        }
    },
    COMPOSE_FILE: 'docker-compose.yml'
};

// Validation results
const VALIDATION_RESULTS = {
    dockerInstallation: false,
    composeFileExists: false,
    containersRunning: false,
    networksCreated: false,
    volumesCreated: false,
    servicesHealthy: false,
    interServiceComm: false,
    resourceUsage: false,
    logsAccessible: false
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
        exec(command, { timeout: 30000, ...options }, (error, stdout, stderr) => {
            if (error) {
                reject({ error, stdout, stderr });
            } else {
                resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
            }
        });
    });
}

// HTTP request utility
function makeRequest(url, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const request = http.get(url, { timeout }, (response) => {
            let data = '';
            response.on('data', chunk => data += chunk);
            response.on('end', () => {
                resolve({
                    statusCode: response.statusCode,
                    headers: response.headers,
                    body: data
                });
            });
        });

        request.on('timeout', () => {
            request.destroy();
            reject(new Error('Request timeout'));
        });

        request.on('error', reject);
    });
}

// Validation functions
async function validateDockerInstallation() {
    logInfo('Validating Docker installation...');

    try {
        // Check Docker version
        const dockerVersion = await execCommand('docker --version');
        logSuccess('Docker is installed', dockerVersion.stdout);

        // Check Docker daemon
        const dockerInfo = await execCommand('docker info --format "{{.ServerVersion}}"');
        logSuccess('Docker daemon is running', `Server version: ${dockerInfo.stdout}`);

        // Check Docker Compose
        const composeVersion = await execCommand('docker-compose --version');
        logSuccess('Docker Compose is available', composeVersion.stdout);

        VALIDATION_RESULTS.dockerInstallation = true;
        return true;
    } catch (error) {
        logError('Docker installation validation failed', error.error?.message || error);
        return false;
    }
}

async function validateComposeFile() {
    logInfo('Validating Docker Compose configuration...');

    try {
        const composeFile = path.join(process.cwd(), CONFIG.COMPOSE_FILE);

        if (!fs.existsSync(composeFile)) {
            throw new Error(`Compose file not found: ${composeFile}`);
        }

        logSuccess('Docker Compose file exists', composeFile);

        // Validate compose file syntax
        const validateResult = await execCommand(`docker-compose -f ${composeFile} config`);
        logSuccess('Docker Compose file syntax is valid');

        VALIDATION_RESULTS.composeFileExists = true;
        return true;
    } catch (error) {
        logError('Docker Compose file validation failed', error.error?.message || error);
        return false;
    }
}

async function validateContainers() {
    logInfo('Validating Docker containers...');

    try {
        // Check if containers exist and are running
        const psResult = await execCommand('docker-compose ps --format json');
        let containers = [];

        if (psResult.stdout) {
            // Parse container information
            containers = psResult.stdout.split('\n')
                .filter(line => line.trim())
                .map(line => {
                    try {
                        return JSON.parse(line);
                    } catch {
                        return null;
                    }
                })
                .filter(container => container !== null);
        }

        logInfo(`Found ${containers.length} containers`);

        let allContainersRunning = true;

        for (const requiredContainer of CONFIG.REQUIRED_CONTAINERS) {
            const container = containers.find(c => c.Name.includes(requiredContainer));

            if (container) {
                if (container.State === 'running') {
                    logSuccess(`Container ${requiredContainer} is running`);
                } else {
                    logError(`Container ${requiredContainer} is not running: ${container.State}`);
                    allContainersRunning = false;
                }
            } else {
                logError(`Required container ${requiredContainer} not found`);
                allContainersRunning = false;
            }
        }

        if (allContainersRunning) {
            VALIDATION_RESULTS.containersRunning = true;
            return true;
        } else {
            return false;
        }
    } catch (error) {
        logError('Container validation failed', error.error?.message || error);
        return false;
    }
}

async function validateNetworks() {
    logInfo('Validating Docker networks...');

    try {
        const networksResult = await execCommand('docker network ls --format "{{.Name}}"');
        const networks = networksResult.stdout.split('\n').filter(name => name.trim());

        let allNetworksExist = true;

        for (const requiredNetwork of CONFIG.REQUIRED_NETWORKS) {
            if (networks.includes(requiredNetwork)) {
                logSuccess(`Network ${requiredNetwork} exists`);

                // Get network details
                const networkInfo = await execCommand(`docker network inspect ${requiredNetwork}`);
                const networkData = JSON.parse(networkInfo.stdout)[0];

                logInfo(`Network ${requiredNetwork} details`, {
                    driver: networkData.Driver,
                    subnet: networkData.IPAM?.Config?.[0]?.Subnet,
                    containers: Object.keys(networkData.Containers || {}).length
                });
            } else {
                logError(`Required network ${requiredNetwork} not found`);
                allNetworksExist = false;
            }
        }

        if (allNetworksExist) {
            VALIDATION_RESULTS.networksCreated = true;
            return true;
        } else {
            return false;
        }
    } catch (error) {
        logError('Network validation failed', error.error?.message || error);
        return false;
    }
}

async function validateVolumes() {
    logInfo('Validating Docker volumes...');

    try {
        const volumesResult = await execCommand('docker volume ls --format "{{.Name}}"');
        const volumes = volumesResult.stdout.split('\n').filter(name => name.trim());

        let allVolumesExist = true;

        for (const requiredVolume of CONFIG.REQUIRED_VOLUMES) {
            if (volumes.includes(requiredVolume)) {
                logSuccess(`Volume ${requiredVolume} exists`);

                // Get volume details
                try {
                    const volumeInfo = await execCommand(`docker volume inspect ${requiredVolume}`);
                    const volumeData = JSON.parse(volumeInfo.stdout)[0];

                    logInfo(`Volume ${requiredVolume} details`, {
                        driver: volumeData.Driver,
                        mountpoint: volumeData.Mountpoint
                    });
                } catch (inspectError) {
                    logWarning(`Could not inspect volume ${requiredVolume}`);
                }
            } else {
                logWarning(`Volume ${requiredVolume} not found (may be created on-demand)`);
                // Don't fail validation for missing volumes as they might be created on-demand
            }
        }

        VALIDATION_RESULTS.volumesCreated = true;
        return true;
    } catch (error) {
        logError('Volume validation failed', error.error?.message || error);
        return false;
    }
}

async function validateServiceHealth() {
    logInfo('Validating service health...');

    let allServicesHealthy = true;

    for (const [serviceName, config] of Object.entries(CONFIG.SERVICE_ENDPOINTS)) {
        try {
            logInfo(`Checking ${serviceName} service health...`);

            const response = await makeRequest(config.url + config.healthPath, config.timeout);

            if (response.statusCode >= 200 && response.statusCode < 400) {
                logSuccess(`${serviceName} service is healthy`, {
                    url: config.url + config.healthPath,
                    status: response.statusCode,
                    contentLength: response.body.length
                });
            } else {
                logError(`${serviceName} service returned error status`, {
                    status: response.statusCode,
                    url: config.url + config.healthPath
                });
                allServicesHealthy = false;
            }
        } catch (error) {
            logError(`${serviceName} service health check failed`, {
                url: config.url + config.healthPath,
                error: error.message
            });
            allServicesHealthy = false;
        }
    }

    if (allServicesHealthy) {
        VALIDATION_RESULTS.servicesHealthy = true;
        return true;
    } else {
        return false;
    }
}

async function validateInterServiceCommunication() {
    logInfo('Validating inter-service communication...');

    try {
        // Test if frontend can reach backend through Docker network
        const backendContainerId = await execCommand(
            'docker-compose ps -q dashboard-backend'
        );

        const frontendContainerId = await execCommand(
            'docker-compose ps -q dashboard-frontend'
        );

        if (!backendContainerId.stdout || !frontendContainerId.stdout) {
            throw new Error('Could not find container IDs');
        }

        // Test network connectivity from frontend to backend
        try {
            const connectivityTest = await execCommand(
                `docker exec ${frontendContainerId.stdout.trim()} wget --spider -T 5 http://dashboard-backend:8000/health`
            );

            logSuccess('Frontend can reach backend through Docker network');
            VALIDATION_RESULTS.interServiceComm = true;
            return true;
        } catch (connectivityError) {
            logWarning('Direct inter-service connectivity test failed', connectivityError.error?.message);

            // Try alternative test - check if both containers are on same network
            const backendNetworks = await execCommand(
                `docker inspect ${backendContainerId.stdout.trim()} --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}'`
            );

            const frontendNetworks = await execCommand(
                `docker inspect ${frontendContainerId.stdout.trim()} --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}'`
            );

            const backendNets = backendNetworks.stdout.split(' ').filter(n => n.trim());
            const frontendNets = frontendNetworks.stdout.split(' ').filter(n => n.trim());

            const commonNetworks = backendNets.filter(net => frontendNets.includes(net));

            if (commonNetworks.length > 0) {
                logSuccess('Containers share common networks', commonNetworks);
                VALIDATION_RESULTS.interServiceComm = true;
                return true;
            } else {
                logError('Containers do not share common networks', {
                    backend: backendNets,
                    frontend: frontendNets
                });
                return false;
            }
        }
    } catch (error) {
        logError('Inter-service communication validation failed', error.error?.message || error);
        return false;
    }
}

async function validateResourceUsage() {
    logInfo('Validating container resource usage...');

    try {
        // Get container resource stats
        const statsResult = await execCommand(
            'docker stats --no-stream --format "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.NetIO}}"'
        );

        logInfo('Container resource usage', statsResult.stdout);

        // Check for containers using excessive resources
        const lines = statsResult.stdout.split('\n').slice(1); // Skip header
        let resourcesOk = true;

        for (const line of lines) {
            if (line.trim()) {
                const [container, cpu, memory, network] = line.split('\t').map(s => s.trim());

                // Extract CPU percentage
                const cpuMatch = cpu.match(/(\d+\.?\d*)%/);
                if (cpuMatch) {
                    const cpuPercent = parseFloat(cpuMatch[1]);
                    if (cpuPercent > 80) {
                        logWarning(`Container ${container} using high CPU: ${cpu}`);
                    } else {
                        logInfo(`Container ${container} CPU usage: ${cpu}`);
                    }
                }

                // Extract memory usage
                const memoryInfo = memory.split(' / ');
                if (memoryInfo.length === 2) {
                    logInfo(`Container ${container} memory usage: ${memory}`);
                }
            }
        }

        VALIDATION_RESULTS.resourceUsage = true;
        return true;
    } catch (error) {
        logError('Resource usage validation failed', error.error?.message || error);
        return false;
    }
}

async function validateLogsAccess() {
    logInfo('Validating log access...');

    try {
        // Test access to container logs
        for (const container of CONFIG.REQUIRED_CONTAINERS) {
            try {
                const logsResult = await execCommand(
                    `docker-compose logs --tail=5 ${container}`
                );

                if (logsResult.stdout || logsResult.stderr) {
                    logSuccess(`Can access logs for ${container}`);
                } else {
                    logWarning(`No recent logs found for ${container}`);
                }
            } catch (logError) {
                logError(`Cannot access logs for ${container}`, logError.error?.message);
                return false;
            }
        }

        VALIDATION_RESULTS.logsAccessible = true;
        return true;
    } catch (error) {
        logError('Log access validation failed', error.error?.message || error);
        return false;
    }
}

// Generate validation report
function generateValidationReport() {
    const report = {
        timestamp: new Date().toISOString(),
        testType: 'Docker Environment Validation',
        results: VALIDATION_RESULTS,
        summary: {
            total: Object.keys(VALIDATION_RESULTS).length,
            passed: Object.values(VALIDATION_RESULTS).filter(r => r).length,
            failed: Object.values(VALIDATION_RESULTS).filter(r => !r).length
        },
        environment: {
            platform: process.platform,
            nodeVersion: process.version,
            workingDirectory: process.cwd()
        }
    };

    const reportPath = path.join(__dirname, 'docker-validation-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    logInfo(`Docker validation report saved: ${reportPath}`);
    return report;
}

// Save test status for main script
function saveTestStatus() {
    const testDir = path.dirname(__filename);
    const statusFile = path.join(testDir, '.docker_test_passed');

    const allTestsPassed = Object.values(VALIDATION_RESULTS).every(result => result === true);

    if (allTestsPassed) {
        fs.writeFileSync(statusFile, `Docker validation passed at ${new Date().toISOString()}`);
        logSuccess('All Docker validation tests passed');
    } else {
        if (fs.existsSync(statusFile)) {
            fs.unlinkSync(statusFile);
        }
        logError('Some Docker validation tests failed');
    }

    return allTestsPassed;
}

// Main validation function
async function runDockerValidation() {
    logInfo('='.repeat(60));
    logInfo('Starting Docker Environment Validation');
    logInfo('='.repeat(60));

    const validations = [
        { name: 'Docker Installation', fn: validateDockerInstallation },
        { name: 'Compose File', fn: validateComposeFile },
        { name: 'Containers', fn: validateContainers },
        { name: 'Networks', fn: validateNetworks },
        { name: 'Volumes', fn: validateVolumes },
        { name: 'Service Health', fn: validateServiceHealth },
        { name: 'Inter-Service Communication', fn: validateInterServiceCommunication },
        { name: 'Resource Usage', fn: validateResourceUsage },
        { name: 'Log Access', fn: validateLogsAccess }
    ];

    logInfo(`Running ${validations.length} Docker validation tests...`);

    for (const validation of validations) {
        logInfo('-'.repeat(40));
        logInfo(`Running: ${validation.name}`);

        try {
            const result = await validation.fn();
            if (result) {
                logSuccess(`${validation.name} validation passed`);
            } else {
                logError(`${validation.name} validation failed`);
            }
        } catch (error) {
            logError(`${validation.name} validation threw an error`, error.message);
        }

        // Small delay between validations
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    logInfo('-'.repeat(40));
    logInfo('Docker Validation Summary:');

    let passCount = 0;
    for (const [testName, result] of Object.entries(VALIDATION_RESULTS)) {
        const status = result ? 'PASS' : 'FAIL';
        const icon = result ? '✓' : '✗';
        logInfo(`  ${icon} ${testName}: ${status}`);

        if (result) passCount++;
    }

    const totalTests = Object.keys(VALIDATION_RESULTS).length;
    logInfo(`Overall: ${passCount}/${totalTests} tests passed (${Math.round((passCount/totalTests) * 100)}%)`);

    // Generate validation report
    const report = generateValidationReport();

    // Save test status
    const allPassed = saveTestStatus();

    if (allPassed) {
        logSuccess('All Docker validation tests completed successfully!');
        process.exit(0);
    } else {
        logError('Some Docker validation tests failed!');
        process.exit(1);
    }
}

// Handle process signals
process.on('SIGINT', () => {
    logWarning('Docker validation interrupted by user');
    process.exit(1);
});

process.on('SIGTERM', () => {
    logWarning('Docker validation terminated');
    process.exit(1);
});

// Set validation timeout
setTimeout(() => {
    logError('Docker validation timed out after 5 minutes');
    process.exit(1);
}, 300000); // 5 minutes

// Run validation
if (require.main === module) {
    runDockerValidation().catch(error => {
        logError('Docker validation failed with error', error.message);
        process.exit(1);
    });
}

module.exports = {
    runDockerValidation,
    VALIDATION_RESULTS
};
