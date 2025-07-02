/**
 * Falco Security Dashboard Integration
 * Integrates Falco security monitoring with the AI Trading Bot dashboard
 */

// Security Event Types
const SecurityEventTypes = {
    CRITICAL: 'CRITICAL',
    EMERGENCY: 'EMERGENCY', 
    ALERT: 'ALERT',
    WARNING: 'WARNING',
    NOTICE: 'NOTICE',
    INFORMATIONAL: 'INFORMATIONAL'
};

// Security Event Categories
const SecurityCategories = {
    CONTAINER_SECURITY: 'Container Security',
    ACCESS_CONTROL: 'Access Control',
    DATA_PROTECTION: 'Data Protection',
    AUTHENTICATION: 'Authentication Security',
    NETWORK_SECURITY: 'Network Security',
    FILE_INTEGRITY: 'File Integrity',
    TRADING_SECURITY: 'Trading Security',
    FINANCIAL_SECURITY: 'Financial Security'
};

class SecurityMonitor {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: config.apiEndpoint || 'http://localhost:8080',
            refreshInterval: config.refreshInterval || 30000, // 30 seconds
            maxEvents: config.maxEvents || 100,
            alertThreshold: config.alertThreshold || 70,
            ...config
        };
        
        this.events = [];
        this.metrics = {
            totalEvents: 0,
            criticalEvents: 0,
            alertEvents: 0,
            warningEvents: 0,
            securityScore: 100
        };
        
        this.callbacks = {
            onSecurityEvent: [],
            onMetricsUpdate: [],
            onSecurityScoreChange: []
        };
        
        this.isRunning = false;
        this.refreshTimer = null;
        
        this.init();
    }
    
    init() {
        this.createSecurityWidget();
        this.setupEventListeners();
        
        // Start monitoring if auto-start is enabled
        if (this.config.autoStart !== false) {
            this.startMonitoring();
        }
    }
    
    createSecurityWidget() {
        // Create security monitoring widget for dashboard
        const securityWidget = document.createElement('div');
        securityWidget.id = 'security-monitor-widget';
        securityWidget.className = 'security-widget';
        securityWidget.innerHTML = `
            <div class="security-header">
                <h3>🛡️ Security Monitor</h3>
                <div class="security-status">
                    <span class="status-indicator" id="security-status-indicator"></span>
                    <span id="security-status-text">Initializing...</span>
                </div>
            </div>
            
            <div class="security-metrics">
                <div class="metric-card">
                    <span class="metric-value" id="security-score">100</span>
                    <span class="metric-label">Security Score</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value" id="total-events">0</span>
                    <span class="metric-label">Total Events</span>
                </div>
                <div class="metric-card critical">
                    <span class="metric-value" id="critical-events">0</span>
                    <span class="metric-label">Critical</span>
                </div>
                <div class="metric-card alert">
                    <span class="metric-value" id="alert-events">0</span>
                    <span class="metric-label">Alerts</span>
                </div>
            </div>
            
            <div class="security-events">
                <div class="events-header">
                    <h4>Recent Security Events</h4>
                    <button id="clear-events-btn" class="btn-small">Clear</button>
                </div>
                <div class="events-list" id="security-events-list">
                    <div class="no-events">No security events</div>
                </div>
            </div>
            
            <div class="security-actions">
                <button id="emergency-stop-btn" class="btn-emergency">Emergency Stop</button>
                <button id="isolate-containers-btn" class="btn-warning">Isolate Containers</button>
                <button id="view-details-btn" class="btn-primary">View Details</button>
            </div>
        `;
        
        // Add to dashboard
        const dashboardContainer = document.querySelector('.dashboard-widgets') || document.body;
        dashboardContainer.appendChild(securityWidget);
        
        // Add CSS styles
        this.addSecurityStyles();
    }
    
    addSecurityStyles() {
        const styles = `
            .security-widget {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 16px;
                margin: 16px 0;
                color: #ffffff;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            }
            
            .security-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                border-bottom: 1px solid #333;
                padding-bottom: 8px;
            }
            
            .security-header h3 {
                margin: 0;
                color: #00ff88;
            }
            
            .security-status {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #00ff88;
                animation: pulse 2s infinite;
            }
            
            .status-indicator.warning { background: #ffaa00; }
            .status-indicator.critical { background: #ff4444; }
            .status-indicator.offline { background: #666666; }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .security-metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                margin-bottom: 16px;
            }
            
            .metric-card {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 12px;
                text-align: center;
            }
            
            .metric-card.critical { border-color: #ff4444; }
            .metric-card.alert { border-color: #ffaa00; }
            
            .metric-value {
                display: block;
                font-size: 24px;
                font-weight: bold;
                color: #00ff88;
                margin-bottom: 4px;
            }
            
            .metric-card.critical .metric-value { color: #ff4444; }
            .metric-card.alert .metric-value { color: #ffaa00; }
            
            .metric-label {
                font-size: 12px;
                color: #cccccc;
            }
            
            .security-events {
                margin-bottom: 16px;
            }
            
            .events-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .events-header h4 {
                margin: 0;
                color: #cccccc;
            }
            
            .events-list {
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #333;
                border-radius: 4px;
                background: #222;
            }
            
            .security-event {
                padding: 8px 12px;
                border-bottom: 1px solid #333;
                font-size: 12px;
            }
            
            .security-event:last-child {
                border-bottom: none;
            }
            
            .security-event.critical {
                background: rgba(255, 68, 68, 0.1);
                border-left: 4px solid #ff4444;
            }
            
            .security-event.alert {
                background: rgba(255, 170, 0, 0.1);
                border-left: 4px solid #ffaa00;
            }
            
            .security-event.warning {
                background: rgba(255, 255, 0, 0.1);
                border-left: 4px solid #ffff00;
            }
            
            .event-time {
                color: #888;
                float: right;
            }
            
            .event-rule {
                font-weight: bold;
                color: #00ff88;
                margin-bottom: 4px;
            }
            
            .event-details {
                color: #cccccc;
            }
            
            .no-events {
                padding: 20px;
                text-align: center;
                color: #666;
            }
            
            .security-actions {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }
            
            .btn-emergency {
                background: #ff4444;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                font-weight: bold;
            }
            
            .btn-emergency:hover {
                background: #ff6666;
            }
            
            .btn-warning {
                background: #ffaa00;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                font-weight: bold;
            }
            
            .btn-warning:hover {
                background: #ffcc33;
            }
            
            .btn-primary {
                background: #0066cc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .btn-primary:hover {
                background: #0088ff;
            }
            
            .btn-small {
                background: #444;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
            
            .btn-small:hover {
                background: #666;
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }
    
    setupEventListeners() {
        // Clear events button
        document.getElementById('clear-events-btn')?.addEventListener('click', () => {
            this.clearEvents();
        });
        
        // Emergency stop button
        document.getElementById('emergency-stop-btn')?.addEventListener('click', () => {
            this.handleEmergencyStop();
        });
        
        // Isolate containers button
        document.getElementById('isolate-containers-btn')?.addEventListener('click', () => {
            this.handleIsolateContainers();
        });
        
        // View details button
        document.getElementById('view-details-btn')?.addEventListener('click', () => {
            this.openSecurityDashboard();
        });
    }
    
    startMonitoring() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.updateStatus('Monitoring...', 'normal');
        
        // Initial data fetch
        this.fetchSecurityData();
        
        // Set up periodic refresh
        this.refreshTimer = setInterval(() => {
            this.fetchSecurityData();
        }, this.config.refreshInterval);
        
        console.log('Security monitoring started');
    }
    
    stopMonitoring() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
        
        this.updateStatus('Stopped', 'offline');
        console.log('Security monitoring stopped');
    }
    
    async fetchSecurityData() {
        try {
            // Fetch recent events
            const eventsResponse = await fetch(`${this.config.apiEndpoint}/events?limit=20`);
            const eventsData = await eventsResponse.json();
            
            if (eventsData.events) {
                this.processEvents(eventsData.events);
            }
            
            // Fetch metrics (if available)
            try {
                const metricsResponse = await fetch(`${this.config.apiEndpoint}/metrics`);
                if (metricsResponse.ok) {
                    const metricsText = await metricsResponse.text();
                    this.processMetrics(metricsText);
                }
            } catch (e) {
                // Metrics endpoint might not be available
            }
            
            this.updateStatus('Active', 'normal');
            
        } catch (error) {
            console.error('Failed to fetch security data:', error);
            this.updateStatus('Error', 'critical');
        }
    }
    
    processEvents(events) {
        // Sort events by timestamp (newest first)
        const sortedEvents = events.sort((a, b) => 
            new Date(b.timestamp) - new Date(a.timestamp)
        );
        
        // Update events list
        this.events = sortedEvents.slice(0, this.config.maxEvents);
        
        // Update metrics
        this.updateMetrics();
        
        // Update UI
        this.updateEventsDisplay();
        
        // Check for high-priority events
        this.checkSecurityAlerts();
        
        // Trigger callbacks
        this.callbacks.onSecurityEvent.forEach(callback => {
            callback(this.events);
        });
    }
    
    processMetrics(metricsText) {
        // Parse Prometheus metrics format
        const lines = metricsText.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('#')) continue;
            
            const [metricName, value] = line.split(' ');
            
            if (metricName?.includes('falco_trading_bot_security_score')) {
                const score = parseFloat(value);
                if (!isNaN(score)) {
                    this.metrics.securityScore = score;
                }
            }
        }
        
        this.updateMetricsDisplay();
    }
    
    updateMetrics() {
        const criticalEvents = this.events.filter(e => 
            e.original_event.priority === 'CRITICAL' || 
            e.original_event.priority === 'EMERGENCY'
        ).length;
        
        const alertEvents = this.events.filter(e => 
            e.original_event.priority === 'ALERT'
        ).length;
        
        const warningEvents = this.events.filter(e => 
            e.original_event.priority === 'WARNING'
        ).length;
        
        this.metrics = {
            ...this.metrics,
            totalEvents: this.events.length,
            criticalEvents,
            alertEvents,
            warningEvents
        };
        
        // Calculate security score based on recent events
        if (criticalEvents > 0) {
            this.metrics.securityScore = Math.max(0, 100 - (criticalEvents * 20) - (alertEvents * 10) - (warningEvents * 5));
        } else if (alertEvents > 0) {
            this.metrics.securityScore = Math.max(60, 100 - (alertEvents * 10) - (warningEvents * 5));
        } else if (warningEvents > 0) {
            this.metrics.securityScore = Math.max(80, 100 - (warningEvents * 5));
        } else {
            this.metrics.securityScore = Math.min(100, this.metrics.securityScore + 1);
        }
        
        this.updateMetricsDisplay();
    }
    
    updateMetricsDisplay() {
        document.getElementById('security-score').textContent = Math.round(this.metrics.securityScore);
        document.getElementById('total-events').textContent = this.metrics.totalEvents;
        document.getElementById('critical-events').textContent = this.metrics.criticalEvents;
        document.getElementById('alert-events').textContent = this.metrics.alertEvents;
        
        // Update security score color
        const scoreElement = document.getElementById('security-score');
        if (this.metrics.securityScore >= 80) {
            scoreElement.style.color = '#00ff88';
        } else if (this.metrics.securityScore >= 60) {
            scoreElement.style.color = '#ffaa00';
        } else {
            scoreElement.style.color = '#ff4444';
        }
        
        // Trigger callbacks
        this.callbacks.onMetricsUpdate.forEach(callback => {
            callback(this.metrics);
        });
    }
    
    updateEventsDisplay() {
        const eventsList = document.getElementById('security-events-list');
        
        if (this.events.length === 0) {
            eventsList.innerHTML = '<div class="no-events">No security events</div>';
            return;
        }
        
        const eventsHtml = this.events.slice(0, 10).map(event => {
            const timestamp = new Date(event.timestamp).toLocaleTimeString();
            const priority = event.original_event.priority.toLowerCase();
            
            return `
                <div class="security-event ${priority}">
                    <div class="event-time">${timestamp}</div>
                    <div class="event-rule">${event.original_event.rule}</div>
                    <div class="event-details">
                        Container: ${event.original_event.output_fields.container_name || 'Unknown'} | 
                        Score: ${event.severity_score}/100
                    </div>
                </div>
            `;
        }).join('');
        
        eventsList.innerHTML = eventsHtml;
    }
    
    updateStatus(text, type) {
        const statusText = document.getElementById('security-status-text');
        const statusIndicator = document.getElementById('security-status-indicator');
        
        if (statusText) statusText.textContent = text;
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${type}`;
        }
    }
    
    checkSecurityAlerts() {
        const recentCriticalEvents = this.events.filter(event => {
            const eventTime = new Date(event.timestamp);
            const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
            return eventTime > fiveMinutesAgo && 
                   (event.original_event.priority === 'CRITICAL' || 
                    event.original_event.priority === 'EMERGENCY');
        });
        
        if (recentCriticalEvents.length > 0) {
            this.showSecurityAlert(recentCriticalEvents);
        }
        
        // Check security score threshold
        if (this.metrics.securityScore < this.config.alertThreshold) {
            this.showSecurityScoreAlert();
        }
    }
    
    showSecurityAlert(events) {
        const alertMessage = `🚨 Critical Security Alert!\n\n${events.length} critical security event(s) detected:\n\n${
            events.map(e => `• ${e.original_event.rule}`).join('\n')
        }\n\nImmediate action may be required.`;
        
        // Show browser notification if permitted
        if (Notification.permission === 'granted') {
            new Notification('Security Alert', {
                body: `${events.length} critical security events detected`,
                icon: '🚨',
                requireInteraction: true
            });
        }
        
        // Update status
        this.updateStatus('Security Alert!', 'critical');
        
        console.warn('Security Alert:', events);
    }
    
    showSecurityScoreAlert() {
        this.updateStatus(`Low Security Score: ${Math.round(this.metrics.securityScore)}`, 'warning');
    }
    
    clearEvents() {
        this.events = [];
        this.updateEventsDisplay();
        this.updateMetrics();
    }
    
    async handleEmergencyStop() {
        if (!confirm('Are you sure you want to trigger an emergency stop? This will halt all trading operations.')) {
            return;
        }
        
        try {
            // Send emergency stop signal to trading bot
            const response = await fetch('/api/emergency/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason: 'Security emergency stop' })
            });
            
            if (response.ok) {
                alert('Emergency stop activated successfully');
            } else {
                alert('Failed to activate emergency stop');
            }
        } catch (error) {
            console.error('Emergency stop failed:', error);
            alert('Emergency stop failed: ' + error.message);
        }
    }
    
    async handleIsolateContainers() {
        if (!confirm('Are you sure you want to isolate trading containers? This will disconnect them from the network.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/security/isolate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ containers: ['ai-trading-bot', 'bluefin-service'] })
            });
            
            if (response.ok) {
                alert('Containers isolated successfully');
            } else {
                alert('Failed to isolate containers');
            }
        } catch (error) {
            console.error('Container isolation failed:', error);
            alert('Container isolation failed: ' + error.message);
        }
    }
    
    openSecurityDashboard() {
        // Open security monitoring dashboard in new tab
        window.open('http://localhost:8080', '_blank');
    }
    
    // Public API methods
    addEventListener(eventType, callback) {
        if (this.callbacks[eventType]) {
            this.callbacks[eventType].push(callback);
        }
    }
    
    removeEventListener(eventType, callback) {
        if (this.callbacks[eventType]) {
            const index = this.callbacks[eventType].indexOf(callback);
            if (index > -1) {
                this.callbacks[eventType].splice(index, 1);
            }
        }
    }
    
    getMetrics() {
        return { ...this.metrics };
    }
    
    getEvents() {
        return [...this.events];
    }
    
    getSecurityScore() {
        return this.metrics.securityScore;
    }
}

// Initialize security monitoring when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
    
    // Initialize security monitor
    window.securityMonitor = new SecurityMonitor({
        apiEndpoint: 'http://localhost:8080',
        refreshInterval: 30000,
        alertThreshold: 70,
        autoStart: true
    });
    
    console.log('Security monitoring dashboard integration loaded');
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SecurityMonitor, SecurityEventTypes, SecurityCategories };
}