/**
 * WebSocket Connectivity Test for Dashboard Frontend
 * 
 * This script can be run in the browser console to test WebSocket connectivity
 * across different environments and configurations.
 */

class ConnectivityTester {
  constructor() {
    this.results = [];
    this.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.hostname = window.location.hostname;
    this.port = window.location.port;
    this.host = window.location.host;
  }

  /**
   * Generate test URLs for different scenarios
   */
  generateTestUrls() {
    const urls = [];

    // Environment variable URLs
    const envWsUrl = window.__WS_URL__ || window.__RUNTIME_CONFIG__?.WS_URL;
    if (envWsUrl) {
      urls.push({ url: envWsUrl, description: 'Environment Variable URL' });
    }

    // Vite environment URLs
    if (typeof import !== 'undefined' && import.meta?.env?.VITE_WS_URL) {
      urls.push({ url: import.meta.env.VITE_WS_URL, description: 'Vite Environment URL' });
    }

    // Docker development scenarios
    if (this.hostname === 'localhost' || this.hostname === '127.0.0.1') {
      urls.push({ url: `${this.protocol}//${this.hostname}:8000/ws`, description: 'Direct Backend Container' });
      urls.push({ url: `${this.protocol}//${this.hostname}:8080/api/ws`, description: 'Nginx Proxy WebSocket' });
      urls.push({ url: `${this.protocol}//${this.host}/api/ws`, description: 'Current Host Nginx Proxy' });
    }

    // Standard fallbacks
    urls.push({ url: `${this.protocol}//${this.host}/ws`, description: 'Current Host Direct' });
    urls.push({ url: `${this.protocol}//${this.host}/api/ws`, description: 'Current Host API Proxy' });

    return urls;
  }

  /**
   * Test a single WebSocket URL
   */
  async testUrl(url, description, timeout = 5000) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      let ws;
      let timeoutId;

      const cleanup = () => {
        if (ws) {
          ws.close();
        }
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
      };

      const result = {
        url,
        description,
        success: false,
        error: null,
        latency: null,
        readyState: null
      };

      try {
        ws = new WebSocket(url);

        timeoutId = setTimeout(() => {
          result.error = 'Connection timeout';
          result.readyState = ws.readyState;
          cleanup();
          resolve(result);
        }, timeout);

        ws.onopen = () => {
          result.success = true;
          result.latency = Date.now() - startTime;
          result.readyState = ws.readyState;
          cleanup();
          resolve(result);
        };

        ws.onerror = (error) => {
          result.error = 'Connection error';
          result.readyState = ws.readyState;
          cleanup();
          resolve(result);
        };

        ws.onclose = (event) => {
          if (!result.success) {
            result.error = `Connection closed: ${event.code} ${event.reason}`;
            result.readyState = ws.readyState;
          }
          cleanup();
          resolve(result);
        };

      } catch (error) {
        result.error = error.message;
        cleanup();
        resolve(result);
      }
    });
  }

  /**
   * Test all URLs and return results
   */
  async testAll() {
    const urls = this.generateTestUrls();
    console.log(`🔗 Testing ${urls.length} WebSocket URLs...`);

    this.results = [];
    for (const { url, description } of urls) {
      console.log(`   Testing: ${description} (${url})`);
      const result = await this.testUrl(url, description);
      this.results.push(result);
      
      if (result.success) {
        console.log(`   ✅ Success (${result.latency}ms)`);
      } else {
        console.log(`   ❌ Failed: ${result.error}`);
      }
    }

    return this.results;
  }

  /**
   * Display formatted results
   */
  displayResults() {
    console.log('\n📊 WebSocket Connectivity Test Results:');
    console.log('='*50);

    const successful = this.results.filter(r => r.success);
    const failed = this.results.filter(r => !r.success);

    console.log(`✅ Successful connections: ${successful.length}`);
    console.log(`❌ Failed connections: ${failed.length}`);

    if (successful.length > 0) {
      console.log('\n🔗 Working URLs:');
      successful.forEach(result => {
        console.log(`   ${result.description}: ${result.url} (${result.latency}ms)`);
      });
    }

    if (failed.length > 0) {
      console.log('\n💥 Failed URLs:');
      failed.forEach(result => {
        console.log(`   ${result.description}: ${result.url}`);
        console.log(`      Error: ${result.error}`);
      });
    }

    // Recommendations
    console.log('\n💡 Recommendations:');
    if (successful.length === 0) {
      console.log('   - Check if the backend service is running');
      console.log('   - Verify network connectivity');
      console.log('   - Check firewall settings');
      console.log('   - Ensure WebSocket support is enabled');
    } else {
      const bestResult = successful.sort((a, b) => a.latency - b.latency)[0];
      console.log(`   - Best connection: ${bestResult.description} (${bestResult.url})`);
      console.log(`   - Consider using this URL in your environment configuration`);
    }

    return this.results;
  }

  /**
   * Get environment information
   */
  getEnvironmentInfo() {
    console.log('\n🌍 Environment Information:');
    console.log(`   Protocol: ${window.location.protocol}`);
    console.log(`   Hostname: ${window.location.hostname}`);
    console.log(`   Port: ${window.location.port || 'default'}`);
    console.log(`   Host: ${window.location.host}`);
    console.log(`   User Agent: ${navigator.userAgent}`);
    
    // Check for environment variables
    console.log('\n🔧 Environment Variables:');
    console.log(`   VITE_WS_URL: ${window.__WS_URL__ || 'not set'}`);
    console.log(`   Runtime Config: ${JSON.stringify(window.__RUNTIME_CONFIG__ || 'not set')}`);
    console.log(`   Docker Env: ${window.__DOCKER_ENV__ || 'not set'}`);
  }
}

// Auto-run if this script is executed directly
if (typeof window !== 'undefined') {
  window.testWebSocketConnectivity = async function() {
    const tester = new ConnectivityTester();
    tester.getEnvironmentInfo();
    await tester.testAll();
    return tester.displayResults();
  };

  console.log('🚀 WebSocket Connectivity Tester loaded!');
  console.log('   Run: testWebSocketConnectivity()');
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ConnectivityTester;
}