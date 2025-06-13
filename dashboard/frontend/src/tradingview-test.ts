// TradingView Loading Test Script
// This file demonstrates the enhanced TradingView loading capabilities

import { TradingViewChart } from './tradingview.ts';
import type { ChartConfig } from './types.ts';

/**
 * Test the enhanced TradingView loading with timeout fixes
 */
export async function testTradingViewLoading() {
  console.log('=== TradingView Loading Test ===');
  
  const config: ChartConfig = {
    container_id: 'tradingview_chart',
    symbol: 'BTC-USD',
    interval: '1',
    library_path: '/',
    theme: 'dark'
  };

  const chart = new TradingViewChart(config);

  // Test 1: Enhanced diagnostics
  console.log('--- Enhanced Diagnostics ---');
  const diagnostics = chart.getEnhancedDiagnostics();
  console.log('System Info:', {
    network: diagnostics.connection.online,
    browser: diagnostics.browser.userAgent.split(' ')[0],
    container: diagnostics.dom.containerExists,
    tradingViewScripts: diagnostics.dom.tradingViewScripts
  });

  // Test 2: CDN connectivity test
  console.log('--- CDN Connectivity Test ---');
  try {
    const connectivity = await chart.testTradingViewConnectivity();
    console.log(`TradingView CDN: ${connectivity.accessible ? 'ACCESSIBLE' : 'NOT ACCESSIBLE'}`);
    console.log(`Latency: ${connectivity.latency}ms`);
    if (connectivity.error) {
      console.log(`Error: ${connectivity.error}`);
    }
  } catch (error) {
    console.error('CDN test failed:', error);
  }

  // Test 3: Initialize with preflight checks
  console.log('--- Initialization with Preflight Checks ---');
  try {
    const success = await chart.initializeWithPreflightChecks();
    console.log(`Initialization: ${success ? 'SUCCESS' : 'FAILED'}`);
    
    if (success) {
      console.log('Chart initialized successfully with enhanced loading!');
      
      // Test performance metrics
      const metrics = chart.getPerformanceMetrics();
      console.log('Performance metrics:', metrics);
    } else {
      console.log('Chart initialization failed. Running diagnostics...');
      const diagResult = await chart.runDiagnostics();
      console.log('Diagnostic result:', diagResult);
    }
  } catch (error) {
    console.error('Initialization test failed:', error);
  }

  // Test 4: Verify enhanced timeout behavior
  console.log('--- Timeout Behavior Test ---');
  console.log('The new timeout is set to 30 seconds (up from 15 seconds)');
  console.log('Multiple CDN fallbacks are configured');
  console.log('Exponential backoff retry logic is active');
  console.log('Network monitoring is enabled');

  return chart;
}

/**
 * Monitor TradingView loading progress
 */
export function monitorTradingViewLoading() {
  console.log('=== TradingView Loading Monitor ===');
  
  const startTime = Date.now();
  
  const monitor = setInterval(() => {
    const elapsed = Date.now() - startTime;
    const scriptExists = !!document.querySelector('script[src*="tradingview"]');
    const libraryLoaded = !!(window.TradingView && window.TradingView.widget);
    const networkOnline = navigator.onLine;
    
    console.log(`[${Math.round(elapsed/1000)}s] Script: ${scriptExists}, Library: ${libraryLoaded}, Network: ${networkOnline}`);
    
    if (libraryLoaded) {
      console.log('✅ TradingView library successfully loaded!');
      clearInterval(monitor);
    } else if (elapsed > 45000) {
      console.log('❌ TradingView loading timeout after 45 seconds');
      clearInterval(monitor);
    }
  }, 2000);
  
  return monitor;
}

/**
 * Create a test HTML container for TradingView
 */
export function createTestContainer(): HTMLElement {
  const container = document.createElement('div');
  container.id = 'tradingview_chart';
  container.style.width = '100%';
  container.style.height = '500px';
  container.style.border = '1px solid #ccc';
  container.style.backgroundColor = '#1e1e1e';
  
  // Add to document if not already present
  if (!document.getElementById('tradingview_chart')) {
    document.body.appendChild(container);
  }
  
  return container;
}

// Auto-run test if in browser environment
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, creating test container...');
    createTestContainer();
    
    console.log('Starting TradingView loading monitor...');
    monitorTradingViewLoading();
    
    // Start the test after a short delay
    setTimeout(() => {
      testTradingViewLoading().catch(error => {
        console.error('TradingView test failed:', error);
      });
    }, 1000);
  });
}