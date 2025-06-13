/**
 * TradingView Schema Validation Test
 * 
 * This script tests the TradingView schema validation fixes
 * to ensure there are no "unknown data type" errors.
 */

import { TradingViewChart } from './tradingview.ts';
import type { ChartConfig } from './types.ts';

// Test configuration
const testConfig: ChartConfig = {
  container_id: 'test-container',
  symbol: 'BTC-USD',
  interval: '1',
  library_path: '/',
  theme: 'dark',
  autosize: true
};

// Create test container element
function createTestContainer(): void {
  const container = document.createElement('div');
  container.id = 'test-container';
  container.style.width = '800px';
  container.style.height = '600px';
  document.body.appendChild(container);
}

// Test TradingView schema compliance
export async function runSchemaValidationTests(): Promise<void> {
  console.log('🧪 Starting TradingView Schema Validation Tests...');
  
  try {
    // Create test container
    createTestContainer();
    
    // Initialize TradingView chart with schema validation
    const chart = new TradingViewChart(testConfig, 'http://localhost:8000');
    
    // Run schema compliance tests
    const complianceResults = chart.testSchemaCompliance();
    
    if (complianceResults.success) {
      console.log('✅ All schema validation tests passed!');
      console.log('📊 Validation results:', complianceResults.validations);
    } else {
      console.error('❌ Schema validation tests failed:');
      complianceResults.issues.forEach((issue, index) => {
        console.error(`  ${index + 1}. ${issue}`);
      });
    }
    
    // Test widget initialization (without actually creating it)
    console.log('🔧 Testing widget configuration creation...');
    try {
      // This should not throw any schema validation errors
      const widgetConfig = (chart as any).createValidatedWidgetConfig();
      console.log('✅ Widget configuration created without errors');
      console.log('📝 Sample config keys:', Object.keys(widgetConfig));
    } catch (error) {
      console.error('❌ Widget configuration creation failed:', error);
    }
    
    // Test UDF datafeed creation
    console.log('🔧 Testing UDF datafeed creation...');
    try {
      const datafeed = (chart as any).createUDFDatafeed();
      console.log('✅ UDF datafeed created without errors');
      console.log('📝 Datafeed methods:', Object.keys(datafeed));
    } catch (error) {
      console.error('❌ UDF datafeed creation failed:', error);
    }
    
    // Test data validation functions
    console.log('🔧 Testing data validation functions...');
    
    // Test bar data validation
    try {
      const testBar = (chart as any).validateBarData({
        time: Date.now(),
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 1000
      });
      console.log('Bar validation result:', testBar);
      console.log('✅ Bar data validation passed');
    } catch (error) {
      console.error('❌ Bar data validation failed:', error);
    }
    
    // Test symbol info validation
    try {
      const testSymbolInfo = (chart as any).validateSymbolInfo({
        name: 'BTCUSD',
        type: 'crypto',
        pricescale: 100000,
        has_intraday: true
      });
      console.log('Symbol info validation result:', testSymbolInfo);
      console.log('✅ Symbol info validation passed');
    } catch (error) {
      console.error('❌ Symbol info validation failed:', error);
    }
    
    // Test shape options validation
    try {
      const testShapeOptions = (chart as any).validateShapeOptions({
        shape: 'circle',
        overrides: {
          color: '#ff0000',
          fontSize: 12
        }
      });
      console.log('Shape options validation result:', testShapeOptions);
      console.log('✅ Shape options validation passed');
    } catch (error) {
      console.error('❌ Shape options validation failed:', error);
    }
    
    console.log('🎉 Schema validation test suite completed!');
    
  } catch (error) {
    console.error('💥 Test suite failed with error:', error);
  }
}

// Export for use in browser console
(window as any).runSchemaValidationTests = runSchemaValidationTests;

console.log('📋 TradingView Schema Test loaded. Run window.runSchemaValidationTests() to execute tests.');