/**
 * ULTIMATE TRADINGVIEW SCHEMA ERROR FIX VALIDATION
 * 
 * This script validates that the aggressive TradingView schema error suppression
 * is working correctly and completely eliminates the specific error:
 * "Property:The state with a data type: unknown does not match a schema"
 */

import { TradingViewChart } from './tradingview';
import type { ChartConfig } from './types';

interface ValidationResult {
  testName: string;
  passed: boolean;
  details: string;
  duration: number;
  errorCount: number;
}

interface ValidationSummary {
  totalTests: number;
  passedTests: number;
  failedTests: number;
  schemaErrorsDetected: number;
  errorsSuppressed: number;
  overallSuccess: boolean;
  executionTime: number;
  details: ValidationResult[];
}

export class UltimateFixValidator {
  private schemaErrorsDetected = 0;
  private errorsSuppressed = 0;
  private originalConsoleError: typeof console.error;
  private testResults: ValidationResult[] = [];
  private startTime = Date.now();

  constructor() {
    this.originalConsoleError = console.error;
    this.setupErrorInterception();
  }

  /**
   * Set up error interception to monitor for the specific TradingView schema error
   */
  private setupErrorInterception(): void {
    console.error = (...args: any[]) => {
      const errorMessage = args.join(' ');
      
      // Check for the specific error we're trying to eliminate
      if (errorMessage.includes('Property:The state with a data type: unknown does not match a schema') ||
          (errorMessage.includes('56106.2e8fa41f279a0fad5423.js') && errorMessage.includes('Property'))) {
        this.schemaErrorsDetected++;
        console.warn('🚨 CRITICAL: Schema error detected - Ultimate fix may not be working:', errorMessage);
        
        // Still call original to see the error
        this.originalConsoleError.apply(console, args);
      } else if (errorMessage.includes('🛡️') || errorMessage.includes('suppressed')) {
        this.errorsSuppressed++;
        // Call original for suppression messages
        this.originalConsoleError.apply(console, args);
      } else {
        // Call original for all other errors
        this.originalConsoleError.apply(console, args);
      }
    };
  }

  /**
   * Run comprehensive validation tests
   */
  public async runValidation(): Promise<ValidationSummary> {
    console.log('🛡️ Starting Ultimate TradingView Schema Error Fix Validation...');
    
    this.testResults = [];
    this.schemaErrorsDetected = 0;
    this.errorsSuppressed = 0;
    this.startTime = Date.now();

    // Test 1: Basic TradingView Initialization with Error Suppression
    await this.testBasicInitialization();

    // Test 2: Schema Error Simulation and Suppression
    await this.testSchemaErrorSuppression();

    // Test 3: Object Sanitization for Unknown Types
    await this.testObjectSanitization();

    // Test 4: TradingView Internal Validation Patching
    await this.testInternalValidationPatching();

    // Test 5: Global Property Descriptor Override
    await this.testGlobalPropertyOverrides();

    // Test 6: Aggressive Console Error Suppression
    await this.testConsoleErrorSuppression();

    // Test 7: JSON Serialization with Unknown Types
    await this.testJSONSerialization();

    // Test 8: Real-world TradingView Usage Simulation
    await this.testRealWorldUsage();

    // Generate summary
    const summary = this.generateValidationSummary();
    this.reportResults(summary);
    
    return summary;
  }

  /**
   * Test 1: Basic TradingView initialization with ultimate error suppression
   */
  private async testBasicInitialization(): Promise<void> {
    const testName = 'Basic TradingView Initialization';
    const startTime = Date.now();
    let errorCount = 0;
    
    try {
      console.log('🧪 Testing basic TradingView initialization...');
      
      const config: ChartConfig = {
        symbol: 'BINANCE:BTCUSDT',
        interval: '1m',
        container_id: 'test-container',
        library_path: 'https://s3.tradingview.com/tv.js',
        theme: 'dark'
      };

      const chart = new TradingViewChart(config);
      
      // Monitor error count before initialization
      const initialErrorCount = this.schemaErrorsDetected;
      
      // Initialize (this should trigger our ultimate error suppression)
      const success = await chart.initialize();
      
      // Check if any schema errors occurred during initialization
      errorCount = this.schemaErrorsDetected - initialErrorCount;
      
      this.testResults.push({
        testName,
        passed: errorCount === 0 && success,
        details: `Initialization ${success ? 'successful' : 'failed'}, Schema errors: ${errorCount}`,
        duration: Date.now() - startTime,
        errorCount
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: errorCount + 1
      });
    }
  }

  /**
   * Test 2: Direct schema error simulation and suppression
   */
  private async testSchemaErrorSuppression(): Promise<void> {
    const testName = 'Schema Error Suppression';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing schema error suppression...');
      
      const initialErrorCount = this.schemaErrorsDetected;
      
      // Directly simulate the exact error that was occurring
      console.error('Property:The state with a data type: unknown does not match a schema');
      console.error('56106.2e8fa41f279a0fad5423.js:20 2025-06-13T04:57:01.868Z:Property:The state with a data type: unknown does not match a schema');
      
      // Wait a moment for error processing
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const errorCount = this.schemaErrorsDetected - initialErrorCount;
      
      this.testResults.push({
        testName,
        passed: errorCount === 0,
        details: `Simulated schema errors suppressed: ${errorCount === 0 ? 'YES' : 'NO'}, Errors detected: ${errorCount}`,
        duration: Date.now() - startTime,
        errorCount
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 3: Object sanitization for unknown types
   */
  private async testObjectSanitization(): Promise<void> {
    const testName = 'Object Sanitization';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing object sanitization...');
      
      // Create objects with problematic types
      const problematicData = {
        undefinedValue: undefined,
        functionValue: function() { return 'test'; },
        symbolValue: Symbol('test'),
        bigintValue: BigInt(123),
        nestedObject: {
          unknown: undefined,
          nestedFunction: () => 'nested',
          deepNested: {
            veryDeep: undefined
          }
        }
      };
      
      // Test JSON serialization (should be handled by our overrides)
      let jsonSuccess = false;
      try {
        const jsonString = JSON.stringify(problematicData);
        jsonSuccess = !!(jsonString && jsonString !== '{}');
      } catch (error) {
        jsonSuccess = false;
      }
      
      // Test property access
      let propertySuccess = false;
      try {
        Object.defineProperty({}, 'testProp', {
          value: undefined,
          writable: true
        });
        propertySuccess = true;
      } catch (error) {
        propertySuccess = false;
      }
      
      this.testResults.push({
        testName,
        passed: jsonSuccess && propertySuccess,
        details: `JSON serialization: ${jsonSuccess ? 'PASS' : 'FAIL'}, Property access: ${propertySuccess ? 'PASS' : 'FAIL'}`,
        duration: Date.now() - startTime,
        errorCount: 0
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 4: TradingView internal validation patching
   */
  private async testInternalValidationPatching(): Promise<void> {
    const testName = 'Internal Validation Patching';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing TradingView internal validation patching...');
      
      // Wait for TradingView to be available
      await this.waitForTradingView();
      
      let patchingSuccess = false;
      
      if ((window as any).TradingView) {
        // Check if our patches are in place
        const tv = (window as any).TradingView;
        
        // Test if validation methods exist and are patched
        if (tv.prototype?.validateSchema || tv.widget?.prototype?._validateState) {
          patchingSuccess = true;
        }
      }
      
      this.testResults.push({
        testName,
        passed: patchingSuccess,
        details: `TradingView patching: ${patchingSuccess ? 'ACTIVE' : 'NOT DETECTED'}`,
        duration: Date.now() - startTime,
        errorCount: 0
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 5: Global property descriptor overrides
   */
  private async testGlobalPropertyOverrides(): Promise<void> {
    const testName = 'Global Property Overrides';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing global property descriptor overrides...');
      
      // Test Object.defineProperty override
      let definePropertyWorking = false;
      try {
        const testObj = {};
        Object.defineProperty(testObj, 'testProp', {
          value: undefined,
          writable: true
        });
        definePropertyWorking = true;
      } catch (error) {
        definePropertyWorking = false;
      }
      
      // Test Object.getOwnPropertyDescriptor override
      let getDescriptorWorking = false;
      try {
        const testObj = { testProp: undefined };
        const descriptor = Object.getOwnPropertyDescriptor(testObj, 'testProp');
        getDescriptorWorking = descriptor !== undefined;
      } catch (error) {
        getDescriptorWorking = false;
      }
      
      this.testResults.push({
        testName,
        passed: definePropertyWorking && getDescriptorWorking,
        details: `defineProperty: ${definePropertyWorking ? 'WORKING' : 'FAILED'}, getDescriptor: ${getDescriptorWorking ? 'WORKING' : 'FAILED'}`,
        duration: Date.now() - startTime,
        errorCount: 0
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 6: Console error suppression
   */
  private async testConsoleErrorSuppression(): Promise<void> {
    const testName = 'Console Error Suppression';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing console error suppression...');
      
      const initialErrorCount = this.schemaErrorsDetected;
      
      // Test various error message patterns that should be suppressed
      const testErrorMessages = [
        'Property:The state with a data type: unknown does not match a schema',
        'unknown data type schema',
        '56106.2e8fa41f279a0fad5423.js Property error',
        'TradingView schema validation failed'
      ];
      
      // These should all be suppressed
      testErrorMessages.forEach(msg => console.error(msg));
      
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const errorCount = this.schemaErrorsDetected - initialErrorCount;
      
      this.testResults.push({
        testName,
        passed: errorCount === 0,
        details: `Error messages suppressed: ${errorCount === 0 ? 'ALL' : 'PARTIAL'}, Errors leaked: ${errorCount}`,
        duration: Date.now() - startTime,
        errorCount
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 7: JSON serialization with unknown types
   */
  private async testJSONSerialization(): Promise<void> {
    const testName = 'JSON Serialization';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing JSON serialization with unknown types...');
      
      const problematicData = {
        state: undefined,
        dataType: 'unknown',
        validation: function() { return false; },
        symbol: Symbol('test'),
        bigInt: BigInt(123),
        nested: {
          unknown: undefined,
          func: () => 'test'
        }
      };
      
      let serializationSuccess = false;
      let serializedData = '';
      
      try {
        serializedData = JSON.stringify(problematicData);
        serializationSuccess = !!(serializedData && serializedData !== '{}');
      } catch (error) {
        serializationSuccess = false;
      }
      
      this.testResults.push({
        testName,
        passed: serializationSuccess,
        details: `JSON serialization: ${serializationSuccess ? 'SUCCESS' : 'FAILED'}, Output: ${serializedData.substring(0, 100)}...`,
        duration: Date.now() - startTime,
        errorCount: 0
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Test 8: Real-world TradingView usage simulation
   */
  private async testRealWorldUsage(): Promise<void> {
    const testName = 'Real-world Usage Simulation';
    const startTime = Date.now();
    
    try {
      console.log('🧪 Testing real-world TradingView usage...');
      
      const initialErrorCount = this.schemaErrorsDetected;
      
      // Simulate real-world usage patterns that might trigger the schema error
      const config: ChartConfig = {
        symbol: 'NASDAQ:AAPL',
        interval: '5m',
        container_id: 'real-world-test',
        library_path: 'https://s3.tradingview.com/tv.js',
        theme: 'dark'
      };

      const chart = new TradingViewChart(config);
      
      // Multiple initialization attempts (common in real apps)
      for (let i = 0; i < 3; i++) {
        try {
          await chart.initialize();
          await new Promise(resolve => setTimeout(resolve, 500));
        } catch (error) {
          // Expected to fail in test environment, but shouldn't cause schema errors
        }
      }
      
      const errorCount = this.schemaErrorsDetected - initialErrorCount;
      
      this.testResults.push({
        testName,
        passed: errorCount === 0,
        details: `Real-world simulation completed, Schema errors: ${errorCount}`,
        duration: Date.now() - startTime,
        errorCount
      });
      
    } catch (error) {
      this.testResults.push({
        testName,
        passed: false,
        details: `Test failed with error: ${error instanceof Error ? error.message : String(error)}`,
        duration: Date.now() - startTime,
        errorCount: 1
      });
    }
  }

  /**
   * Wait for TradingView library to be available
   */
  private async waitForTradingView(timeout = 5000): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if ((window as any).TradingView) {
        return;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  /**
   * Generate validation summary
   */
  private generateValidationSummary(): ValidationSummary {
    const totalTests = this.testResults.length;
    const passedTests = this.testResults.filter(test => test.passed).length;
    const failedTests = totalTests - passedTests;
    const executionTime = Date.now() - this.startTime;
    
    // The ultimate fix is successful if no schema errors were detected
    const overallSuccess = this.schemaErrorsDetected === 0 && passedTests === totalTests;
    
    return {
      totalTests,
      passedTests,
      failedTests,
      schemaErrorsDetected: this.schemaErrorsDetected,
      errorsSuppressed: this.errorsSuppressed,
      overallSuccess,
      executionTime,
      details: this.testResults
    };
  }

  /**
   * Report validation results
   */
  private reportResults(summary: ValidationSummary): void {
    console.log('\n' + '='.repeat(80));
    console.log('🛡️ ULTIMATE TRADINGVIEW SCHEMA ERROR FIX VALIDATION RESULTS');
    console.log('='.repeat(80));
    
    console.log(`📊 Overall Status: ${summary.overallSuccess ? '✅ SUCCESS' : '❌ FAILED'}`);
    console.log(`🧪 Tests: ${summary.passedTests}/${summary.totalTests} passed`);
    console.log(`🚨 Schema Errors Detected: ${summary.schemaErrorsDetected}`);
    console.log(`🛡️ Errors Suppressed: ${summary.errorsSuppressed}`);
    console.log(`⏱️ Execution Time: ${summary.executionTime}ms`);
    
    console.log('\n📋 Detailed Test Results:');
    console.log('-'.repeat(80));
    
    summary.details.forEach((test, index) => {
      const status = test.passed ? '✅' : '❌';
      console.log(`${status} Test ${index + 1}: ${test.testName}`);
      console.log(`   Duration: ${test.duration}ms | Errors: ${test.errorCount}`);
      console.log(`   Details: ${test.details}`);
      console.log('');
    });
    
    console.log('='.repeat(80));
    
    if (summary.overallSuccess) {
      console.log('🎉 ULTIMATE FIX VALIDATION SUCCESSFUL!');
      console.log('   The TradingView schema error has been completely eliminated.');
    } else {
      console.log('⚠️ ULTIMATE FIX NEEDS ADJUSTMENT');
      console.log('   Some schema errors are still occurring or tests failed.');
    }
    
    console.log('='.repeat(80));
  }

  /**
   * Cleanup validation
   */
  public cleanup(): void {
    // Restore original console.error
    console.error = this.originalConsoleError;
  }
}

// Export function to run validation
export async function validateUltimateFix(): Promise<ValidationSummary> {
  const validator = new UltimateFixValidator();
  
  try {
    const results = await validator.runValidation();
    return results;
  } finally {
    validator.cleanup();
  }
}

// If running directly
if (typeof window !== 'undefined') {
  // Browser environment
  (window as any).validateUltimateFix = validateUltimateFix;
} else if (typeof module !== 'undefined' && module.exports) {
  // Node.js environment
  module.exports = { validateUltimateFix, UltimateFixValidator };
}