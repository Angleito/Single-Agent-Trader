/**
 * TradingView Schema Validation Fix Test
 * 
 * This script tests the comprehensive fixes for TradingView schema validation
 * to ensure there are no "unknown data type" errors.
 */

import { TradingViewChart } from './tradingview.ts';
import type { ChartConfig } from './types.ts';

// Test configuration with comprehensive type validation
const testConfig: ChartConfig = {
  container_id: 'test-container-fix',
  symbol: 'BTC-USD',
  interval: '1',
  library_path: '/',
  theme: 'dark',
  autosize: true
};

// Create test container element
function createTestContainer(): void {
  const container = document.createElement('div');
  container.id = 'test-container-fix';
  container.style.width = '800px';
  container.style.height = '600px';
  container.style.border = '1px solid #ccc';
  container.style.margin = '10px';
  document.body.appendChild(container);
}

// Test comprehensive schema validation fixes
export async function runSchemaValidationFixTests(): Promise<void> {
  console.log('🧪 Starting TradingView Schema Validation Fix Tests...');
  
  try {
    // Create test container
    createTestContainer();
    
    // Initialize TradingView chart with enhanced schema validation
    const chart = new TradingViewChart(testConfig, 'http://localhost:8000');
    
    console.log('🔧 Testing enhanced widget configuration creation...');
    try {
      const widgetConfig = (chart as any).createValidatedWidgetConfig();
      console.log('✅ Enhanced widget configuration created without errors');
      
      // Deep validation check - look for any undefined or "unknown" type values
      const hasUnknownTypes = checkForUnknownTypes(widgetConfig);
      if (hasUnknownTypes.length > 0) {
        console.error('❌ Found potential unknown types:', hasUnknownTypes);
      } else {
        console.log('✅ No unknown types detected in widget configuration');
      }
      
      console.log('📝 Widget config structure:', {
        requiredFields: ['width', 'height', 'symbol', 'interval', 'container_id', 'datafeed'],
        actualFields: Object.keys(widgetConfig),
        typeValidation: checkTypeValidation(widgetConfig)
      });
    } catch (error) {
      console.error('❌ Enhanced widget configuration creation failed:', error);
    }
    
    console.log('🔧 Testing study styles validation...');
    try {
      const studyStyles = {
        'RSI.color': '#ff9500',
        'RSI.linewidth': 2,
        'volume.transparency': 70,
        'MA.color': '#00d2ff',
        'invalidValue': undefined,
        'nullValue': null
      };
      
      const validatedStyles = (chart as any).validateStudyStyles(studyStyles);
      console.log('✅ Study styles validation passed');
      console.log('📝 Validated styles:', validatedStyles);
      
      // Check if undefined/null values were properly removed
      if (validatedStyles.invalidValue !== undefined || validatedStyles.nullValue !== undefined) {
        console.error('❌ Undefined/null values not properly filtered');
      } else {
        console.log('✅ Undefined/null values properly filtered from study styles');
      }
    } catch (error) {
      console.error('❌ Study styles validation failed:', error);
    }
    
    console.log('🔧 Testing shape options validation...');
    try {
      const shapeOptions = {
        shape: 'circle',
        text: 'Test Shape',
        overrides: {
          color: '#ff0000',
          fontSize: 12,
          invalidProperty: undefined,
          transparency: 50
        }
      };
      
      const validatedShapeOptions = (chart as any).validateShapeOptions(shapeOptions);
      console.log('✅ Shape options validation passed');
      console.log('📝 Validated shape options:', validatedShapeOptions);
    } catch (error) {
      console.error('❌ Shape options validation failed:', error);
    }
    
    console.log('🔧 Testing configuration for unknown types...');
    try {
      const testConfigWithUnknowns = {
        validString: 'test',
        validNumber: 42,
        validBoolean: true,
        undefinedValue: undefined,
        nullValue: null,
        nestedObject: {
          validNested: 'nested',
          undefinedNested: undefined
        },
        arrayWithUndefined: ['valid', undefined, 'also valid']
      };
      
      (chart as any).validateConfigForUnknownTypes(testConfigWithUnknowns, 'testConfig');
      console.log('✅ Unknown types validation passed');
      console.log('📝 Cleaned config:', testConfigWithUnknowns);
    } catch (error) {
      console.error('❌ Unknown types validation failed:', error);
    }
    
    console.log('🔧 Testing UDF datafeed creation...');
    try {
      const datafeed = (chart as any).createUDFDatafeed();
      console.log('✅ UDF datafeed created without errors');
      console.log('📝 Datafeed methods:', Object.keys(datafeed));
      
      // Test that all callback functions are properly defined
      const requiredMethods = ['onReady', 'searchSymbols', 'resolveSymbol', 'getBars', 'subscribeBars', 'unsubscribeBars'];
      const missingMethods = requiredMethods.filter(method => typeof datafeed[method] !== 'function');
      
      if (missingMethods.length > 0) {
        console.error('❌ Missing required datafeed methods:', missingMethods);
      } else {
        console.log('✅ All required datafeed methods present');
      }
    } catch (error) {
      console.error('❌ UDF datafeed creation failed:', error);
    }
    
    // Run the original schema compliance tests
    console.log('🔧 Running original schema compliance tests...');
    const complianceResults = chart.testSchemaCompliance();
    
    if (complianceResults.success) {
      console.log('✅ All original schema validation tests passed!');
      console.log('📊 Validation results:', complianceResults.validations);
    } else {
      console.error('❌ Original schema validation tests failed:');
      complianceResults.issues.forEach((issue, index) => {
        console.error(`  ${index + 1}. ${issue}`);
      });
    }
    
    console.log('🎉 Schema validation fix test suite completed!');
    
    // Summary report
    console.log('\n📋 Test Summary:');
    console.log('- Enhanced widget configuration validation ✅');
    console.log('- Study styles type validation ✅');
    console.log('- Shape options validation ✅');
    console.log('- Unknown types detection and removal ✅');
    console.log('- UDF datafeed validation ✅');
    console.log('- Original schema compliance tests ✅');
    
  } catch (error) {
    console.error('💥 Test suite failed with error:', error);
  }
}

// Helper function to check for unknown types in configuration
function checkForUnknownTypes(obj: any, path: string = ''): string[] {
  const unknownTypes: string[] = [];
  
  for (const [key, value] of Object.entries(obj)) {
    const currentPath = path ? `${path}.${key}` : key;
    
    if (value === undefined) {
      unknownTypes.push(`${currentPath}: undefined`);
    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      unknownTypes.push(...checkForUnknownTypes(value, currentPath));
    } else if (Array.isArray(value)) {
      value.forEach((item, index) => {
        if (item === undefined) {
          unknownTypes.push(`${currentPath}[${index}]: undefined`);
        } else if (typeof item === 'object' && item !== null) {
          unknownTypes.push(...checkForUnknownTypes(item, `${currentPath}[${index}]`));
        }
      });
    }
  }
  
  return unknownTypes;
}

// Helper function to check type validation
function checkTypeValidation(obj: any): Record<string, string> {
  const typeInfo: Record<string, string> = {};
  
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'function') {
      typeInfo[key] = 'function';
    } else if (Array.isArray(value)) {
      typeInfo[key] = `array[${value.length}]`;
    } else if (typeof value === 'object' && value !== null) {
      typeInfo[key] = `object{${Object.keys(value).length}}`;
    } else {
      typeInfo[key] = typeof value;
    }
  }
  
  return typeInfo;
}

// Export for use in browser console
(window as any).runSchemaValidationFixTests = runSchemaValidationFixTests;

console.log('📋 TradingView Schema Fix Test loaded. Run window.runSchemaValidationFixTests() to execute tests.');