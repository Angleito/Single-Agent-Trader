/**
 * Quick test runner for the Ultimate TradingView Schema Error Fix
 */

import { validateUltimateFix } from './validate-ultimate-fix';

async function runQuickTest(): Promise<void> {
  console.log('🚀 Starting Ultimate TradingView Schema Error Fix Test...');
  
  try {
    const results = await validateUltimateFix();
    
    console.log('\n📊 QUICK TEST SUMMARY:');
    console.log(`✅ Overall Success: ${results.overallSuccess}`);
    console.log(`🧪 Tests Passed: ${results.passedTests}/${results.totalTests}`);
    console.log(`🚨 Schema Errors: ${results.schemaErrorsDetected}`);
    console.log(`🛡️ Errors Suppressed: ${results.errorsSuppressed}`);
    console.log(`⏱️ Execution Time: ${results.executionTime}ms`);
    
    if (results.overallSuccess) {
      console.log('\n🎉 ULTIMATE FIX IS WORKING CORRECTLY!');
      console.log('   No TradingView schema errors detected.');
    } else {
      console.log('\n⚠️ ULTIMATE FIX NEEDS ATTENTION');
      console.log('   Some tests failed or schema errors were detected.');
      
      // Show failed tests
      const failedTests = results.details.filter(test => !test.passed);
      if (failedTests.length > 0) {
        console.log('\n❌ Failed Tests:');
        failedTests.forEach(test => {
          console.log(`   • ${test.testName}: ${test.details}`);
        });
      }
    }
    
  } catch (error) {
    console.error('❌ Test execution failed:', error);
  }
}

// Run the test
runQuickTest().catch(console.error);