// Simple console check using node-fetch to test the page
import fetch from 'node-fetch';

async function checkDashboard() {
  try {
    console.log('🔍 Checking dashboard accessibility...');
    
    // Test main page
    const response = await fetch('http://localhost:3000');
    console.log(`📱 Dashboard HTTP Status: ${response.status}`);
    
    if (response.ok) {
      console.log('✅ Dashboard is accessible');
      
      // Test if TradingView can be loaded
      const tvResponse = await fetch('https://s3.tradingview.com/tv.js');
      console.log(`📈 TradingView CDN Status: ${tvResponse.status}`);
      
      // Test WebSocket endpoint (will fail but shows if backend is responding)
      try {
        const wsTest = await fetch('http://localhost:8000/health');
        console.log(`🔌 Backend Health Status: ${wsTest.status}`);
      } catch (e) {
        console.log('❌ Backend not accessible:', e.message);
      }
      
      console.log('\n📋 Manual Console Check Required:');
      console.log('1. Open http://localhost:3000 in your browser');
      console.log('2. Press F12 to open Developer Tools');
      console.log('3. Check the Console tab for any errors');
      console.log('4. Look specifically for:');
      console.log('   - Red error messages');
      console.log('   - "unknown data type" messages');
      console.log('   - WebSocket connection errors');
      console.log('   - TradingView schema errors');
      
    } else {
      console.log('❌ Dashboard not accessible');
    }
    
  } catch (error) {
    console.error('❌ Error checking dashboard:', error.message);
  }
}

checkDashboard();