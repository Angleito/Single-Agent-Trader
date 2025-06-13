import { chromium } from 'playwright';

async function checkBrowserConsole() {
  console.log('🔍 Starting browser console check...');
  
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  // Collect console messages
  const consoleMessages = [];
  const errors = [];
  
  page.on('console', (msg) => {
    const type = msg.type();
    const text = msg.text();
    
    consoleMessages.push({
      type,
      text,
      timestamp: new Date().toISOString()
    });
    
    if (type === 'error') {
      errors.push(text);
    }
    
    console.log(`[${type.toUpperCase()}] ${text}`);
  });
  
  page.on('pageerror', (error) => {
    errors.push(`PAGE ERROR: ${error.message}`);
    console.log(`[PAGE ERROR] ${error.message}`);
  });
  
  try {
    console.log('📱 Navigating to dashboard...');
    await page.goto('http://localhost:3000', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Wait for dashboard to initialize
    console.log('⏳ Waiting for dashboard initialization...');
    await page.waitForTimeout(10000);
    
    // Try to check if dashboard object exists
    const dashboardExists = await page.evaluate(() => {
      return typeof window.dashboard !== 'undefined';
    });
    
    console.log(`\n📊 Dashboard object exists: ${dashboardExists}`);
    
    // Get any specific errors
    const tradingViewErrors = consoleMessages.filter(msg => 
      msg.text.includes('TradingView') || 
      msg.text.includes('schema') || 
      msg.text.includes('unknown data type')
    );
    
    const websocketErrors = consoleMessages.filter(msg => 
      msg.text.includes('WebSocket') || 
      msg.text.includes('1006')
    );
    
    console.log(`\n📈 TradingView related messages: ${tradingViewErrors.length}`);
    tradingViewErrors.forEach(msg => console.log(`  - [${msg.type}] ${msg.text}`));
    
    console.log(`\n🔌 WebSocket related messages: ${websocketErrors.length}`);
    websocketErrors.forEach(msg => console.log(`  - [${msg.type}] ${msg.text}`));
    
    console.log(`\n❌ Total errors found: ${errors.length}`);
    if (errors.length > 0) {
      console.log('Error details:');
      errors.forEach((error, i) => console.log(`  ${i + 1}. ${error}`));
    }
    
    console.log(`\n📝 Total console messages: ${consoleMessages.length}`);
    
  } catch (error) {
    console.error('❌ Failed to load page:', error.message);
  } finally {
    await browser.close();
  }
}

checkBrowserConsole().catch(console.error);