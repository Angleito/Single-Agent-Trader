/**
 * TradingView Chart Integration Usage Example
 * 
 * This file demonstrates how to use the enhanced TradingView integration
 * with the AI Trading Bot Dashboard.
 */

import { TradingViewChart } from './tradingview.ts';
import type { ChartConfig, MarketData, TradeAction } from './types.ts';

/**
 * Example: Initialize TradingView Chart with Enhanced Configuration
 */
export async function initializeTradingViewExample(): Promise<TradingViewChart> {
  // Enhanced chart configuration for crypto trading
  const chartConfig: ChartConfig = {
    container_id: 'tradingview-chart',
    symbol: 'DOGE-USD',
    interval: '5', // 5-minute timeframe
    library_path: '/charting_library/',
    theme: 'dark',
    autosize: true,
    charts_storage_url: 'https://saveload.tradingview.com',
    charts_storage_api_version: '1.1',
    client_id: 'ai-trading-bot-dashboard',
    user_id: 'trader_001',
    fullscreen: false
  };

  // Initialize chart with backend connection
  const chart = new TradingViewChart(chartConfig, 'http://localhost:8000');
  
  const success = await chart.initialize();
  if (!success) {
    throw new Error('Failed to initialize TradingView chart');
  }

  console.log('TradingView chart initialized successfully');
  return chart;
}

/**
 * Example: Handle Real-time Market Data Updates
 */
export function handleMarketDataUpdates(chart: TradingViewChart): void {
  // Simulate real-time market data (normally from WebSocket)
  const marketDataExample: MarketData = {
    symbol: 'DOGE-USD',
    price: 0.08234,
    timestamp: new Date().toISOString(),
    volume: 1234567,
    change_24h: 0.0012,
    change_percent_24h: 1.48
  };

  // Update chart with new market data
  chart.updateMarketData(marketDataExample);
  
  console.log('Market data updated:', marketDataExample);
}

/**
 * Example: Add AI Decision Markers
 */
export function addAIDecisionExamples(chart: TradingViewChart): void {
  // High-confidence BUY signal
  const buyDecision: TradeAction = {
    action: 'BUY',
    confidence: 0.89,
    reasoning: 'Strong bullish divergence detected with VuManChu Cipher A crossing above zero, supported by volume spike and RSI oversold recovery',
    timestamp: new Date().toISOString(),
    price: 0.08234,
    quantity: 10000,
    leverage: 5
  };

  chart.addAIDecisionMarker(buyDecision);

  // Medium-confidence SELL signal
  const sellDecision: TradeAction = {
    action: 'SELL',
    confidence: 0.67,
    reasoning: 'Resistance level reached with bearish MACD crossover, moderate selling pressure observed',
    timestamp: new Date(Date.now() + 300000).toISOString(), // 5 minutes later
    price: 0.08456,
    quantity: 10000,
    leverage: 5
  };

  chart.addAIDecisionMarker(sellDecision);

  // Low-confidence HOLD signal
  const holdDecision: TradeAction = {
    action: 'HOLD',
    confidence: 0.45,
    reasoning: 'Mixed signals from indicators, market consolidation expected',
    timestamp: new Date(Date.now() + 600000).toISOString(), // 10 minutes later
    price: 0.08345,
    quantity: 0,
    leverage: 5
  };

  chart.addAIDecisionMarker(holdDecision);

  console.log('AI decision markers added to chart');
}

/**
 * Example: Chart Management and Customization
 */
export function demonstrateChartManagement(chart: TradingViewChart): void {
  // Change chart symbol
  setTimeout(() => {
    chart.changeSymbol('BTC-USD');
    console.log('Chart symbol changed to BTC-USD');
  }, 5000);

  // Change chart interval
  setTimeout(() => {
    chart.changeInterval('15');
    console.log('Chart interval changed to 15 minutes');
  }, 10000);

  // Add custom chart type
  setTimeout(() => {
    chart.setChartType('heikin_ashi');
    console.log('Chart type changed to Heikin Ashi');
  }, 15000);

  // Add drawing tools
  setTimeout(() => {
    chart.addDrawingTool('trend_line');
    console.log('Trend line drawing tool activated');
  }, 20000);

  // Get performance metrics
  setTimeout(() => {
    const metrics = chart.getPerformanceMetrics();
    console.log('Chart performance metrics:', metrics);
  }, 25000);
}

/**
 * Example: Advanced Chart Features
 */
export function demonstrateAdvancedFeatures(chart: TradingViewChart): void {
  // Add custom indicator (Pine Script example)
  const customIndicatorScript = `
    //@version=5
    indicator("AI Confidence", shorttitle="AI_CONF", overlay=false)
    
    confidence = input.float(0.75, "AI Confidence Level", minval=0, maxval=1, step=0.01)
    threshold_high = input.float(0.8, "High Confidence Threshold", minval=0, maxval=1, step=0.01)
    threshold_low = input.float(0.3, "Low Confidence Threshold", minval=0, maxval=1, step=0.01)
    
    confidence_color = confidence > threshold_high ? color.green : 
                      confidence < threshold_low ? color.red : color.yellow
    
    plot(confidence, "AI Confidence", color=confidence_color, linewidth=2)
    hline(threshold_high, "High Threshold", color=color.green, linestyle=hline.style_dashed)
    hline(threshold_low, "Low Threshold", color=color.red, linestyle=hline.style_dashed)
  `;

  chart.addCustomIndicator('AI Confidence Indicator', customIndicatorScript);

  // Save chart layout
  setTimeout(() => {
    const layout = chart.saveChartLayout();
    if (layout) {
      localStorage.setItem('tradingview-layout', layout);
      console.log('Chart layout saved to localStorage');
    }
  }, 5000);

  // Export chart as image
  setTimeout(async () => {
    const screenshot = await chart.exportChart('png');
    if (screenshot) {
      console.log('Chart exported as image:', screenshot.substring(0, 100) + '...');
    }
  }, 10000);
}

/**
 * Example: Error Handling and Recovery
 */
export function setupErrorHandling(chart: TradingViewChart): void {
  // Monitor chart initialization
  if (!chart.initialized) {
    console.warn('Chart not initialized, attempting retry...');
    setTimeout(() => {
      chart.retryChartInitialization();
    }, 2000);
  }

  // Periodic health check
  setInterval(() => {
    const metrics = chart.getPerformanceMetrics();
    
    // Check for performance issues
    if (metrics.queueLength > 100) {
      console.warn('Chart update queue is getting large:', metrics.queueLength);
      chart.clearCache();
    }
    
    // Check for memory leaks
    if (metrics.markerCount > 1000) {
      console.warn('Too many markers on chart, cleaning up...');
      chart.removeAllMarkers();
    }
  }, 30000); // Check every 30 seconds
}

/**
 * Example: WebSocket Integration
 */
export function integrateWithWebSocket(chart: TradingViewChart): void {
  // Example WebSocket message handlers
  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'market_data':
        chart.updateMarketData(message.data);
        break;
        
      case 'trade_action':
        chart.addAIDecisionMarker(message.data);
        break;
        
      case 'chart_command':
        handleChartCommand(chart, message.data);
        break;
        
      default:
        console.log('Unhandled WebSocket message:', message.type);
    }
  };
  
  // Example usage - connect to WebSocket and use the handler
  console.log('WebSocket integration ready with handler:', handleWebSocketMessage);

  // Chart command handler
  const handleChartCommand = (chart: TradingViewChart, command: any) => {
    switch (command.action) {
      case 'change_symbol':
        chart.changeSymbol(command.symbol);
        break;
        
      case 'change_interval':
        chart.changeInterval(command.interval);
        break;
        
      case 'clear_markers':
        chart.removeAllMarkers();
        break;
        
      case 'reset_chart':
        chart.resetChart();
        break;
        
      case 'take_screenshot':
        chart.exportChart().then(screenshot => {
          if (screenshot) {
            // Send screenshot back to server or save locally
            console.log('Chart screenshot captured');
          }
        });
        break;
        
      default:
        console.log('Unknown chart command:', command.action);
    }
  };

  // Simulate WebSocket connection (replace with actual WebSocket)
  console.log('WebSocket integration configured for TradingView chart');
}

/**
 * Complete Example: Full Integration
 */
export async function fullTradingViewIntegration(): Promise<void> {
  try {
    // Initialize chart
    const chart = await initializeTradingViewExample();
    
    // Set up error handling
    setupErrorHandling(chart);
    
    // Integrate with WebSocket
    integrateWithWebSocket(chart);
    
    // Start real-time data simulation
    setInterval(() => {
      handleMarketDataUpdates(chart);
    }, 1000); // Update every second
    
    // Add AI decision examples
    setTimeout(() => {
      addAIDecisionExamples(chart);
    }, 3000);
    
    // Demonstrate chart management
    setTimeout(() => {
      demonstrateChartManagement(chart);
    }, 5000);
    
    // Show advanced features
    setTimeout(() => {
      demonstrateAdvancedFeatures(chart);
    }, 10000);
    
    console.log('Full TradingView integration completed successfully');
    
  } catch (error) {
    console.error('Failed to complete TradingView integration:', error);
  }
}

// Usage Example:
// Call this function after DOM is loaded
// fullTradingViewIntegration();