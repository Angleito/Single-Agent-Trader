#!/usr/bin/env node

/**
 * Mock Data Generator for AI Trading Bot Dashboard Testing
 * 
 * Generates realistic trading data, LLM decisions, and market conditions
 * for comprehensive integration testing
 */

const fs = require('fs');
const path = require('path');

// Configuration for data generation
const CONFIG = {
    SYMBOLS: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD'],
    TIME_RANGES: {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    },
    MARKET_CONDITIONS: ['bull', 'bear', 'sideways', 'volatile'],
    LLM_ACTIONS: ['LONG', 'SHORT', 'CLOSE', 'HOLD'],
    BASE_PRICES: {
        'BTC-USD': 45000,
        'ETH-USD': 2800,
        'SOL-USD': 95,
        'AVAX-USD': 35
    }
};

// Mock data generators
class MockDataGenerator {
    constructor() {
        this.sessionId = `test_session_${Date.now()}`;
        this.tradeCounter = 0;
        this.lastPrices = { ...CONFIG.BASE_PRICES };
        this.marketCondition = 'sideways';
        this.trendDirection = 1; // 1 for up, -1 for down
    }

    // Generate realistic price movement
    generatePriceMovement(symbol, basePrice, volatility = 0.02) {
        const trend = this.trendDirection * (Math.random() * 0.001 + 0.0005);
        const noise = (Math.random() - 0.5) * volatility;
        const movement = trend + noise;
        
        const newPrice = basePrice * (1 + movement);
        this.lastPrices[symbol] = newPrice;
        
        return newPrice;
    }

    // Generate OHLCV candlestick data
    generateOHLCV(symbol, timeframe = '1m', count = 100) {
        const interval = CONFIG.TIME_RANGES[timeframe];
        const basePrice = CONFIG.BASE_PRICES[symbol];
        const data = [];
        
        let currentTime = Date.now() - (count * interval);
        let currentPrice = basePrice;
        
        for (let i = 0; i < count; i++) {
            const open = currentPrice;
            const volatility = this.getVolatilityForCondition();
            
            // Generate high and low
            const range = open * volatility;
            const high = open + (Math.random() * range);
            const low = open - (Math.random() * range);
            
            // Generate close price
            const close = this.generatePriceMovement(symbol, open, volatility * 0.5);
            currentPrice = close;
            
            // Generate volume (higher during volatility)
            const baseVolume = Math.random() * 1000 + 500;
            const volume = baseVolume * (1 + volatility * 10);
            
            data.push({
                time: Math.floor(currentTime / 1000),
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(Math.max(open, high, close).toFixed(2)),
                low: parseFloat(Math.min(open, low, close).toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: Math.floor(volume)
            });
            
            currentTime += interval;
        }
        
        return data;
    }

    // Generate technical indicators
    generateIndicators(symbol, ohlcvData) {
        const closes = ohlcvData.map(d => d.close);
        const highs = ohlcvData.map(d => d.high);
        const lows = ohlcvData.map(d => d.low);
        
        return {
            rsi: this.calculateRSI(closes),
            ma_20: this.calculateSMA(closes, 20),
            ma_50: this.calculateSMA(closes, 50),
            ma_200: this.calculateSMA(closes, 200),
            bb_upper: this.calculateBollingerBands(closes).upper,
            bb_lower: this.calculateBollingerBands(closes).lower,
            macd: this.calculateMACD(closes),
            volume_sma: this.calculateSMA(ohlcvData.map(d => d.volume), 20),
            atr: this.calculateATR(highs, lows, closes),
            stoch_k: Math.random() * 100,
            stoch_d: Math.random() * 100
        };
    }

    // Generate realistic LLM trading decision
    generateLLMDecision(symbol = null, customIndicators = null) {
        const targetSymbol = symbol || CONFIG.SYMBOLS[Math.floor(Math.random() * CONFIG.SYMBOLS.length)];
        const currentPrice = this.lastPrices[targetSymbol];
        
        // Generate indicators if not provided
        let indicators = customIndicators;
        if (!indicators) {
            const ohlcvData = this.generateOHLCV(targetSymbol, '5m', 50);
            indicators = this.generateIndicators(targetSymbol, ohlcvData);
        }
        
        // Generate decision based on indicators and market conditions
        const action = this.generateActionBasedOnIndicators(indicators);
        const sizePct = this.generatePositionSize(action, indicators);
        const rationale = this.generateRationale(action, indicators, this.marketCondition);
        
        this.tradeCounter++;
        
        return {
            type: 'llm_decision',
            timestamp: new Date().toISOString(),
            event_type: 'trading_decision',
            source: 'mock_generator',
            data: {
                action: action,
                size_pct: sizePct,
                rationale: rationale,
                symbol: targetSymbol,
                current_price: parseFloat(currentPrice.toFixed(2)),
                indicators: indicators,
                session_id: this.sessionId,
                trade_id: `mock_trade_${this.tradeCounter}`,
                market_condition: this.marketCondition,
                confidence: Math.random() * 0.4 + 0.6, // 60-100% confidence
                risk_level: this.calculateRiskLevel(indicators),
                timestamp_ms: Date.now()
            }
        };
    }

    // Generate LLM completion event
    generateLLMCompletion(requestType = 'trading_decision') {
        const responseTime = Math.floor(Math.random() * 5000) + 1000; // 1-6 seconds
        const tokens = Math.floor(Math.random() * 500) + 100;
        const cost = (tokens / 1000) * 0.03; // Rough cost estimate
        
        return {
            type: 'llm_completion',
            timestamp: new Date().toISOString(),
            event_type: 'llm_response',
            source: 'mock_generator',
            data: {
                request_id: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                session_id: this.sessionId,
                request_type: requestType,
                model: 'gpt-4',
                response_time_ms: responseTime,
                tokens_used: tokens,
                cost_estimate_usd: parseFloat(cost.toFixed(4)),
                success: Math.random() > 0.05, // 95% success rate
                timestamp_ms: Date.now()
            }
        };
    }

    // Generate position update
    generatePositionUpdate(symbol = null, action = null) {
        const targetSymbol = symbol || CONFIG.SYMBOLS[Math.floor(Math.random() * CONFIG.SYMBOLS.length)];
        const currentPrice = this.lastPrices[targetSymbol];
        const side = action || (Math.random() > 0.5 ? 'long' : 'short');
        
        const entryPrice = currentPrice * (1 + (Math.random() - 0.5) * 0.02);
        const size = Math.random() * 0.1 + 0.01; // 1-11% position size
        const unrealizedPnl = (currentPrice - entryPrice) * size * (side === 'long' ? 1 : -1);
        
        return {
            type: 'position_update',
            timestamp: new Date().toISOString(),
            source: 'mock_generator',
            data: {
                symbol: targetSymbol,
                side: side,
                size: parseFloat(size.toFixed(4)),
                entry_price: parseFloat(entryPrice.toFixed(2)),
                current_price: parseFloat(currentPrice.toFixed(2)),
                unrealized_pnl: parseFloat(unrealizedPnl.toFixed(2)),
                margin_used: parseFloat((entryPrice * size * 0.2).toFixed(2)), // 5x leverage
                leverage: 5,
                timestamp_ms: Date.now()
            }
        };
    }

    // Generate account balance update
    generateAccountUpdate() {
        const balance = 50000 + (Math.random() - 0.5) * 20000; // $30k-$70k range
        const availableBalance = balance * (0.7 + Math.random() * 0.3); // 70-100% available
        const totalPnl = (Math.random() - 0.5) * 5000; // ±$5k PnL
        
        return {
            type: 'account_update',
            timestamp: new Date().toISOString(),
            source: 'mock_generator',
            data: {
                balance: parseFloat(balance.toFixed(2)),
                available_balance: parseFloat(availableBalance.toFixed(2)),
                margin_used: parseFloat((balance - availableBalance).toFixed(2)),
                total_unrealized_pnl: parseFloat(totalPnl.toFixed(2)),
                total_realized_pnl: parseFloat((Math.random() - 0.4) * 3000),
                currency: 'USD',
                timestamp_ms: Date.now()
            }
        };
    }

    // Generate performance metrics
    generatePerformanceMetrics() {
        const totalTrades = Math.floor(Math.random() * 100) + 50;
        const winningTrades = Math.floor(totalTrades * (0.4 + Math.random() * 0.4)); // 40-80% win rate
        const losingTrades = totalTrades - winningTrades;
        
        return {
            type: 'performance_metrics',
            timestamp: new Date().toISOString(),
            source: 'mock_generator',
            data: {
                total_trades: totalTrades,
                winning_trades: winningTrades,
                losing_trades: losingTrades,
                win_rate: parseFloat(((winningTrades / totalTrades) * 100).toFixed(2)),
                total_pnl: parseFloat((Math.random() - 0.3) * 10000),
                avg_win: parseFloat((Math.random() * 200 + 100).toFixed(2)),
                avg_loss: parseFloat((-Math.random() * 150 - 50).toFixed(2)),
                profit_factor: parseFloat((1 + Math.random() * 2).toFixed(2)),
                sharpe_ratio: parseFloat((Math.random() * 2).toFixed(2)),
                max_drawdown: parseFloat((-Math.random() * 2000).toFixed(2)),
                timestamp_ms: Date.now()
            }
        };
    }

    // Generate market condition simulation
    simulateMarketCondition(condition = null, duration = 60000) {
        const targetCondition = condition || CONFIG.MARKET_CONDITIONS[Math.floor(Math.random() * CONFIG.MARKET_CONDITIONS.length)];
        this.marketCondition = targetCondition;
        
        switch (targetCondition) {
            case 'bull':
                this.trendDirection = 1;
                break;
            case 'bear':
                this.trendDirection = -1;
                break;
            case 'sideways':
                this.trendDirection = Math.random() > 0.5 ? 1 : -1;
                break;
            case 'volatile':
                this.trendDirection = Math.random() > 0.5 ? 1 : -1;
                break;
        }
        
        return {
            type: 'market_condition',
            timestamp: new Date().toISOString(),
            source: 'mock_generator',
            data: {
                condition: targetCondition,
                trend_direction: this.trendDirection,
                volatility: this.getVolatilityForCondition(),
                duration_ms: duration,
                timestamp_ms: Date.now()
            }
        };
    }

    // Helper methods for calculations
    calculateRSI(prices, period = 14) {
        if (prices.length < period) return 50; // Default neutral RSI
        
        let gains = 0;
        let losses = 0;
        
        for (let i = 1; i <= period; i++) {
            const change = prices[prices.length - i] - prices[prices.length - i - 1];
            if (change > 0) gains += change;
            else losses -= change;
        }
        
        const avgGain = gains / period;
        const avgLoss = losses / period;
        const rs = avgGain / (avgLoss || 1);
        
        return parseFloat((100 - (100 / (1 + rs))).toFixed(2));
    }

    calculateSMA(values, period) {
        if (values.length < period) return values[values.length - 1];
        
        const sum = values.slice(-period).reduce((a, b) => a + b, 0);
        return parseFloat((sum / period).toFixed(2));
    }

    calculateBollingerBands(prices, period = 20, multiplier = 2) {
        const sma = this.calculateSMA(prices, period);
        if (prices.length < period) return { upper: sma * 1.02, lower: sma * 0.98 };
        
        const squared_diffs = prices.slice(-period).map(price => Math.pow(price - sma, 2));
        const variance = squared_diffs.reduce((a, b) => a + b, 0) / period;
        const std_dev = Math.sqrt(variance);
        
        return {
            upper: parseFloat((sma + (std_dev * multiplier)).toFixed(2)),
            lower: parseFloat((sma - (std_dev * multiplier)).toFixed(2))
        };
    }

    calculateMACD(prices, fastPeriod = 12, slowPeriod = 26) {
        const fastEMA = this.calculateEMA(prices, fastPeriod);
        const slowEMA = this.calculateEMA(prices, slowPeriod);
        return parseFloat((fastEMA - slowEMA).toFixed(4));
    }

    calculateEMA(prices, period) {
        if (prices.length === 0) return 0;
        if (prices.length === 1) return prices[0];
        
        const multiplier = 2 / (period + 1);
        let ema = prices[0];
        
        for (let i = 1; i < Math.min(prices.length, period * 2); i++) {
            ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
        }
        
        return parseFloat(ema.toFixed(2));
    }

    calculateATR(highs, lows, closes, period = 14) {
        if (highs.length < 2) return 1;
        
        const trueRanges = [];
        for (let i = 1; i < Math.min(highs.length, period + 1); i++) {
            const tr1 = highs[i] - lows[i];
            const tr2 = Math.abs(highs[i] - closes[i - 1]);
            const tr3 = Math.abs(lows[i] - closes[i - 1]);
            trueRanges.push(Math.max(tr1, tr2, tr3));
        }
        
        return parseFloat((trueRanges.reduce((a, b) => a + b, 0) / trueRanges.length).toFixed(2));
    }

    // Generate trading decision based on indicators
    generateActionBasedOnIndicators(indicators) {
        const rsi = indicators.rsi;
        const macdSignal = indicators.macd > 0 ? 1 : -1;
        const trendSignal = indicators.ma_20 > indicators.ma_50 ? 1 : -1;
        
        let bullishSignals = 0;
        let bearishSignals = 0;
        
        // RSI signals
        if (rsi < 30) bullishSignals++;
        if (rsi > 70) bearishSignals++;
        
        // MACD signals
        if (macdSignal > 0) bullishSignals++;
        else bearishSignals++;
        
        // Trend signals
        if (trendSignal > 0) bullishSignals++;
        else bearishSignals++;
        
        // Market condition influence
        if (this.marketCondition === 'bull') bullishSignals++;
        if (this.marketCondition === 'bear') bearishSignals++;
        
        const netSignal = bullishSignals - bearishSignals;
        
        if (netSignal > 1) return 'LONG';
        if (netSignal < -1) return 'SHORT';
        if (Math.random() > 0.7) return 'CLOSE'; // 30% chance to close
        return 'HOLD';
    }

    generatePositionSize(action, indicators) {
        if (action === 'HOLD' || action === 'CLOSE') return 0;
        
        const riskLevel = this.calculateRiskLevel(indicators);
        const baseSize = 0.02; // 2% base position
        const riskMultiplier = riskLevel < 0.3 ? 1.5 : riskLevel > 0.7 ? 0.5 : 1;
        
        return parseFloat((baseSize * riskMultiplier * (0.5 + Math.random())).toFixed(4));
    }

    calculateRiskLevel(indicators) {
        let riskScore = 0;
        
        // Volatility indicators
        const atr = indicators.atr || 1;
        const avgPrice = (indicators.ma_20 + indicators.ma_50) / 2;
        const volatility = atr / avgPrice;
        
        if (volatility > 0.02) riskScore += 0.3;
        if (this.marketCondition === 'volatile') riskScore += 0.3;
        if (indicators.rsi > 80 || indicators.rsi < 20) riskScore += 0.2;
        
        return Math.min(riskScore, 1.0);
    }

    generateRationale(action, indicators, marketCondition) {
        const rationales = {
            'LONG': [
                `RSI oversold at ${indicators.rsi.toFixed(1)}, expecting bullish reversal`,
                `MACD showing positive momentum with ${marketCondition} market conditions`,
                `Price above MA20 (${indicators.ma_20}) indicating upward trend`,
                `Strong support at MA50 level, good risk-reward setup`,
                `Bollinger bands expansion suggesting breakout potential`
            ],
            'SHORT': [
                `RSI overbought at ${indicators.rsi.toFixed(1)}, expecting bearish correction`,
                `MACD divergence indicating weakening momentum`,
                `Price rejection at MA200 resistance level`,
                `High volatility in ${marketCondition} market suggests downside risk`,
                `Volume declining on recent highs, potential distribution`
            ],
            'CLOSE': [
                `Taking profits at key resistance level`,
                `Risk management: position reached target profit`,
                `Market volatility increasing, reducing exposure`,
                `Technical indicators showing divergence`,
                `Rebalancing portfolio allocation`
            ],
            'HOLD': [
                `Consolidation pattern forming, waiting for clear direction`,
                `Mixed signals from technical indicators`,
                `Low volume suggesting lack of conviction`,
                `Waiting for market structure confirmation`,
                `Position sizing at maximum allocation`
            ]
        };
        
        const actionRationales = rationales[action] || rationales['HOLD'];
        return actionRationales[Math.floor(Math.random() * actionRationales.length)];
    }

    getVolatilityForCondition() {
        switch (this.marketCondition) {
            case 'bull': return 0.015;
            case 'bear': return 0.02;
            case 'sideways': return 0.01;
            case 'volatile': return 0.035;
            default: return 0.015;
        }
    }

    // Generate complete trading session data
    generateTradingSession(duration = 3600000, updateInterval = 60000) {
        const session = {
            session_id: this.sessionId,
            start_time: new Date().toISOString(),
            duration_ms: duration,
            market_condition: this.marketCondition,
            events: []
        };
        
        const eventCount = Math.floor(duration / updateInterval);
        let currentTime = Date.now();
        
        for (let i = 0; i < eventCount; i++) {
            // Generate different types of events
            const eventType = Math.random();
            
            if (eventType < 0.3) {
                // LLM decision event
                session.events.push(this.generateLLMDecision());
            } else if (eventType < 0.5) {
                // LLM completion event
                session.events.push(this.generateLLMCompletion());
            } else if (eventType < 0.7) {
                // Position update
                session.events.push(this.generatePositionUpdate());
            } else if (eventType < 0.85) {
                // Account update
                session.events.push(this.generateAccountUpdate());
            } else {
                // Performance metrics
                session.events.push(this.generatePerformanceMetrics());
            }
            
            currentTime += updateInterval;
        }
        
        return session;
    }
}

// CLI interface for generating test data
function main() {
    const generator = new MockDataGenerator();
    
    if (process.argv.length < 3) {
        console.log('Usage: node mock-data-generator.js <command> [options]');
        console.log('Commands:');
        console.log('  llm-decision [symbol] - Generate LLM trading decision');
        console.log('  ohlcv <symbol> [timeframe] [count] - Generate OHLCV data');
        console.log('  session [duration] - Generate complete trading session');
        console.log('  indicators <symbol> - Generate technical indicators');
        console.log('  stream [count] - Generate stream of events');
        console.log('  market-condition [condition] - Simulate market condition');
        return;
    }
    
    const command = process.argv[2];
    const outputDir = path.join(__dirname, 'data');
    
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    switch (command) {
        case 'llm-decision':
            const symbol = process.argv[3] || null;
            const decision = generator.generateLLMDecision(symbol);
            console.log(JSON.stringify(decision, null, 2));
            break;
            
        case 'ohlcv':
            const ohlcvSymbol = process.argv[3] || 'BTC-USD';
            const timeframe = process.argv[4] || '5m';
            const count = parseInt(process.argv[5]) || 100;
            const ohlcvData = generator.generateOHLCV(ohlcvSymbol, timeframe, count);
            console.log(JSON.stringify(ohlcvData, null, 2));
            break;
            
        case 'session':
            const duration = parseInt(process.argv[3]) || 3600000; // 1 hour default
            const session = generator.generateTradingSession(duration);
            const sessionFile = path.join(outputDir, `session_${Date.now()}.json`);
            fs.writeFileSync(sessionFile, JSON.stringify(session, null, 2));
            console.log(`Trading session saved to: ${sessionFile}`);
            break;
            
        case 'indicators':
            const indSymbol = process.argv[3] || 'BTC-USD';
            const ohlcv = generator.generateOHLCV(indSymbol, '5m', 100);
            const indicators = generator.generateIndicators(indSymbol, ohlcv);
            console.log(JSON.stringify(indicators, null, 2));
            break;
            
        case 'stream':
            const streamCount = parseInt(process.argv[3]) || 10;
            const events = [];
            for (let i = 0; i < streamCount; i++) {
                const eventType = Math.random();
                if (eventType < 0.4) {
                    events.push(generator.generateLLMDecision());
                } else if (eventType < 0.6) {
                    events.push(generator.generatePositionUpdate());
                } else if (eventType < 0.8) {
                    events.push(generator.generateAccountUpdate());
                } else {
                    events.push(generator.generatePerformanceMetrics());
                }
            }
            console.log(JSON.stringify(events, null, 2));
            break;
            
        case 'market-condition':
            const condition = process.argv[3] || null;
            const marketSim = generator.simulateMarketCondition(condition);
            console.log(JSON.stringify(marketSim, null, 2));
            break;
            
        default:
            console.log(`Unknown command: ${command}`);
            break;
    }
}

// Export for use in other modules
module.exports = MockDataGenerator;

// Run CLI if called directly
if (require.main === module) {
    main();
}