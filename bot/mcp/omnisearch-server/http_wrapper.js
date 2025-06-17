#!/usr/bin/env node

/**
 * HTTP Wrapper for MCP OmniSearch Server
 * 
 * This wrapper provides an HTTP API interface to the MCP OmniSearch server,
 * allowing the AI Trading Bot to communicate with it via REST API calls.
 */

import express from 'express';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';
import { readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Environment variables are now loaded from docker-compose.yml env_file
// No need to manually load .env file in container

const app = express();
app.use(express.json());

const PORT = process.env.MCP_SERVER_PORT || 8767;
const MCP_SERVER_PATH = path.join(__dirname, 'dist', 'index.js');

// MCP server process management
let mcpProcess = null;
let isConnected = false;
let requestId = 0;
const pendingRequests = new Map();

// Start MCP server process
async function startMCPServer() {
    return new Promise((resolve, reject) => {
        console.log('Starting MCP OmniSearch server...');
        
        mcpProcess = spawn('node', [MCP_SERVER_PATH], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env }  // Pass all environment variables to the child process
        });

        mcpProcess.stdout.on('data', (data) => {
            const lines = data.toString().split('\n').filter(line => line.trim());
            for (const line of lines) {
                try {
                    const message = JSON.parse(line);
                    handleMCPResponse(message);
                } catch (err) {
                    console.error('Failed to parse MCP response:', line);
                }
            }
        });

        mcpProcess.stderr.on('data', (data) => {
            console.error('MCP Server Error:', data.toString());
        });

        mcpProcess.on('close', (code) => {
            console.log(`MCP server process exited with code ${code}`);
            isConnected = false;
            mcpProcess = null;
        });

        // Initialize MCP connection
        setTimeout(async () => {
            try {
                await initializeMCP();
                isConnected = true;
                console.log('MCP OmniSearch server connected successfully');
                resolve();
            } catch (err) {
                reject(err);
            }
        }, 1000);
    });
}

// Initialize MCP protocol
async function initializeMCP() {
    const initMessage = {
        jsonrpc: "2.0",
        id: ++requestId,
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05",
            capabilities: {
                tools: {}
            },
            clientInfo: {
                name: "omnisearch-http-wrapper",
                version: "1.0.0"
            }
        }
    };

    return sendMCPRequest(initMessage);
}

// Send request to MCP server
function sendMCPRequest(message) {
    return new Promise((resolve, reject) => {
        if (!mcpProcess || !mcpProcess.stdin) {
            reject(new Error('MCP server not connected'));
            return;
        }

        const messageId = message.id || ++requestId;
        message.id = messageId;

        pendingRequests.set(messageId, { resolve, reject });

        mcpProcess.stdin.write(JSON.stringify(message) + '\n');
        
        // Timeout after 30 seconds
        setTimeout(() => {
            if (pendingRequests.has(messageId)) {
                pendingRequests.delete(messageId);
                reject(new Error('Request timeout'));
            }
        }, 30000);
    });
}

// Handle MCP server responses
function handleMCPResponse(message) {
    if (message.id && pendingRequests.has(message.id)) {
        const { resolve, reject } = pendingRequests.get(message.id);
        pendingRequests.delete(message.id);

        if (message.error) {
            reject(new Error(message.error.message || 'MCP error'));
        } else {
            resolve(message.result);
        }
    }
}

// Call MCP tool
async function callMCPTool(toolName, args) {
    const message = {
        jsonrpc: "2.0",
        id: ++requestId,
        method: "tools/call",
        params: {
            name: toolName,
            arguments: args
        }
    };

    return sendMCPRequest(message);
}

// Health check endpoint
app.get('/health', async (req, res) => {
    let availableTools = [];
    
    if (isConnected) {
        try {
            // Query available tools from MCP server
            const toolsMessage = {
                jsonrpc: "2.0",
                id: ++requestId,
                method: "tools/list"
            };
            
            const toolsResult = await sendMCPRequest(toolsMessage);
            availableTools = toolsResult.tools || [];
        } catch (error) {
            console.error('Error getting tools:', error);
        }
    }
    
    res.json({
        status: isConnected ? 'healthy' : 'unhealthy',
        connected: isConnected,
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        available_tools: availableTools.map(tool => tool.name),
        total_tools: availableTools.length
    });
});

// Financial news search endpoint
app.all('/financial-news', async (req, res) => {
    try {
        const params = req.method === 'GET' ? req.query : req.body;
        const { q, limit = 5, timeframe = '24h', include_sentiment = true } = params;

        if (!q) {
            return res.status(400).json({ error: 'Query parameter "q" is required' });
        }

        // Use Tavily search through MCP
        const result = await callMCPTool('tavily_search', {
            query: `${q} financial news ${timeframe}`
        });

        // Parse the MCP result and transform to expected format
        let searchResults = [];
        try {
            // Handle MCP response format which wraps content in an array
            if (result && result.content && Array.isArray(result.content)) {
                // Extract JSON from the text content
                const textContent = result.content[0]?.text;
                if (textContent) {
                    const parsedContent = JSON.parse(textContent);
                    searchResults = Array.isArray(parsedContent) ? parsedContent : [parsedContent];
                }
            } else {
                const mcpData = typeof result === 'string' ? JSON.parse(result) : result;
                searchResults = mcpData.results || mcpData || [];
            }
        } catch (err) {
            console.error('Error parsing MCP search results:', err);
            searchResults = [];
        }

        const results = {
            results: searchResults.slice(0, limit).map(item => ({
                title: item.title || '',
                url: item.url || '',
                snippet: item.content || item.snippet || '',
                source: item.url ? new URL(item.url).hostname : '',
                relevance_score: item.score || 0.5,
                sentiment: include_sentiment ? analyzeSentiment(item.content || item.snippet || '') : null,
                mentioned_symbols: extractSymbols(item.content || item.snippet || ''),
                category: 'market_news',
                impact_level: 'medium'
            }))
        };

        res.json(results);
    } catch (error) {
        console.error('Financial news search error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Crypto sentiment endpoint
app.post('/crypto-sentiment', async (req, res) => {
    try {
        const { symbol, include_social = true, include_news = true, include_technical = true } = req.body;

        if (!symbol) {
            return res.status(400).json({ error: 'Symbol parameter is required' });
        }

        // Use Perplexity AI through MCP for sentiment analysis
        const result = await callMCPTool('perplexity_search', {
            query: `What is the current market sentiment for ${symbol} cryptocurrency? Include bullish/bearish indicators, news sentiment, and technical analysis sentiment. Provide a numerical sentiment score from -1 (very bearish) to 1 (very bullish).`
        });

        // Parse AI response and extract sentiment data
        let responseContent = '';
        try {
            // Handle MCP response format which wraps content in an array
            if (result && result.content && Array.isArray(result.content)) {
                // Extract content from the first text item
                const textContent = result.content[0]?.text;
                if (textContent) {
                    // For Perplexity, the response might be JSON-wrapped
                    try {
                        const parsedContent = JSON.parse(textContent);
                        // Extract the snippet from the first result if it's an array
                        if (Array.isArray(parsedContent) && parsedContent[0]?.snippet) {
                            responseContent = parsedContent[0].snippet;
                        } else {
                            responseContent = textContent;
                        }
                    } catch {
                        responseContent = textContent;
                    }
                } else {
                    responseContent = JSON.stringify(result.content);
                }
            } else {
                const mcpData = typeof result === 'string' ? JSON.parse(result) : result;
                responseContent = mcpData.content || mcpData.response || mcpData.answer || JSON.stringify(mcpData);
            }
        } catch (err) {
            console.error('Error parsing MCP response:', err);
            responseContent = typeof result === 'string' ? result : JSON.stringify(result);
        }
        
        const sentiment = parseSentimentFromAI(responseContent);

        res.json({
            sentiment: {
                overall: sentiment.overall,
                score: sentiment.score,
                confidence: sentiment.confidence,
                source_count: 5,
                news_sentiment: include_news ? sentiment.news_sentiment : null,
                social_sentiment: include_social ? sentiment.social_sentiment : null,
                technical_sentiment: include_technical ? sentiment.technical_sentiment : null,
                key_drivers: sentiment.key_drivers,
                risk_factors: sentiment.risk_factors
            }
        });
    } catch (error) {
        console.error('Crypto sentiment error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Market sentiment endpoint
app.post('/market-sentiment', async (req, res) => {
    try {
        const { market = 'nasdaq', include_indices = true, include_sectors = true } = req.body;

        // Use Perplexity AI for market sentiment
        const result = await callMCPTool('perplexity_search', {
            query: `What is the current ${market} market sentiment? Include overall market mood, sector performance, and key indices. Provide a sentiment score from -1 (very bearish) to 1 (very bullish).`
        });

        // Parse AI response and extract sentiment data
        let responseContent = '';
        try {
            // Handle MCP response format which wraps content in an array
            if (result && result.content && Array.isArray(result.content)) {
                // Extract content from the first text item
                const textContent = result.content[0]?.text;
                if (textContent) {
                    // For Perplexity, the response might be JSON-wrapped
                    try {
                        const parsedContent = JSON.parse(textContent);
                        // Extract the snippet from the first result if it's an array
                        if (Array.isArray(parsedContent) && parsedContent[0]?.snippet) {
                            responseContent = parsedContent[0].snippet;
                        } else {
                            responseContent = textContent;
                        }
                    } catch {
                        responseContent = textContent;
                    }
                } else {
                    responseContent = JSON.stringify(result.content);
                }
            } else {
                const mcpData = typeof result === 'string' ? JSON.parse(result) : result;
                responseContent = mcpData.content || mcpData.response || mcpData.answer || JSON.stringify(mcpData);
            }
        } catch (err) {
            console.error('Error parsing MCP response:', err);
            responseContent = typeof result === 'string' ? result : JSON.stringify(result);
        }
        
        const sentiment = parseSentimentFromAI(responseContent);

        res.json({
            sentiment: {
                overall: sentiment.overall,
                score: sentiment.score,
                confidence: sentiment.confidence,
                source_count: 10,
                news_sentiment: sentiment.news_sentiment,
                social_sentiment: sentiment.social_sentiment,
                technical_sentiment: sentiment.technical_sentiment,
                key_drivers: sentiment.key_drivers,
                risk_factors: sentiment.risk_factors
            }
        });
    } catch (error) {
        console.error('Market sentiment error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Market correlation endpoint
app.post('/market-correlation', async (req, res) => {
    try {
        const { symbol1, symbol2, timeframe = '30d', include_beta = true } = req.body;

        if (!symbol1 || !symbol2) {
            return res.status(400).json({ error: 'Both symbol1 and symbol2 are required' });
        }

        // Use AI to analyze correlation
        const result = await callMCPTool('perplexity_search', {
            query: `Analyze the correlation between ${symbol1} and ${symbol2} over the past ${timeframe}. Provide correlation coefficient, beta, and direction of correlation.`
        });

        // Parse AI response and extract correlation data
        let responseContent = '';
        try {
            // Handle MCP response format which wraps content in an array
            if (result && result.content && Array.isArray(result.content)) {
                // Extract content from the first text item
                const textContent = result.content[0]?.text;
                if (textContent) {
                    // For Perplexity, the response might be JSON-wrapped
                    try {
                        const parsedContent = JSON.parse(textContent);
                        // Extract the snippet from the first result if it's an array
                        if (Array.isArray(parsedContent) && parsedContent[0]?.snippet) {
                            responseContent = parsedContent[0].snippet;
                        } else {
                            responseContent = textContent;
                        }
                    } catch {
                        responseContent = textContent;
                    }
                } else {
                    responseContent = JSON.stringify(result.content);
                }
            } else {
                const mcpData = typeof result === 'string' ? JSON.parse(result) : result;
                responseContent = mcpData.content || mcpData.response || mcpData.answer || JSON.stringify(mcpData);
            }
        } catch (err) {
            console.error('Error parsing MCP response:', err);
            responseContent = typeof result === 'string' ? result : JSON.stringify(result);
        }
        
        const correlation = parseCorrelationFromAI(responseContent);

        res.json({
            correlation: {
                coefficient: correlation.coefficient,
                beta: include_beta ? correlation.beta : null,
                r_squared: correlation.r_squared
            }
        });
    } catch (error) {
        console.error('Market correlation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Helper functions
function analyzeSentiment(text) {
    const positiveWords = ['bullish', 'positive', 'growth', 'surge', 'rally', 'gain'];
    const negativeWords = ['bearish', 'negative', 'decline', 'fall', 'drop', 'loss'];
    
    const textLower = text.toLowerCase();
    const positiveCount = positiveWords.filter(word => textLower.includes(word)).length;
    const negativeCount = negativeWords.filter(word => textLower.includes(word)).length;
    
    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
}

function extractSymbols(text) {
    const symbols = [];
    const matches = text.match(/\b[A-Z]{2,5}\b/g);
    if (matches) {
        symbols.push(...new Set(matches));
    }
    return symbols;
}

function parseSentimentFromAI(content) {
    try {
        // Simple parsing logic - in production, use more sophisticated NLP
        let contentString = '';
        
        
        if (typeof content === 'string') {
            contentString = content;
        } else if (content === null || content === undefined) {
            contentString = '';
        } else if (typeof content === 'object') {
            contentString = JSON.stringify(content);
        } else {
            contentString = String(content);
        }
        
        if (typeof contentString !== 'string') {
            console.error('contentString is not a string:', typeof contentString, contentString);
            return {
                overall: 'neutral',
                score: 0,
                confidence: 0.5,
                news_sentiment: 0,
                social_sentiment: 0,
                technical_sentiment: 0,
                key_drivers: ['Content type error'],
                risk_factors: ['Analysis unavailable']
            };
        }
        
        const contentLower = contentString.toLowerCase();
        
        let score = 0;
        let overall = 'neutral';
        
        if (contentLower.includes('very bullish')) {
            score = 0.8;
            overall = 'bullish';
        } else if (contentLower.includes('bullish')) {
            score = 0.5;
            overall = 'bullish';
        } else if (contentLower.includes('very bearish')) {
            score = -0.8;
            overall = 'bearish';
        } else if (contentLower.includes('bearish')) {
            score = -0.5;
            overall = 'bearish';
        }
        
        return {
            overall,
            score,
            confidence: 0.7,
            news_sentiment: score * 0.8,
            social_sentiment: score * 0.9,
            technical_sentiment: score * 0.7,
            key_drivers: ['Market analysis from AI'],
            risk_factors: ['AI-based analysis']
        };
    } catch (error) {
        console.error('Error in parseSentimentFromAI:', error, 'Content:', content);
        return {
            overall: 'neutral',
            score: 0,
            confidence: 0.5,
            news_sentiment: 0,
            social_sentiment: 0,
            technical_sentiment: 0,
            key_drivers: ['Error parsing AI response'],
            risk_factors: ['Analysis unavailable']
        };
    }
}

function parseCorrelationFromAI(content) {
    try {
        // Simple parsing - extract numbers from AI response
        let contentString = '';
        
        
        if (typeof content === 'string') {
            contentString = content;
        } else if (content === null || content === undefined) {
            contentString = '';
        } else if (typeof content === 'object') {
            contentString = JSON.stringify(content);
        } else {
            contentString = String(content);
        }
        
        if (typeof contentString !== 'string') {
            console.error('contentString is not a string:', typeof contentString, contentString);
            return {
                coefficient: 0,
                beta: 1,
                r_squared: 0
            };
        }
        
        const numbers = contentString.match(/-?\d+\.?\d*/g) || [];
        
        return {
            coefficient: parseFloat(numbers[0]) || 0,
            beta: parseFloat(numbers[1]) || 1,
            r_squared: parseFloat(numbers[2]) || 0
        };
    } catch (error) {
        console.error('Error in parseCorrelationFromAI:', error, 'Content:', content);
        return {
            coefficient: 0,
            beta: 1,
            r_squared: 0
        };
    }
}

// Start server
async function start() {
    try {
        await startMCPServer();
        
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`OmniSearch HTTP wrapper listening on port ${PORT}`);
        });
    } catch (error) {
        console.error('Failed to start:', error);
        process.exit(1);
    }
}

// Handle shutdown
process.on('SIGINT', () => {
    console.log('Shutting down...');
    if (mcpProcess) {
        mcpProcess.kill();
    }
    process.exit(0);
});

start();