#!/usr/bin/env python3
"""
Test script for validating LLM log parsing capabilities
"""

import json
import re

def test_log_parsing():
    """Test JSON parsing from LLM completion logs"""
    
    # Read the log file
    with open('/app/trading-logs/llm_completions.log', 'r') as f:
        lines = f.readlines()

    print(f'Total log lines: {len(lines)}')

    # Parse different log types
    llm_requests = []
    llm_responses = []
    trading_decisions = []
    performance_metrics = []

    for line in lines:
        if 'LLM_REQUEST:' in line:
            json_start = line.find('{')
            if json_start != -1:
                try:
                    json_data = json.loads(line[json_start:].strip())
                    llm_requests.append(json_data)
                except Exception as e:
                    print(f"Error parsing LLM_REQUEST: {e}")
                    
        elif 'LLM_RESPONSE:' in line:
            json_start = line.find('{')
            if json_start != -1:
                try:
                    json_data = json.loads(line[json_start:].strip())
                    llm_responses.append(json_data)
                except Exception as e:
                    print(f"Error parsing LLM_RESPONSE: {e}")
                    
        elif 'TRADING_DECISION:' in line:
            json_start = line.find('{')
            if json_start != -1:
                try:
                    json_data = json.loads(line[json_start:].strip())
                    trading_decisions.append(json_data)
                except Exception as e:
                    print(f"Error parsing TRADING_DECISION: {e}")
                    
        elif 'PERFORMANCE:' in line:
            json_start = line.find('{')
            if json_start != -1:
                try:
                    json_data = json.loads(line[json_start:].strip())
                    performance_metrics.append(json_data)
                except Exception as e:
                    print(f"Error parsing PERFORMANCE: {e}")

    print(f'LLM Requests: {len(llm_requests)}')
    print(f'LLM Responses: {len(llm_responses)}')
    print(f'Trading Decisions: {len(trading_decisions)}')
    print(f'Performance Metrics: {len(performance_metrics)}')

    # Show sample structure
    if llm_requests:
        print('\nSample LLM Request:')
        sample = llm_requests[0]
        print(f'  Model: {sample.get("model", "unknown")}')
        print(f'  Temperature: {sample.get("temperature", "unknown")}')
        print(f'  Tokens: {sample.get("prompt_length", "unknown")}')
        print(f'  Session: {sample.get("session_id", "unknown")}')
        print(f'  Request ID: {sample.get("request_id", "unknown")}')

    if trading_decisions:
        print('\nSample Trading Decision:')
        sample = trading_decisions[0]
        print(f'  Action: {sample.get("action", "unknown")}')
        print(f'  Symbol: {sample.get("symbol", "unknown")}')
        print(f'  Price: {sample.get("current_price", "unknown")}')
        print(f'  Rationale: {sample.get("rationale", "unknown")}')
        
    if llm_responses:
        print('\nSample LLM Response:')
        sample = llm_responses[0]
        print(f'  Success: {sample.get("success", "unknown")}')
        print(f'  Response Time: {sample.get("response_time_ms", "unknown")}ms')
        print(f'  Cost: ${sample.get("cost_estimate_usd", "unknown")}')
        
    return {
        'llm_requests': len(llm_requests),
        'llm_responses': len(llm_responses), 
        'trading_decisions': len(trading_decisions),
        'performance_metrics': len(performance_metrics),
        'total_lines': len(lines)
    }

if __name__ == "__main__":
    test_log_parsing()