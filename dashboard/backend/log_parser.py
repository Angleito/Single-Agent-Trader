"""
Docker log parser module for extracting and parsing AI trading bot logs.

This module provides functionality to parse Docker logs from the ai-trading-bot container
and extract structured trading data, AI decisions, and system events.
"""

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Generator, Callable, Any
import json
import logging
from queue import Queue, Empty

# Set up logging for this module
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradeAction(Enum):
    """Trade action enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class ParsedLogEntry:
    """Base class for parsed log entries."""
    timestamp: datetime
    log_level: LogLevel
    logger_name: str
    raw_message: str
    container_name: str = "ai-trading-bot"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'log_level': self.log_level.value,
            'logger_name': self.logger_name,
            'raw_message': self.raw_message,
            'container_name': self.container_name,
            'type': self.__class__.__name__
        }


@dataclass
class AIDecisionLog(ParsedLogEntry):
    """Parsed AI trading decision log entry."""
    trade_action: TradeAction
    reasoning: str
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'trade_action': self.trade_action.value,
            'reasoning': self.reasoning,
            'confidence': self.confidence
        })
        return base_dict


@dataclass
class TradingLoopLog(ParsedLogEntry):
    """Parsed trading loop status log entry."""
    loop_number: int
    price: float
    action: TradeAction
    confidence: Optional[float] = None
    risk_status: Optional[str] = None
    symbol: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'loop_number': self.loop_number,
            'price': self.price,
            'action': self.action.value,
            'confidence': self.confidence,
            'risk_status': self.risk_status,
            'symbol': self.symbol
        })
        return base_dict


@dataclass
class SystemStatusLog(ParsedLogEntry):
    """Parsed system status log entry."""
    component: str
    status: str
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'component': self.component,
            'status': self.status,
            'details': self.details
        })
        return base_dict


@dataclass
class ErrorLog(ParsedLogEntry):
    """Parsed error log entry."""
    error_type: str
    error_message: str
    component: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'error_type': self.error_type,
            'error_message': self.error_message,
            'component': self.component
        })
        return base_dict


@dataclass
class PerformanceLog(ParsedLogEntry):
    """Parsed performance metrics log entry."""
    metric_name: str
    metric_value: float
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'unit': self.unit
        })
        return base_dict


class DockerLogParser:
    """Parser for Docker logs from AI trading bot container."""
    
    # Regex patterns for different log types
    LOG_PATTERN = re.compile(
        r'^(?P<container>\S+)\s*\|\s*'
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s*-\s*'
        r'(?P<logger>[\w\.]+)\s*-\s*'
        r'(?P<level>\w+)\s*-\s*'
        r'(?P<message>.*)'
    )
    
    # AI decision pattern
    AI_DECISION_PATTERN = re.compile(
        r'Generated trade action:\s*(?P<action>LONG|SHORT|HOLD|CLOSE)(?:\s*-\s*(?P<reasoning>.+?))?$'
    )
    
    # Trading loop pattern
    LOOP_PATTERN = re.compile(
        r'Loop\s+(?P<loop>\d+):\s*'
        r'Price=\$(?P<price>[\d\.]+)(?:\s*\|\s*'
        r'Action=(?P<action>LONG|SHORT|HOLD|CLOSE)(?:\s*\((?P<confidence>\d+)%\))?)?(?:\s*\|\s*'
        r'Risk=(?P<risk>[^|]+))?'
    )
    
    # WebSocket error pattern
    WEBSOCKET_ERROR_PATTERN = re.compile(
        r'WebSocket\s+error:\s*(?P<error>.+)'
    )
    
    # Performance metric pattern
    PERFORMANCE_PATTERN = re.compile(
        r'(?P<metric>PnL|Return|Sharpe|Drawdown|Win Rate):\s*(?P<value>[\d\.\-\+%]+)(?:\s*(?P<unit>%|\$))?'
    )
    
    def __init__(self, container_name: str = "ai-trading-bot"):
        """Initialize the Docker log parser."""
        self.container_name = container_name
        self._stop_event = threading.Event()
        self._log_queue = Queue()
        self._parsing_thread: Optional[threading.Thread] = None
        
    def parse_log_line(self, line: str) -> Optional[ParsedLogEntry]:
        """Parse a single log line into a structured log entry."""
        try:
            # First, match the basic log structure
            match = self.LOG_PATTERN.match(line.strip())
            if not match:
                logger.debug(f"Failed to match log pattern for line: {line}")
                return None
            
            # Extract basic components
            container = match.group('container')
            timestamp_str = match.group('timestamp')
            logger_name = match.group('logger')
            level_str = match.group('level')
            message = match.group('message')
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            except ValueError:
                logger.warning(f"Failed to parse timestamp: {timestamp_str}")
                timestamp = datetime.now()
            
            # Parse log level
            try:
                log_level = LogLevel(level_str)
            except ValueError:
                logger.warning(f"Unknown log level: {level_str}")
                log_level = LogLevel.INFO
            
            # Parse specific log types based on content
            parsed_entry = self._parse_specific_log_type(
                timestamp, log_level, logger_name, message, container
            )
            
            return parsed_entry
            
        except Exception as e:
            logger.error(f"Error parsing log line '{line}': {e}")
            return None
    
    def _parse_specific_log_type(
        self, 
        timestamp: datetime, 
        log_level: LogLevel, 
        logger_name: str, 
        message: str,
        container: str
    ) -> ParsedLogEntry:
        """Parse specific log types based on message content."""
        
        # Check for AI decision logs
        ai_match = self.AI_DECISION_PATTERN.search(message)
        if ai_match:
            action = TradeAction(ai_match.group('action'))
            reasoning = ai_match.group('reasoning') or ""
            return AIDecisionLog(
                timestamp=timestamp,
                log_level=log_level,
                logger_name=logger_name,
                raw_message=message,
                container_name=container,
                trade_action=action,
                reasoning=reasoning.strip()
            )
        
        # Check for trading loop logs
        loop_match = self.LOOP_PATTERN.search(message)
        if loop_match:
            loop_num = int(loop_match.group('loop'))
            price = float(loop_match.group('price'))
            action_str = loop_match.group('action')
            confidence_str = loop_match.group('confidence')
            risk_status = loop_match.group('risk')
            
            action = TradeAction(action_str) if action_str else TradeAction.HOLD
            confidence = float(confidence_str) if confidence_str else None
            
            return TradingLoopLog(
                timestamp=timestamp,
                log_level=log_level,
                logger_name=logger_name,
                raw_message=message,
                container_name=container,
                loop_number=loop_num,
                price=price,
                action=action,
                confidence=confidence,
                risk_status=risk_status.strip() if risk_status else None
            )
        
        # Check for WebSocket errors
        ws_error_match = self.WEBSOCKET_ERROR_PATTERN.search(message)
        if ws_error_match:
            error_msg = ws_error_match.group('error')
            return ErrorLog(
                timestamp=timestamp,
                log_level=log_level,
                logger_name=logger_name,
                raw_message=message,
                container_name=container,
                error_type="WebSocket Error",
                error_message=error_msg,
                component="WebSocket"
            )
        
        # Check for performance metrics
        perf_match = self.PERFORMANCE_PATTERN.search(message)
        if perf_match:
            metric_name = perf_match.group('metric')
            value_str = perf_match.group('value').replace('%', '').replace('$', '')
            unit = perf_match.group('unit')
            
            try:
                metric_value = float(value_str)
                return PerformanceLog(
                    timestamp=timestamp,
                    log_level=log_level,
                    logger_name=logger_name,
                    raw_message=message,
                    container_name=container,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    unit=unit
                )
            except ValueError:
                pass
        
        # Check for error logs
        if log_level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            component = logger_name.split('.')[-1] if '.' in logger_name else logger_name
            return ErrorLog(
                timestamp=timestamp,
                log_level=log_level,
                logger_name=logger_name,
                raw_message=message,
                container_name=container,
                error_type="System Error",
                error_message=message,
                component=component
            )
        
        # Check for system status logs
        if any(keyword in message.lower() for keyword in ['started', 'stopped', 'connected', 'disconnected', 'initialized']):
            component = logger_name.split('.')[-1] if '.' in logger_name else logger_name
            status = "running" if any(word in message.lower() for word in ['started', 'connected', 'initialized']) else "stopped"
            
            return SystemStatusLog(
                timestamp=timestamp,
                log_level=log_level,
                logger_name=logger_name,
                raw_message=message,
                container_name=container,
                component=component,
                status=status,
                details=message
            )
        
        # Default to generic parsed entry
        return ParsedLogEntry(
            timestamp=timestamp,
            log_level=log_level,
            logger_name=logger_name,
            raw_message=message,
            container_name=container
        )
    
    def get_historical_logs(self, tail_lines: int = 100) -> List[ParsedLogEntry]:
        """Get historical logs from the container."""
        try:
            # Run docker logs command to get historical logs
            cmd = ["docker", "logs", "--tail", str(tail_lines), self.container_name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Docker logs command failed: {result.stderr}")
                return []
            
            # Parse each line
            parsed_logs = []
            for line in result.stdout.splitlines():
                if line.strip():
                    parsed_entry = self.parse_log_line(line)
                    if parsed_entry:
                        parsed_logs.append(parsed_entry)
            
            # Sort by timestamp
            parsed_logs.sort(key=lambda x: x.timestamp)
            return parsed_logs
            
        except subprocess.TimeoutExpired:
            logger.error("Docker logs command timed out")
            return []
        except FileNotFoundError:
            logger.error("Docker command not found. Make sure Docker is installed and in PATH")
            return []
        except Exception as e:
            logger.error(f"Error getting historical logs: {e}")
            return []
    
    def stream_logs(self, callback: Callable[[ParsedLogEntry], None]) -> None:
        """Stream live logs from the container."""
        def _stream_worker():
            try:
                # Start docker logs in follow mode
                cmd = ["docker", "logs", "-f", "--tail", "0", self.container_name]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                logger.info(f"Started streaming logs from {self.container_name}")
                
                while not self._stop_event.is_set():
                    try:
                        # Read line with timeout
                        line = process.stdout.readline()
                        if not line:
                            if process.poll() is not None:
                                break
                            continue
                        
                        # Parse and callback
                        parsed_entry = self.parse_log_line(line)
                        if parsed_entry and callback:
                            callback(parsed_entry)
                            
                    except Exception as e:
                        logger.error(f"Error processing log line: {e}")
                        continue
                
                # Clean up process
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                
            except FileNotFoundError:
                logger.error("Docker command not found. Make sure Docker is installed and in PATH")
            except Exception as e:
                logger.error(f"Error in log streaming worker: {e}")
        
        # Start streaming in background thread
        self._parsing_thread = threading.Thread(target=_stream_worker, daemon=True)
        self._parsing_thread.start()
    
    def stop_streaming(self) -> None:
        """Stop the log streaming."""
        self._stop_event.set()
        if self._parsing_thread and self._parsing_thread.is_alive():
            self._parsing_thread.join(timeout=5)
    
    def is_container_running(self) -> bool:
        """Check if the container is running."""
        try:
            cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return self.container_name in result.stdout
            
        except Exception as e:
            logger.error(f"Error checking container status: {e}")
            return False


class LogFormatter:
    """Utility class for formatting log data for different outputs."""
    
    @staticmethod
    def format_for_websocket(log_entry: ParsedLogEntry) -> Dict[str, Any]:
        """Format log entry for WebSocket transmission."""
        data = log_entry.to_dict()
        
        # Add WebSocket-specific formatting
        data['id'] = f"{log_entry.timestamp.timestamp()}_{hash(log_entry.raw_message) % 10000}"
        data['display_time'] = log_entry.timestamp.strftime('%H:%M:%S')
        
        # Add color coding based on log level
        color_map = {
            LogLevel.DEBUG: 'gray',
            LogLevel.INFO: 'blue',
            LogLevel.WARNING: 'orange',
            LogLevel.ERROR: 'red',
            LogLevel.CRITICAL: 'darkred'
        }
        data['color'] = color_map.get(log_entry.log_level, 'black')
        
        return data
    
    @staticmethod
    def format_for_json_export(log_entries: List[ParsedLogEntry]) -> str:
        """Format log entries for JSON export."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_entries': len(log_entries),
            'entries': [entry.to_dict() for entry in log_entries]
        }
        return json.dumps(data, indent=2)
    
    @staticmethod
    def get_summary_stats(log_entries: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Get summary statistics from log entries."""
        if not log_entries:
            return {}
        
        # Count by type
        type_counts = {}
        level_counts = {}
        action_counts = {}
        
        ai_decisions = []
        trading_loops = []
        errors = []
        
        for entry in log_entries:
            # Count by type
            entry_type = entry.__class__.__name__
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
            
            # Count by level
            level_counts[entry.log_level.value] = level_counts.get(entry.log_level.value, 0) + 1
            
            # Collect specific types
            if isinstance(entry, AIDecisionLog):
                ai_decisions.append(entry)
                action_counts[entry.trade_action.value] = action_counts.get(entry.trade_action.value, 0) + 1
            elif isinstance(entry, TradingLoopLog):
                trading_loops.append(entry)
            elif isinstance(entry, ErrorLog):
                errors.append(entry)
        
        # Calculate time range
        timestamps = [entry.timestamp for entry in log_entries]
        time_range = {
            'start': min(timestamps).isoformat() if timestamps else None,
            'end': max(timestamps).isoformat() if timestamps else None,
            'duration_minutes': (max(timestamps) - min(timestamps)).total_seconds() / 60 if len(timestamps) > 1 else 0
        }
        
        return {
            'total_entries': len(log_entries),
            'time_range': time_range,
            'type_counts': type_counts,
            'level_counts': level_counts,
            'action_counts': action_counts,
            'ai_decisions_count': len(ai_decisions),
            'trading_loops_count': len(trading_loops),
            'errors_count': len(errors),
            'latest_price': trading_loops[-1].price if trading_loops else None,
            'latest_action': trading_loops[-1].action.value if trading_loops else None
        }


# Convenience functions for easy usage
def parse_docker_logs(container_name: str = "ai-trading-bot", tail_lines: int = 100) -> List[ParsedLogEntry]:
    """Parse historical Docker logs from the AI trading bot container."""
    parser = DockerLogParser(container_name)
    return parser.get_historical_logs(tail_lines)


def stream_docker_logs(
    callback: Callable[[ParsedLogEntry], None],
    container_name: str = "ai-trading-bot"
) -> DockerLogParser:
    """Stream live Docker logs from the AI trading bot container."""
    parser = DockerLogParser(container_name)
    parser.stream_logs(callback)
    return parser


def get_log_summary(container_name: str = "ai-trading-bot", tail_lines: int = 100) -> Dict[str, Any]:
    """Get a summary of recent logs from the AI trading bot container."""
    logs = parse_docker_logs(container_name, tail_lines)
    return LogFormatter.get_summary_stats(logs)


# Example usage and testing
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test parsing historical logs
    print("Testing historical log parsing...")
    parser = DockerLogParser()
    
    if parser.is_container_running():
        print(f"Container {parser.container_name} is running")
        logs = parser.get_historical_logs(50)
        print(f"Parsed {len(logs)} log entries")
        
        # Print summary
        summary = LogFormatter.get_summary_stats(logs)
        print("Summary:", json.dumps(summary, indent=2))
        
        # Print first few entries
        for i, log in enumerate(logs[:5]):
            print(f"{i+1}. {log}")
        
        # Test streaming (for 10 seconds)
        print("\nTesting live log streaming for 10 seconds...")
        def print_log(entry):
            print(f"LIVE: {entry}")
        
        parser.stream_logs(print_log)
        time.sleep(10)
        parser.stop_streaming()
        
    else:
        print(f"Container {parser.container_name} is not running")
        
        # Test with sample log lines
        print("Testing with sample log lines...")
        sample_logs = [
            "ai-trading-bot | 2025-06-12 21:44:53,633 - bot.strategy.llm_agent - INFO - Generated trade action: HOLD - Neutral RSI and flat EMAs; no clear edge, stay sidelined",
            "ai-trading-bot | 2025-06-12 21:44:53,634 - __main__ - INFO - Loop 1: Price=$0.18195 | Action=HOLD (75%) | Risk=Risk approved",
            "ai-trading-bot | 2025-06-12 21:44:48,758 - bot.data.market - ERROR - WebSocket error: authentication failure"
        ]
        
        for line in sample_logs:
            parsed = parser.parse_log_line(line)
            print(f"Parsed: {parsed}")