#!/usr/bin/env python3
"""
FastAPI server for AI Trading Bot Dashboard

Provides WebSocket streaming of bot logs and REST API endpoints for monitoring
the AI trading bot running in Docker container.
"""

import asyncio
import json
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

# Optional imports for fallback functionality
try:
    import psutil
except ImportError:
    psutil = None

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llm_log_parser import AlertThresholds, create_llm_log_parser

# Import TradingView data feed and LLM log parser
from tradingview_feed import generate_sample_data, tradingview_feed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.log_buffer: list[dict] = []
        self.max_buffer_size = 1000

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection and send buffered logs"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            f"WebSocket connection established. Total connections: {len(self.active_connections)}"
        )

        # Send buffered logs to new connection
        for log_entry in self.log_buffer[-50:]:  # Send last 50 entries
            try:
                await websocket.send_text(json.dumps(log_entry))
            except Exception as e:
                logger.error(f"Error sending buffered log to new connection: {e}")
                break

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(
            f"WebSocket connection closed. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        # Validate message structure before broadcasting
        try:
            # Ensure message has required fields
            if not isinstance(message, dict):
                logger.warning(f"Invalid message type for broadcast: {type(message)}")
                return
                
            # Ensure timestamp is present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
                
            # Ensure type is present
            if "type" not in message:
                message["type"] = "unknown"
                
            # Test JSON serialization
            json.dumps(message)
            
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid message for broadcast: {e}")
            # Create a safe fallback message
            message = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "message": "Invalid message format",
                "source": "dashboard-api"
            }

        # Add to buffer
        self.log_buffer.append(message)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)

        # Broadcast to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.discard(conn)


# Global connection manager instance
manager = ConnectionManager()

# Global LLM log parser
llm_parser = create_llm_log_parser(
    log_file="/app/trading-logs/llm_completions.log",
    alert_thresholds=AlertThresholds(
        max_response_time_ms=30000,
        max_cost_per_hour=10.0,
        min_success_rate=0.90,
        max_consecutive_failures=3,
    ),
)


# Background task for log streaming
class LogStreamer:
    """Streams Docker container logs in background"""

    def __init__(self, container_name: str = "ai-trading-bot"):
        self.container_name = container_name
        self.process: subprocess.Popen | None = None
        self.running = False

    async def start(self):
        """Start log streaming in background"""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting log streamer for container: {self.container_name}")

        try:
            # Start docker logs process
            self.process = subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "100", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Stream logs in background
            asyncio.create_task(self._stream_logs())

        except Exception as e:
            logger.error(f"Failed to start log streaming: {e}")
            self.running = False

    async def _stream_logs(self):
        """Stream logs from Docker container"""
        try:
            while self.running and self.process:
                line = self.process.stdout.readline()
                if not line:
                    break

                # Parse and broadcast log entry
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": line.strip(),
                    "source": "docker-logs",
                }

                # Try to parse log level from message
                if any(
                    level in line.upper()
                    for level in ["ERROR", "WARN", "INFO", "DEBUG"]
                ):
                    for level in ["ERROR", "WARN", "INFO", "DEBUG"]:
                        if level in line.upper():
                            log_entry["level"] = level
                            break

                await manager.broadcast(log_entry)

        except Exception as e:
            logger.error(f"Error in log streaming: {e}")
        finally:
            self.running = False
            if self.process:
                self.process.terminate()

    def stop(self):
        """Stop log streaming"""
        self.running = False
        if self.process:
            self.process.terminate()


# Global log streamer
log_streamer = LogStreamer()


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting FastAPI dashboard server...")

    # Try to start log streamer, but don't fail if target container doesn't exist
    try:
        # Start log streamer in background without awaiting
        asyncio.create_task(log_streamer.start())
        logger.info("Log streamer initialization scheduled")
    except Exception as e:
        logger.warning(
            f"Failed to start log streamer: {e}. Continuing without log streaming."
        )

    # Initialize TradingView feed with sample data for demo
    generate_sample_data()
    logger.info("Initialized TradingView data feed with sample data")

    # Initialize LLM log parser
    try:
        # Parse existing logs
        counts = llm_parser.parse_log_file()
        logger.info(f"Parsed existing LLM logs: {counts}")

        # Set up callback to broadcast LLM events to WebSocket clients
        def llm_event_callback(event_data):
            # Map event types to WebSocket message types
            ws_event_type = "llm_decision" if event_data.get("event_type") == "trading_decision" else "llm_event"
            
            # Extract key fields for easier frontend consumption
            formatted_event = {
                "timestamp": event_data.get("timestamp"),
                "type": ws_event_type,
                "event_type": event_data.get("event_type"),
                "source": "llm_parser",
            }
            
            # Add specific fields based on event type
            if event_data.get("event_type") == "trading_decision":
                formatted_event.update({
                    "action": event_data.get("action"),
                    "size_pct": event_data.get("size_pct"),
                    "rationale": event_data.get("rationale"),
                    "symbol": event_data.get("symbol"),
                    "current_price": event_data.get("current_price"),
                    "indicators": event_data.get("indicators", {}),
                    "session_id": event_data.get("session_id"),
                })
            elif event_data.get("event_type") == "performance_metrics":
                formatted_event.update({
                    "total_completions": event_data.get("total_completions"),
                    "avg_response_time_ms": event_data.get("avg_response_time_ms"),
                    "total_cost_estimate_usd": event_data.get("total_cost_estimate_usd"),
                })
            
            # Always include full data for detailed views
            formatted_event["data"] = event_data
            
            asyncio.create_task(manager.broadcast(formatted_event))

        llm_parser.add_callback(llm_event_callback)

        # Start real-time monitoring
        llm_parser.start_real_time_monitoring(poll_interval=1.0)
        logger.info("Started LLM log real-time monitoring")

    except Exception as e:
        logger.warning(
            f"Failed to initialize LLM log parser: {e}. Continuing without LLM monitoring."
        )

    yield

    # Shutdown
    logger.info("Shutting down FastAPI dashboard server...")
    log_streamer.stop()

    # Stop LLM log monitoring
    try:
        llm_parser.stop_real_time_monitoring()
        logger.info("Stopped LLM log monitoring")
    except Exception as e:
        logger.warning(f"Error stopping LLM log monitoring: {e}")


# Create FastAPI app
app = FastAPI(
    title="AI Trading Bot Dashboard",
    description="WebSocket and REST API for monitoring AI trading bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoint for real-time log streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")

            # Parse and validate client message
            try:
                parsed_data = json.loads(data) if data.strip().startswith('{') else {"raw": data}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from client: {e}")
                parsed_data = {"raw": data, "error": "invalid_json"}

            # Echo back or handle client commands with proper error handling
            try:
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "echo",
                    "message": f"Server received: {data}",
                    "parsed": parsed_data
                }
                await websocket.send_text(json.dumps(response))
            except Exception as e:
                logger.error(f"Failed to send WebSocket response: {e}")
                # Try to send a simple error message
                try:
                    error_response = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "error",
                        "message": "Failed to process message"
                    }
                    await websocket.send_text(json.dumps(error_response))
                except Exception:
                    # If we can't send error response, break the connection
                    break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# REST API Endpoints


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "AI Trading Bot Dashboard",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "running",
    }


@app.get("/status")
async def get_status():
    """Get bot health check and status information"""
    try:
        # Check if Docker container is running with error handling
        container_status = "unknown"
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "name=ai-trading-bot",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                status_text = result.stdout.strip()
                if "Up" in status_text:
                    container_status = "running"
                elif "Exited" in status_text:
                    container_status = "stopped"
            else:
                container_status = "not_found"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Failed to check container status: {e}")
            container_status = "unavailable"

        # Get basic system info with graceful fallback
        system_uptime = "unknown"
        try:
            uptime_result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=3)
            if uptime_result.returncode == 0:
                system_uptime = uptime_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Failed to get system uptime: {e}")
            # Provide a simple alternative if psutil is available
            if psutil:
                try:
                    boot_time = datetime.fromtimestamp(psutil.boot_time())
                    uptime_seconds = (datetime.now() - boot_time).total_seconds()
                    hours = int(uptime_seconds // 3600)
                    minutes = int((uptime_seconds % 3600) // 60)
                    system_uptime = f"up {hours}:{minutes:02d}"
                except Exception:
                    system_uptime = "unknown"

        return {
            "timestamp": datetime.now().isoformat(),
            "bot_status": container_status,
            "system_uptime": system_uptime,
            "websocket_connections": len(manager.active_connections),
            "log_buffer_size": len(manager.log_buffer),
            "log_streamer_running": log_streamer.running,
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        # Return a minimal but functional response instead of raising
        return {
            "timestamp": datetime.now().isoformat(),
            "bot_status": "error",
            "system_uptime": "unknown",
            "websocket_connections": len(manager.active_connections),
            "log_buffer_size": len(manager.log_buffer),
            "log_streamer_running": log_streamer.running,
            "error": str(e)
        }


@app.get("/trading-data")
async def get_trading_data():
    """Get current trading information and metrics"""
    try:
        # Try to get container stats with graceful fallback
        cpu_usage = "N/A"
        memory_usage = "N/A"
        
        try:
            stats_result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "ai-trading-bot",
                    "--no-stream",
                    "--format",
                    "table {{.CPUPerc}}\t{{.MemUsage}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if stats_result.returncode == 0:
                lines = stats_result.stdout.strip().split("\n")
                if len(lines) > 1:  # Skip header
                    stats = lines[1].split("\t")
                    if len(stats) >= 2:
                        cpu_usage = stats[0]
                        memory_usage = stats[1]
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"Failed to get container stats: {e}")
            # Try to get system stats as fallback
            if psutil:
                try:
                    cpu_usage = f"{psutil.cpu_percent(interval=0.1):.1f}%"
                    memory = psutil.virtual_memory()
                    memory_usage = f"{memory.used / (1024**3):.1f}GiB / {memory.total / (1024**3):.1f}GiB"
                except Exception:
                    pass  # Keep N/A values

        # Mock trading data - in real implementation, this would come from the bot
        trading_data = {
            "timestamp": datetime.now().isoformat(),
            "account": {
                "balance": "10000.00",
                "currency": "USD",
                "available_balance": "8500.00",
                "unrealized_pnl": "150.00",
            },
            "current_position": {
                "symbol": "BTC-USD",
                "side": "long",
                "size": "0.05",
                "entry_price": "65000.00",
                "current_price": "65500.00",
                "unrealized_pnl": "25.00",
            },
            "recent_trades": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": "BTC-USD",
                    "side": "buy",
                    "quantity": "0.05",
                    "price": "65000.00",
                    "status": "filled",
                }
            ],
            "performance": {
                "total_trades": 42,
                "winning_trades": 28,
                "losing_trades": 14,
                "win_rate": "66.67%",
                "total_pnl": "1250.00",
            },
            "system": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "last_update": datetime.now().isoformat(),
            },
        }

        return trading_data

    except Exception as e:
        logger.error(f"Error getting trading data: {e}")
        # Return mock data even on error to keep dashboard functional
        return {
            "timestamp": datetime.now().isoformat(),
            "account": {
                "balance": "0.00",
                "currency": "USD",
                "available_balance": "0.00",
                "unrealized_pnl": "0.00",
            },
            "current_position": {
                "symbol": "BTC-USD",
                "side": "flat",
                "size": "0.00",
                "entry_price": "0.00",
                "current_price": "0.00",
                "unrealized_pnl": "0.00",
            },
            "recent_trades": [],
            "performance": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": "0.00%",
                "total_pnl": "0.00",
            },
            "system": {
                "cpu_usage": "N/A",
                "memory_usage": "N/A",
                "last_update": datetime.now().isoformat(),
            },
            "error": str(e)
        }


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent logs from buffer"""
    try:
        logs = manager.log_buffer[-limit:] if manager.log_buffer else []
        return {
            "timestamp": datetime.now().isoformat(),
            "total_logs": len(manager.log_buffer),
            "returned_logs": len(logs),
            "logs": logs,
        }
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting logs: {str(e)}"
        ) from e


# LLM Monitoring Endpoints


@app.get("/llm/status")
async def get_llm_status():
    """Get LLM completion monitoring status and summary"""
    try:
        from datetime import timedelta

        # Gracefully handle missing LLM parser or log files
        if not llm_parser:
            logger.warning("LLM parser not available")
            return {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": False,
                "log_file": "not_available",
                "total_parsed": {
                    "requests": 0,
                    "responses": 0,
                    "decisions": 0,
                    "alerts": 0,
                    "performance_metrics": 0,
                },
                "metrics": {
                    "1_hour": {},
                    "24_hours": {},
                    "all_time": {},
                },
                "active_alerts": 0,
                "recent_activity": {
                    "decisions_last_hour": 0,
                    "decision_distribution": {},
                    "last_decision": None,
                },
                "error": "LLM monitoring not available"
            }

        # Get aggregated metrics for different time windows with error handling
        try:
            metrics_1h = llm_parser.get_aggregated_metrics(timedelta(hours=1))
            metrics_24h = llm_parser.get_aggregated_metrics(timedelta(hours=24))
            metrics_all = llm_parser.get_aggregated_metrics()
        except Exception as e:
            logger.warning(f"Failed to get aggregated metrics: {e}")
            metrics_1h = metrics_24h = metrics_all = {}

        # Calculate derived metrics from available data with safety checks
        try:
            total_decisions = len(llm_parser.decisions) if hasattr(llm_parser, 'decisions') else 0
            recent_decisions = []
            if hasattr(llm_parser, 'decisions') and llm_parser.decisions:
                recent_decisions = [
                    d for d in llm_parser.decisions 
                    if d.timestamp >= datetime.now() - timedelta(hours=1)
                ]
            
            # Calculate decision distribution
            decision_distribution = {}
            if hasattr(llm_parser, 'decisions') and llm_parser.decisions:
                for decision in llm_parser.decisions:
                    action = getattr(decision, 'action', 'unknown')
                    decision_distribution[action] = decision_distribution.get(action, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to calculate decision metrics: {e}")
            total_decisions = 0
            recent_decisions = []
            decision_distribution = {}

        # Get last decision safely
        last_decision = None
        try:
            if hasattr(llm_parser, 'decisions') and llm_parser.decisions:
                last_decision = llm_parser.decisions[-1].to_dict()
        except Exception as e:
            logger.warning(f"Failed to get last decision: {e}")

        # Get active alerts safely
        active_alerts_count = 0
        try:
            if hasattr(llm_parser, 'alerts') and llm_parser.alerts:
                active_alerts_count = len([
                    a for a in llm_parser.alerts
                    if a.timestamp >= datetime.now() - timedelta(hours=1)
                ])
        except Exception as e:
            logger.warning(f"Failed to count active alerts: {e}")

        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": True,
            "log_file": str(getattr(llm_parser, 'log_file', 'unknown')),
            "total_parsed": {
                "requests": len(getattr(llm_parser, 'requests', [])),
                "responses": len(getattr(llm_parser, 'responses', [])),
                "decisions": len(getattr(llm_parser, 'decisions', [])),
                "alerts": len(getattr(llm_parser, 'alerts', [])),
                "performance_metrics": len(getattr(llm_parser, 'metrics', [])),
            },
            "metrics": {
                "1_hour": metrics_1h,
                "24_hours": metrics_24h,
                "all_time": metrics_all,
            },
            "active_alerts": active_alerts_count,
            "recent_activity": {
                "decisions_last_hour": len(recent_decisions),
                "decision_distribution": decision_distribution,
                "last_decision": last_decision,
            },
        }

    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        # Return a functional response instead of raising an exception
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": False,
            "log_file": "error",
            "total_parsed": {
                "requests": 0,
                "responses": 0,
                "decisions": 0,
                "alerts": 0,
                "performance_metrics": 0,
            },
            "metrics": {
                "1_hour": {},
                "24_hours": {},
                "all_time": {},
            },
            "active_alerts": 0,
            "recent_activity": {
                "decisions_last_hour": 0,
                "decision_distribution": {},
                "last_decision": None,
            },
            "error": str(e)
        }


@app.get("/llm/metrics")
async def get_llm_metrics(
    time_window: str | None = Query(
        None, description="Time window: 1h, 24h, 7d, or all"
    )
):
    """Get detailed LLM performance metrics"""
    try:
        from datetime import timedelta

        # Parse time window
        window_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "all": None,
        }

        time_delta = window_map.get(time_window, timedelta(hours=24))
        
        # Get metrics with error handling
        metrics = {}
        if llm_parser:
            try:
                metrics = llm_parser.get_aggregated_metrics(time_delta)
            except Exception as e:
                logger.warning(f"Failed to get aggregated metrics: {e}")
                metrics = {
                    "total_requests": 0,
                    "total_responses": 0,
                    "success_rate": 0.0,
                    "avg_response_time_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "error": str(e)
                }
        else:
            metrics = {
                "total_requests": 0,
                "total_responses": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "total_cost_usd": 0.0,
                "error": "LLM parser not available"
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "time_window": time_window or "24h",
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Error getting LLM metrics: {e}")
        # Return empty metrics instead of raising
        return {
            "timestamp": datetime.now().isoformat(),
            "time_window": time_window or "24h",
            "metrics": {
                "total_requests": 0,
                "total_responses": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "total_cost_usd": 0.0,
                "error": str(e)
            },
        }


@app.get("/llm/activity")
async def get_llm_activity(limit: int = Query(50, ge=1, le=500)):
    """Get recent LLM activity across all event types"""
    try:
        activity = []
        if llm_parser:
            try:
                activity = llm_parser.get_recent_activity(limit)
            except Exception as e:
                logger.warning(f"Failed to get recent activity: {e}")
                # Return empty activity list instead of failing
                activity = []
        else:
            logger.warning("LLM parser not available for activity")

        return {
            "timestamp": datetime.now().isoformat(),
            "limit": limit,
            "total_events": len(activity),
            "activity": activity,
        }

    except Exception as e:
        logger.error(f"Error getting LLM activity: {e}")
        # Return empty activity instead of raising
        return {
            "timestamp": datetime.now().isoformat(),
            "limit": limit,
            "total_events": 0,
            "activity": [],
            "error": str(e)
        }


@app.get("/llm/decisions")
async def get_llm_decisions(
    limit: int = Query(50, ge=1, le=500),
    action_filter: str | None = Query(None, description="Filter by action: LONG, SHORT, CLOSE, HOLD"),
):
    """Get recent LLM trading decisions"""
    try:
        decisions = llm_parser.decisions
        
        # Apply action filter if specified
        if action_filter:
            decisions = [d for d in decisions if d.action == action_filter.upper()]
        
        # Get most recent decisions
        recent_decisions = decisions[-limit:] if len(decisions) > limit else decisions
        recent_decisions.reverse()  # Most recent first
        
        # Calculate statistics
        action_counts = {}
        for decision in decisions:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_decisions": len(decisions),
            "returned_decisions": len(recent_decisions),
            "action_distribution": action_counts,
            "decisions": [d.to_dict() for d in recent_decisions],
        }
    
    except Exception as e:
        logger.error(f"Error getting LLM decisions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM decisions: {str(e)}"
        ) from e


@app.get("/llm/alerts")
async def get_llm_alerts(
    active_only: bool = Query(False, description="Return only recent alerts"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get LLM monitoring alerts"""
    try:
        from datetime import timedelta

        alerts = llm_parser.alerts

        if active_only:
            # Only alerts from last hour
            cutoff = datetime.now() - timedelta(hours=1)
            alerts = [a for a in alerts if a.timestamp >= cutoff]

        # Apply limit
        alerts = alerts[-limit:] if len(alerts) > limit else alerts

        return {
            "timestamp": datetime.now().isoformat(),
            "active_only": active_only,
            "total_alerts": len(alerts),
            "alerts": [alert.to_dict() for alert in alerts],
        }

    except Exception as e:
        logger.error(f"Error getting LLM alerts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM alerts: {str(e)}"
        ) from e


@app.get("/llm/sessions")
async def get_llm_sessions():
    """Get LLM session information and statistics"""
    try:
        from collections import defaultdict

        # Group by session ID
        sessions = defaultdict(
            lambda: {"requests": [], "responses": [], "decisions": []}
        )

        for req in llm_parser.requests:
            sessions[req.session_id]["requests"].append(req.to_dict())

        for resp in llm_parser.responses:
            sessions[resp.session_id]["responses"].append(resp.to_dict())

        for decision in llm_parser.decisions:
            sessions[decision.session_id]["decisions"].append(decision.to_dict())

        # Calculate session stats
        session_stats = []
        for session_id, data in sessions.items():
            requests = data["requests"]
            responses = data["responses"]
            decisions = data["decisions"]

            if requests:
                start_time = min(req["timestamp"] for req in requests)
                end_time = max(req["timestamp"] for req in requests)

                session_stats.append(
                    {
                        "session_id": session_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "total_requests": len(requests),
                        "total_responses": len(responses),
                        "total_decisions": len(decisions),
                        "success_rate": (
                            sum(1 for r in responses if r["success"]) / len(responses)
                            if responses
                            else 0
                        ),
                        "avg_response_time_ms": (
                            sum(r["response_time_ms"] for r in responses)
                            / len(responses)
                            if responses
                            else 0
                        ),
                    }
                )

        # Sort by start time (most recent first)
        session_stats.sort(key=lambda x: x["start_time"], reverse=True)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_sessions": len(session_stats),
            "sessions": session_stats,
        }

    except Exception as e:
        logger.error(f"Error getting LLM sessions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM sessions: {str(e)}"
        ) from e


@app.get("/llm/cost-analysis")
async def get_llm_cost_analysis():
    """Get detailed cost analysis and projections"""
    try:
        from collections import defaultdict

        # Calculate costs by time period
        costs_by_hour = defaultdict(float)
        costs_by_day = defaultdict(float)
        costs_by_model = defaultdict(float)

        for response in llm_parser.responses:
            if response.cost_estimate_usd > 0:
                hour_key = response.timestamp.strftime("%Y-%m-%d-%H")
                day_key = response.timestamp.strftime("%Y-%m-%d")

                costs_by_hour[hour_key] += response.cost_estimate_usd
                costs_by_day[day_key] += response.cost_estimate_usd

        # Get model info from requests
        for request in llm_parser.requests:
            # Find matching response
            matching_responses = [
                r for r in llm_parser.responses if r.request_id == request.request_id
            ]
            if matching_responses:
                cost = matching_responses[0].cost_estimate_usd
                costs_by_model[request.model] += cost

        # Calculate current hour and day costs
        now = datetime.now()
        current_hour_key = now.strftime("%Y-%m-%d-%H")
        current_day_key = now.strftime("%Y-%m-%d")

        # Project daily and monthly costs
        hourly_costs = list(costs_by_hour.values())[-24:]  # Last 24 hours
        avg_hourly_cost = sum(hourly_costs) / len(hourly_costs) if hourly_costs else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "current_hour_cost": costs_by_hour.get(current_hour_key, 0),
            "current_day_cost": costs_by_day.get(current_day_key, 0),
            "total_cost": sum(r.cost_estimate_usd for r in llm_parser.responses),
            "avg_hourly_cost": avg_hourly_cost,
            "projected_daily_cost": avg_hourly_cost * 24,
            "projected_monthly_cost": avg_hourly_cost * 24 * 30,
            "costs_by_model": dict(costs_by_model),
            "hourly_breakdown": dict(
                list(costs_by_hour.items())[-24:]
            ),  # Last 24 hours
            "daily_breakdown": dict(list(costs_by_day.items())[-7:]),  # Last 7 days
        }

    except Exception as e:
        logger.error(f"Error getting LLM cost analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting LLM cost analysis: {str(e)}"
        ) from e


@app.get("/llm/export")
async def export_llm_data(
    format: str = Query("json", description="Export format: json")
):
    """Export all LLM data for analysis"""
    try:
        if format.lower() != "json":
            raise HTTPException(
                status_code=400, detail="Only JSON format is currently supported"
            )

        exported_data = llm_parser.export_data(format)

        return {
            "timestamp": datetime.now().isoformat(),
            "format": format,
            "data": json.loads(exported_data),
        }

    except Exception as e:
        logger.error(f"Error exporting LLM data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error exporting LLM data: {str(e)}"
        ) from e


@app.post("/llm/alerts/configure")
async def configure_llm_alerts(thresholds: dict[str, Any]):
    """Configure LLM alert thresholds"""
    try:
        # Update alert thresholds
        if "max_response_time_ms" in thresholds:
            llm_parser.alert_thresholds.max_response_time_ms = thresholds[
                "max_response_time_ms"
            ]
        if "max_cost_per_hour" in thresholds:
            llm_parser.alert_thresholds.max_cost_per_hour = thresholds[
                "max_cost_per_hour"
            ]
        if "min_success_rate" in thresholds:
            llm_parser.alert_thresholds.min_success_rate = thresholds[
                "min_success_rate"
            ]
        if "max_consecutive_failures" in thresholds:
            llm_parser.alert_thresholds.max_consecutive_failures = thresholds[
                "max_consecutive_failures"
            ]

        return {
            "status": "success",
            "message": "Alert thresholds updated",
            "current_thresholds": {
                "max_response_time_ms": llm_parser.alert_thresholds.max_response_time_ms,
                "max_cost_per_hour": llm_parser.alert_thresholds.max_cost_per_hour,
                "min_success_rate": llm_parser.alert_thresholds.min_success_rate,
                "max_consecutive_failures": llm_parser.alert_thresholds.max_consecutive_failures,
            },
        }

    except Exception as e:
        logger.error(f"Error configuring LLM alerts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error configuring alerts: {str(e)}"
        ) from e


@app.post("/control/restart")
async def restart_bot():
    """Restart the trading bot container"""
    try:
        result = subprocess.run(
            ["docker", "restart", "ai-trading-bot"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Broadcast restart notification
            await manager.broadcast(
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Trading bot container restarted",
                    "source": "dashboard-api",
                }
            )

            return {"status": "success", "message": "Bot restarted successfully"}
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to restart bot: {result.stderr}"
            )

    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail="Timeout restarting bot") from e
    except Exception as e:
        logger.error(f"Error restarting bot: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error restarting bot: {str(e)}"
        ) from e


# Add missing /api/trading-data endpoint
@app.get("/api/trading-data")
async def get_api_trading_data():
    """API endpoint for trading data"""
    return await get_trading_data()

# Add missing /api/llm/status endpoint
@app.get("/api/llm/status")
async def get_api_llm_status():
    """API endpoint for LLM status"""
    return await get_llm_status()

# Add missing /api/llm/metrics endpoint
@app.get("/api/llm/metrics")
async def get_api_llm_metrics(time_window: str | None = Query(None)):
    """API endpoint for LLM metrics"""
    return await get_llm_metrics(time_window)

# Add missing /api/llm/activity endpoint
@app.get("/api/llm/activity")
async def get_api_llm_activity(limit: int = Query(50, ge=1, le=500)):
    """API endpoint for LLM activity"""
    return await get_llm_activity(limit)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


# TradingView UDF API Endpoints


@app.get("/udf/config")
async def udf_config():
    """TradingView UDF configuration"""
    return {
        "supports_search": True,
        "supports_group_request": False,
        "supports_marks": True,
        "supports_timescale_marks": True,
        "supports_time": True,
        "exchanges": [
            {
                "value": "COINBASE",
                "name": "Coinbase",
                "desc": "Coinbase Pro/Advanced Trade",
            }
        ],
        "symbols_types": [{"value": "crypto", "name": "Cryptocurrency"}],
        "supported_resolutions": ["1", "5", "15", "60", "240", "1D", "1W", "1M"],
    }


@app.get("/udf/symbols")
async def udf_symbols(symbol: str):
    """Get symbol information"""
    symbol_info = tradingview_feed.get_symbol_info(symbol)
    if not symbol_info:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return symbol_info


@app.get("/udf/search")
async def udf_search(query: str, limit: int = Query(10, ge=1, le=100)):
    """Search for symbols"""
    symbols = tradingview_feed.get_symbols_list()

    # Filter symbols based on query
    filtered_symbols = [
        symbol
        for symbol in symbols
        if query.upper() in symbol["symbol"].upper()
        or query.upper() in symbol["description"].upper()
    ]

    # Limit results
    return filtered_symbols[:limit]


@app.get("/udf/history")
async def udf_history(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
    countback: int | None = None,
):
    """Get historical data"""
    return tradingview_feed.get_history(symbol, resolution, from_, to, countback)


@app.get("/udf/marks")
async def udf_marks(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get AI decision marks"""
    return tradingview_feed.get_marks(symbol, from_, to, resolution)


@app.get("/udf/timescale_marks")
async def udf_timescale_marks(
    symbol: str,
    resolution: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get timescale marks"""
    return tradingview_feed.get_timescale_marks(symbol, from_, to, resolution)


@app.get("/udf/time")
async def udf_time():
    """Get server time"""
    return int(time.time())


# Additional TradingView API endpoints with LLM integration


@app.get("/tradingview/symbols")
async def get_tradingview_symbols():
    """Get all available symbols for TradingView"""
    return {
        "symbols": tradingview_feed.get_symbols_list(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tradingview/indicators/{symbol}")
async def get_indicator_data(
    symbol: str,
    indicator: str,
    from_: int = Query(..., alias="from"),
    to: int = Query(...),
):
    """Get technical indicator data"""
    data = tradingview_feed.get_indicator_values(symbol, indicator, from_, to)
    return {
        "symbol": symbol,
        "indicator": indicator,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tradingview/realtime/{symbol}")
async def get_realtime_data(symbol: str, resolution: str = "1"):
    """Get real-time bar data"""
    bar = tradingview_feed.get_real_time_bar(symbol, resolution)
    if not bar:
        raise HTTPException(status_code=404, detail="No real-time data available")
    return bar


@app.post("/tradingview/update/{symbol}")
async def update_trading_data(symbol: str, data: dict[str, Any]):
    """Update trading data from bot (for real-time updates)"""
    try:
        # Process different types of updates
        if "price_data" in data:
            # Handle OHLCV data updates
            resolution = data.get("resolution", "1")
            if "new_bar" in data:
                tradingview_feed.create_new_bar(symbol, resolution, data["new_bar"])
            elif "bar_update" in data:
                tradingview_feed.update_real_time_bar(
                    symbol, resolution, data["bar_update"]
                )

        if "ai_decision" in data:
            # Handle AI decision updates
            from tradingview_feed import convert_bot_trade_to_ai_decision

            decision = convert_bot_trade_to_ai_decision(data["ai_decision"], symbol)
            if decision:
                tradingview_feed.add_ai_decision(symbol, decision)

        if "indicator" in data:
            # Handle technical indicator updates
            from tradingview_feed import convert_indicator_data_to_technical_indicator

            indicator_name = data["indicator"].get("name", "unknown")
            indicator = convert_indicator_data_to_technical_indicator(
                data["indicator"], indicator_name, symbol
            )
            if indicator:
                tradingview_feed.add_technical_indicator(symbol, indicator)

        # Broadcast update to WebSocket clients
        await manager.broadcast(
            {
                "timestamp": datetime.now().isoformat(),
                "type": "tradingview_update",
                "symbol": symbol,
                "data": data,
            }
        )

        return {"status": "success", "message": "Data updated successfully"}

    except Exception as e:
        logger.error(f"Error updating trading data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error updating data: {str(e)}"
        ) from e


@app.get("/tradingview/summary")
async def get_tradingview_summary():
    """Get summary of all TradingView data"""
    return tradingview_feed.export_data_summary()


@app.post("/tradingview/llm-decision")
async def add_llm_decision_to_chart(decision_data: dict[str, Any]):
    """Add LLM trading decision to TradingView chart"""
    try:
        from tradingview_feed import convert_bot_trade_to_ai_decision

        symbol = decision_data.get("symbol", "BTC-USD")

        # Convert LLM decision to TradingView marker
        marker = convert_bot_trade_to_ai_decision(decision_data, symbol)

        if marker:
            tradingview_feed.add_ai_decision(symbol, marker)

            # Broadcast to WebSocket clients
            await manager.broadcast(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "tradingview_decision",
                    "symbol": symbol,
                    "decision": marker.__dict__,
                }
            )

            return {"status": "success", "message": "Decision added to chart"}
        else:
            raise HTTPException(status_code=400, detail="Invalid decision data")

    except Exception as e:
        logger.error(f"Error adding LLM decision to chart: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error adding decision: {str(e)}"
        ) from e


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import os

    # Disable reload in container environments for better stability
    is_container = os.getenv("DOCKER_ENV") or os.getenv("CONTAINER")
    enable_reload = not is_container

    logger.info(f"Starting uvicorn server with reload={enable_reload}")

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=enable_reload, log_level="info"
    )
