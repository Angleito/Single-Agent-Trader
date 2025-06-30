#!/usr/bin/env python3
"""
Demo VPS Monitoring Script for AI Trading Bot
Shows what the real monitoring would look like on deployment
"""

import random
import time
from datetime import datetime


def print_dashboard():
    """Print a demo real-time dashboard"""

    # Simulate metrics
    cpu_percent = random.uniform(15, 75)
    memory_percent = random.uniform(45, 85)
    disk_percent = random.uniform(20, 60)
    load_avg = random.uniform(0.3, 1.8)

    # Container status
    containers = [
        {
            "name": "ai-trading-bot",
            "status": "running",
            "cpu": random.uniform(10, 40),
            "memory": random.uniform(200, 450),
        },
        {
            "name": "dashboard-backend",
            "status": "running",
            "cpu": random.uniform(5, 20),
            "memory": random.uniform(50, 120),
        },
        {
            "name": "bluefin-service",
            "status": "running",
            "cpu": random.uniform(8, 25),
            "memory": random.uniform(80, 200),
        },
        {
            "name": "prometheus",
            "status": "running",
            "cpu": random.uniform(2, 15),
            "memory": random.uniform(40, 100),
        },
    ]

    # Trading metrics
    balance = 10000 + random.uniform(-500, 1500)
    pnl = random.uniform(-200, 300)
    pnl_percent = (pnl / balance) * 100

    # Clear screen
    print("\033[2J\033[H", end="")

    print("=" * 80)
    print(
        f"🚀 AI Trading Bot VPS Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 80)

    # System overview
    print("\n📊 SYSTEM PERFORMANCE")
    print(f"CPU:      {cpu_percent:5.1f}% | Cores: 1")
    print(f"Memory:   {memory_percent:5.1f}% | {memory_percent * 10:.0f}MB / 1000MB")
    print(f"Disk:     {disk_percent:5.1f}% | {disk_percent * 0.2:.1f}GB / 20.0GB")
    print(f"Load Avg: {load_avg:5.2f} | 1.20 | 1.15")
    print(
        f"Network:  ↑{random.uniform(5, 50):6.1f}MB ↓{random.uniform(20, 200):6.1f}MB"
    )

    # Container status
    print("\n🐳 CONTAINER STATUS")
    print(f"{'Container':<20} {'Status':<10} {'CPU%':<8} {'Memory':<15} {'Health':<10}")
    print("-" * 70)

    for container in containers:
        status_icon = "✅" if container["status"] == "running" else "❌"
        health_icon = "💚"

        print(
            f"{container['name']:<20} {status_icon}running   {container['cpu']:6.1f}% {container['memory']:6.0f}MB/512MB  {health_icon}healthy  "
        )

    # Trading performance
    print("\n💰 TRADING PERFORMANCE")
    pnl_icon = "📈" if pnl >= 0 else "📉"
    print(f"Balance:       ${balance:,.2f}")
    print(f"P&L:           {pnl_icon} ${pnl:,.2f} ({pnl_percent:+.2f}%)")
    print(f"Positions:     {random.randint(0, 2)}")
    print(f"Pending Orders:{random.randint(0, 3)}")
    print(f"Trades Today:  {random.randint(5, 25)}")
    print(f"Win Rate:      {random.uniform(45, 75):.1f}%")
    print(f"Last Trade:    {datetime.now().strftime('%H:%M:%S')}")

    # Performance indicators
    print("\n🚨 STATUS INDICATORS")
    cpu_status = "🔥" if cpu_percent > 80 else "✅"
    mem_status = "🔥" if memory_percent > 85 else "✅"
    container_status = "✅"

    print(f"CPU Load:      {cpu_status} {'Critical' if cpu_percent > 80 else 'Normal'}")
    print(
        f"Memory Usage:  {mem_status} {'Critical' if memory_percent > 85 else 'Normal'}"
    )
    print(f"Containers:    {container_status} 4/4 running")

    # Alerts
    if cpu_percent > 75:
        print(f"\n🚨 ALERT: High CPU usage: {cpu_percent:.1f}%")
    if memory_percent > 80:
        print(f"\n🚨 ALERT: High memory usage: {memory_percent:.1f}%")

    print("\n" + "=" * 80)
    print("Press Ctrl+C to stop monitoring (DEMO MODE)")
    print("=" * 80)


def main():
    """Main demo loop"""
    print("🚀 Starting VPS Monitor Demo...")
    print("This shows what real-time monitoring looks like during deployment")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            print_dashboard()
            time.sleep(3)  # Update every 3 seconds for demo
    except KeyboardInterrupt:
        print("\n\n✅ Demo monitoring stopped")
        print("\nIn real deployment, this would run continuously on your VPS")


if __name__ == "__main__":
    main()
