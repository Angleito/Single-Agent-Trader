#!/usr/bin/env python3
"""
Network connectivity diagnostics for trading bot services.

This script checks connectivity between services and helps diagnose
WebSocket connection issues, service discovery problems, and network
configuration errors.
"""

import asyncio
import socket
import sys

import aiohttp
import websockets
from websockets.exceptions import InvalidURI, WebSocketException


class NetworkDiagnostics:
    """Network connectivity diagnostics for trading bot services."""

    def __init__(self):
        """Initialize diagnostics."""
        self.services = {
            "dashboard-backend": {
                "http": [
                    "http://localhost:8000",
                    "http://127.0.0.1:8000",
                    "http://dashboard-backend:8000",
                ],
                "ws": [
                    "ws://localhost:8000/ws",
                    "ws://127.0.0.1:8000/ws",
                    "ws://dashboard-backend:8000/ws",
                ],
                "port": 8000,
                "container": "dashboard-backend",
            },
            "bluefin-service": {
                "http": [
                    "http://localhost:8081",
                    "http://127.0.0.1:8081",
                    "http://bluefin-service:8080",
                ],
                "port": 8081,
                "container": "bluefin-service",
            },
            "mcp-memory": {
                "http": [
                    "http://localhost:8765",
                    "http://127.0.0.1:8765",
                    "http://mcp-memory:8765",
                ],
                "port": 8765,
                "container": "mcp-memory-server",
            },
            "mcp-omnisearch": {
                "http": [
                    "http://localhost:8767",
                    "http://127.0.0.1:8767",
                    "http://mcp-omnisearch:8767",
                ],
                "port": 8767,
                "container": "mcp-omnisearch-server",
            },
        }

        self.bluefin_websockets = {
            "mainnet": {
                "notification": "wss://notifications.api.sui-prod.bluefin.io",
                "dapi": "wss://dapi.api.sui-prod.bluefin.io",
            },
            "testnet": {
                "notification": "wss://notifications.api.sui-staging.bluefin.io",
                "dapi": "wss://dapi.api.sui-staging.bluefin.io",
            },
        }

    def check_port_open(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a port is open on a host."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    async def check_http_endpoint(
        self, url: str, timeout: float = 5.0
    ) -> tuple[bool, str]:
        """Check HTTP endpoint connectivity."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(url) as response:
                    return response.status in [200, 404], f"Status: {response.status}"
        except aiohttp.ClientConnectorError as e:
            return False, f"Connection error: {e!s}"
        except TimeoutError:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, f"Error: {e!s}"

    async def check_websocket_endpoint(
        self, url: str, timeout: float = 10.0
    ) -> tuple[bool, str]:
        """Check WebSocket endpoint connectivity."""
        try:
            async with asyncio.timeout(timeout):
                async with websockets.connect(
                    url, ping_interval=5, ping_timeout=3
                ) as ws:
                    # Send a test ping
                    pong = await ws.ping()
                    await asyncio.wait_for(pong, timeout=3)
                    return True, "Connected and responsive"
        except InvalidURI:
            return False, "Invalid WebSocket URI"
        except WebSocketException as e:
            return False, f"WebSocket error: {e!s}"
        except TimeoutError:
            return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, f"Error: {e!s}"

    def check_dns_resolution(self, hostname: str) -> tuple[bool, str]:
        """Check if hostname can be resolved."""
        try:
            ip = socket.gethostbyname(hostname)
            return True, f"Resolved to {ip}"
        except socket.gaierror:
            return False, "DNS resolution failed"
        except Exception as e:
            return False, f"Error: {e!s}"

    async def diagnose_service(self, service_name: str, service_config: dict) -> dict:
        """Diagnose connectivity for a single service."""
        results = {
            "name": service_name,
            "port_check": {},
            "http_check": {},
            "ws_check": {},
            "overall": "unknown",
        }

        # Check port accessibility
        if "port" in service_config:
            port = service_config["port"]
            for host in ["localhost", "127.0.0.1"]:
                is_open = self.check_port_open(host, port)
                results["port_check"][f"{host}:{port}"] = (
                    "✓ Open" if is_open else "✗ Closed"
                )

        # Check HTTP endpoints
        if "http" in service_config:
            for url in service_config["http"]:
                success, message = await self.check_http_endpoint(url)
                results["http_check"][url] = (
                    "✓ " + message if success else "✗ " + message
                )

        # Check WebSocket endpoints
        if "ws" in service_config:
            for url in service_config["ws"]:
                success, message = await self.check_websocket_endpoint(url)
                results["ws_check"][url] = "✓ " + message if success else "✗ " + message

        # Determine overall status
        any_success = (
            any("✓" in v for v in results["port_check"].values())
            or any("✓" in v for v in results["http_check"].values())
            or any("✓" in v for v in results["ws_check"].values())
        )
        results["overall"] = "✓ Available" if any_success else "✗ Unavailable"

        return results

    async def diagnose_bluefin_websockets(self, network: str = "mainnet") -> dict:
        """Diagnose Bluefin WebSocket connectivity."""
        results = {"network": network, "endpoints": {}}

        if network not in self.bluefin_websockets:
            results["error"] = f"Unknown network: {network}"
            return results

        endpoints = self.bluefin_websockets[network]

        for name, url in endpoints.items():
            # Check DNS resolution
            hostname = url.split("://")[1].split("/")[0]
            dns_ok, dns_msg = self.check_dns_resolution(hostname)

            # Check WebSocket connectivity
            ws_ok, ws_msg = await self.check_websocket_endpoint(url)

            results["endpoints"][name] = {
                "url": url,
                "dns": "✓ " + dns_msg if dns_ok else "✗ " + dns_msg,
                "websocket": "✓ " + ws_msg if ws_ok else "✗ " + ws_msg,
                "status": "✓ Healthy" if dns_ok and ws_ok else "✗ Issues detected",
            }

        return results

    def print_results(self, results: list[dict]):
        """Print diagnostic results in a formatted way."""
        print("\n" + "=" * 80)
        print("NETWORK CONNECTIVITY DIAGNOSTICS")
        print("=" * 80 + "\n")

        # Local services
        print("LOCAL SERVICES:")
        print("-" * 80)

        for result in results:
            if "endpoints" in result:  # Skip Bluefin results
                continue

            print(f"\n{result['name'].upper()} - {result['overall']}")

            if result["port_check"]:
                print("  Port checks:")
                for endpoint, status in result["port_check"].items():
                    print(f"    {endpoint}: {status}")

            if result["http_check"]:
                print("  HTTP checks:")
                for endpoint, status in result["http_check"].items():
                    print(f"    {endpoint}: {status}")

            if result["ws_check"]:
                print("  WebSocket checks:")
                for endpoint, status in result["ws_check"].items():
                    print(f"    {endpoint}: {status}")

        # Bluefin WebSockets
        bluefin_results = [r for r in results if "endpoints" in r]
        if bluefin_results:
            print("\n" + "-" * 80)
            print("BLUEFIN WEBSOCKETS:")
            print("-" * 80)

            for result in bluefin_results:
                print(f"\nNetwork: {result['network']}")

                if "error" in result:
                    print(f"  Error: {result['error']}")
                    continue

                for name, endpoint in result["endpoints"].items():
                    print(f"\n  {name.upper()} - {endpoint['status']}")
                    print(f"    URL: {endpoint['url']}")
                    print(f"    DNS: {endpoint['dns']}")
                    print(f"    WebSocket: {endpoint['websocket']}")

    def print_recommendations(self, results: list[dict]):
        """Print recommendations based on diagnostic results."""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80 + "\n")

        issues_found = False

        # Check local services
        for result in results:
            if "endpoints" in result:  # Skip Bluefin results
                continue

            if "✗" in result["overall"]:
                issues_found = True
                service = result["name"]

                print(f"• {service.upper()} is not accessible:")

                # Check if any endpoint works
                any_working = any(
                    "✓" in v for v in result["http_check"].values()
                ) or any("✓" in v for v in result["ws_check"].values())

                if not any_working:
                    print(
                        f"  - Check if {service} container is running: docker ps | grep {service}"
                    )
                    print(f"  - Check container logs: docker logs {service}")
                    print("  - Ensure Docker network 'trading-network' exists")
                    print(f"  - Try: docker-compose up -d {service}")
                else:
                    print(
                        "  - Service is partially accessible, check specific endpoints"
                    )

                print()

        # Check Bluefin WebSockets
        bluefin_results = [r for r in results if "endpoints" in r]
        for result in bluefin_results:
            if "error" not in result:
                for name, endpoint in result["endpoints"].items():
                    if "✗" in endpoint["status"]:
                        issues_found = True
                        print(f"• Bluefin {name} WebSocket issues:")

                        if "✗" in endpoint["dns"]:
                            print(
                                "  - DNS resolution failed, check internet connectivity"
                            )
                            print(
                                "  - Try: nslookup "
                                + endpoint["url"].split("://")[1].split("/")[0]
                            )

                        if "✗" in endpoint["websocket"]:
                            print("  - WebSocket connection failed")
                            print(
                                "  - Check firewall settings for outbound WSS connections"
                            )
                            print("  - Verify network allows WebSocket protocol")

                        print()

        if not issues_found:
            print("✓ All services are accessible and healthy!")
        else:
            print("\nGENERAL TROUBLESHOOTING:")
            print("-" * 40)
            print("1. Restart Docker services: docker-compose restart")
            print("2. Check Docker network: docker network inspect trading-network")
            print("3. View container logs: docker-compose logs -f")
            print("4. Ensure .env file has correct configuration")
            print("5. Check system firewall settings")
            print("6. Verify Docker daemon is running properly")

    async def run_diagnostics(self):
        """Run all diagnostics."""
        print("Starting network diagnostics...")
        print("This may take a few moments...\n")

        results = []

        # Diagnose local services
        for service_name, service_config in self.services.items():
            result = await self.diagnose_service(service_name, service_config)
            results.append(result)

        # Diagnose Bluefin WebSockets
        for network in ["mainnet", "testnet"]:
            result = await self.diagnose_bluefin_websockets(network)
            results.append(result)

        # Print results
        self.print_results(results)
        self.print_recommendations(results)

        # Return overall status
        any_issues = any(
            "✗" in r.get("overall", "")
            or any(
                "✗" in ep.get("status", "") for ep in r.get("endpoints", {}).values()
            )
            for r in results
        )

        return 1 if any_issues else 0


async def main():
    """Main entry point."""
    diagnostics = NetworkDiagnostics()
    exit_code = await diagnostics.run_diagnostics()

    print("\n" + "=" * 80)
    print("Diagnostics complete.")
    print("=" * 80 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
