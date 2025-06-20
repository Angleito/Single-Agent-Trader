#!/usr/bin/env python3
"""
Performance test runner for the AI Trading Bot.

This script provides a convenient way to run performance tests with
different configurations and generate comprehensive reports.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_suite import BenchmarkSuite, run_benchmark_suite
from load_tests import LoadTestConfig, run_load_tests


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("performance_tests.log"),
        ],
    )


def save_results_to_file(results: dict[str, Any], output_file: Path):
    """Save test results to JSON file."""
    with output_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")


def print_benchmark_summary(suite: BenchmarkSuite):
    """Print benchmark suite summary."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'=' * 80}")

    summary = suite.get_summary()
    print(f"Suite: {summary['suite_name']}")
    print(f"Description: {summary['description']}")
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
    print(f"{'=' * 80}")

    # Sort results by execution time
    results = sorted(
        summary["results"], key=lambda x: x["execution_time_ms"], reverse=True
    )

    for result in results:
        print(f"\n📊 {result['name']}")
        print(f"   {result['description']}")
        print(f"   ⏱️  Total Time: {result['execution_time_ms']:.2f} ms")
        print(f"   🔄 Avg Per Iteration: {result['avg_time_per_iteration_ms']:.2f} ms")
        print(f"   📈 Iterations: {result['iterations']}")

        if result.get("throughput_per_sec"):
            print(f"   🚀 Throughput: {result['throughput_per_sec']:.2f} ops/sec")

        if result.get("memory_usage_mb"):
            print(f"   💾 Memory Usage: {result['memory_usage_mb']:.2f} MB")

        if result.get("peak_memory_mb"):
            print(f"   📊 Peak Memory: {result['peak_memory_mb']:.2f} MB")

        # Show top 3 additional metrics
        if result.get("additional_metrics"):
            metrics = list(result["additional_metrics"].items())[:3]
            for key, value in metrics:
                if isinstance(value, int | float):
                    print(f"   📝 {key}: {value:.2f}")
                else:
                    print(f"   📝 {key}: {value}")


def print_load_test_summary(results):
    """Print load test results summary."""
    print(f"\n{'=' * 80}")
    print("LOAD TEST RESULTS SUMMARY")
    print(f"{'=' * 80}")

    for result in results:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result

        print(f"\n🔥 {result_dict['test_name']}")
        print(f"   {result_dict['description']}")
        print(f"   ⏱️  Duration: {result_dict['duration_seconds']:.2f}s")
        print(
            f"   📊 Operations: {result_dict['total_operations']} "
            f"(✅ {result_dict['successful_operations']}, ❌ {result_dict['failed_operations']})"
        )
        print(f"   🚀 Throughput: {result_dict['operations_per_second']:.2f} ops/sec")
        print(f"   ⚡ Avg Response: {result_dict['avg_response_time_ms']:.2f} ms")
        print(f"   📈 P95 Response: {result_dict['p95_response_time_ms']:.2f} ms")
        print(f"   📊 P99 Response: {result_dict['p99_response_time_ms']:.2f} ms")
        print(f"   💾 Memory Usage: {result_dict['memory_usage_mb']:.2f} MB")
        print(f"   📊 Peak Memory: {result_dict['peak_memory_mb']:.2f} MB")
        print(f"   ❌ Error Rate: {result_dict['error_rate_percent']:.2f}%")

        # Show key additional metrics
        if result_dict.get("additional_metrics"):
            metrics = result_dict["additional_metrics"]
            for key in [
                "target_ops_per_sec",
                "memory_growth_mb",
                "max_load_level_tested",
            ]:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, int | float):
                        print(f"   📝 {key}: {value:.2f}")
                    else:
                        print(f"   📝 {key}: {value}")


async def run_performance_tests(args):
    """Run performance tests based on arguments."""
    results = {}

    # Run benchmark tests
    if args.benchmarks:
        print("🚀 Running benchmark suite...")
        benchmark_suite = run_benchmark_suite()
        results["benchmarks"] = benchmark_suite.get_summary()
        print_benchmark_summary(benchmark_suite)

    # Run load tests
    if args.load_tests:
        print("\n🔥 Running load tests...")
        config = LoadTestConfig(
            duration_seconds=args.duration,
            concurrent_users=args.concurrent_users,
            operations_per_second=args.ops_per_second,
        )
        load_results = await run_load_tests(config)
        results["load_tests"] = [
            r.to_dict() if hasattr(r, "to_dict") else r for r in load_results
        ]
        print_load_test_summary(load_results)

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        save_results_to_file(results, output_path)

    return results


def generate_html_report(results: dict[str, Any], output_file: Path):
    """Generate HTML performance report."""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Bot - Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .metric { background: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }
        .benchmark { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .good { color: green; }
        .warning { color: orange; }
        .critical { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Trading Bot - Performance Test Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>

    {content}
</body>
</html>
    """

    content_sections = []

    # Benchmark results
    if "benchmarks" in results:
        benchmarks = results["benchmarks"]
        content_sections.append(
            f"""
        <div class="section">
            <h2>Benchmark Results</h2>
            <p>Total Benchmarks: {benchmarks.get("total_benchmarks", 0)}</p>
            <p>Total Duration: {benchmarks.get("total_duration_seconds", 0):.2f} seconds</p>

            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Avg Time (ms)</th>
                    <th>Throughput (ops/sec)</th>
                    <th>Memory (MB)</th>
                    <th>Status</th>
                </tr>
        """
        )

        for result in benchmarks.get("results", []):
            status_class = "good"
            if result.get("avg_time_per_iteration_ms", 0) > 100:
                status_class = "warning"
            if result.get("avg_time_per_iteration_ms", 0) > 500:
                status_class = "critical"

            content_sections.append(
                f"""
                <tr>
                    <td>{result.get("name", "Unknown")}</td>
                    <td class="{status_class}">{result.get("avg_time_per_iteration_ms", 0):.2f}</td>
                    <td>{result.get("throughput_per_sec", 0):.2f}</td>
                    <td>{result.get("memory_usage_mb", 0):.2f}</td>
                    <td class="{status_class}">{"OK" if status_class == "good" else "SLOW" if status_class == "warning" else "CRITICAL"}</td>
                </tr>
            """
            )

        content_sections.append("</table></div>")

    # Load test results
    if "load_tests" in results:
        content_sections.append(
            """
        <div class="section">
            <h2>Load Test Results</h2>
        """
        )

        for result in results["load_tests"]:
            error_rate = result.get("error_rate_percent", 0)
            status_class = (
                "good"
                if error_rate < 1
                else "warning"
                if error_rate < 5
                else "critical"
            )

            content_sections.append(
                f"""
            <div class="benchmark">
                <h3>{result.get("test_name", "Unknown Test")}</h3>
                <p>{result.get("description", "")}</p>
                <div class="metric">Duration: {result.get("duration_seconds", 0):.2f}s</div>
                <div class="metric">Throughput: {result.get("operations_per_second", 0):.2f} ops/sec</div>
                <div class="metric">Avg Response Time: {result.get("avg_response_time_ms", 0):.2f} ms</div>
                <div class="metric">P95 Response Time: {result.get("p95_response_time_ms", 0):.2f} ms</div>
                <div class="metric">Memory Usage: {result.get("memory_usage_mb", 0):.2f} MB</div>
                <div class="metric {status_class}">Error Rate: {error_rate:.2f}%</div>
            </div>
            """
            )

        content_sections.append("</div>")

    # Generate final HTML
    content = "".join(content_sections)
    html_content = html_template.format(
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        content=content,
    )

    with output_file.open("w") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AI Trading Bot performance tests")

    # Test selection
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmark tests")
    parser.add_argument("--load-tests", action="store_true", help="Run load tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    # Load test configuration
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Load test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=5,
        help="Number of concurrent users for load tests (default: 5)",
    )
    parser.add_argument(
        "--ops-per-second",
        type=int,
        default=10,
        help="Target operations per second (default: 10)",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, help="Output file for results (JSON format)"
    )
    parser.add_argument("--html-report", type=str, help="Generate HTML report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Set defaults if --all is specified
    if args.all:
        args.benchmarks = True
        args.load_tests = True

    # Require at least one test type
    if not args.benchmarks and not args.load_tests:
        parser.error("Must specify --benchmarks, --load-tests, or --all")

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Run tests
        results = asyncio.run(run_performance_tests(args))

        # Generate HTML report if requested
        if args.html_report:
            html_path = Path(args.html_report)
            generate_html_report(results, html_path)

        print(f"\n{'=' * 80}")
        print("🎉 Performance testing completed successfully!")

        # Show summary statistics
        if "benchmarks" in results:
            benchmark_count = results["benchmarks"].get("total_benchmarks", 0)
            print(f"📊 Benchmarks completed: {benchmark_count}")

        if "load_tests" in results:
            load_test_count = len(results["load_tests"])
            print(f"🔥 Load tests completed: {load_test_count}")

        print(f"{'=' * 80}")

    except KeyboardInterrupt:
        print("\n❌ Performance testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
