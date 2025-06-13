#!/usr/bin/env python3
"""
Docker Log Validation for AI Trading Bot

Comprehensive log analysis tool that:
- Parses structured logs from Docker containers
- Validates log completeness and quality
- Analyzes performance metrics from logs
- Checks for errors and warnings
- Generates detailed log analysis reports

Usage:
    python scripts/validate_docker_logs.py --analyze-patterns
    python scripts/validate_docker_logs.py --container ai-trading-bot --since 1h
    python scripts/validate_docker_logs.py --export-analysis --format json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()

class DockerLogValidator:
    """Comprehensive Docker log validator and analyzer."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.docker_client = docker.from_env()
        self.log_patterns = {
            'trading_bot': {
                'startup_patterns': [
                    r'Starting AI Trading Bot',
                    r'Configuration loaded.*successfully',
                    r'Market data.*connected',
                    r'Trading engine.*initialized'
                ],
                'indicator_patterns': [
                    r'VuManChu.*Cipher.*A.*calculated',
                    r'VuManChu.*Cipher.*B.*calculated',
                    r'WaveTrend.*oscillator.*calculated',
                    r'EMA.*Ribbon.*calculated',
                    r'RSI.*MFI.*calculated',
                    r'Stochastic.*RSI.*calculated',
                    r'Schaff.*Trend.*Cycle.*calculated'
                ],
                'signal_patterns': [
                    r'Signal.*generated.*BUY',
                    r'Signal.*generated.*SELL',
                    r'Diamond.*pattern.*detected',
                    r'Yellow.*cross.*signal',
                    r'Bull.*candle.*pattern',
                    r'Bear.*candle.*pattern'
                ],
                'trading_patterns': [
                    r'Trade.*action.*BUY',
                    r'Trade.*action.*SELL',
                    r'Trade.*action.*HOLD',
                    r'Position.*opened',
                    r'Position.*closed',
                    r'Risk.*management.*triggered'
                ],
                'performance_patterns': [
                    r'Performance.*metrics.*calculated',
                    r'Portfolio.*value.*\$[\d,]+\.?\d*',
                    r'P&L.*[\+\-]?\$[\d,]+\.?\d*',
                    r'Win.*rate.*\d+\.?\d*%'
                ],
                'error_patterns': [
                    r'ERROR|CRITICAL|FATAL',
                    r'Exception.*:',
                    r'Traceback.*:',
                    r'Failed.*to.*connect',
                    r'API.*error',
                    r'Network.*timeout'
                ],
                'warning_patterns': [
                    r'WARNING|WARN',
                    r'Retrying.*attempt',
                    r'Rate.*limit.*exceeded',
                    r'Connection.*unstable'
                ]
            },
            'dashboard': {
                'startup_patterns': [
                    r'FastAPI.*application.*startup',
                    r'Application.*startup.*complete',
                    r'WebSocket.*server.*started',
                    r'API.*server.*listening'
                ],
                'api_patterns': [
                    r'GET.*\/health.*200',
                    r'GET.*\/api\/.*200',
                    r'WebSocket.*connection.*established',
                    r'Real.*time.*data.*streaming'
                ],
                'error_patterns': [
                    r'ERROR|CRITICAL|500',
                    r'Internal.*server.*error',
                    r'Database.*connection.*failed',
                    r'WebSocket.*connection.*failed'
                ]
            }
        }
        
        self.structured_log_schema = {
            'required_fields': ['timestamp', 'level', 'message'],
            'optional_fields': ['module', 'function', 'line', 'component', 'trade_id', 'symbol']
        }
    
    def validate_container_logs(
        self, 
        container_name: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        analyze_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Validate logs from a specific container.
        
        Args:
            container_name: Name of the container to analyze
            since: Start time for log analysis (e.g., '1h', '30m')
            until: End time for log analysis
            analyze_patterns: Whether to perform pattern analysis
            
        Returns:
            Validation results dictionary
        """
        console.print(f"[yellow]Validating logs for container: {container_name}[/yellow]")
        
        try:
            container = self.docker_client.containers.get(container_name)
        except docker.errors.NotFound:
            return {'error': f'Container {container_name} not found'}
        
        # Get logs with specified time range
        log_kwargs = {}
        if since:
            log_kwargs['since'] = since
        if until:
            log_kwargs['until'] = until
        
        logs = container.logs(timestamps=True, **log_kwargs).decode('utf-8', errors='ignore')
        
        # Basic log statistics
        validation_result = {
            'container': container_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_stats': self._analyze_basic_stats(logs),
            'structured_logs': self._analyze_structured_logs(logs),
            'log_levels': self._analyze_log_levels(logs),
            'time_analysis': self._analyze_log_timing(logs)
        }
        
        if analyze_patterns:
            validation_result['pattern_analysis'] = self._analyze_log_patterns(logs, container_name)
            validation_result['performance_metrics'] = self._extract_performance_metrics(logs)
            validation_result['error_analysis'] = self._analyze_errors_and_warnings(logs)
        
        return validation_result
    
    def _analyze_basic_stats(self, logs: str) -> Dict[str, Any]:
        """Analyze basic log statistics."""
        lines = logs.strip().split('\n')
        return {
            'total_lines': len(lines),
            'total_characters': len(logs),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'first_log_timestamp': self._extract_first_timestamp(logs),
            'last_log_timestamp': self._extract_last_timestamp(logs)
        }
    
    def _analyze_structured_logs(self, logs: str) -> Dict[str, Any]:
        """Analyze structured logging compliance."""
        lines = logs.strip().split('\n')
        structured_count = 0
        json_logs = []
        schema_violations = []
        
        for i, line in enumerate(lines, 1):
            # Remove timestamp prefix if present
            clean_line = re.sub(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+', '', line)
            
            if clean_line.strip().startswith('{') and clean_line.strip().endswith('}'):
                try:
                    log_entry = json.loads(clean_line.strip())
                    structured_count += 1
                    json_logs.append(log_entry)
                    
                    # Validate schema
                    missing_required = [
                        field for field in self.structured_log_schema['required_fields']
                        if field not in log_entry
                    ]
                    
                    if missing_required:
                        schema_violations.append({
                            'line': i,
                            'missing_fields': missing_required
                        })
                        
                except json.JSONDecodeError as e:
                    schema_violations.append({
                        'line': i,
                        'error': f'JSON decode error: {e}',
                        'content': clean_line[:100]
                    })
        
        return {
            'total_structured_logs': structured_count,
            'structured_percentage': (structured_count / len(lines)) * 100 if lines else 0,
            'schema_violations': schema_violations,
            'schema_compliance_rate': ((structured_count - len(schema_violations)) / structured_count * 100) 
                                    if structured_count else 0
        }
    
    def _analyze_log_levels(self, logs: str) -> Dict[str, Any]:
        """Analyze log level distribution."""
        level_patterns = {
            'DEBUG': r'\bDEBUG\b',
            'INFO': r'\bINFO\b',
            'WARNING': r'\bWARN(?:ING)?\b',
            'ERROR': r'\bERROR\b',
            'CRITICAL': r'\bCRITICAL\b',
            'FATAL': r'\bFATAL\b'
        }
        
        level_counts = {}
        total_lines = len(logs.strip().split('\n'))
        
        for level, pattern in level_patterns.items():
            matches = re.findall(pattern, logs, re.IGNORECASE)
            level_counts[level] = len(matches)
        
        return {
            'level_distribution': level_counts,
            'level_percentages': {
                level: (count / total_lines) * 100 if total_lines else 0
                for level, count in level_counts.items()
            }
        }
    
    def _analyze_log_timing(self, logs: str) -> Dict[str, Any]:
        """Analyze log timing patterns."""
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)'
        timestamps = re.findall(timestamp_pattern, logs)
        
        if not timestamps:
            return {'error': 'No timestamps found in logs'}
        
        # Parse timestamps
        parsed_timestamps = []
        for ts in timestamps:
            try:
                # Handle different timestamp formats
                if ts.endswith('Z'):
                    dt = datetime.fromisoformat(ts[:-1])
                else:
                    dt = datetime.fromisoformat(ts)
                parsed_timestamps.append(dt)
            except ValueError:
                continue
        
        if len(parsed_timestamps) < 2:
            return {'error': 'Insufficient timestamps for analysis'}
        
        # Calculate intervals
        intervals = [
            (parsed_timestamps[i] - parsed_timestamps[i-1]).total_seconds()
            for i in range(1, len(parsed_timestamps))
        ]
        
        return {
            'total_timespan_seconds': (parsed_timestamps[-1] - parsed_timestamps[0]).total_seconds(),
            'average_log_interval_seconds': sum(intervals) / len(intervals),
            'min_interval_seconds': min(intervals),
            'max_interval_seconds': max(intervals),
            'logs_per_minute': len(parsed_timestamps) / ((parsed_timestamps[-1] - parsed_timestamps[0]).total_seconds() / 60),
            'quiet_periods': [interval for interval in intervals if interval > 60]  # Gaps > 1 minute
        }
    
    def _analyze_log_patterns(self, logs: str, container_type: str) -> Dict[str, Any]:
        """Analyze specific log patterns based on container type."""
        # Determine pattern set based on container name
        if 'trading-bot' in container_type or 'ai-trading-bot' in container_type:
            patterns = self.log_patterns['trading_bot']
        elif 'dashboard' in container_type:
            patterns = self.log_patterns['dashboard']
        else:
            patterns = {}
        
        pattern_results = {}
        
        for category, pattern_list in patterns.items():
            category_results = {}
            
            for pattern in pattern_list:
                matches = re.findall(pattern, logs, re.IGNORECASE | re.MULTILINE)
                category_results[pattern] = {
                    'count': len(matches),
                    'matches': matches[:10] if matches else []  # First 10 matches for context
                }
            
            pattern_results[category] = category_results
        
        return pattern_results
    
    def _extract_performance_metrics(self, logs: str) -> Dict[str, Any]:
        """Extract performance metrics from logs."""
        metrics = {
            'indicator_calculation_times': [],
            'signal_generation_times': [],
            'api_response_times': [],
            'memory_usage_reports': [],
            'cpu_usage_reports': []
        }
        
        # Pattern to extract calculation times
        calc_time_pattern = r'calculated.*in\s+([\d.]+)\s*(?:ms|seconds?)'
        calc_times = re.findall(calc_time_pattern, logs, re.IGNORECASE)
        metrics['indicator_calculation_times'] = [float(t) for t in calc_times]
        
        # Pattern to extract API response times
        api_time_pattern = r'API.*response.*time.*?(\d+\.?\d*)\s*ms'
        api_times = re.findall(api_time_pattern, logs, re.IGNORECASE)
        metrics['api_response_times'] = [float(t) for t in api_times]
        
        # Pattern to extract memory usage
        memory_pattern = r'Memory.*usage.*?(\d+\.?\d*)\s*(?:MB|GB)'
        memory_usage = re.findall(memory_pattern, logs, re.IGNORECASE)
        metrics['memory_usage_reports'] = [float(m) for m in memory_usage]
        
        # Pattern to extract CPU usage
        cpu_pattern = r'CPU.*usage.*?(\d+\.?\d*)\s*%'
        cpu_usage = re.findall(cpu_pattern, logs, re.IGNORECASE)
        metrics['cpu_usage_reports'] = [float(c) for c in cpu_usage]
        
        return metrics
    
    def _analyze_errors_and_warnings(self, logs: str) -> Dict[str, Any]:
        """Analyze errors and warnings in detail."""
        errors = []
        warnings = []
        
        lines = logs.split('\n')
        
        for i, line in enumerate(lines):
            # Check for errors
            if re.search(r'ERROR|CRITICAL|FATAL|Exception', line, re.IGNORECASE):
                error_context = {
                    'line_number': i + 1,
                    'message': line.strip(),
                    'context_before': lines[max(0, i-2):i] if i > 0 else [],
                    'context_after': lines[i+1:min(len(lines), i+3)] if i < len(lines)-1 else []
                }
                errors.append(error_context)
            
            # Check for warnings
            elif re.search(r'WARNING|WARN', line, re.IGNORECASE):
                warning_context = {
                    'line_number': i + 1,
                    'message': line.strip(),
                    'context_before': lines[max(0, i-1):i] if i > 0 else [],
                    'context_after': lines[i+1:min(len(lines), i+2)] if i < len(lines)-1 else []
                }
                warnings.append(warning_context)
        
        # Categorize errors
        error_categories = defaultdict(list)
        for error in errors:
            if 'connection' in error['message'].lower():
                error_categories['connection'].append(error)
            elif 'api' in error['message'].lower():
                error_categories['api'].append(error)
            elif 'timeout' in error['message'].lower():
                error_categories['timeout'].append(error)
            else:
                error_categories['other'].append(error)
        
        return {
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'error_categories': dict(error_categories),
            'error_details': errors[:20],  # First 20 errors for detailed analysis
            'warning_details': warnings[:20],  # First 20 warnings for detailed analysis
            'error_frequency_per_hour': self._calculate_error_frequency(errors, logs)
        }
    
    def _calculate_error_frequency(self, errors: List[Dict], logs: str) -> float:
        """Calculate error frequency per hour."""
        if not errors:
            return 0.0
        
        # Extract timespan from logs
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)'
        timestamps = re.findall(timestamp_pattern, logs)
        
        if len(timestamps) < 2:
            return 0.0
        
        try:
            first_ts = datetime.fromisoformat(timestamps[0].rstrip('Z'))
            last_ts = datetime.fromisoformat(timestamps[-1].rstrip('Z'))
            timespan_hours = (last_ts - first_ts).total_seconds() / 3600
            
            return len(errors) / timespan_hours if timespan_hours > 0 else 0.0
        except ValueError:
            return 0.0
    
    def _extract_first_timestamp(self, logs: str) -> Optional[str]:
        """Extract first timestamp from logs."""
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)'
        match = re.search(timestamp_pattern, logs)
        return match.group(1) if match else None
    
    def _extract_last_timestamp(self, logs: str) -> Optional[str]:
        """Extract last timestamp from logs."""
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)'
        matches = re.findall(timestamp_pattern, logs)
        return matches[-1] if matches else None
    
    def validate_all_containers(self, analyze_patterns: bool = True) -> Dict[str, Any]:
        """Validate logs from all trading bot related containers."""
        console.print("[yellow]Validating logs from all containers...[/yellow]")
        
        # Find all trading bot containers
        containers = self.docker_client.containers.list(
            filters={'name': 'trading-bot'}
        ) + self.docker_client.containers.list(
            filters={'name': 'dashboard'}
        )
        
        if not containers:
            return {'error': 'No trading bot containers found'}
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            
            for container in containers:
                task = progress.add_task(f"Analyzing {container.name}...")
                results[container.name] = self.validate_container_logs(
                    container.name, 
                    analyze_patterns=analyze_patterns
                )
                progress.update(task, completed=100)
        
        # Generate summary
        results['summary'] = self._generate_validation_summary(results)
        
        return results
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        summary = {
            'containers_analyzed': len([k for k in results.keys() if k != 'summary']),
            'total_log_lines': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'structured_logs_percentage': 0,
            'containers_with_issues': []
        }
        
        container_results = {k: v for k, v in results.items() if k != 'summary'}
        
        for container_name, container_result in container_results.items():
            if 'error' in container_result:
                summary['containers_with_issues'].append({
                    'container': container_name,
                    'issue': container_result['error']
                })
                continue
            
            # Aggregate statistics
            basic_stats = container_result.get('basic_stats', {})
            summary['total_log_lines'] += basic_stats.get('total_lines', 0)
            
            error_analysis = container_result.get('error_analysis', {})
            summary['total_errors'] += error_analysis.get('total_errors', 0)
            summary['total_warnings'] += error_analysis.get('total_warnings', 0)
            
            structured_logs = container_result.get('structured_logs', {})
            struct_percentage = structured_logs.get('structured_percentage', 0)
            
            # Check for issues
            if error_analysis.get('total_errors', 0) > 10:
                summary['containers_with_issues'].append({
                    'container': container_name,
                    'issue': f"High error count: {error_analysis['total_errors']}"
                })
            
            if struct_percentage < 50:
                summary['containers_with_issues'].append({
                    'container': container_name,
                    'issue': f"Low structured logging: {struct_percentage:.1f}%"
                })
        
        # Calculate average structured logging percentage
        if container_results:
            avg_struct_percentage = sum(
                result.get('structured_logs', {}).get('structured_percentage', 0)
                for result in container_results.values()
                if 'error' not in result
            ) / len([r for r in container_results.values() if 'error' not in r])
            
            summary['structured_logs_percentage'] = avg_struct_percentage
        
        return summary
    
    def export_analysis(self, results: Dict[str, Any], format: str = 'json', output_path: Optional[Path] = None) -> Path:
        """Export analysis results to file."""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.project_dir / f'log_analysis_{timestamp}.{format}'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format == 'csv':
            # Convert to tabular format for CSV
            rows = []
            for container_name, container_result in results.items():
                if container_name == 'summary' or 'error' in container_result:
                    continue
                
                basic_stats = container_result.get('basic_stats', {})
                error_analysis = container_result.get('error_analysis', {})
                structured_logs = container_result.get('structured_logs', {})
                
                rows.append({
                    'container': container_name,
                    'total_lines': basic_stats.get('total_lines', 0),
                    'total_errors': error_analysis.get('total_errors', 0),
                    'total_warnings': error_analysis.get('total_warnings', 0),
                    'structured_percentage': structured_logs.get('structured_percentage', 0),
                    'first_timestamp': basic_stats.get('first_log_timestamp', ''),
                    'last_timestamp': basic_stats.get('last_log_timestamp', '')
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        elif format == 'md':
            # Generate markdown report
            self._generate_markdown_report(results, output_path)
        
        console.print(f"[green]✓ Analysis exported to: {output_path}[/green]")
        return output_path
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path):
        """Generate markdown report from analysis results."""
        lines = []
        lines.append("# Docker Log Analysis Report\n")
        lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
        
        # Summary section
        if 'summary' in results:
            summary = results['summary']
            lines.append("## Summary\n")
            lines.append(f"- **Containers analyzed:** {summary['containers_analyzed']}\n")
            lines.append(f"- **Total log lines:** {summary['total_log_lines']:,}\n")
            lines.append(f"- **Total errors:** {summary['total_errors']}\n")
            lines.append(f"- **Total warnings:** {summary['total_warnings']}\n")
            lines.append(f"- **Structured logging:** {summary['structured_logs_percentage']:.1f}%\n")
            
            if summary['containers_with_issues']:
                lines.append("\n### Issues Found\n")
                for issue in summary['containers_with_issues']:
                    lines.append(f"- **{issue['container']}:** {issue['issue']}\n")
        
        # Container details
        lines.append("\n## Container Details\n")
        
        for container_name, container_result in results.items():
            if container_name == 'summary' or 'error' in container_result:
                continue
            
            lines.append(f"\n### {container_name}\n")
            
            # Basic statistics
            basic_stats = container_result.get('basic_stats', {})
            lines.append("#### Basic Statistics\n")
            lines.append(f"- Total lines: {basic_stats.get('total_lines', 0):,}\n")
            lines.append(f"- Total characters: {basic_stats.get('total_characters', 0):,}\n")
            lines.append(f"- Average line length: {basic_stats.get('average_line_length', 0):.1f}\n")
            lines.append(f"- First log: {basic_stats.get('first_log_timestamp', 'N/A')}\n")
            lines.append(f"- Last log: {basic_stats.get('last_log_timestamp', 'N/A')}\n")
            
            # Error analysis
            error_analysis = container_result.get('error_analysis', {})
            if error_analysis:
                lines.append("\n#### Error Analysis\n")
                lines.append(f"- Total errors: {error_analysis.get('total_errors', 0)}\n")
                lines.append(f"- Total warnings: {error_analysis.get('total_warnings', 0)}\n")
                lines.append(f"- Error frequency: {error_analysis.get('error_frequency_per_hour', 0):.2f}/hour\n")
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
    
    def display_analysis_summary(self, results: Dict[str, Any]):
        """Display analysis summary in rich format."""
        if 'summary' in results:
            summary = results['summary']
            
            # Create summary table
            table = Table(title="Log Analysis Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Containers Analyzed", str(summary['containers_analyzed']))
            table.add_row("Total Log Lines", f"{summary['total_log_lines']:,}")
            table.add_row("Total Errors", str(summary['total_errors']))
            table.add_row("Total Warnings", str(summary['total_warnings']))
            table.add_row("Structured Logging", f"{summary['structured_logs_percentage']:.1f}%")
            
            console.print(table)
            
            # Display issues if any
            if summary['containers_with_issues']:
                console.print("\n[bold red]Issues Found:[/bold red]")
                for issue in summary['containers_with_issues']:
                    console.print(f"  • [red]{issue['container']}[/red]: {issue['issue']}")
        
        # Display container details
        container_results = {k: v for k, v in results.items() if k != 'summary'}
        
        if container_results:
            console.print("\n[bold]Container Details:[/bold]")
            
            for container_name, container_result in container_results.items():
                if 'error' in container_result:
                    console.print(f"  [red]✗ {container_name}[/red]: {container_result['error']}")
                    continue
                
                basic_stats = container_result.get('basic_stats', {})
                error_analysis = container_result.get('error_analysis', {})
                
                status = "✓" if error_analysis.get('total_errors', 0) == 0 else "⚠"
                console.print(f"  {status} [bold]{container_name}[/bold]:")
                console.print(f"    Lines: {basic_stats.get('total_lines', 0):,}")
                console.print(f"    Errors: {error_analysis.get('total_errors', 0)}")
                console.print(f"    Warnings: {error_analysis.get('total_warnings', 0)}")


def main():
    """Main entry point for log validation."""
    parser = argparse.ArgumentParser(description="Docker Log Validator for AI Trading Bot")
    parser.add_argument(
        '--container',
        type=str,
        help='Specific container to analyze'
    )
    parser.add_argument(
        '--since',
        type=str,
        default='1h',
        help='Analyze logs since this time (e.g., 1h, 30m, 2023-01-01T10:00:00)'
    )
    parser.add_argument(
        '--until',
        type=str,
        help='Analyze logs until this time'
    )
    parser.add_argument(
        '--analyze-patterns',
        action='store_true',
        default=True,
        help='Perform detailed pattern analysis'
    )
    parser.add_argument(
        '--export-analysis',
        action='store_true',
        help='Export analysis results to file'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'md'],
        default='json',
        help='Export format'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path'
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Project directory path'
    )
    
    args = parser.parse_args()
    
    try:
        validator = DockerLogValidator(args.project_dir)
        
        # Validate specific container or all containers
        if args.container:
            results = {
                args.container: validator.validate_container_logs(
                    args.container,
                    since=args.since,
                    until=args.until,
                    analyze_patterns=args.analyze_patterns
                )
            }
        else:
            results = validator.validate_all_containers(analyze_patterns=args.analyze_patterns)
        
        # Display results
        validator.display_analysis_summary(results)
        
        # Export results if requested
        if args.export_analysis:
            validator.export_analysis(results, args.format, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        logger.exception("Log validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())