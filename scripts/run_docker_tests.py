#!/usr/bin/env python3
"""
Docker E2E Test Runner for VuManChu Implementation.

This script orchestrates the complete E2E testing suite in Docker environment,
providing commands to run different test profiles and manage test results.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockerTestRunner:
    """Orchestrate E2E tests in Docker environment."""
    
    def __init__(self, project_root: Path):
        """Initialize test runner."""
        self.project_root = project_root
        self.docker_dir = project_root / "docker"
        self.test_compose_file = self.docker_dir / "test-compose.yml"
        self.test_config_file = self.docker_dir / "test-config.yml"
        self.results_dir = project_root / "test_results"
        self.logs_dir = project_root / "logs"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Load test configuration
        self.config = self._load_test_config()
    
    def _load_test_config(self) -> Dict:
        """Load test configuration."""
        if self.test_config_file.exists():
            with open(self.test_config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Test config not found: {self.test_config_file}")
            return {}
    
    def _run_docker_compose(self, command: List[str], 
                          capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run docker-compose command."""
        full_command = [
            "docker-compose",
            "-f", str(self.test_compose_file),
            "-p", "vumanchu-e2e-tests"
        ] + command
        
        logger.info(f"Running: {' '.join(full_command)}")
        
        try:
            result = subprocess.run(
                full_command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error("Docker command timed out after 30 minutes")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker command failed: {e}")
            raise
    
    def setup_test_environment(self) -> bool:
        """Set up test environment and validate requirements."""
        logger.info("Setting up test environment")
        
        # Check Docker availability
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error("Docker not available")
                return False
            logger.info(f"Docker version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker not found or not responding")
            return False
        
        # Check docker-compose availability
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error("Docker Compose not available")
                return False
            logger.info(f"Docker Compose version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker Compose not found or not responding")
            return False
        
        # Validate test files exist
        required_files = [
            self.test_compose_file,
            self.project_root / "tests" / "test_e2e_vumanchu_docker.py",
            self.project_root / "tests" / "data" / "generate_test_market_data.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required test file missing: {file_path}")
                return False
        
        logger.info("Test environment setup completed")
        return True
    
    def generate_test_data(self, scenarios: List[str] = None, 
                          sizes: List[int] = None) -> bool:
        """Generate test data using the data generator service."""
        logger.info("Generating test data")
        
        if scenarios is None:
            scenarios = ["default", "trending", "ranging", "volatile", "gap_data"]
        if sizes is None:
            sizes = [1000, 5000, 10000]
        
        # Set environment variables for data generation
        env_vars = {
            "TEST_DATA_SCENARIOS": ",".join(scenarios),
            "TEST_DATA_SIZES": ",".join(map(str, sizes))
        }
        
        # Update environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            # Run data generator
            result = self._run_docker_compose([
                "--profile", "datagen",
                "up", "--build", "test-data-generator"
            ])
            
            if result.returncode == 0:
                logger.info("Test data generation completed")
                return True
            else:
                logger.error("Test data generation failed")
                return False
                
        finally:
            # Clean up environment variables
            for key in env_vars.keys():
                if key in os.environ:
                    del os.environ[key]
    
    def run_test_profile(self, profile: str = "standard", 
                        build: bool = True) -> Dict[str, bool]:
        """Run specific test profile."""
        logger.info(f"Running test profile: {profile}")
        
        # Map profiles to docker-compose profiles
        profile_mapping = {
            "quick": ["e2e-test-runner"],
            "standard": ["e2e-test-runner"],
            "comprehensive": ["e2e-test-runner", "performance", "signals", "integration"],
            "performance": ["performance"],
            "memory": ["memory"],
            "signals": ["signals"],
            "integration": ["integration"],
            "errors": ["errors"],
            "full": ["e2e-test-runner", "performance", "memory", "signals", "integration", "errors"]
        }
        
        if profile not in profile_mapping:
            logger.error(f"Unknown test profile: {profile}")
            return {"error": True}
        
        profiles = profile_mapping[profile]
        results = {}
        
        try:
            # Build images if requested
            if build:
                logger.info("Building test images")
                build_result = self._run_docker_compose(["build"])
                if build_result.returncode != 0:
                    logger.error("Failed to build test images")
                    return {"build_failed": True}
            
            # Run each profile
            for docker_profile in profiles:
                logger.info(f"Running Docker profile: {docker_profile}")
                
                # Set profile-specific environment
                if docker_profile == "performance":
                    os.environ["PERFORMANCE_TESTING"] = "true"
                elif docker_profile == "memory":
                    os.environ["MEMORY_TESTING"] = "true"
                elif docker_profile == "signals":
                    os.environ["SIGNAL_TESTING"] = "true"
                elif docker_profile == "integration":
                    os.environ["INTEGRATION_TESTING"] = "true"
                elif docker_profile == "errors":
                    os.environ["ERROR_TESTING"] = "true"
                
                try:
                    # Run the profile
                    result = self._run_docker_compose([
                        "--profile", docker_profile,
                        "up", "--abort-on-container-exit"
                    ])
                    
                    results[docker_profile] = result.returncode == 0
                    
                    if result.returncode == 0:
                        logger.info(f"Profile {docker_profile} completed successfully")
                    else:
                        logger.error(f"Profile {docker_profile} failed")
                
                finally:
                    # Clean up environment variables
                    env_vars_to_clean = [
                        "PERFORMANCE_TESTING", "MEMORY_TESTING", 
                        "SIGNAL_TESTING", "INTEGRATION_TESTING", "ERROR_TESTING"
                    ]
                    for env_var in env_vars_to_clean:
                        if env_var in os.environ:
                            del os.environ[env_var]
        
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            results["exception"] = str(e)
        
        return results
    
    def collect_test_results(self) -> Dict:
        """Collect and aggregate test results."""
        logger.info("Collecting test results")
        
        results_summary = {
            "collection_time": datetime.now().isoformat(),
            "test_files": [],
            "summary": {}
        }
        
        # Look for result files
        result_patterns = [
            "**/*_results.xml",
            "**/*_report.json",
            "**/*.xml",
            "**/test_summary.json",
            "**/e2e_test_report.json"
        ]
        
        for pattern in result_patterns:
            for result_file in self.results_dir.glob(pattern):
                file_info = {
                    "filename": result_file.name,
                    "path": str(result_file),
                    "size_bytes": result_file.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        result_file.stat().st_mtime
                    ).isoformat()
                }
                
                # Try to extract summary information
                if result_file.suffix == ".json":
                    try:
                        with open(result_file, 'r') as f:
                            content = json.load(f)
                        file_info["content_type"] = "json"
                        if "test_suite" in content:
                            file_info["test_suite"] = content["test_suite"]
                    except Exception:
                        pass
                
                results_summary["test_files"].append(file_info)
        
        # Aggregate summary
        results_summary["summary"] = {
            "total_files": len(results_summary["test_files"]),
            "xml_files": len([f for f in results_summary["test_files"] 
                             if f["filename"].endswith(".xml")]),
            "json_files": len([f for f in results_summary["test_files"] 
                              if f["filename"].endswith(".json")]),
            "total_size_mb": sum(f["size_bytes"] for f in results_summary["test_files"]) / 1024 / 1024
        }
        
        # Save results summary
        summary_file = self.results_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results collected: {results_summary['summary']}")
        return results_summary
    
    def cleanup_test_environment(self) -> bool:
        """Clean up test environment and containers."""
        logger.info("Cleaning up test environment")
        
        try:
            # Stop and remove containers
            stop_result = self._run_docker_compose([
                "down", "-v", "--remove-orphans"
            ])
            
            if stop_result.returncode == 0:
                logger.info("Test environment cleanup completed")
                return True
            else:
                logger.warning("Test environment cleanup had issues")
                return False
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def get_test_logs(self, service: str = None) -> str:
        """Get logs from test containers."""
        logger.info(f"Getting test logs for service: {service or 'all'}")
        
        try:
            command = ["logs"]
            if service:
                command.append(service)
            
            result = self._run_docker_compose(command, capture_output=True)
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error("Failed to get logs")
                return result.stderr
                
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return str(e)
    
    def run_comprehensive_test_suite(self) -> Dict:
        """Run the complete comprehensive test suite."""
        logger.info("Starting comprehensive E2E test suite")
        
        test_start_time = datetime.now()
        overall_results = {
            "start_time": test_start_time.isoformat(),
            "stages": {},
            "success": False
        }
        
        try:
            # Stage 1: Environment setup
            logger.info("Stage 1: Environment setup")
            setup_success = self.setup_test_environment()
            overall_results["stages"]["setup"] = setup_success
            
            if not setup_success:
                logger.error("Environment setup failed")
                return overall_results
            
            # Stage 2: Test data generation
            logger.info("Stage 2: Test data generation")
            data_success = self.generate_test_data()
            overall_results["stages"]["data_generation"] = data_success
            
            if not data_success:
                logger.error("Test data generation failed")
                return overall_results
            
            # Stage 3: Run comprehensive tests
            logger.info("Stage 3: Comprehensive test execution")
            test_results = self.run_test_profile("comprehensive", build=True)
            overall_results["stages"]["testing"] = test_results
            
            # Stage 4: Collect results
            logger.info("Stage 4: Results collection")
            collection_results = self.collect_test_results()
            overall_results["stages"]["collection"] = collection_results
            
            # Stage 5: Cleanup
            logger.info("Stage 5: Environment cleanup")
            cleanup_success = self.cleanup_test_environment()
            overall_results["stages"]["cleanup"] = cleanup_success
            
            # Determine overall success
            test_success = all(
                result is True if isinstance(result, bool) else 
                not result.get("error", False) and not result.get("exception")
                for result in test_results.values()
            )
            
            overall_results["success"] = (
                setup_success and 
                data_success and 
                test_success and
                collection_results is not None
            )
            
        except Exception as e:
            logger.error(f"Comprehensive test suite failed: {e}")
            overall_results["error"] = str(e)
        
        finally:
            overall_results["end_time"] = datetime.now().isoformat()
            overall_results["duration_minutes"] = (
                datetime.now() - test_start_time
            ).total_seconds() / 60
        
        # Save overall results
        results_file = self.results_dir / "comprehensive_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        logger.info(f"Comprehensive test suite completed: {overall_results['success']}")
        return overall_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run VuManChu E2E tests in Docker environment"
    )
    
    parser.add_argument(
        'command',
        choices=[
            'setup', 'generate-data', 'run-tests', 'collect-results', 
            'cleanup', 'logs', 'comprehensive'
        ],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--profile',
        choices=[
            'quick', 'standard', 'comprehensive', 'performance', 
            'memory', 'signals', 'integration', 'errors', 'full'
        ],
        default='standard',
        help='Test profile to run'
    )
    
    parser.add_argument(
        '--scenarios',
        type=str,
        default='default,trending,ranging,volatile',
        help='Comma-separated list of test scenarios'
    )
    
    parser.add_argument(
        '--sizes',
        type=str,
        default='1000,5000,10000',
        help='Comma-separated list of data sizes'
    )
    
    parser.add_argument(
        '--no-build',
        action='store_true',
        help='Skip building Docker images'
    )
    
    parser.add_argument(
        '--service',
        type=str,
        help='Specific service for logs command'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    project_root = Path(__file__).parent.parent
    runner = DockerTestRunner(project_root)
    
    # Execute command
    try:
        if args.command == 'setup':
            success = runner.setup_test_environment()
            sys.exit(0 if success else 1)
            
        elif args.command == 'generate-data':
            scenarios = [s.strip() for s in args.scenarios.split(',')]
            sizes = [int(s.strip()) for s in args.sizes.split(',')]
            success = runner.generate_test_data(scenarios, sizes)
            sys.exit(0 if success else 1)
            
        elif args.command == 'run-tests':
            results = runner.run_test_profile(args.profile, build=not args.no_build)
            success = all(
                result is True if isinstance(result, bool) else
                not result.get("error", False)
                for result in results.values()
            )
            print(json.dumps(results, indent=2))
            sys.exit(0 if success else 1)
            
        elif args.command == 'collect-results':
            results = runner.collect_test_results()
            print(json.dumps(results, indent=2))
            sys.exit(0)
            
        elif args.command == 'cleanup':
            success = runner.cleanup_test_environment()
            sys.exit(0 if success else 1)
            
        elif args.command == 'logs':
            logs = runner.get_test_logs(args.service)
            print(logs)
            sys.exit(0)
            
        elif args.command == 'comprehensive':
            results = runner.run_comprehensive_test_suite()
            print(json.dumps(results, indent=2))
            sys.exit(0 if results.get("success", False) else 1)
            
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        runner.cleanup_test_environment()
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()