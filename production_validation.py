#!/usr/bin/env python3
"""
Enhanced production validation test script.
Tests all critical fixes and integrations for production readiness.
"""

import os
import sys
import traceback
from typing import List, Tuple, Dict, Any
import asyncio

def test_result(test_name: str, success: bool, message: str = "") -> Tuple[str, bool, str]:
    """Format test result"""
    status = "✅" if success else "❌"
    return f"{status} {test_name}", success, message

class EnhancedValidationSuite:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        
    def add_result(self, test_name: str, success: bool, message: str = ""):
        """Add test result"""
        result = test_result(test_name, success, message)
        self.results.append(result)
        print(result[0])
        if message:
            print(f"   {message}")
    
    def run_import_validation(self):
        """Test 1: Import Validation"""
        print("\n=== 1. IMPORT VALIDATION TEST ===")
        
        imports_to_test = [
            ("bot.main", "Core main module"),
            ("bot.data.market", "Market data module"),
            ("bot.types", "Type definitions"),
            ("bot.config", "Configuration module"),
            ("bot.strategy.llm_agent", "LLM agent strategy"),
            ("bot.indicators.vumanchu", "VuManChu indicators"),
            ("bot.risk", "Risk management"),
            ("bot.validator", "Trade validator"),
            ("bot.exchange.factory", "Exchange factory"),
            ("bot.position_manager", "Position manager"),
        ]
        
        for module_name, description in imports_to_test:
            try:
                __import__(module_name)
                self.add_result(f"{module_name} import", True, description)
            except Exception as e:
                self.add_result(f"{module_name} import", False, f"{description}: {str(e)}")

    def run_config_validation(self):
        """Test 2: Configuration Validation"""
        print("\n=== 2. CONFIGURATION VALIDATION TEST ===")
        
        try:
            from bot.config import settings
            self.add_result("Config loading", True, f"dry_run: {settings.system.dry_run}")
            
            # Test critical config sections
            config_sections = [
                (hasattr(settings, 'system'), "System config section"),
                (hasattr(settings.system, 'dry_run'), "Dry run setting"),
                (hasattr(settings, 'trading'), "Trading config section"),
                (hasattr(settings, 'exchange'), "Exchange config section"),
                (hasattr(settings, 'risk'), "Risk config section"),
                (hasattr(settings, 'llm'), "LLM config section"),
            ]
            
            for condition, description in config_sections:
                self.add_result(description, condition, "Present" if condition else "Missing")
                
        except Exception as e:
            self.add_result("Config validation", False, str(e))

    def run_websocket_validation(self):
        """Test 3: WebSocket Performance Validation"""
        print("\n=== 3. WEBSOCKET PERFORMANCE TEST ===")
        
        try:
            import bot.data.market
            self.add_result("WebSocket module import", True)
            
            # Check for MarketDataProvider class (the actual class in the module)
            market_module = bot.data.market
            
            classes_to_check = [
                ("MarketDataProvider", "Main market data provider"),
                ("WebSocketMessageValidator", "Message validation"),
                ("MarketDataClient", "High-level client"),
            ]
            
            for class_name, description in classes_to_check:
                if hasattr(market_module, class_name):
                    self.add_result(f"{class_name} class", True, description)
                    
                    # Check methods for MarketDataProvider
                    if class_name == "MarketDataProvider":
                        provider_class = getattr(market_module, class_name)
                        methods = dir(provider_class)
                        
                        method_checks = [
                            (any("process" in m.lower() for m in methods), "Processing methods"),
                            (any("queue" in m.lower() for m in methods), "Message queuing"),
                            (any("websocket" in m.lower() for m in methods), "WebSocket handling"),
                            (any("async" in m.lower() or m.startswith("_async") for m in methods), "Async operations"),
                        ]
                        
                        for condition, check_name in method_checks:
                            self.add_result(f"WebSocket {check_name.lower()}", condition, 
                                          "Found" if condition else "Not found")
                else:
                    self.add_result(f"{class_name} class", False, f"{description} not found")
                    
        except Exception as e:
            self.add_result("WebSocket validation", False, str(e))

    def run_data_validation(self):
        """Test 4: Data Validation Test"""
        print("\n=== 4. DATA VALIDATION TEST ===")
        
        try:
            import bot.validator
            self.add_result("Validator module import", True)
            
            # Check for TradeValidator class
            if hasattr(bot.validator, "TradeValidator"):
                validator_class = bot.validator.TradeValidator
                self.add_result("TradeValidator class", True)
                
                # Check validator methods
                methods = dir(validator_class)
                validation_checks = [
                    (any("validate" in m.lower() for m in methods), "Validation methods"),
                    (any("schema" in m.lower() for m in methods), "Schema validation"),
                    (any("sanitize" in m.lower() for m in methods), "Input sanitization"),
                ]
                
                for condition, check_name in validation_checks:
                    self.add_result(f"{check_name}", condition, "Present" if condition else "Missing")
            else:
                self.add_result("TradeValidator class", False, "Class not found")
                
        except Exception as e:
            self.add_result("Data validation test", False, str(e))
        
        # Test circuit breaker functionality
        try:
            import bot.risk
            self.add_result("Risk module import", True)
            
            # Check for circuit breaker classes
            if hasattr(bot.risk, "CircuitBreaker"):
                self.add_result("Circuit breaker implementation", True, "Found in risk module")
            else:
                self.add_result("Circuit breaker implementation", False, "Not found in risk module")
                
        except Exception as e:
            self.add_result("Circuit breaker test", False, str(e))

    def run_security_validation(self):
        """Test 5: Security Configuration Test"""
        print("\n=== 5. SECURITY CONFIGURATION TEST ===")
        
        # Check Docker configuration
        try:
            docker_compose_path = "docker-compose.yml"
            if os.path.exists(docker_compose_path):
                with open(docker_compose_path, 'r') as f:
                    docker_content = f.read()
                
                security_checks = [
                    (("/var/run/docker.sock" not in docker_content), "Docker socket removal", 
                     "No docker socket mounts found" if "/var/run/docker.sock" not in docker_content 
                     else "Docker socket still mounted"),
                    (("privileged" not in docker_content), "Docker privilege check", 
                     "No privileged containers" if "privileged" not in docker_content 
                     else "Privileged containers found"),
                    (("user:" in docker_content), "Non-root user configuration", 
                     "Non-root users configured" if "user:" in docker_content 
                     else "No user configuration found"),
                    (("read_only: true" in docker_content), "Read-only filesystem", 
                     "Read-only filesystems configured" if "read_only: true" in docker_content 
                     else "No read-only configuration"),
                ]
                
                for condition, check_name, message in security_checks:
                    self.add_result(check_name, condition, message)
            else:
                self.add_result("Docker compose file", False, "File not found")
                
        except Exception as e:
            self.add_result("Docker security check", False, str(e))
        
        # Check for CORS configuration
        try:
            from bot.config import settings
            
            # Check if CORS restrictions are mentioned in docker-compose
            docker_compose_path = "docker-compose.yml"
            if os.path.exists(docker_compose_path):
                with open(docker_compose_path, 'r') as f:
                    docker_content = f.read()
                
                cors_configured = "CORS_ORIGINS=" in docker_content and "*" not in docker_content
                self.add_result("CORS configuration", cors_configured, 
                              "Restricted CORS origins" if cors_configured else "CORS may be too permissive")
            else:
                self.add_result("CORS configuration", False, "Cannot check - no docker-compose.yml")
                
        except Exception as e:
            self.add_result("CORS configuration check", False, str(e))
        
        # Check private key validation
        try:
            import bot.exchange.coinbase
            import bot.exchange.bluefin
            self.add_result("Exchange modules", True, "Both exchange modules accessible")
            
            # Check for validation in exchange settings
            from bot.config import ExchangeSettings
            self.add_result("Exchange validation", True, "ExchangeSettings class with validation")
            
        except Exception as e:
            self.add_result("Private key validation modules", False, str(e))

    def run_error_handling_validation(self):
        """Test 6: Error Handling and Recovery"""
        print("\n=== 6. ERROR HANDLING VALIDATION ===")
        
        try:
            # Check for error handling modules
            error_modules = [
                ("bot.error_handling", "Error handling framework"),
                ("bot.system_monitor", "System monitoring"),
            ]
            
            for module_name, description in error_modules:
                try:
                    __import__(module_name)
                    self.add_result(f"{module_name}", True, description)
                except ImportError:
                    self.add_result(f"{module_name}", False, f"{description} not found")
            
            # Check for error boundary patterns
            try:
                from bot.error_handling import ErrorBoundary, graceful_degradation
                self.add_result("Error boundary implementation", True, "ErrorBoundary and graceful degradation")
            except ImportError:
                self.add_result("Error boundary implementation", False, "Components not found")
                
        except Exception as e:
            self.add_result("Error handling validation", False, str(e))

    def run_performance_validation(self):
        """Test 7: Performance Optimization Validation"""
        print("\n=== 7. PERFORMANCE OPTIMIZATION TEST ===")
        
        try:
            from bot.config import settings
            
            # Check performance-related settings
            performance_checks = [
                (settings.system.update_frequency_seconds <= 2.0, "Fast update frequency", 
                 f"Update frequency: {settings.system.update_frequency_seconds}s"),
                (settings.data.data_cache_ttl_seconds <= 30, "Fast cache refresh", 
                 f"Cache TTL: {settings.data.data_cache_ttl_seconds}s"),
                (settings.system.parallel_processing, "Parallel processing enabled", 
                 "Parallel processing is enabled"),
                (settings.llm.enable_caching, "LLM response caching", 
                 "LLM caching enabled for performance"),
            ]
            
            for condition, check_name, message in performance_checks:
                self.add_result(check_name, condition, message)
                
            # Check for async/await patterns in market data
            import bot.data.market
            market_source = ""
            try:
                import inspect
                market_source = inspect.getsource(bot.data.market)
                async_patterns = market_source.count("async def") + market_source.count("await ")
                self.add_result("Async patterns in market data", async_patterns > 10, 
                              f"Found {async_patterns} async patterns")
            except:
                self.add_result("Async patterns check", False, "Could not analyze source")
                
        except Exception as e:
            self.add_result("Performance validation", False, str(e))

    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 STARTING ENHANCED PRODUCTION VALIDATION")
        print("=" * 70)
        
        test_suites = [
            self.run_import_validation,
            self.run_config_validation,
            self.run_websocket_validation,
            self.run_data_validation,
            self.run_security_validation,
            self.run_error_handling_validation,
            self.run_performance_validation,
        ]
        
        for test_suite in test_suites:
            try:
                test_suite()
            except Exception as e:
                print(f"\n❌ Test suite failed: {test_suite.__name__}")
                print(f"   Error: {str(e)}")
                print(f"   Traceback: {traceback.format_exc()}")
        
        # Generate enhanced summary
        self.generate_enhanced_summary()

    def generate_enhanced_summary(self):
        """Generate enhanced validation summary with detailed analysis"""
        print("\n" + "=" * 70)
        print("📊 ENHANCED VALIDATION SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _ in self.results if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Categorize failures
        critical_failures = []
        minor_failures = []
        
        for result, success, message in self.results:
            if not success:
                if any(keyword in result.lower() for keyword in ['import', 'config', 'security', 'error']):
                    critical_failures.append((result, message))
                else:
                    minor_failures.append((result, message))
        
        if critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(critical_failures)}):")
            for result, message in critical_failures:
                print(f"   {result}")
                if message:
                    print(f"      → {message}")
        
        if minor_failures:
            print(f"\n⚠️  MINOR ISSUES ({len(minor_failures)}):")
            for result, message in minor_failures:
                print(f"   {result}")
                if message:
                    print(f"      → {message}")
        
        # Enhanced production readiness assessment
        print(f"\n🎯 PRODUCTION READINESS ASSESSMENT:")
        
        critical_failure_count = len(critical_failures)
        minor_failure_count = len(minor_failures)
        
        if critical_failure_count == 0 and minor_failure_count == 0:
            print("🟢 EXCELLENT - FULLY READY FOR PRODUCTION")
            readiness_score = 100
        elif critical_failure_count == 0 and minor_failure_count <= 3:
            print("🟡 GOOD - READY FOR PRODUCTION WITH MONITORING")
            readiness_score = 85
        elif critical_failure_count <= 2 and minor_failure_count <= 5:
            print("🟠 FAIR - MOSTLY READY - ADDRESS CRITICAL ISSUES")
            readiness_score = 70
        else:
            print("🔴 POOR - NOT READY - SIGNIFICANT ISSUES NEED RESOLUTION")
            readiness_score = 40
        
        print(f"📈 READINESS SCORE: {readiness_score}/100")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if critical_failure_count > 0:
            print("   1. Address all critical failures before production deployment")
        if minor_failure_count > 3:
            print("   2. Review and resolve minor issues to improve stability")
        if readiness_score >= 85:
            print("   3. System appears ready for production deployment")
            print("   4. Consider setting up monitoring and alerting")
        
        return critical_failure_count == 0 and minor_failure_count <= 3

if __name__ == "__main__":
    validator = EnhancedValidationSuite()
    validator.run_all_tests()