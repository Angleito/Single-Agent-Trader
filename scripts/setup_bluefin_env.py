#!/usr/bin/env python3
"""
Bluefin Environment Setup Helper Script

This script helps users set up their environment for Bluefin DEX trading by:
1. Generating secure API keys
2. Creating environment files from templates
3. Validating configuration
4. Providing setup guidance

Usage:
    python scripts/setup_bluefin_env.py [--interactive] [--validate-only]
"""

import argparse
import json
import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


class BluefinEnvSetup:
    """Helper class for setting up Bluefin environment configuration."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / ".env"
        self.env_example = self.project_root / ".env.example"
        self.env_bluefin_example = self.project_root / ".env.bluefin.example"
        
    def print_header(self):
        """Print the setup script header."""
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print("🚀 Bluefin DEX Environment Setup Helper")
        print("🔧 Setting up your AI Trading Bot for Bluefin DEX")
        print(f"{'='*70}{Colors.END}\n")

    def generate_api_key(self, length: int = 32) -> str:
        """Generate a cryptographically secure API key."""
        return secrets.token_urlsafe(length)

    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met."""
        print(f"{Colors.BOLD}📋 Checking Prerequisites...{Colors.END}\n")
        
        checks = {
            "project_structure": self.project_root.exists(),
            "example_files": self.env_example.exists() or self.env_bluefin_example.exists(),
            "services_directory": (self.project_root / "services").exists(),
            "validation_script": (self.project_root / "services" / "scripts" / "validate_env.py").exists(),
        }
        
        for check, passed in checks.items():
            status = f"{Colors.GREEN}✅" if passed else f"{Colors.RED}❌"
            readable_name = check.replace("_", " ").title()
            print(f"  {status} {readable_name}{Colors.END}")
        
        print()
        return checks

    def create_env_file(self, template: str = "bluefin", interactive: bool = False) -> bool:
        """Create .env file from template."""
        print(f"{Colors.BOLD}📝 Creating Environment Configuration...{Colors.END}\n")
        
        # Determine source template
        if template == "bluefin" and self.env_bluefin_example.exists():
            source_template = self.env_bluefin_example
            print(f"Using Bluefin-specific template: {source_template.name}")
        elif self.env_example.exists():
            source_template = self.env_example
            print(f"Using general template: {source_template.name}")
        else:
            print(f"{Colors.RED}❌ No template files found!{Colors.END}")
            return False

        # Check if .env already exists
        if self.env_file.exists():
            if interactive:
                response = input(f"{Colors.YELLOW}⚠️ .env file already exists. Overwrite? (y/N): {Colors.END}")
                if response.lower() != 'y':
                    print("Keeping existing .env file.")
                    return True
            else:
                print(f"{Colors.YELLOW}⚠️ .env file already exists. Use --interactive to overwrite.{Colors.END}")
                return True

        # Copy template to .env
        try:
            shutil.copy2(source_template, self.env_file)
            print(f"{Colors.GREEN}✅ Created .env from {source_template.name}{Colors.END}")
            
            # Set secure file permissions
            os.chmod(self.env_file, 0o600)
            print(f"{Colors.GREEN}✅ Set secure file permissions (600){Colors.END}")
            
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to create .env file: {e}{Colors.END}")
            return False

    def generate_bluefin_api_key(self) -> Optional[str]:
        """Generate a new Bluefin service API key."""
        print(f"{Colors.BOLD}🔑 Generating Bluefin Service API Key...{Colors.END}\n")
        
        try:
            api_key = self.generate_api_key(32)
            print(f"Generated API key: {Colors.GREEN}{api_key}{Colors.END}")
            print(f"\n{Colors.YELLOW}📝 Add this to your .env file:{Colors.END}")
            print(f"BLUEFIN_SERVICE_API_KEY={api_key}")
            print()
            
            return api_key
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to generate API key: {e}{Colors.END}")
            return None

    def update_env_with_api_key(self, api_key: str) -> bool:
        """Update .env file with the generated API key."""
        if not self.env_file.exists():
            print(f"{Colors.RED}❌ .env file not found. Create it first.{Colors.END}")
            return False
        
        try:
            # Read current .env content
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            
            # Update or add the API key
            updated = False
            for i, line in enumerate(lines):
                if line.startswith('BLUEFIN_SERVICE_API_KEY='):
                    lines[i] = f'BLUEFIN_SERVICE_API_KEY={api_key}\n'
                    updated = True
                    break
            
            if not updated:
                # Add the API key at the end
                lines.append(f'\nBLUEFIN_SERVICE_API_KEY={api_key}\n')
            
            # Write back to file
            with open(self.env_file, 'w') as f:
                f.writelines(lines)
            
            print(f"{Colors.GREEN}✅ Updated .env file with API key{Colors.END}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to update .env file: {e}{Colors.END}")
            return False

    def validate_configuration(self) -> bool:
        """Run the environment validation script."""
        print(f"{Colors.BOLD}🔍 Validating Configuration...{Colors.END}\n")
        
        validation_script = self.project_root / "services" / "scripts" / "validate_env.py"
        if not validation_script.exists():
            print(f"{Colors.RED}❌ Validation script not found: {validation_script}{Colors.END}")
            return False
        
        # Change to project directory for validation
        original_cwd = os.getcwd()
        try:
            os.chdir(self.project_root)
            
            # Run validation script
            import subprocess
            result = subprocess.run(
                [sys.executable, str(validation_script)], 
                capture_output=True, 
                text=True
            )
            
            # Display validation output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"{Colors.YELLOW}Warnings:{Colors.END}")
                print(result.stderr)
            
            success = result.returncode == 0
            if success:
                print(f"\n{Colors.GREEN}✅ Configuration validation passed!{Colors.END}")
            else:
                print(f"\n{Colors.YELLOW}⚠️ Configuration validation found issues. Please review above.{Colors.END}")
            
            return success
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to run validation: {e}{Colors.END}")
            return False
        finally:
            os.chdir(original_cwd)

    def provide_next_steps(self):
        """Provide guidance on next steps."""
        print(f"\n{Colors.BOLD}🎯 Next Steps:{Colors.END}\n")
        
        steps = [
            "🔐 Add your Sui wallet private key to EXCHANGE__BLUEFIN_PRIVATE_KEY",
            "🤖 Add your OpenAI API key to LLM__OPENAI_API_KEY", 
            "🔍 Review and adjust trading parameters (leverage, symbols, etc.)",
            "✅ Ensure SYSTEM__DRY_RUN=true for safe testing",
            "🚀 Start the bot: python -m bot.main live",
            "📊 Monitor the dashboard: http://localhost:8000",
            "📝 Check logs for any issues: tail -f logs/bot.log"
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\n{Colors.BOLD}📚 Documentation:{Colors.END}")
        print(f"  • Environment Guide: docs/Environment_Setup_Guide.md")
        print(f"  • Bluefin Integration: docs/bluefin_integration.md")
        print(f"  • Quick Start: docs/BLUEFIN_QUICK_START.md")

    def provide_security_warnings(self):
        """Display important security warnings."""
        print(f"\n{Colors.BOLD}{Colors.RED}🚨 SECURITY WARNINGS:{Colors.END}\n")
        
        warnings = [
            "NEVER commit your .env file to version control",
            "Keep your Sui private key secure - anyone with it can access your funds",
            "Always start with SYSTEM__DRY_RUN=true for testing",
            "Use testnet (EXCHANGE__BLUEFIN_NETWORK=testnet) for development",
            "Regularly rotate your API keys",
            "Set proper file permissions: chmod 600 .env"
        ]
        
        for warning in warnings:
            print(f"  {Colors.RED}⚠️{Colors.END} {warning}")

    def run_interactive_setup(self) -> bool:
        """Run interactive setup process."""
        self.print_header()
        
        # Check prerequisites
        prereqs = self.check_prerequisites()
        if not all(prereqs.values()):
            print(f"{Colors.RED}❌ Some prerequisites are missing. Please check your installation.{Colors.END}")
            return False
        
        print(f"{Colors.GREEN}✅ All prerequisites met!{Colors.END}\n")
        
        # Create .env file
        if not self.create_env_file(template="bluefin", interactive=True):
            return False
        
        # Generate API key
        api_key = self.generate_bluefin_api_key()
        if api_key:
            response = input(f"{Colors.YELLOW}Update .env file with this API key? (Y/n): {Colors.END}")
            if response.lower() != 'n':
                self.update_env_with_api_key(api_key)
        
        # Validate configuration
        print(f"\n{Colors.BOLD}Would you like to validate your configuration now?{Colors.END}")
        response = input(f"{Colors.YELLOW}Run validation? (Y/n): {Colors.END}")
        if response.lower() != 'n':
            self.validate_configuration()
        
        # Show next steps
        self.provide_next_steps()
        self.provide_security_warnings()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Setup complete! Happy trading!{Colors.END}")
        return True

    def run_validation_only(self) -> bool:
        """Run only the validation process."""
        self.print_header()
        print(f"{Colors.BOLD}Running validation only...{Colors.END}\n")
        
        if not self.env_file.exists():
            print(f"{Colors.RED}❌ .env file not found. Please create it first.{Colors.END}")
            return False
        
        return self.validate_configuration()


def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(
        description="Bluefin Environment Setup Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_bluefin_env.py --interactive    # Full interactive setup
  python scripts/setup_bluefin_env.py --validate-only # Just validate existing config
  python scripts/setup_bluefin_env.py                 # Quick setup with defaults
        """
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run interactive setup with prompts"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only run validation on existing configuration"
    )
    
    args = parser.parse_args()
    
    setup = BluefinEnvSetup()
    
    try:
        if args.validate_only:
            success = setup.run_validation_only()
        elif args.interactive:
            success = setup.run_interactive_setup()
        else:
            # Quick setup with defaults
            setup.print_header()
            success = True
            
            # Create .env file if it doesn't exist
            if not setup.env_file.exists():
                success &= setup.create_env_file(template="bluefin")
            
            # Generate API key
            api_key = setup.generate_bluefin_api_key()
            if api_key:
                success &= setup.update_env_with_api_key(api_key)
            
            # Show next steps
            setup.provide_next_steps()
            setup.provide_security_warnings()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Setup interrupted by user.{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()