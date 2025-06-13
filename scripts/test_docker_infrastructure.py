#!/usr/bin/env python3
"""
Docker Testing Infrastructure Validation

Simple test script to verify that the Docker testing infrastructure is working correctly.
This script validates the environment and runs basic functionality tests.

Usage:
    python scripts/test_docker_infrastructure.py
"""

import sys
import subprocess
import importlib
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def test_python_dependencies():
    """Test that all required Python dependencies are available."""
    console.print("[yellow]Testing Python dependencies...[/yellow]")
    
    required_modules = [
        'docker',
        'rich', 
        'pandas',
        'psutil',
        'sqlite3'  # Built-in, but test anyway
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            console.print(f"  ✓ {module}")
        except ImportError:
            missing_modules.append(module)
            console.print(f"  ✗ {module} [red](missing)[/red]")
    
    if missing_modules:
        console.print(f"[red]Missing dependencies: {', '.join(missing_modules)}[/red]")
        console.print("[yellow]Install with: pip install docker rich pandas psutil[/yellow]")
        return False
    
    console.print("[green]✓ All Python dependencies available[/green]")
    return True

def test_docker_availability():
    """Test Docker daemon and docker-compose availability."""
    console.print("[yellow]Testing Docker availability...[/yellow]")
    
    # Test Docker daemon
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            console.print("  ✓ Docker daemon accessible")
        else:
            console.print("  ✗ Docker daemon not accessible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("  ✗ Docker command not found")
        return False
    
    # Test docker-compose
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            console.print("  ✓ docker-compose available")
        else:
            console.print("  ✗ docker-compose not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("  ✗ docker-compose command not found")
        return False
    
    console.print("[green]✓ Docker environment ready[/green]")
    return True

def test_project_structure():
    """Test that required project files exist."""
    console.print("[yellow]Testing project structure...[/yellow]")
    
    project_dir = Path(__file__).parent.parent
    required_files = [
        'docker-compose.yml',
        'Dockerfile.minimal',
        'scripts/run_docker_tests.py',
        'scripts/validate_docker_logs.py', 
        'scripts/monitor_test_performance.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_dir / file_path
        if full_path.exists():
            console.print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            console.print(f"  ✗ {file_path} [red](missing)[/red]")
    
    if missing_files:
        console.print(f"[red]Missing files: {', '.join(missing_files)}[/red]")
        return False
    
    console.print("[green]✓ Project structure valid[/green]")
    return True

def test_script_functionality():
    """Test basic script functionality."""
    console.print("[yellow]Testing script functionality...[/yellow]")
    
    project_dir = Path(__file__).parent.parent
    
    # Test script help functions
    scripts_to_test = [
        'scripts/run_docker_tests.py',
        'scripts/validate_docker_logs.py',
        'scripts/monitor_test_performance.py'
    ]
    
    for script in scripts_to_test:
        script_path = project_dir / script
        try:
            # Test that script can show help without errors
            result = subprocess.run([
                sys.executable, str(script_path), '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                console.print(f"  ✓ {script} (help works)")
            else:
                console.print(f"  ✗ {script} (help failed): {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            console.print(f"  ✗ {script} (execution failed): {e}")
            return False
    
    console.print("[green]✓ All scripts functional[/green]")
    return True

def test_directory_creation():
    """Test that required directories can be created."""
    console.print("[yellow]Testing directory creation...[/yellow]")
    
    project_dir = Path(__file__).parent.parent
    required_dirs = ['reports', 'data']
    
    for dir_name in required_dirs:
        dir_path = project_dir / dir_name
        try:
            dir_path.mkdir(exist_ok=True)
            if dir_path.exists():
                console.print(f"  ✓ {dir_name}/ directory ready")
            else:
                console.print(f"  ✗ {dir_name}/ directory creation failed")
                return False
        except PermissionError:
            console.print(f"  ✗ {dir_name}/ permission denied")
            return False
    
    console.print("[green]✓ Required directories ready[/green]")
    return True

def test_environment_file():
    """Test that .env file exists or can be created from example."""
    console.print("[yellow]Testing environment configuration...[/yellow]")
    
    project_dir = Path(__file__).parent.parent
    env_file = project_dir / '.env'
    example_env = project_dir / 'example.env'
    
    if env_file.exists():
        console.print("  ✓ .env file exists")
    elif example_env.exists():
        console.print("  ✓ example.env found (copy to .env for testing)")
    else:
        console.print("  ⚠ No .env or example.env file found")
        console.print("    [yellow]Create .env file with required environment variables[/yellow]")
        return False
    
    console.print("[green]✓ Environment configuration available[/green]")
    return True

def run_infrastructure_test():
    """Run complete infrastructure test suite."""
    console.print(Panel.fit(
        "[bold blue]Docker Testing Infrastructure Validation[/bold blue]",
        border_style="blue"
    ))
    
    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("Docker Availability", test_docker_availability), 
        ("Project Structure", test_project_structure),
        ("Script Functionality", test_script_functionality),
        ("Directory Creation", test_directory_creation),
        ("Environment Configuration", test_environment_file)
    ]
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        
        for test_name, test_func in tests:
            task = progress.add_task(f"Running {test_name}...")
            
            try:
                results[test_name] = test_func()
            except Exception as e:
                console.print(f"[red]Error in {test_name}: {e}[/red]")
                results[test_name] = False
            
            progress.update(task, completed=100)
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]Test Summary[/bold]")
    console.print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    console.print("="*60)
    console.print(f"[bold]Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("\n[bold green]✓ Docker testing infrastructure is ready![/bold green]")
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("1. Ensure Docker containers are running: docker-compose up -d")
        console.print("2. Run quick test: python scripts/run_docker_tests.py --quick")
        console.print("3. Check the documentation: scripts/README_DOCKER_TESTING.md")
        return True
    else:
        console.print(f"\n[bold red]✗ {total - passed} test(s) failed - fix issues before proceeding[/bold red]")
        return False

def main():
    """Main entry point."""
    try:
        success = run_infrastructure_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Test failed with error: {e}[/red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())