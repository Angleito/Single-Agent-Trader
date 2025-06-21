#!/bin/bash
"""
Quick Market Making Validation Check Script

This script provides quick validation workflows for common market making deployment scenarios.
It's designed to be run before deployment to ensure everything is properly configured.

Usage:
    ./scripts/quick-market-making-check.sh [scenario]

Scenarios:
    pre-deploy     - Quick pre-deployment checklist (default)
    health         - Component health check
    full           - Complete validation suite
    testnet        - Testnet deployment validation
    mainnet        - Production mainnet validation
    emergency      - Emergency procedures test
    performance    - Performance benchmarking
    indicators     - VuManChu indicator validation
    fees           - Fee calculation testing
    connectivity   - Exchange connectivity tests

Examples:
    ./scripts/quick-market-making-check.sh
    ./scripts/quick-market-making-check.sh full
    ./scripts/quick-market-making-check.sh testnet
    ./scripts/quick-market-making-check.sh mainnet
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validation script
VALIDATOR_SCRIPT="$SCRIPT_DIR/validate-market-making-setup.py"

# Default scenario
SCENARIO="${1:-pre-deploy}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Market Making Validation Check                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${CYAN}Scenario: ${SCENARIO}${NC}"
echo -e "${CYAN}Project Root: ${PROJECT_ROOT}${NC}"
echo -e "${CYAN}Timestamp: $(date)${NC}\n"

# Change to project root
cd "$PROJECT_ROOT"

# Function to check if required files exist
check_required_files() {
    echo -e "${YELLOW}🔍 Checking required files...${NC}"

    local missing_files=()

    # Core files
    [ ! -f ".env" ] && missing_files+=(".env")
    [ ! -f "pyproject.toml" ] && missing_files+=("pyproject.toml")
    [ ! -f "config/market_making.json" ] && missing_files+=("config/market_making.json")
    [ ! -f "$VALIDATOR_SCRIPT" ] && missing_files+=("scripts/validate-market-making-setup.py")

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required files:${NC}"
        for file in "${missing_files[@]}"; do
            echo -e "   ${RED}- $file${NC}"
        done

        echo -e "\n${YELLOW}Please ensure all required files are present before running validation.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ All required files present${NC}"
    fi
}

# Function to check Python environment
check_python_env() {
    echo -e "\n${YELLOW}🐍 Checking Python environment...${NC}"

    # Check if Poetry is available
    if command -v poetry &> /dev/null; then
        echo -e "${GREEN}✅ Poetry found: $(poetry --version)${NC}"

        # Check if virtual environment is active or can be activated
        if poetry env info --path &> /dev/null; then
            echo -e "${GREEN}✅ Poetry virtual environment available${NC}"
        else
            echo -e "${YELLOW}⚠️ Poetry virtual environment not found, will attempt to create${NC}"
        fi
    else
        echo -e "${RED}❌ Poetry not found. Please install Poetry first.${NC}"
        echo -e "${YELLOW}Install with: curl -sSL https://install.python-poetry.org | python3 -${NC}"
        exit 1
    fi

    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python version: $python_version${NC}"
}

# Function to install dependencies if needed
install_dependencies() {
    echo -e "\n${YELLOW}📦 Installing dependencies...${NC}"

    # Install poetry dependencies
    poetry install --no-dev --quiet

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
}

# Function to run validation based on scenario
run_validation() {
    local scenario="$1"
    local report_file="validation_reports/quick_check_$(date +%Y%m%d_%H%M%S).json"

    echo -e "\n${YELLOW}🚀 Running validation scenario: ${scenario}${NC}"

    # Create reports directory
    mkdir -p validation_reports

    case "$scenario" in
        "pre-deploy")
            echo -e "${BLUE}Running pre-deployment checklist...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --pre-deployment --fix-suggestions --export-report "$report_file"
            ;;

        "health")
            echo -e "${BLUE}Running component health checks...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --health-check --export-report "$report_file"
            ;;

        "full")
            echo -e "${BLUE}Running complete validation suite...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --full --fix-suggestions --export-report "$report_file"
            ;;

        "testnet")
            echo -e "${BLUE}Running testnet deployment validation...${NC}"
            # Check if we're configured for testnet
            if grep -q "EXCHANGE__BLUEFIN_NETWORK=testnet" .env 2>/dev/null; then
                poetry run python "$VALIDATOR_SCRIPT" --full --fix-suggestions --export-report "$report_file"
            else
                echo -e "${YELLOW}⚠️ Environment not configured for testnet. Please set EXCHANGE__BLUEFIN_NETWORK=testnet${NC}"
                poetry run python "$VALIDATOR_SCRIPT" --pre-deployment --export-report "$report_file"
            fi
            ;;

        "mainnet")
            echo -e "${BLUE}Running production mainnet validation...${NC}"
            echo -e "${RED}⚠️ WARNING: This will validate for PRODUCTION MAINNET deployment${NC}"
            echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
            read -r

            # Check if we're configured for mainnet
            if grep -q "EXCHANGE__BLUEFIN_NETWORK=mainnet" .env 2>/dev/null; then
                poetry run python "$VALIDATOR_SCRIPT" --full --fix-suggestions --export-report "$report_file"
            else
                echo -e "${YELLOW}⚠️ Environment not configured for mainnet. Please set EXCHANGE__BLUEFIN_NETWORK=mainnet${NC}"
                poetry run python "$VALIDATOR_SCRIPT" --pre-deployment --export-report "$report_file"
            fi
            ;;

        "emergency")
            echo -e "${BLUE}Running emergency procedures test...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --emergency-test --export-report "$report_file"
            ;;

        "performance")
            echo -e "${BLUE}Running performance benchmarking...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --performance-bench --export-report "$report_file"
            ;;

        "indicators")
            echo -e "${BLUE}Running VuManChu indicator validation...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --indicator-test --export-report "$report_file"
            ;;

        "fees")
            echo -e "${BLUE}Running fee calculation testing...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --fee-test --export-report "$report_file"
            ;;

        "connectivity")
            echo -e "${BLUE}Running exchange connectivity tests...${NC}"
            poetry run python "$VALIDATOR_SCRIPT" --connectivity-test --export-report "$report_file"
            ;;

        *)
            echo -e "${RED}❌ Unknown scenario: $scenario${NC}"
            echo -e "${YELLOW}Available scenarios: pre-deploy, health, full, testnet, mainnet, emergency, performance, indicators, fees, connectivity${NC}"
            exit 1
            ;;
    esac

    local exit_code=$?

    echo -e "\n${CYAN}📄 Validation report saved to: $report_file${NC}"

    # Print summary based on exit code
    case $exit_code in
        0)
            echo -e "\n${GREEN}✅ VALIDATION PASSED${NC}"
            echo -e "${GREEN}Your market making setup is ready for deployment!${NC}"
            ;;
        1)
            echo -e "\n${RED}❌ VALIDATION FAILED${NC}"
            echo -e "${RED}Critical issues found. Please resolve them before deployment.${NC}"
            ;;
        2)
            echo -e "\n${YELLOW}⚠️ VALIDATION PASSED WITH WARNINGS${NC}"
            echo -e "${YELLOW}Deployment possible but address warnings for optimal performance.${NC}"
            ;;
        130)
            echo -e "\n${YELLOW}⏸️ VALIDATION INTERRUPTED${NC}"
            echo -e "${YELLOW}Validation was cancelled by user.${NC}"
            ;;
        *)
            echo -e "\n${RED}❌ VALIDATION ERROR${NC}"
            echo -e "${RED}Unexpected error occurred during validation.${NC}"
            ;;
    esac

    return $exit_code
}

# Function to show post-validation recommendations
show_recommendations() {
    local exit_code=$1

    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                              Recommendations                                 ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"

    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}🎯 READY FOR DEPLOYMENT${NC}"
        echo -e "${GREEN}Your market making setup has passed validation. Consider these next steps:${NC}"
        echo -e "${CYAN}• Start with paper trading mode (SYSTEM__DRY_RUN=true)${NC}"
        echo -e "${CYAN}• Monitor performance for the first few hours${NC}"
        echo -e "${CYAN}• Set up alerts for balance and performance monitoring${NC}"
        echo -e "${CYAN}• Keep the emergency stop procedures readily available${NC}"

    elif [ $exit_code -eq 2 ]; then
        echo -e "\n${YELLOW}⚠️ DEPLOYMENT WITH CAUTIONS${NC}"
        echo -e "${YELLOW}Address these warnings for optimal performance:${NC}"
        echo -e "${CYAN}• Review the validation report for specific warnings${NC}"
        echo -e "${CYAN}• Consider testing in paper trading mode first${NC}"
        echo -e "${CYAN}• Monitor closely during initial operation${NC}"

    else
        echo -e "\n${RED}🚫 DEPLOYMENT NOT RECOMMENDED${NC}"
        echo -e "${RED}Critical issues must be resolved:${NC}"
        echo -e "${CYAN}• Review the validation report for specific failures${NC}"
        echo -e "${CYAN}• Fix all critical issues before attempting deployment${NC}"
        echo -e "${CYAN}• Re-run validation after making fixes${NC}"
        echo -e "${CYAN}• Consider reaching out for support if issues persist${NC}"
    fi

    echo -e "\n${BLUE}🔧 USEFUL COMMANDS:${NC}"
    echo -e "${CYAN}• Re-run this check: ./scripts/quick-market-making-check.sh $SCENARIO${NC}"
    echo -e "${CYAN}• Full validation:   ./scripts/quick-market-making-check.sh full${NC}"
    echo -e "${CYAN}• Health check:      ./scripts/quick-market-making-check.sh health${NC}"
    echo -e "${CYAN}• Monitor mode:      poetry run python scripts/validate-market-making-setup.py --monitor${NC}"
    echo -e "${CYAN}• View logs:         tail -f logs/bot.log${NC}"
}

# Function to show available scenarios
show_help() {
    echo -e "\n${BLUE}Available Scenarios:${NC}"
    echo -e "${CYAN}  pre-deploy     - Quick pre-deployment checklist (default)${NC}"
    echo -e "${CYAN}  health         - Component health check${NC}"
    echo -e "${CYAN}  full           - Complete validation suite${NC}"
    echo -e "${CYAN}  testnet        - Testnet deployment validation${NC}"
    echo -e "${CYAN}  mainnet        - Production mainnet validation${NC}"
    echo -e "${CYAN}  emergency      - Emergency procedures test${NC}"
    echo -e "${CYAN}  performance    - Performance benchmarking${NC}"
    echo -e "${CYAN}  indicators     - VuManChu indicator validation${NC}"
    echo -e "${CYAN}  fees           - Fee calculation testing${NC}"
    echo -e "${CYAN}  connectivity   - Exchange connectivity tests${NC}"

    echo -e "\n${BLUE}Examples:${NC}"
    echo -e "${CYAN}  ./scripts/quick-market-making-check.sh${NC}"
    echo -e "${CYAN}  ./scripts/quick-market-making-check.sh full${NC}"
    echo -e "${CYAN}  ./scripts/quick-market-making-check.sh testnet${NC}"
}

# Handle help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Main execution flow
main() {
    echo -e "${YELLOW}🔍 Starting market making validation check...${NC}"

    # Step 1: Check required files
    check_required_files

    # Step 2: Check Python environment
    check_python_env

    # Step 3: Install dependencies
    install_dependencies

    # Step 4: Run validation
    run_validation "$SCENARIO"
    local validation_exit_code=$?

    # Step 5: Show recommendations
    show_recommendations $validation_exit_code

    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                           Validation Complete                                ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"

    exit $validation_exit_code
}

# Trap to handle interruption
trap 'echo -e "\n${YELLOW}⏸️ Validation interrupted by user${NC}"; exit 130' INT

# Run main function
main
