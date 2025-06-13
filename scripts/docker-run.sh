#!/bin/bash

# AI Trading Bot - Docker Run Script
# Runs the Docker container with proper environment handling and configuration

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="ai-trading-bot"
CONTAINER_NAME="ai-trading-bot"
DEFAULT_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
AI Trading Bot - Docker Run Script

Usage: $0 [OPTIONS] [COMMAND]

OPTIONS:
    -t, --tag TAG           Use specific image tag (default: latest)
    -n, --name NAME         Set container name (default: ai-trading-bot)
    -d, --detach            Run in detached mode
    -i, --interactive       Run in interactive mode with TTY
    -e, --env-file FILE     Use custom env file (default: .env)
    --dry-run               Force dry-run mode (safe)
    --live                  Enable live trading (DANGEROUS!)
    --dev                   Run in development mode
    -p, --port PORT         Expose port for web interface
    -v, --volume SRC:DEST   Add custom volume mount
    --symbol SYMBOL         Trading symbol (default: BTC-USD)
    --interval INTERVAL     Candle interval (default: 1m)
    --clean                 Remove existing container first
    -h, --help              Show this help message

COMMANDS:
    live                    Start live trading (default)
    backtest               Run backtesting
    init                   Initialize configuration
    shell                  Open shell in container

EXAMPLES:
    $0                              # Run in dry-run mode
    $0 --live                       # Run live trading (DANGEROUS!)
    $0 -d                          # Run detached
    $0 --dev -i                    # Development mode with TTY
    $0 -p 8080                     # Expose port 8080
    $0 backtest --symbol ETH-USD   # Run backtest for ETH
    $0 shell                       # Open shell in container

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY        Registry URL for image
    TRADING_ENV            Trading environment (dev/prod)

SAFETY FEATURES:
    - Defaults to dry-run mode for safety
    - Requires explicit --live flag for real trading
    - Validates environment files
    - Creates necessary directories
    - Health checks and monitoring

EOF
}

# Default values
TAG="$DEFAULT_TAG"
DETACH=false
INTERACTIVE=false
ENV_FILE=".env"
DRY_RUN=true
DEV_MODE=false
PORT=""
CUSTOM_VOLUMES=()
SYMBOL="BTC-USD"
INTERVAL="1m"
CLEAN=false
COMMAND="live"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --live)
            DRY_RUN=false
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -v|--volume)
            CUSTOM_VOLUMES+=("$2")
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        live|backtest|init|shell)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate Docker installation
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p logs data config

# Check for environment file
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "example.env" ]]; then
        log_warning "Environment file $ENV_FILE not found"
        log_info "Creating $ENV_FILE from example.env"
        cp "example.env" "$ENV_FILE"
        log_warning "Please edit $ENV_FILE with your API keys before running!"
        exit 1
    else
        log_error "Environment file $ENV_FILE not found and no example.env available"
        exit 1
    fi
fi

# Validate critical environment variables (OpenAI key + at least one Coinbase credential scheme)
if ! grep -q "LLM__OPENAI_API_KEY=" "$ENV_FILE"; then
    log_error "Missing OpenAI API key (LLM__OPENAI_API_KEY) in $ENV_FILE"
    exit 1
fi

# Check for Coinbase credentials – either legacy Advanced-Trade OR CDP keys must be present
if ! grep -q "EXCHANGE__CB_API_KEY=" "$ENV_FILE" && ! grep -q "EXCHANGE__CDP_API_KEY_NAME=" "$ENV_FILE"; then
    log_error "Missing Coinbase API credentials in $ENV_FILE"
    log_info "Add legacy vars (EXCHANGE__CB_API_KEY, EXCHANGE__CB_API_SECRET, EXCHANGE__CB_PASSPHRASE) OR CDP vars (EXCHANGE__CDP_API_KEY_NAME, EXCHANGE__CDP_PRIVATE_KEY)"
    exit 1
fi

# Safety check for live trading
if [[ "$DRY_RUN" == false ]]; then
    log_warning "LIVE TRADING MODE ENABLED - REAL MONEY AT RISK!"
    echo -e "${RED}This will trade with real money on your Coinbase account.${NC}"
    echo -e "${RED}Are you absolutely sure you want to continue? (type 'YES' to confirm)${NC}"
    read -r confirmation
    if [[ "$confirmation" != "YES" ]]; then
        log_info "Live trading cancelled by user"
        exit 0
    fi
fi

# Set image name
if [[ "$DEV_MODE" == true ]]; then
    FULL_IMAGE_NAME="${IMAGE_NAME}-dev:${TAG}"
    CONTAINER_NAME="${CONTAINER_NAME}-dev"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
fi

# Check if image exists
if ! docker image inspect "$FULL_IMAGE_NAME" &> /dev/null; then
    log_error "Image $FULL_IMAGE_NAME not found"
    log_info "Build the image first: ./scripts/docker-build.sh"
    exit 1
fi

# Clean existing container if requested
if [[ "$CLEAN" == true ]]; then
    if docker container inspect "$CONTAINER_NAME" &> /dev/null; then
        log_info "Removing existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME"
    fi
fi

# Build Docker run arguments
RUN_ARGS=()
RUN_ARGS+=(--name "$CONTAINER_NAME")
RUN_ARGS+=(--env-file "$ENV_FILE")

# Override environment variables
RUN_ARGS+=(--env "DRY_RUN=$DRY_RUN")
RUN_ARGS+=(--env "SYMBOL=$SYMBOL")
RUN_ARGS+=(--env "INTERVAL=$INTERVAL")

if [[ "$DEV_MODE" == true ]]; then
    RUN_ARGS+=(--env "DEBUG=true")
    RUN_ARGS+=(--env "TESTING=true")
fi

# Volume mounts
RUN_ARGS+=(--volume "$PWD/logs:/app/logs")
RUN_ARGS+=(--volume "$PWD/data:/app/data")
RUN_ARGS+=(--volume "$PWD/config:/app/config:ro")

# Add custom volumes
for volume in "${CUSTOM_VOLUMES[@]}"; do
    RUN_ARGS+=(--volume "$volume")
done

# Port mapping
if [[ -n "$PORT" ]]; then
    RUN_ARGS+=(--publish "${PORT}:8080")
fi

# Network
RUN_ARGS+=(--network "trading-network")

# Runtime options
if [[ "$DETACH" == true ]]; then
    RUN_ARGS+=(--detach)
fi

if [[ "$INTERACTIVE" == true ]]; then
    RUN_ARGS+=(--interactive --tty)
fi

# Restart policy for detached containers
if [[ "$DETACH" == true ]]; then
    RUN_ARGS+=(--restart "unless-stopped")
fi

# Set command based on request
case "$COMMAND" in
    "live")
        if [[ "$DRY_RUN" == true ]]; then
            CMD_ARGS=("python" "-m" "bot.main" "live" "--dry-run" "--symbol" "$SYMBOL" "--interval" "$INTERVAL")
        else
            CMD_ARGS=("python" "-m" "bot.main" "live" "--symbol" "$SYMBOL" "--interval" "$INTERVAL")
        fi
        ;;
    "backtest")
        CMD_ARGS=("python" "-m" "bot.main" "backtest" "--symbol" "$SYMBOL")
        ;;
    "init")
        CMD_ARGS=("python" "-m" "bot.main" "init")
        ;;
    "shell")
        CMD_ARGS=("/bin/bash")
        INTERACTIVE=true
        RUN_ARGS+=(--interactive --tty)
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        exit 1
        ;;
esac

# Create network if it doesn't exist
if ! docker network inspect trading-network &> /dev/null; then
    log_info "Creating Docker network: trading-network"
    docker network create trading-network
fi

# Display run information
log_info "Starting AI Trading Bot container..."
log_info "Image: $FULL_IMAGE_NAME"
log_info "Container: $CONTAINER_NAME"
log_info "Command: $COMMAND"
log_info "Dry Run: $DRY_RUN"
log_info "Symbol: $SYMBOL"
log_info "Interval: $INTERVAL"

if [[ "$DRY_RUN" == false ]]; then
    log_warning "LIVE TRADING MODE - REAL MONEY AT RISK!"
fi

# Run the container
log_info "Running: docker run ${RUN_ARGS[*]} $FULL_IMAGE_NAME ${CMD_ARGS[*]}"
docker run "${RUN_ARGS[@]}" "$FULL_IMAGE_NAME" "${CMD_ARGS[@]}"

# Show status for detached containers
if [[ "$DETACH" == true ]]; then
    log_success "Container started successfully"
    log_info "Container status:"
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    log_info "View logs: ./scripts/docker-logs.sh"
    log_info "Stop container: docker stop $CONTAINER_NAME"
fi

exit 0