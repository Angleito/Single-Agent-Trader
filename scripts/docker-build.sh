#!/bin/bash

# AI Trading Bot - Docker Build Script
# Builds the Docker image with proper tagging and optimization

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="ai-trading-bot"
REGISTRY="${DOCKER_REGISTRY:-}"
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
AI Trading Bot - Docker Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --tag TAG           Set image tag (default: latest)
    -r, --registry URL      Set Docker registry URL
    -p, --push              Push image to registry after build
    -c, --clean             Clean build (no cache)
    -d, --dev               Build development image
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Build with default settings
    $0 -t v1.0.0           # Build with specific tag
    $0 -t v1.0.0 -p        # Build and push to registry
    $0 -c                  # Clean build without cache
    $0 -d                  # Build development image

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY        Default registry URL
    DOCKER_BUILDKIT        Enable BuildKit (default: 1)

EOF
}

# Parse command line arguments
TAG="$DEFAULT_TAG"
PUSH=false
CLEAN=false
DEV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--dev)
            DEV=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
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

# Enable Docker BuildKit for better performance
export DOCKER_BUILDKIT=1

# Change to project directory
cd "$PROJECT_DIR"

# Check for required files
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found in project directory"
    exit 1
fi

if [[ ! -f "pyproject.toml" ]]; then
    log_error "pyproject.toml not found in project directory"
    exit 1
fi

# Set image name with registry if provided
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

# Add development suffix for dev builds
if [[ "$DEV" == true ]]; then
    FULL_IMAGE_NAME="${FULL_IMAGE_NAME}-dev"
    TARGET="production"  # We still use production stage but with dev tag
else
    TARGET="production"
fi

FULL_IMAGE_TAG="${FULL_IMAGE_NAME}:${TAG}"

log_info "Building AI Trading Bot Docker image..."
log_info "Image: $FULL_IMAGE_TAG"
log_info "Target: $TARGET"
log_info "Clean build: $CLEAN"

# Build arguments
BUILD_ARGS=()
BUILD_ARGS+=(--tag "$FULL_IMAGE_TAG")
BUILD_ARGS+=(--target "$TARGET")
BUILD_ARGS+=(--file "Dockerfile")

# Add build metadata
BUILD_ARGS+=(--label "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
BUILD_ARGS+=(--label "org.opencontainers.image.version=$TAG")
BUILD_ARGS+=(--label "org.opencontainers.image.revision=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')")

# Clean build option
if [[ "$CLEAN" == true ]]; then
    BUILD_ARGS+=(--no-cache)
    log_info "Building without cache..."
fi

# Add current directory as build context
BUILD_ARGS+=(.)

# Execute build
log_info "Running: docker build ${BUILD_ARGS[*]}"
if docker build "${BUILD_ARGS[@]}"; then
    log_success "Image built successfully: $FULL_IMAGE_TAG"
else
    log_error "Build failed"
    exit 1
fi

# Tag as latest if not dev build and not already latest
if [[ "$TAG" != "latest" && "$DEV" != true ]]; then
    LATEST_TAG="${FULL_IMAGE_NAME}:latest"
    log_info "Tagging as latest: $LATEST_TAG"
    docker tag "$FULL_IMAGE_TAG" "$LATEST_TAG"
fi

# Show image info
log_info "Image details:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"

# Push to registry if requested
if [[ "$PUSH" == true ]]; then
    if [[ -z "$REGISTRY" ]]; then
        log_error "Cannot push: no registry specified"
        exit 1
    fi
    
    log_info "Pushing image to registry..."
    if docker push "$FULL_IMAGE_TAG"; then
        log_success "Image pushed successfully: $FULL_IMAGE_TAG"
        
        # Push latest tag if created
        if [[ "$TAG" != "latest" && "$DEV" != true ]]; then
            LATEST_TAG="${FULL_IMAGE_NAME}:latest"
            docker push "$LATEST_TAG"
            log_success "Latest tag pushed: $LATEST_TAG"
        fi
    else
        log_error "Push failed"
        exit 1
    fi
fi

# Security scan suggestion
log_info "Build completed successfully!"
log_warning "Consider running security scan: docker scout cves $FULL_IMAGE_TAG"
log_info "To run the container: ./scripts/docker-run.sh"

exit 0