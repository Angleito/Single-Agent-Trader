#!/bin/bash

# AI Trading Bot Dashboard Docker Setup Script
# This script helps set up and manage the Docker environment for the dashboard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    mkdir -p logs data
    mkdir -p backend/logs backend/data
    mkdir -p nginx/conf.d

    print_success "Directories created"
}

# Function to build services
build_services() {
    print_status "Building Docker services..."

    cd ..  # Go to project root
    docker-compose build --no-cache
    cd dashboard

    print_success "Services built successfully"
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."

    cd ..  # Go to project root
    docker-compose up -d ai-trading-bot dashboard-backend dashboard-frontend
    cd dashboard

    print_success "Development environment started"
    print_status "Dashboard available at:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
}

# Function to start production environment
start_prod() {
    print_status "Starting production environment..."

    cd ..  # Go to project root
    docker-compose --profile production up -d
    cd dashboard

    print_success "Production environment started"
    print_status "Dashboard available at:"
    echo "  - Main access: http://localhost:8080"
    echo "  - Direct API: http://localhost:8000"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."

    cd ..  # Go to project root
    docker-compose down
    cd dashboard

    print_success "Services stopped"
}

# Function to view logs
view_logs() {
    local service=${1:-""}

    cd ..  # Go to project root

    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    fi

    cd dashboard
}

# Function to show status
show_status() {
    print_status "Service status:"

    cd ..  # Go to project root
    docker-compose ps
    cd dashboard
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."

    cd ..  # Go to project root
    docker-compose down -v
    docker system prune -f
    cd dashboard

    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "AI Trading Bot Dashboard Docker Setup"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Initial setup and build"
    echo "  dev       - Start development environment"
    echo "  prod      - Start production environment"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  logs      - View logs for all services"
    echo "  logs SERVICE - View logs for specific service"
    echo "  status    - Show service status"
    echo "  cleanup   - Clean up Docker resources"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 dev"
    echo "  $0 logs dashboard-backend"
    echo "  $0 prod"
}

# Main script logic
main() {
    local command=${1:-"help"}

    case $command in
        "setup")
            check_docker
            check_docker_compose
            create_directories
            build_services
            ;;
        "dev")
            check_docker
            start_dev
            ;;
        "prod")
            check_docker
            start_prod
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            start_dev
            ;;
        "logs")
            view_logs "$2"
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
