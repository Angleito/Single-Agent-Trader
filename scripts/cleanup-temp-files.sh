#!/usr/bin/env bash

# Clean Up Temporary Files Script
# This script safely removes temporary files, cache directories, and old logs
# from the AI Trading Bot project

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}🧹 Starting cleanup of temporary files in: ${PROJECT_ROOT}${NC}"
echo ""

# Counter for tracking deletions
TOTAL_DELETED=0
TOTAL_SIZE_FREED=0

# Function to calculate size before deletion
get_size() {
    if [[ -d "$1" ]]; then
        du -sh "$1" 2>/dev/null | cut -f1 || echo "0"
    elif [[ -f "$1" ]]; then
        ls -lh "$1" 2>/dev/null | awk '{print $5}' || echo "0"
    else
        echo "0"
    fi
}

# Function to safely delete files/directories
safe_delete() {
    local path="$1"
    local description="$2"

    if [[ -e "$path" ]]; then
        local size=$(get_size "$path")
        if rm -rf "$path" 2>/dev/null; then
            echo -e "${GREEN}✓ Deleted${NC} $description (freed: $size)"
            ((TOTAL_DELETED++))
        else
            echo -e "${RED}✗ Failed to delete${NC} $description"
        fi
    fi
}

# 1. Find and remove .tmp, .bak, .orig files
echo -e "${YELLOW}1. Cleaning temporary files (.tmp, .bak, .orig)...${NC}"
while IFS= read -r -d '' file; do
    safe_delete "$file" "$(basename "$file")"
done < <(find "$PROJECT_ROOT" -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*.orig" \) \
    -not -path "*/dashboard_venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/venv/*" \
    -not -path "*/.git/*" \
    -print0 2>/dev/null)
echo ""

# 2. Clean up __pycache__ directories
echo -e "${YELLOW}2. Cleaning __pycache__ directories...${NC}"
while IFS= read -r -d '' dir; do
    safe_delete "$dir" "__pycache__ in $(dirname "$dir" | sed "s|$PROJECT_ROOT/||")"
done < <(find "$PROJECT_ROOT" -type d -name "__pycache__" \
    -not -path "*/dashboard_venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/venv/*" \
    -not -path "*/.git/*" \
    -print0 2>/dev/null)
echo ""

# 3. Remove test artifacts in /tmp
echo -e "${YELLOW}3. Cleaning test artifacts in /tmp...${NC}"
# Look for files that might be related to this project
for pattern in "cursorprod" "ai-trading-bot" "trading-bot" "vumanchu"; do
    while IFS= read -r file; do
        if [[ -f "$file" || -d "$file" ]]; then
            safe_delete "$file" "$(basename "$file") from /tmp"
        fi
    done < <(find /tmp -maxdepth 3 -name "*${pattern}*" 2>/dev/null || true)
done
echo ""

# 4. Clear old log files (older than 7 days)
echo -e "${YELLOW}4. Cleaning old log files (>7 days)...${NC}"
if [[ -d "$PROJECT_ROOT/logs" ]]; then
    while IFS= read -r -d '' file; do
        safe_delete "$file" "$(basename "$file") (old log)"
    done < <(find "$PROJECT_ROOT/logs" -name "*.log" -type f -mtime +7 -print0 2>/dev/null)

    # Also clean old JSONL files in logs/trades
    while IFS= read -r -d '' file; do
        safe_delete "$file" "$(basename "$file") (old trade log)"
    done < <(find "$PROJECT_ROOT/logs/trades" -name "*.jsonl" -type f -mtime +7 -print0 2>/dev/null)
fi
echo ""

# 5. Remove any .pyc files
echo -e "${YELLOW}5. Cleaning .pyc files...${NC}"
while IFS= read -r -d '' file; do
    safe_delete "$file" "$(basename "$file")"
done < <(find "$PROJECT_ROOT" -name "*.pyc" -type f \
    -not -path "*/dashboard_venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/venv/*" \
    -not -path "*/.git/*" \
    -print0 2>/dev/null)
echo ""

# 6. Clean Docker dangling images
echo -e "${YELLOW}6. Cleaning Docker dangling images...${NC}"
if command -v docker &> /dev/null; then
    # Count dangling images before pruning
    DANGLING_COUNT=$(docker images -f "dangling=true" -q | wc -l | tr -d ' ')

    if [[ $DANGLING_COUNT -gt 0 ]]; then
        echo -e "${BLUE}Found $DANGLING_COUNT dangling images${NC}"
        if docker image prune -f &> /dev/null; then
            echo -e "${GREEN}✓ Removed${NC} $DANGLING_COUNT dangling Docker images"
            ((TOTAL_DELETED+=$DANGLING_COUNT))
        else
            echo -e "${RED}✗ Failed to prune Docker images${NC}"
        fi
    else
        echo -e "${GREEN}✓ No dangling Docker images found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Docker not found, skipping image cleanup${NC}"
fi
echo ""

# Additional cleanup options (disabled by default for safety)
echo -e "${YELLOW}7. Optional cleanup tasks:${NC}"

# Clean pytest cache
if [[ -d "$PROJECT_ROOT/.pytest_cache" ]]; then
    echo -e "${BLUE}Found .pytest_cache directory${NC}"
    read -p "Delete .pytest_cache? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_delete "$PROJECT_ROOT/.pytest_cache" ".pytest_cache"
    fi
fi

# Clean mypy cache
if [[ -d "$PROJECT_ROOT/.mypy_cache" ]]; then
    echo -e "${BLUE}Found .mypy_cache directory${NC}"
    read -p "Delete .mypy_cache? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_delete "$PROJECT_ROOT/.mypy_cache" ".mypy_cache"
    fi
fi

# Clean ruff cache
if [[ -d "$PROJECT_ROOT/.ruff_cache" ]]; then
    echo -e "${BLUE}Found .ruff_cache directory${NC}"
    read -p "Delete .ruff_cache? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_delete "$PROJECT_ROOT/.ruff_cache" ".ruff_cache"
    fi
fi

# Clean coverage data
if [[ -f "$PROJECT_ROOT/.coverage" ]]; then
    echo -e "${BLUE}Found .coverage file${NC}"
    read -p "Delete .coverage file? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_delete "$PROJECT_ROOT/.coverage" ".coverage"
    fi
fi

# Summary
echo ""
echo -e "${GREEN}================== Cleanup Summary ==================${NC}"
echo -e "${BLUE}Total items deleted:${NC} $TOTAL_DELETED"
echo -e "${BLUE}Project directory:${NC} $PROJECT_ROOT"
echo ""

# Final disk usage report
echo -e "${YELLOW}Current disk usage of common directories:${NC}"
echo -e "Logs:     $(du -sh "$PROJECT_ROOT/logs" 2>/dev/null | cut -f1 || echo "N/A")"
echo -e "Data:     $(du -sh "$PROJECT_ROOT/data" 2>/dev/null | cut -f1 || echo "N/A")"
echo -e "Project:  $(du -sh "$PROJECT_ROOT" 2>/dev/null | cut -f1 || echo "N/A")"

echo ""
echo -e "${GREEN}✨ Cleanup completed!${NC}"

# Safety reminder
echo ""
echo -e "${YELLOW}Safety notes:${NC}"
echo "• Virtual environment directories were preserved"
echo "• Git repository data was preserved"
echo "• Active log files were preserved (only >7 days removed)"
echo "• Configuration files were preserved"
