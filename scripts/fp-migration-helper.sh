#!/bin/bash

# Functional Programming Migration Helper Script
# Assists with converting legacy imperative code to functional programming patterns

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "MIGRATION") echo -e "${PURPLE}🔄 $2${NC}" ;;
        "ANALYZE") echo -e "${CYAN}🔍 $2${NC}" ;;
    esac
}

echo "🔄 Functional Programming Migration Helper"
echo "========================================"
echo ""

# Configuration
MIGRATION_LOG_DIR="logs/fp/migration"
MIGRATION_DATA_DIR="data/fp/migration_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MIGRATION_REPORT="$MIGRATION_DATA_DIR/migration_analysis_$TIMESTAMP.json"
BACKUP_DIR="backups/fp_migration_$TIMESTAMP"

# Ensure directories exist
mkdir -p "$MIGRATION_LOG_DIR" "$MIGRATION_DATA_DIR" "$BACKUP_DIR"

# Initialize migration report
cat > "$MIGRATION_REPORT" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "analysis": {},
    "recommendations": {},
    "migration_candidates": {},
    "risks": {},
    "progress": {
        "total_files": 0,
        "analyzed": 0,
        "migration_ready": 0,
        "complex_migrations": 0,
        "blocked": 0
    }
}
EOF

# Function to update migration report
update_migration_report() {
    local section="$1"
    local key="$2"
    local value="$3"

    python3 -c "
import json
import sys
try:
    with open('$MIGRATION_REPORT', 'r') as f:
        report = json.load(f)

    if '$section' not in report:
        report['$section'] = {}

    report['$section']['$key'] = '$value'

    with open('$MIGRATION_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except Exception as e:
    print(f'Error updating report: {e}', file=sys.stderr)
    "
}

# Function to analyze code for migration readiness
analyze_migration_readiness() {
    local file_path="$1"
    local analysis_output="$MIGRATION_LOG_DIR/analysis_$(basename "$file_path" .py).json"

    python3 -c "
import ast
import json
import sys
from pathlib import Path

def analyze_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        analysis = {
            'file': str(file_path),
            'migration_score': 0,
            'complexity': 'unknown',
            'issues': [],
            'recommendations': [],
            'patterns': {
                'exceptions': 0,
                'none_returns': 0,
                'async_functions': 0,
                'side_effects': 0,
                'mutable_operations': 0
            }
        }

        # Analyze AST for migration complexity
        for node in ast.walk(tree):
            # Exception handling patterns
            if isinstance(node, ast.Raise):
                analysis['patterns']['exceptions'] += 1
                analysis['issues'].append('Uses exception raising (convert to Result type)')

            if isinstance(node, ast.ExceptHandler):
                analysis['issues'].append('Exception handling found (consider Result/Either)')

            # Return None patterns
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None:
                analysis['patterns']['none_returns'] += 1
                analysis['issues'].append('Returns None (convert to Maybe type)')

            # Async patterns
            if isinstance(node, ast.AsyncFunctionDef):
                analysis['patterns']['async_functions'] += 1
                analysis['issues'].append('Async function (convert to IO monad)')

            # Side effects (print, file operations, etc.)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['print', 'open', 'input']:
                    analysis['patterns']['side_effects'] += 1
                    analysis['issues'].append(f'Side effect: {node.func.id} (wrap in IO)')

            # Mutable operations
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                analysis['patterns']['mutable_operations'] += 1

        # Calculate migration score
        total_issues = sum(analysis['patterns'].values())
        if total_issues == 0:
            analysis['migration_score'] = 100
            analysis['complexity'] = 'easy'
            analysis['recommendations'].append('Ready for immediate migration')
        elif total_issues <= 3:
            analysis['migration_score'] = 80
            analysis['complexity'] = 'medium'
            analysis['recommendations'].append('Minor refactoring needed before migration')
        elif total_issues <= 8:
            analysis['migration_score'] = 50
            analysis['complexity'] = 'complex'
            analysis['recommendations'].append('Significant refactoring required')
        else:
            analysis['migration_score'] = 20
            analysis['complexity'] = 'hard'
            analysis['recommendations'].append('Consider gradual migration with adapters')

        # Specific recommendations
        if analysis['patterns']['exceptions'] > 0:
            analysis['recommendations'].append('Replace exceptions with Result[T, Error] types')

        if analysis['patterns']['none_returns'] > 0:
            analysis['recommendations'].append('Replace None returns with Maybe[T] types')

        if analysis['patterns']['async_functions'] > 0:
            analysis['recommendations'].append('Wrap async operations in IO monad')

        if analysis['patterns']['side_effects'] > 0:
            analysis['recommendations'].append('Isolate side effects using IO and effects system')

        return analysis

    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'migration_score': 0,
            'complexity': 'error'
        }

# Analyze the file
result = analyze_file('$file_path')

# Save analysis
with open('$analysis_output', 'w') as f:
    json.dump(result, f, indent=2)

# Print summary
print(f\"File: {result['file']}\")
print(f\"Migration Score: {result['migration_score']}/100\")
print(f\"Complexity: {result['complexity']}\")
print(f\"Issues Found: {len(result.get('issues', []))}\")

# Exit with score-based code
if result['migration_score'] >= 80:
    sys.exit(0)  # Easy migration
elif result['migration_score'] >= 50:
    sys.exit(1)  # Medium complexity
else:
    sys.exit(2)  # Hard migration
    "
}

# Function to generate migration code suggestions
generate_migration_suggestions() {
    local file_path="$1"
    local suggestions_file="$MIGRATION_LOG_DIR/suggestions_$(basename "$file_path" .py).py"

    python3 -c "
import ast
import sys
from pathlib import Path

def generate_fp_version(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Generate FP version suggestions
        fp_suggestions = []
        fp_suggestions.append('# Functional Programming Migration Suggestions')
        fp_suggestions.append(f'# Original file: {file_path}')
        fp_suggestions.append('# Generated by fp-migration-helper.sh')
        fp_suggestions.append('')
        fp_suggestions.append('# Required FP imports:')
        fp_suggestions.append('from bot.fp.types.result import Result, Success, Failure')
        fp_suggestions.append('from bot.fp.types.base import Maybe, Some, Nothing')
        fp_suggestions.append('from bot.fp.core.io import IO')
        fp_suggestions.append('from bot.fp.core.either import Either, Left, Right')
        fp_suggestions.append('')

        # Analyze original code and suggest FP patterns
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fp_suggestions.append(f'# Original function: {node.name}')

                # Check for exception patterns
                has_raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                has_none_return = any(
                    isinstance(n, ast.Return) and
                    isinstance(n.value, ast.Constant) and
                    n.value.value is None
                    for n in ast.walk(node)
                )

                if has_raises:
                    fp_suggestions.append(f'def {node.name}_fp(*args, **kwargs) -> Result[T, str]:')
                    fp_suggestions.append('    \"\"\"FP version using Result instead of exceptions.\"\"\"')
                    fp_suggestions.append('    try:')
                    fp_suggestions.append(f'        result = original_{node.name}(*args, **kwargs)')
                    fp_suggestions.append('        return Success(result)')
                    fp_suggestions.append('    except Exception as e:')
                    fp_suggestions.append('        return Failure(str(e))')
                elif has_none_return:
                    fp_suggestions.append(f'def {node.name}_fp(*args, **kwargs) -> Maybe[T]:')
                    fp_suggestions.append('    \"\"\"FP version using Maybe instead of None.\"\"\"')
                    fp_suggestions.append(f'    result = original_{node.name}(*args, **kwargs)')
                    fp_suggestions.append('    return Some(result) if result is not None else Nothing()')
                else:
                    fp_suggestions.append(f'def {node.name}_fp(*args, **kwargs) -> IO[T]:')
                    fp_suggestions.append('    \"\"\"FP version wrapped in IO monad.\"\"\"')
                    fp_suggestions.append(f'    return IO.pure(original_{node.name}(*args, **kwargs))')

                fp_suggestions.append('')

        # Additional migration patterns
        fp_suggestions.append('# Common migration patterns:')
        fp_suggestions.append('#')
        fp_suggestions.append('# 1. Exception handling -> Result type:')
        fp_suggestions.append('#    try: ... except: ... -> Result[T, Error]')
        fp_suggestions.append('#')
        fp_suggestions.append('# 2. None returns -> Maybe type:')
        fp_suggestions.append('#    return None -> Nothing()')
        fp_suggestions.append('#    return value -> Some(value)')
        fp_suggestions.append('#')
        fp_suggestions.append('# 3. Side effects -> IO monad:')
        fp_suggestions.append('#    print(...) -> IO.of(lambda: print(...))')
        fp_suggestions.append('#    file operations -> IO operations')
        fp_suggestions.append('#')
        fp_suggestions.append('# 4. State mutations -> immutable data:')
        fp_suggestions.append('#    x = x + 1 -> new_x = x + 1')
        fp_suggestions.append('#    list.append(x) -> new_list = list + [x]')

        return '\\n'.join(fp_suggestions)

    except Exception as e:
        return f'# Error generating suggestions: {e}'

# Generate and save suggestions
suggestions = generate_fp_version('$file_path')
with open('$suggestions_file', 'w') as f:
    f.write(suggestions)

print(f'Suggestions saved to: $suggestions_file')
    "
}

# Function to create backup
create_backup() {
    local file_path="$1"
    local backup_path="$BACKUP_DIR/$(basename "$file_path")"

    cp "$file_path" "$backup_path"
    print_status "INFO" "Backup created: $backup_path"
}

# Main migration analysis function
analyze_codebase() {
    print_status "ANALYZE" "Scanning codebase for migration candidates..."

    # Find all Python files in bot/ excluding bot/fp/
    local candidates=()
    while IFS= read -r -d '' file; do
        candidates+=("$file")
    done < <(find bot/ -name "*.py" -not -path "bot/fp/*" -not -path "bot/__pycache__/*" -print0)

    print_status "INFO" "Found ${#candidates[@]} Python files to analyze"

    local easy_migrations=0
    local medium_migrations=0
    local hard_migrations=0
    local total_analyzed=0

    echo "Migration Analysis Report" > "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "========================" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Generated: $(date)" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"

    for file in "${candidates[@]}"; do
        if [[ "$file" == *"__init__.py" ]] || [[ "$file" == *"__pycache__"* ]]; then
            continue
        fi

        print_status "ANALYZE" "Analyzing $(basename "$file")..."

        if analyze_migration_readiness "$file"; then
            case $? in
                0)
                    easy_migrations=$((easy_migrations + 1))
                    echo "EASY: $file" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
                    ;;
                1)
                    medium_migrations=$((medium_migrations + 1))
                    echo "MEDIUM: $file" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
                    ;;
                *)
                    hard_migrations=$((hard_migrations + 1))
                    echo "HARD: $file" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
                    ;;
            esac
        fi

        total_analyzed=$((total_analyzed + 1))
    done

    # Generate summary
    echo "" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Summary:" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Total files analyzed: $total_analyzed" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Easy migrations: $easy_migrations" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Medium complexity: $medium_migrations" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"
    echo "Hard migrations: $hard_migrations" >> "$MIGRATION_LOG_DIR/analysis_summary.txt"

    print_status "SUCCESS" "Analysis complete!"
    print_status "INFO" "Easy migrations: $easy_migrations"
    print_status "INFO" "Medium complexity: $medium_migrations"
    print_status "INFO" "Hard migrations: $hard_migrations"

    # Update progress in main report
    python3 -c "
import json
try:
    with open('$MIGRATION_REPORT', 'r') as f:
        report = json.load(f)

    report['progress']['total_files'] = $total_analyzed
    report['progress']['analyzed'] = $total_analyzed
    report['progress']['migration_ready'] = $easy_migrations
    report['progress']['complex_migrations'] = $medium_migrations
    report['progress']['blocked'] = $hard_migrations

    with open('$MIGRATION_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except Exception as e:
    print(f'Error updating progress: {e}')
    "
}

# Function to migrate specific file
migrate_file() {
    local file_path="$1"
    local force="$2"

    if [ ! -f "$file_path" ]; then
        print_status "ERROR" "File not found: $file_path"
        return 1
    fi

    print_status "MIGRATION" "Migrating $file_path..."

    # Create backup
    create_backup "$file_path"

    # Analyze migration readiness
    analyze_migration_readiness "$file_path"
    local complexity_exit_code=$?

    if [ $complexity_exit_code -eq 2 ] && [ "$force" != "--force" ]; then
        print_status "WARNING" "File has high migration complexity. Use --force to proceed anyway."
        print_status "INFO" "Run: $0 migrate $file_path --force"
        return 1
    fi

    # Generate migration suggestions
    generate_migration_suggestions "$file_path"

    print_status "SUCCESS" "Migration analysis complete for $file_path"
    print_status "INFO" "Check logs/fp/migration/ for detailed analysis and suggestions"

    return 0
}

# Interactive migration mode
interactive_migration() {
    print_status "MIGRATION" "Starting interactive migration mode..."

    # Get list of easy migration candidates
    local easy_files=()
    while IFS= read -r -d '' file; do
        if analyze_migration_readiness "$file" &>/dev/null && [ $? -eq 0 ]; then
            easy_files+=("$file")
        fi
    done < <(find bot/ -name "*.py" -not -path "bot/fp/*" -not -path "*__pycache__*" -print0)

    if [ ${#easy_files[@]} -eq 0 ]; then
        print_status "INFO" "No easy migration candidates found. Run analysis first."
        return 0
    fi

    print_status "INFO" "Found ${#easy_files[@]} easy migration candidates:"
    for i in "${!easy_files[@]}"; do
        echo "  $((i+1)). ${easy_files[$i]}"
    done

    echo ""
    read -p "Select file to migrate (1-${#easy_files[@]}, 'a' for all, 'q' to quit): " choice

    case "$choice" in
        q|Q)
            print_status "INFO" "Migration cancelled"
            return 0
            ;;
        a|A)
            for file in "${easy_files[@]}"; do
                migrate_file "$file"
            done
            ;;
        [1-9]*)
            if [ "$choice" -le "${#easy_files[@]}" ] && [ "$choice" -gt 0 ]; then
                migrate_file "${easy_files[$((choice-1))]}"
            else
                print_status "ERROR" "Invalid selection"
                return 1
            fi
            ;;
        *)
            print_status "ERROR" "Invalid choice"
            return 1
            ;;
    esac
}

# Main script logic
case "${1:-help}" in
    "analyze"|"analyse")
        analyze_codebase
        ;;
    "migrate")
        if [ -z "$2" ]; then
            interactive_migration
        else
            migrate_file "$2" "$3"
        fi
        ;;
    "suggestions")
        if [ -z "$2" ]; then
            print_status "ERROR" "Please specify a file: $0 suggestions <file>"
            exit 1
        fi
        generate_migration_suggestions "$2"
        ;;
    "backup")
        if [ -z "$2" ]; then
            print_status "ERROR" "Please specify a file: $0 backup <file>"
            exit 1
        fi
        create_backup "$2"
        ;;
    "status"|"report")
        if [ -f "$MIGRATION_LOG_DIR/analysis_summary.txt" ]; then
            cat "$MIGRATION_LOG_DIR/analysis_summary.txt"
        else
            print_status "WARNING" "No analysis report found. Run: $0 analyze"
        fi
        ;;
    "clean")
        print_status "INFO" "Cleaning migration logs and temporary files..."
        rm -rf "$MIGRATION_LOG_DIR"/* "$MIGRATION_DATA_DIR"/*
        print_status "SUCCESS" "Migration logs cleaned"
        ;;
    "help"|*)
        echo "FP Migration Helper - Usage:"
        echo ""
        echo "  $0 analyze              # Analyze entire codebase for migration readiness"
        echo "  $0 migrate [file]       # Migrate specific file or interactive mode"
        echo "  $0 migrate file --force # Force migration of complex file"
        echo "  $0 suggestions <file>   # Generate FP conversion suggestions"
        echo "  $0 backup <file>        # Create backup of file"
        echo "  $0 status               # Show migration analysis report"
        echo "  $0 clean                # Clean migration logs"
        echo "  $0 help                 # Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 analyze                           # Scan all code"
        echo "  $0 migrate                           # Interactive migration"
        echo "  $0 migrate bot/strategy/core.py      # Migrate specific file"
        echo "  $0 suggestions bot/risk/manager.py   # Get FP suggestions"
        echo ""
        echo "Migration Complexity:"
        echo "  Easy    - Ready for immediate FP conversion"
        echo "  Medium  - Minor refactoring needed"
        echo "  Hard    - Significant changes required"
        echo ""
        ;;
esac

print_status "MIGRATION" "Migration helper complete!"
print_status "INFO" "Logs: $MIGRATION_LOG_DIR"
print_status "INFO" "Reports: $MIGRATION_DATA_DIR"
print_status "INFO" "Backups: $BACKUP_DIR"
