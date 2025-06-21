#!/usr/bin/env python3
"""Script to identify and fix logging format type mismatches in the codebase."""

import re
from pathlib import Path


def find_logging_issues(file_path: Path) -> list[tuple[int, str, str]]:
    """Find potential logging format issues in a file."""
    issues = []

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    patterns = [
        # %d with potential string values
        (r'logger\.\w+\(".*%d.*",.*getattr\(', "Potential string from getattr with %d"),
        (
            r'logger\.\w+\(".*%d.*",.*\blen\(.*\).*,.*[^)]',
            "%d with len() followed by non-numeric",
        ),
        (r'logger\.\w+\(".*%s.*",.*\blen\(', "Should use %d for len()"),
        (
            r'logger\.\w+\(".*%s.*",.*\.status[^_]',
            "HTTP status might be int, consider %d",
        ),
        (
            r'logger\.\w+\(".*%s.*",.*\.(ttl_seconds|max_entries|cleanup_interval|poll_interval)',
            "Config value might need int conversion",
        ),
        # Float formatting
        (
            r'logger\.\w+\(".*%.?\d?f.*",.*[^0-9.]',
            "Float format with potential non-numeric",
        ),
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern, description in patterns:
            if re.search(pattern, line):
                issues.append((line_num, line.strip(), description))

    return issues


def scan_codebase():
    """Scan the entire bot/ directory for logging issues."""
    bot_dir = Path(__file__).parent.parent / "bot"

    all_issues = {}

    for py_file in bot_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        issues = find_logging_issues(py_file)
        if issues:
            all_issues[py_file] = issues

    return all_issues


def main():
    """Main function to scan and report logging issues."""
    print("Scanning for logging format issues...")

    issues = scan_codebase()

    if not issues:
        print("No logging format issues found!")
        return

    print(f"\nFound issues in {len(issues)} files:\n")

    for file_path, file_issues in issues.items():
        rel_path = file_path.relative_to(Path.cwd())
        print(f"\n{rel_path}:")
        for line_num, line, description in file_issues:
            print(f"  Line {line_num}: {description}")
            print(f"    {line[:100]}{'...' if len(line) > 100 else ''}")

    print(f"\nTotal issues found: {sum(len(v) for v in issues.values())}")

    # Generate fix suggestions
    print("\n\nSuggested fixes:")
    print("1. For len() with %s: Change %s to %d")
    print("2. For config values: Wrap with int() or float() as appropriate")
    print("3. For HTTP status: Check if response.status is int, use %d if so")
    print("4. For getattr with defaults: Ensure default values match expected type")


if __name__ == "__main__":
    main()
