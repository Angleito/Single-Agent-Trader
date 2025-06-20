#!/usr/bin/env python3
"""
Comprehensive script to fix all G004 linting errors in the codebase.
Converts f-string logging statements to % formatting.
"""

import re
import subprocess
from pathlib import Path


def get_g004_files() -> list[str]:
    """Get all files with G004 errors using ruff."""
    try:
        result = subprocess.run(
            [
                "poetry",
                "run",
                "ruff",
                "check",
                ".",
                "--select",
                "G004",
                "--output-format",
                "concise",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=False,
        )

        files_with_errors = set()
        for line in result.stdout.split("\n"):
            if "G004" in line and ":" in line:
                file_path = line.split(":")[0]
                files_with_errors.add(file_path)

        return sorted(list(files_with_errors))
    except Exception as e:
        print(f"Error running ruff: {e}")
        return []


def fix_simple_f_string_logging(content: str) -> str:
    """Fix simple single-line f-string logging statements."""

    # Pattern for single-line logger calls with f-strings
    patterns = [
        # logger.info(f"text {var}")
        (
            r'(logger\.(info|warning|error|debug|exception|critical))\s*\(\s*f"([^"]*?)"\s*\)',
            r"\1",
        ),
        # logger.info(f'text {var}')
        (
            r"(logger\.(info|warning|error|debug|exception|critical))\s*\(\s*f'([^']*?)'\s*\)",
            r"\1",
        ),
    ]

    def replace_f_string(match):
        logger_call = match.group(1)
        log_level = match.group(2)
        f_string_content = match.group(3)

        # Find all {var} patterns
        variables = []
        var_pattern = r"\{([^}]+?)\}"

        # Extract variables and replace with %s
        def replace_var(var_match):
            var_expr = var_match.group(1)
            variables.append(var_expr)
            return "%s"

        new_string = re.sub(var_pattern, replace_var, f_string_content)

        if variables:
            var_list = ", ".join(variables)
            return f'{logger_call}("{new_string}", {var_list})'
        else:
            return f'{logger_call}("{new_string}")'

    # Apply the replacements
    for pattern, _ in patterns:
        content = re.sub(pattern, replace_f_string, content)

    return content


def fix_multiline_f_string_logging(content: str) -> str:
    """Fix multi-line f-string logging statements."""
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for logger calls followed by f-strings
        logger_match = re.search(
            r"(\s*)(logger\.(info|warning|error|debug|exception|critical))\s*\(\s*$",
            line,
        )
        if logger_match and i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            # Check if next line starts with f"
            if next_line.startswith('f"') or next_line.startswith("f'"):
                indent = logger_match.group(1)
                logger_func = logger_match.group(2)

                # Collect all lines until closing parenthesis
                content_lines = []
                j = i + 1
                paren_count = 0

                while j < len(lines):
                    current_line = lines[j]
                    content_lines.append(current_line.strip())

                    # Count parentheses (simple approach)
                    paren_count += current_line.count("(") - current_line.count(")")
                    if paren_count <= 0 and ")" in current_line:
                        break
                    j += 1

                if j < len(lines):
                    # Join content and process
                    full_content = " ".join(content_lines)

                    # Remove f-string prefix
                    full_content = re.sub(r'f(["\'])', r"\1", full_content)

                    # Find variables
                    variables = []
                    var_pattern = r"\{([^}]+?)\}"

                    def replace_var(var_match):
                        var_expr = var_match.group(1)
                        variables.append(var_expr)
                        return "%s"

                    new_content = re.sub(var_pattern, replace_var, full_content)

                    # Reconstruct the logger call
                    if variables:
                        var_list = ", ".join(variables)
                        new_call = f"{indent}{logger_func}({new_content}, {var_list})"
                    else:
                        new_call = f"{indent}{logger_func}({new_content})"

                    result_lines.append(new_call)
                    i = j + 1
                    continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def fix_file(file_path: str) -> bool:
    """Fix a single file's G004 errors."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return False

        content = path.read_text(encoding="utf-8")
        original_content = content

        # Apply fixes
        content = fix_simple_f_string_logging(content)
        content = fix_multiline_f_string_logging(content)

        # Only write if changes were made
        if content != original_content:
            path.write_text(content, encoding="utf-8")
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⚪ No changes: {file_path}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix all G004 errors."""
    print("🔍 Finding files with G004 errors...")
    files_with_errors = get_g004_files()

    if not files_with_errors:
        print("✅ No G004 errors found!")
        return

    print(f"📁 Found {len(files_with_errors)} files with G004 errors")

    fixed_count = 0
    for file_path in files_with_errors:
        if fix_file(file_path):
            fixed_count += 1

    print("\n📊 Summary:")
    print(f"  Total files processed: {len(files_with_errors)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files unchanged: {len(files_with_errors) - fixed_count}")

    # Check remaining errors
    print("\n🔍 Checking remaining G004 errors...")
    try:
        result = subprocess.run(
            [
                "poetry",
                "run",
                "ruff",
                "check",
                ".",
                "--select",
                "G004",
                "--output-format",
                "concise",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=False,
        )

        remaining_errors = len(
            [line for line in result.stdout.split("\n") if "G004" in line]
        )
        print(f"🎯 Remaining G004 errors: {remaining_errors}")

        if remaining_errors == 0:
            print("🎉 All G004 errors fixed!")
        else:
            print("⚠️  Some complex cases may need manual fixing")

    except Exception as e:
        print(f"Error checking remaining errors: {e}")


if __name__ == "__main__":
    main()
