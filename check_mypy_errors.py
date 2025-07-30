#!/usr/bin/env python3
"""
Script to analyze type errors in bot/validation/pipeline.py
"""

import ast


def analyze_file():
    """Analyze the pipeline.py file for potential type issues."""

    # Read the file
    with open("bot/validation/pipeline.py") as f:
        content = f.read()

    # Parse AST
    tree = ast.parse(content)

    print("=== Type Analysis for bot/validation/pipeline.py ===\n")

    # Find function definitions without type annotations
    print("Functions without complete type annotations:")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has return annotation
            has_return_type = node.returns is not None

            # Check if all parameters have annotations
            all_params_typed = all(
                arg.annotation is not None
                for arg in node.args.args
                if arg.arg != "self"
            )

            if not has_return_type or not all_params_typed:
                print(f"  - {node.name} (line {node.lineno})")
                if not has_return_type:
                    print("    Missing return type annotation")
                if not all_params_typed:
                    untyped_params = [
                        arg.arg
                        for arg in node.args.args
                        if arg.annotation is None and arg.arg != "self"
                    ]
                    print(f"    Missing parameter annotations: {untyped_params}")

    print("\n" + "=" * 50 + "\n")

    # Find class methods without type annotations
    print("Class methods needing type annotations:")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name

                    # Skip special methods like __init__
                    if method_name.startswith("_") and method_name.endswith("_"):
                        continue

                    has_return_type = item.returns is not None
                    all_params_typed = all(
                        arg.annotation is not None
                        for arg in item.args.args
                        if arg.arg != "self"
                    )

                    if not has_return_type or not all_params_typed:
                        print(f"  - {class_name}.{method_name} (line {item.lineno})")

    print("\n" + "=" * 50 + "\n")

    # Find generic usage that might need fixing
    print("Potential generic type issues:")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in ["dict", "list", "tuple", "set"]:
                print(
                    f"  - Found '{node.id}' usage at line {node.lineno} (should use Dict, List, etc. from typing)"
                )

    print("\n" + "=" * 50 + "\n")

    # Find attribute access on unions
    print("Potential union attribute access issues:")
    print(
        "  - Look for patterns like 'result.success()' or 'result.failure()' without type guards"
    )
    print("  - These need proper type checking with is_success()/is_failure() first")


if __name__ == "__main__":
    analyze_file()
