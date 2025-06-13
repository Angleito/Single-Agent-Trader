#!/usr/bin/env python3
"""
Comprehensive fix for all code quality issues in bot/config_utils.py
"""

import re
from pathlib import Path


def fix_all_issues():
    file_path = Path("bot/config_utils.py")

    with open(file_path) as f:
        content = f.read()

    # Remove unused imports
    content = content.replace("import json\n", "")
    content = re.sub(
        r"from typing import.*Tuple.*\n",
        "from typing import Dict, List, Optional, Any\n",
        content,
    )

    # Fix broken string concatenations from previous script
    # Fix line 46-47
    content = content.replace(
        'issues.append("OpenAI API key is" \n                              "required when using OpenAI provider")',
        'issues.append("OpenAI API key is required when using OpenAI provider")',
    )

    # Fix line 50-51
    content = content.replace(
        'issues.append("Anthropic API key is" \n                              "required when using Anthropic provider")',
        'issues.append(\n                    "Anthropic API key is required when using Anthropic provider")',
    )

    # Fix line 54-55
    content = content.replace(
        'issues.append("Ollama base URL is" \n                              "required when using Ollama provider")',
        'issues.append(\n                    "Ollama base URL is required when using Ollama provider")',
    )

    # Fix line 62-63
    content = content.replace(
        'issues.append("Coinbase API secret is" \n                              "required for live trading")',
        'issues.append(\n                    "Coinbase API secret is required for live trading")',
    )

    # Fix line 65-66
    content = content.replace(
        'issues.append("Coinbase passphrase is" \n                              "required for live trading")',
        'issues.append(\n                    "Coinbase passphrase is required for live trading")',
    )

    # Fix line 71-72
    content = content.replace(
        'issues.append("Production environment should not" \n                              "use dry run mode")',
        'issues.append(\n                    "Production environment should not use dry run mode")',
    )

    # Fix line 74-75
    content = content.replace(
        'issues.append("Production environment should" \n                              "not use sandbox exchange")',
        'issues.append(\n                    "Production environment should not use sandbox exchange")',
    )

    # Fix line 87-88
    content = content.replace(
        'issues.append(f"LLM provider ({self.settings.llm.provider})" \n                          "connectivity failed: {llm_status[\'error\']}")',
        'issues.append(\n                f"LLM provider ({self.settings.llm.provider}) "\n                f"connectivity failed: {llm_status[\'error\']}")',
    )

    # Fix line 94-95
    content = content.replace(
        'issues.append(f"Exchange connectivity" \n                              "failed: {exchange_status[\'error\']}")',
        "issues.append(\n                f\"Exchange connectivity failed: {exchange_status['error']}\")",
    )

    # Fix line 106-107 - the broken f-string
    content = content.replace(
        'issues.append(f"Python 3.8+" \n                          "required, found {sys.version_info.major}.{sys.version_info.minor}")',
        'issues.append(\n                f"Python 3.8+ required, found "\n                f"{sys.version_info.major}.{sys.version_info.minor}")',
    )

    # Fix the broken string concatenations that were created throughout the file
    # We need to fix all the multi-line strings that got broken

    # Fix all remaining line length violations by splitting long lines properly
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if len(line) > 79:
            # Handle various types of long lines

            # Long if conditions
            if "if any(keyword in issue.lower() for keyword in" in line:
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + "if any(keyword in issue.lower()")
                fixed_lines.append(
                    " " * (indent + 7)
                    + "for keyword in ['required', 'failed', 'cannot', 'not installed']):"
                )
                continue

            elif (
                "if any(keyword in issue.lower() for keyword in ['extremely risky']):"
                in line
            ):
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + "if any(keyword in issue.lower()")
                fixed_lines.append(
                    " " * (indent + 7) + "for keyword in ['extremely risky']):"
                )
                continue

            # Long function calls and assignments
            elif (
                "if settings.trading.leverage > 10 and settings.risk.max_daily_loss_pct > 5.0:"
                in line
            ):
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(
                    " " * indent + "if (settings.trading.leverage > 10 and"
                )
                fixed_lines.append(
                    " " * (indent + 8) + "settings.risk.max_daily_loss_pct > 5.0):"
                )
                continue

            elif (
                "risk_reward_ratio = settings.risk.default_take_profit_pct / settings.risk.default_stop_loss_pct"
                in line
            ):
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + "risk_reward_ratio = (")
                fixed_lines.append(
                    " " * (indent + 4) + "settings.risk.default_take_profit_pct /"
                )
                fixed_lines.append(
                    " " * (indent + 4) + "settings.risk.default_stop_loss_pct"
                )
                fixed_lines.append(" " * indent + ")")
                continue

            elif 'issues.append(f"Risk/reward ratio of {risk_reward_ratio:.2f}' in line:
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + "issues.append(")
                fixed_lines.append(
                    " " * (indent + 4)
                    + 'f"Risk/reward ratio of {risk_reward_ratio:.2f} "'
                )
                fixed_lines.append(
                    " " * (indent + 4) + 'f"may not be profitable long-term"'
                )
                fixed_lines.append(" " * indent + ")")
                continue

            # Handle very long variable assignments and other cases
            elif '= f"' in line and len(line) > 79:
                # Split f-string assignments
                indent = len(line) - len(line.lstrip())
                equals_pos = line.find("=")
                var_part = line[:equals_pos].strip()
                value_part = line[equals_pos + 1 :].strip()

                if len(var_part) + len(value_part) > 70:  # Need to split
                    fixed_lines.append(" " * indent + var_part + " = (")
                    fixed_lines.append(" " * (indent + 4) + value_part)
                    fixed_lines.append(" " * indent + ")")
                    continue

            # Generic line splitting for remaining cases
            elif len(line) > 79:
                # Try to split at logical points
                if ", " in line and not line.strip().startswith("#"):
                    # Split at comma
                    parts = line.split(", ")
                    if len(parts) > 1:
                        indent = len(line) - len(line.lstrip())
                        first_part = parts[0] + ","
                        if len(first_part) <= 79:
                            fixed_lines.append(first_part)
                            remaining = ", ".join(parts[1:])
                            if len(" " * (indent + 4) + remaining) <= 79:
                                fixed_lines.append(" " * (indent + 4) + remaining)
                                continue

                # If we can't split nicely, keep as is for now
                # The critical errors will be handled by specific patterns above

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    # Remove all trailing whitespace from blank lines
    content = re.sub(r"^\s+$", "", content, flags=re.MULTILINE)

    # Fix the f-string without placeholders on line 998
    content = content.replace(
        'logger.info(f"Configuration loaded successfully:")',
        'logger.info("Configuration loaded successfully:")',
    )

    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"

    # Write back to file
    with open(file_path, "w") as f:
        f.write(content)

    print("Comprehensive fix applied to config_utils.py")


if __name__ == "__main__":
    fix_all_issues()
