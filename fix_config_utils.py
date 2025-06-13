#!/usr/bin/env python3
"""
Script to fix all code quality issues in bot/config_utils.py
This will fix line length, missing newline, and improve code formatting.
"""

from pathlib import Path


def fix_file():
    file_path = Path("bot/config_utils.py")

    with open(file_path) as f:
        content = f.read()

    # Already fixed issues that don't need changes:
    # - Removed unused imports json and Tuple
    # - Removed trailing whitespace
    # - Fixed f-string issue

    # Fix remaining line length violations by splitting long lines
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if len(line) > 79:
            # Handle specific long line patterns
            if "issues.append(" in line and len(line) > 79:
                # Split long append statements
                indent = len(line) - len(line.lstrip())
                if '"' in line:
                    # Find the quoted string and split it
                    start_quote = line.find('"')
                    if start_quote != -1:
                        before_quote = line[: start_quote + 1]
                        after_quote = line[start_quote + 1 :]
                        end_quote = after_quote.find('"')
                        if end_quote != -1:
                            quote_content = after_quote[:end_quote]
                            after_end = after_quote[end_quote:]

                            # Split the quoted content
                            if len(before_quote + quote_content + after_end) > 79:
                                words = quote_content.split()
                                if len(words) > 1:
                                    mid = len(words) // 2
                                    first_part = " ".join(words[:mid])
                                    second_part = " ".join(words[mid:])
                                    fixed_lines.append(before_quote + first_part + '" ')
                                    fixed_lines.append(
                                        " " * (indent + 14)
                                        + '"'
                                        + second_part
                                        + after_end
                                    )
                                    continue

            elif "if any(" in line and len(line) > 79:
                # Split long if statements with any()
                indent = len(line) - len(line.lstrip())
                if "for keyword in" in line:
                    parts = line.split("for keyword in")
                    if len(parts) == 2:
                        first_part = parts[0] + "for keyword in"
                        second_part = "   " + parts[1].strip()
                        fixed_lines.append(first_part)
                        fixed_lines.append(" " * (indent + 7) + second_part)
                        continue

            elif 'f"' in line and len(line) > 79:
                # Split long f-strings
                indent = len(line) - len(line.lstrip())
                if "connectivity failed:" in line:
                    fixed_lines.append(
                        line.replace(
                            "f\"LLM provider ({self.settings.llm.provider}) connectivity failed: {llm_status['error']}\"",
                            'f"LLM provider ({self.settings.llm.provider}) "\n'
                            + " " * (indent + 14)
                            + "f\"connectivity failed: {llm_status['error']}\"",
                        )
                    )
                    continue
                elif "Exchange connectivity failed:" in line:
                    fixed_lines.append(
                        line.replace(
                            "f\"Exchange connectivity failed: {exchange_status['error']}\"",
                            'f"Exchange connectivity failed: "\n'
                            + " " * (indent + 14)
                            + "f\"{exchange_status['error']}\"",
                        )
                    )
                    continue
                elif "Python 3.8+ required, found" in line:
                    fixed_lines.append(
                        line.replace(
                            'f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}"',
                            'f"Python 3.8+ required, found "\n'
                            + " " * (indent + 14)
                            + 'f"{sys.version_info.major}."\n'
                            + " " * (indent + 14)
                            + 'f"{sys.version_info.minor}"',
                        )
                    )
                    continue

            elif "headers = {" in line and len(line) > 79:
                # Split long header definitions
                indent = len(line) - len(line.lstrip())
                if "Authorization" in line:
                    fixed_lines.append(" " * indent + "headers = {")
                    fixed_lines.append(
                        " " * (indent + 4)
                        + '"Authorization": f"Bearer {self.settings.llm.openai_api_key}"'
                    )
                    fixed_lines.append(" " * indent + "}")
                    continue
                elif "x-api-key" in line:
                    fixed_lines.append(" " * indent + "headers = {")
                    fixed_lines.append(
                        " " * (indent + 4)
                        + '"x-api-key": str(self.settings.llm.anthropic_api_key)'
                    )
                    fixed_lines.append(" " * indent + "}")
                    continue

            elif "response = requests.get(" in line and len(line) > 79:
                # Split long requests.get calls
                indent = len(line) - len(line.lstrip())
                if "ollama_base_url" in line:
                    fixed_lines.append(" " * indent + "response = requests.get(")
                    fixed_lines.append(
                        " " * (indent + 4)
                        + 'f"{self.settings.llm.ollama_base_url}/api/tags",'
                    )
                    fixed_lines.append(" " * (indent + 4) + "timeout=10")
                    fixed_lines.append(" " * indent + ")")
                    continue

            elif "return {" in line and '"Connectivity test skipped' in line:
                # Split long return statements
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + "return {")
                fixed_lines.append(" " * (indent + 4) + '"success": True,')
                fixed_lines.append(
                    " " * (indent + 4) + '"error": "Connectivity test skipped "'
                )
                fixed_lines.append(
                    " " * (indent + 13) + '"(requests module not available)"'
                )
                fixed_lines.append(" " * indent + "}")
                continue

            elif (
                "if self.settings.exchange.cb_api_key and self.settings.exchange.cb_api_secret:"
                in line
            ):
                # Split long if conditions
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(
                    " " * indent + "if (self.settings.exchange.cb_api_key and"
                )
                fixed_lines.append(
                    " " * (indent + 8) + "self.settings.exchange.cb_api_secret):"
                )
                continue

            elif "if memory.available < 512 * 1024 * 1024:" in line:
                # Split long memory check
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(
                    " " * indent + "if memory.available < 512 * 1024 * 1024:  # 512MB"
                )
                continue

            elif 'return "Low available memory' in line:
                # Split long return statements
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(
                    " " * indent + 'return ("Low available memory (<512MB). "'
                )
                fixed_lines.append(
                    " " * (indent + 8) + '"Performance may be affected.")'
                )
                continue

            elif '"python_version": f"' in line and len(line) > 79:
                # Split long python version string
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(" " * indent + '"python_version": (')
                fixed_lines.append(" " * (indent + 4) + 'f"{sys.version_info.major}."')
                fixed_lines.append(" " * (indent + 4) + 'f"{sys.version_info.minor}."')
                fixed_lines.append(" " * (indent + 4) + 'f"{sys.version_info.micro}"')
                fixed_lines.append(" " * indent + "),")
                continue

            # Add more specific patterns as needed...
            # For now, if we haven't handled the line specifically, keep it as is
            # The linter will catch remaining issues

        fixed_lines.append(line)

    # Join lines back together
    content = "\n".join(fixed_lines)

    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"

    # Write back to file
    with open(file_path, "w") as f:
        f.write(content)

    print("Fixed config_utils.py")


if __name__ == "__main__":
    fix_file()
