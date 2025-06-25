"""Log format string parser for automatic type conversion."""

import re
from typing import Any


class FormatSpec:
    """Parse and validate format specifiers."""

    # Regex patterns for different format specifiers
    PATTERNS = {
        r"%[+-]?\d*\.?\d*[dioxX]": int,  # Integer formats
        r"%[+-]?\d*\.?\d*[fFeEgG]": float,  # Float formats
        r"%[+-]?\d*\.?\d*[sc]": str,  # String formats
    }

    @classmethod
    def parse_format(cls, fmt: str) -> list[type[Any]]:
        """
        Extract expected types from format string.

        Args:
            fmt: Format string with % specifiers

        Returns:
            List of expected types in order
        """
        # Track position in string to maintain order
        format_positions = []

        for pattern, expected_type in cls.PATTERNS.items():
            for match in re.finditer(pattern, fmt):
                format_positions.append((match.start(), expected_type))

        # Sort by position to maintain order
        format_positions.sort(key=lambda x: x[0])
        return [typ for _, typ in format_positions]

    @classmethod
    def convert_args(cls, fmt: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """
        Convert arguments to match format string types.

        Args:
            fmt: Format string
            args: Arguments to convert

        Returns:
            Tuple of converted arguments
        """
        expected_types = cls.parse_format(fmt)
        converted: list[Any] = []

        # Convert each argument to its expected type
        for i, arg in enumerate(args):
            if i < len(expected_types):
                expected = expected_types[i]
                try:
                    if expected == int:
                        # Convert through float to handle decimal strings
                        converted.append(int(float(str(arg))))
                    elif expected == float:
                        converted.append(float(str(arg)))
                    else:
                        converted.append(str(arg))
                except (ValueError, TypeError):
                    # If conversion fails, keep as string
                    converted.append(str(arg))
            else:
                # No format specifier for this arg, keep as-is
                converted.append(arg)

        return tuple(converted)

    @classmethod
    def validate_args(cls, fmt: str, args: tuple[Any, ...]) -> bool:
        """
        Validate that arguments match format string.

        Args:
            fmt: Format string
            args: Arguments to validate

        Returns:
            True if arguments match format string
        """
        expected_types = cls.parse_format(fmt)

        # Check if we have the right number of arguments
        if len(args) != len(expected_types):
            return False

        # Check if each argument can be converted to expected type
        for arg, expected in zip(args, expected_types, strict=False):
            try:
                if expected == int:
                    int(float(str(arg)))
                elif expected == float:
                    float(str(arg))
                else:
                    str(arg)
            except (ValueError, TypeError):
                return False

        return True
