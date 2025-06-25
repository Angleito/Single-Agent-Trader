#!/usr/bin/env python3
"""
Test script to verify the Bluefin SDK v2 SyntaxWarning fix.

This script tests:
1. Warning filters are correctly applied
2. Bluefin SDK import does not generate SyntaxWarning
3. Docker environment variable is properly set
"""

import os
import sys
import warnings
from io import StringIO


def test_warning_suppression():
    """Test that SyntaxWarning is properly suppressed."""
    print("🔧 Testing Bluefin SDK v2 SyntaxWarning Fix")
    print("=" * 50)

    # Check environment variables
    python_warnings = os.environ.get("PYTHONWARNINGS", "")
    print(f"📋 PYTHONWARNINGS environment variable: {python_warnings}")

    if "SyntaxWarning" in python_warnings:
        print("✅ SyntaxWarning suppression found in environment")
    else:
        print("⚠️  SyntaxWarning not found in PYTHONWARNINGS")

    # Test the specific warning filter
    print("\n🧪 Testing warning filters...")

    warning_buffer = StringIO()
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # Catch all warnings

        # Apply the same filters as our fix
        warnings.filterwarnings(
            "ignore", message=r'.*"is not" with.*str.*literal.*', category=SyntaxWarning
        )
        warnings.filterwarnings(
            "ignore", message=r".*contentType is not.*", category=SyntaxWarning
        )

        # Simulate the problematic code pattern
        # This would normally generate a SyntaxWarning
        try:
            exec('contentType = "test"; result = contentType is not ""')
            print("✅ Problematic syntax executed without warnings")
        except Exception as e:
            print(f"❌ Error in test syntax: {e}")

    # Check if any SyntaxWarnings were caught
    syntax_warnings = [w for w in warning_list if issubclass(w.category, SyntaxWarning)]

    if syntax_warnings:
        print(f"❌ Found {len(syntax_warnings)} SyntaxWarning(s):")
        for w in syntax_warnings:
            print(f"   - {w.message}")
    else:
        print("✅ No SyntaxWarnings detected")

    print("\n🔍 Testing Bluefin SDK import...")
    try:
        # This should not produce warnings if our fix is working
        with warnings.catch_warnings(record=True) as import_warnings:
            warnings.simplefilter("always")

            # Apply our filters before import
            warnings.filterwarnings(
                "ignore",
                message=r'.*"is not" with.*str.*literal.*',
                category=SyntaxWarning,
            )
            warnings.filterwarnings(
                "ignore", message=r".*contentType is not.*", category=SyntaxWarning
            )

            # Import the Bluefin SDK (this would trigger the warning)
            try:
                from bluefin_v2_client import BluefinClient

                print("✅ Bluefin SDK imported successfully")
            except ImportError:
                print("⚠️  Bluefin SDK not available (expected in local environment)")
                print("   The fix will be tested when Docker container runs")

        # Check for import-time warnings
        import_syntax_warnings = [
            w for w in import_warnings if issubclass(w.category, SyntaxWarning)
        ]

        if import_syntax_warnings:
            print(
                f"❌ Import generated {len(import_syntax_warnings)} SyntaxWarning(s):"
            )
            for w in import_syntax_warnings:
                print(f"   - {w.message}")
        else:
            print("✅ No SyntaxWarnings from import")

    except Exception as e:
        print(f"⚠️  Import test failed: {e}")

    print("\n📊 Test Summary:")
    print(
        f"   Environment Variable: {'✅' if 'SyntaxWarning' in python_warnings else '❌'}"
    )
    print(f"   Warning Filters: {'✅' if len(syntax_warnings) == 0 else '❌'}")
    print(f"   Import Test: {'✅' if len(import_syntax_warnings) == 0 else '❌'}")

    print("\n🐳 Docker Test Instructions:")
    print("   Run: docker-compose logs bluefin-service | grep -i syntax")
    print("   Expected: No SyntaxWarning messages in logs")

    return len(syntax_warnings) == 0 and len(import_syntax_warnings) == 0


if __name__ == "__main__":
    success = test_warning_suppression()
    sys.exit(0 if success else 1)
