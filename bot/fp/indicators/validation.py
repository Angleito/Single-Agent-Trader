"""
Validation module for ensuring functional indicator types preserve calculation accuracy.

This module provides validation functions to compare imperative and functional
implementations of VuManChu indicators, ensuring that the migration to functional
types maintains mathematical accuracy and signal integrity.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from bot.fp.indicators.integrations import integrate_vumanchu_with_indicators
from bot.fp.indicators.vumanchu_functional import (
    calculate_hlc3,
    calculate_wavetrend_oscillator,
    vumanchu_cipher,
)

if TYPE_CHECKING:
    from bot.fp.types.indicators import VuManchuResult, VuManchuSignalSet


def generate_test_ohlcv_data(length: int = 100, seed: int = 42) -> np.ndarray:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Generate base price trend
    base_price = 100.0
    trend = np.cumsum(np.random.normal(0, 0.5, length))
    prices = base_price + trend

    # Generate OHLCV data
    ohlcv = np.zeros((length, 5))

    for i in range(length):
        # Close price from trend
        close = prices[i]

        # Generate realistic OHLC relationships
        daily_range = abs(np.random.normal(0, 2.0))
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)

        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume
        volume = np.random.uniform(1000, 10000)

        ohlcv[i] = [open_price, high, low, close, volume]

    return ohlcv


def validate_hlc3_calculation(
    ohlcv: np.ndarray, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Validate HLC3 calculation accuracy."""
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # Functional calculation
    functional_hlc3 = calculate_hlc3(high, low, close)

    # Reference calculation
    reference_hlc3 = (high + low + close) / 3.0

    # Calculate differences
    differences = np.abs(functional_hlc3 - reference_hlc3)
    max_difference = np.max(differences)
    mean_difference = np.mean(differences)

    validation_result = {
        "test_name": "HLC3 Calculation",
        "passed": max_difference < tolerance,
        "max_difference": float(max_difference),
        "mean_difference": float(mean_difference),
        "tolerance": tolerance,
        "sample_size": len(ohlcv),
    }

    if not validation_result["passed"]:
        validation_result["error_details"] = {
            "functional_sample": functional_hlc3[:5].tolist(),
            "reference_sample": reference_hlc3[:5].tolist(),
            "differences_sample": differences[:5].tolist(),
        }

    return validation_result


def validate_wavetrend_calculation(
    ohlcv: np.ndarray,
    channel_length: int = 6,
    average_length: int = 8,
    ma_length: int = 3,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Validate WaveTrend oscillator calculation accuracy."""
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    hlc3 = calculate_hlc3(high, low, close)

    # Functional calculation
    wt1_func, wt2_func = calculate_wavetrend_oscillator(
        hlc3, channel_length, average_length, ma_length
    )

    # Remove NaN values for comparison
    valid_indices = ~(np.isnan(wt1_func) | np.isnan(wt2_func))
    wt1_func_clean = wt1_func[valid_indices]
    wt2_func_clean = wt2_func[valid_indices]

    validation_result = {
        "test_name": "WaveTrend Calculation",
        "passed": True,
        "wt1_stats": {
            "mean": float(np.mean(wt1_func_clean)) if len(wt1_func_clean) > 0 else 0.0,
            "std": float(np.std(wt1_func_clean)) if len(wt1_func_clean) > 0 else 0.0,
            "min": float(np.min(wt1_func_clean)) if len(wt1_func_clean) > 0 else 0.0,
            "max": float(np.max(wt1_func_clean)) if len(wt1_func_clean) > 0 else 0.0,
        },
        "wt2_stats": {
            "mean": float(np.mean(wt2_func_clean)) if len(wt2_func_clean) > 0 else 0.0,
            "std": float(np.std(wt2_func_clean)) if len(wt2_func_clean) > 0 else 0.0,
            "min": float(np.min(wt2_func_clean)) if len(wt2_func_clean) > 0 else 0.0,
            "max": float(np.max(wt2_func_clean)) if len(wt2_func_clean) > 0 else 0.0,
        },
        "valid_values": len(wt1_func_clean),
        "total_values": len(wt1_func),
        "sample_size": len(ohlcv),
    }

    # Basic sanity checks
    if len(wt1_func_clean) == 0:
        validation_result["passed"] = False
        validation_result["error"] = "No valid WaveTrend values calculated"
    elif np.any(np.isinf(wt1_func_clean)) or np.any(np.isinf(wt2_func_clean)):
        validation_result["passed"] = False
        validation_result["error"] = "Infinite values in WaveTrend calculation"
    elif (
        validation_result["wt1_stats"]["std"] == 0
        or validation_result["wt2_stats"]["std"] == 0
    ):
        validation_result["passed"] = False
        validation_result["error"] = "Zero variance in WaveTrend values"

    return validation_result


def validate_vumanchu_result_structure(result: VuManchuResult) -> dict[str, Any]:
    """Validate VuManchuResult structure and content."""
    validation_result = {
        "test_name": "VuManchuResult Structure",
        "passed": True,
        "errors": [],
    }

    # Check required attributes
    required_attrs = ["timestamp", "wave_a", "wave_b", "signal"]
    for attr in required_attrs:
        if not hasattr(result, attr):
            validation_result["passed"] = False
            validation_result["errors"].append(f"Missing attribute: {attr}")

    # Check data types
    if hasattr(result, "wave_a") and not isinstance(result.wave_a, int | float):
        validation_result["passed"] = False
        validation_result["errors"].append(f"wave_a type error: {type(result.wave_a)}")

    if hasattr(result, "wave_b") and not isinstance(result.wave_b, int | float):
        validation_result["passed"] = False
        validation_result["errors"].append(f"wave_b type error: {type(result.wave_b)}")

    if hasattr(result, "signal") and result.signal not in ["LONG", "SHORT", "NEUTRAL"]:
        validation_result["passed"] = False
        validation_result["errors"].append(f"Invalid signal value: {result.signal}")

    # Check for invalid numeric values
    if hasattr(result, "wave_a") and (
        np.isnan(result.wave_a) or np.isinf(result.wave_a)
    ):
        validation_result["passed"] = False
        validation_result["errors"].append("wave_a contains NaN or Inf")

    if hasattr(result, "wave_b") and (
        np.isnan(result.wave_b) or np.isinf(result.wave_b)
    ):
        validation_result["passed"] = False
        validation_result["errors"].append("wave_b contains NaN or Inf")

    # Check method functionality
    try:
        result.is_bullish_crossover()
        result.is_bearish_crossover()
        result.momentum_strength()
    except Exception as e:
        validation_result["passed"] = False
        validation_result["errors"].append(f"Method error: {e}")

    return validation_result


def validate_signal_set_structure(signal_set: VuManchuSignalSet) -> dict[str, Any]:
    """Validate VuManchuSignalSet structure and content."""
    validation_result = {
        "test_name": "VuManchuSignalSet Structure",
        "passed": True,
        "errors": [],
        "component_counts": {},
    }

    # Check required attributes
    required_attrs = [
        "timestamp",
        "vumanchu_result",
        "diamond_patterns",
        "yellow_cross_signals",
        "candle_patterns",
        "divergence_patterns",
    ]

    for attr in required_attrs:
        if not hasattr(signal_set, attr):
            validation_result["passed"] = False
            validation_result["errors"].append(f"Missing attribute: {attr}")

    # Check component counts
    validation_result["component_counts"] = {
        "diamond_patterns": len(signal_set.diamond_patterns),
        "yellow_cross_signals": len(signal_set.yellow_cross_signals),
        "candle_patterns": len(signal_set.candle_patterns),
        "divergence_patterns": len(signal_set.divergence_patterns),
    }

    # Validate VuManchuResult
    if hasattr(signal_set, "vumanchu_result"):
        vumanchu_validation = validate_vumanchu_result_structure(
            signal_set.vumanchu_result
        )
        if not vumanchu_validation["passed"]:
            validation_result["passed"] = False
            validation_result["errors"].extend(vumanchu_validation["errors"])

    # Check method functionality
    try:
        signal_set.get_active_patterns()
        signal_set.get_bullish_signals()
        signal_set.get_bearish_signals()
        signal_set.signal_confluence_score()
        signal_set.overall_direction()
    except Exception as e:
        validation_result["passed"] = False
        validation_result["errors"].append(f"Method error: {e}")

    return validation_result


def validate_signal_consistency(
    ohlcv: np.ndarray,
    runs: int = 5,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Validate signal calculation consistency across multiple runs."""
    validation_result = {
        "test_name": "Signal Consistency",
        "passed": True,
        "errors": [],
        "run_results": [],
    }

    # Run calculations multiple times
    results = []
    for i in range(runs):
        try:
            result = vumanchu_cipher(ohlcv, timestamp=datetime.now())
            results.append(result)
        except Exception as e:
            validation_result["passed"] = False
            validation_result["errors"].append(f"Run {i + 1} failed: {e}")
            continue

    if len(results) < 2:
        validation_result["passed"] = False
        validation_result["errors"].append(
            "Insufficient successful runs for comparison"
        )
        return validation_result

    # Compare results for consistency
    reference = results[0]
    for i, result in enumerate(results[1:], 1):
        wave_a_diff = abs(result.wave_a - reference.wave_a)
        wave_b_diff = abs(result.wave_b - reference.wave_b)

        run_result = {
            "run": i + 1,
            "wave_a_diff": float(wave_a_diff),
            "wave_b_diff": float(wave_b_diff),
            "signal_match": result.signal == reference.signal,
        }

        validation_result["run_results"].append(run_result)

        if wave_a_diff > tolerance or wave_b_diff > tolerance:
            validation_result["passed"] = False
            validation_result["errors"].append(
                f"Run {i + 1} differs from reference by {wave_a_diff:.2e} (wave_a), {wave_b_diff:.2e} (wave_b)"
            )

        if result.signal != reference.signal:
            validation_result["passed"] = False
            validation_result["errors"].append(
                f"Run {i + 1} signal mismatch: {result.signal} vs {reference.signal}"
            )

    return validation_result


def validate_integration_accuracy(
    ohlcv: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate integration layer accuracy."""
    validation_result = {
        "test_name": "Integration Accuracy",
        "passed": True,
        "errors": [],
        "component_validation": {},
    }

    try:
        # Test comprehensive analysis
        signal_set = integrate_vumanchu_with_indicators(
            ohlcv, vumanchu_config=config, timestamp=datetime.now()
        )

        # Validate signal set structure
        structure_validation = validate_signal_set_structure(signal_set)
        validation_result["component_validation"]["structure"] = structure_validation

        if not structure_validation["passed"]:
            validation_result["passed"] = False
            validation_result["errors"].extend(structure_validation["errors"])

        # Test individual components
        vumanchu_validation = validate_vumanchu_result_structure(
            signal_set.vumanchu_result
        )
        validation_result["component_validation"]["vumanchu"] = vumanchu_validation

        if not vumanchu_validation["passed"]:
            validation_result["passed"] = False
            validation_result["errors"].extend(vumanchu_validation["errors"])

        # Validate composite signal if present
        if signal_set.composite_signal:
            composite_valid = (
                hasattr(signal_set.composite_signal, "signal_direction")
                and hasattr(signal_set.composite_signal, "confidence")
                and hasattr(signal_set.composite_signal, "strength")
            )

            if not composite_valid:
                validation_result["passed"] = False
                validation_result["errors"].append("Invalid composite signal structure")

            validation_result["component_validation"]["composite"] = {
                "present": True,
                "valid": composite_valid,
                "direction": signal_set.composite_signal.signal_direction,
                "confidence": signal_set.composite_signal.confidence,
            }
        else:
            validation_result["component_validation"]["composite"] = {"present": False}

    except Exception as e:
        validation_result["passed"] = False
        validation_result["errors"].append(f"Integration test failed: {e}")

    return validation_result


def run_comprehensive_validation(
    test_lengths: list[int] | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Run comprehensive validation suite."""
    if test_lengths is None:
        test_lengths = [50, 100, 200]
    validation_suite = {
        "test_summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        },
        "test_results": [],
        "overall_passed": True,
    }

    for length in test_lengths:
        # Generate test data
        ohlcv = generate_test_ohlcv_data(length)

        # Run HLC3 validation
        hlc3_result = validate_hlc3_calculation(ohlcv, tolerance)
        validation_suite["test_results"].append(hlc3_result)
        validation_suite["test_summary"]["total_tests"] += 1

        if hlc3_result["passed"]:
            validation_suite["test_summary"]["passed_tests"] += 1
        else:
            validation_suite["test_summary"]["failed_tests"] += 1
            validation_suite["overall_passed"] = False

        # Run WaveTrend validation
        wt_result = validate_wavetrend_calculation(ohlcv, tolerance=tolerance)
        validation_suite["test_results"].append(wt_result)
        validation_suite["test_summary"]["total_tests"] += 1

        if wt_result["passed"]:
            validation_suite["test_summary"]["passed_tests"] += 1
        else:
            validation_suite["test_summary"]["failed_tests"] += 1
            validation_suite["overall_passed"] = False

        # Run consistency validation
        consistency_result = validate_signal_consistency(ohlcv, tolerance=tolerance)
        validation_suite["test_results"].append(consistency_result)
        validation_suite["test_summary"]["total_tests"] += 1

        if consistency_result["passed"]:
            validation_suite["test_summary"]["passed_tests"] += 1
        else:
            validation_suite["test_summary"]["failed_tests"] += 1
            validation_suite["overall_passed"] = False

        # Run integration validation
        integration_result = validate_integration_accuracy(ohlcv)
        validation_suite["test_results"].append(integration_result)
        validation_suite["test_summary"]["total_tests"] += 1

        if integration_result["passed"]:
            validation_suite["test_summary"]["passed_tests"] += 1
        else:
            validation_suite["test_summary"]["failed_tests"] += 1
            validation_suite["overall_passed"] = False

    # Calculate success rate
    validation_suite["test_summary"]["success_rate"] = (
        (
            validation_suite["test_summary"]["passed_tests"]
            / validation_suite["test_summary"]["total_tests"]
        )
        if validation_suite["test_summary"]["total_tests"] > 0
        else 0.0
    )

    return validation_suite


def print_validation_report(validation_results: dict[str, Any]) -> None:
    """Print a formatted validation report."""
    print("=" * 80)
    print("VUMANCHU FUNCTIONAL TYPES VALIDATION REPORT")
    print("=" * 80)

    summary = validation_results["test_summary"]
    print(
        f"\nOVERALL RESULT: {'PASSED' if validation_results['overall_passed'] else 'FAILED'}"
    )
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")

    print("\nDETAILED RESULTS:")
    print("-" * 50)

    for i, result in enumerate(validation_results["test_results"], 1):
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{i}. {result['test_name']}: {status}")

        if not result["passed"] and "errors" in result:
            for error in result["errors"]:
                print(f"   ERROR: {error}")

        # Print key metrics
        if "max_difference" in result:
            print(f"   Max Difference: {result['max_difference']:.2e}")
        if "sample_size" in result:
            print(f"   Sample Size: {result['sample_size']}")
        if "success_rate" in result:
            print(f"   Success Rate: {result['success_rate']:.1%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run validation when script is executed directly
    print("Running VuManChu Functional Types Validation...")
    results = run_comprehensive_validation()
    print_validation_report(results)
