"""Volume data validation and error handling utilities.

This module provides comprehensive validation for volume data used in
trading operations, ensuring data quality and preventing trading errors.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeValidationResult:
    """Result of volume data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    normalized_volume: Decimal | None = None
    confidence_score: float = 0.0


class VolumeValidationError(Exception):
    """Exception raised when volume validation fails critically."""

    def __init__(self, message: str, validation_result: VolumeValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


def validate_volume_data(
    volume: Any,
    min_volume: Decimal = Decimal(0),
    max_volume: Decimal = Decimal(1000000000),
    allow_zero: bool = True,
) -> VolumeValidationResult:
    """Validate volume data for trading operations.

    Args:
        volume: Volume value to validate
        min_volume: Minimum acceptable volume
        max_volume: Maximum acceptable volume
        allow_zero: Whether zero volume is acceptable

    Returns:
        Validation result with errors and normalized volume
    """
    errors = []
    warnings = []
    normalized_volume = None
    confidence_score = 0.0

    try:
        # Convert to Decimal for validation
        if volume is None:
            errors.append("Volume is None")
            return VolumeValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                normalized_volume=None,
                confidence_score=0.0,
            )

        # Handle different input types
        if isinstance(volume, str):
            if not volume.strip():
                errors.append("Volume string is empty")
                return VolumeValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    normalized_volume=None,
                    confidence_score=0.0,
                )
            try:
                normalized_volume = Decimal(volume.strip())
            except (ValueError, decimal.InvalidOperation):
                errors.append(f"Cannot convert volume string '{volume}' to decimal")
                return VolumeValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    normalized_volume=None,
                    confidence_score=0.0,
                )
        elif isinstance(volume, (int, float)):
            normalized_volume = Decimal(str(volume))
        elif isinstance(volume, Decimal):
            normalized_volume = volume
        else:
            errors.append(f"Unsupported volume type: {type(volume)}")
            return VolumeValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                normalized_volume=None,
                confidence_score=0.0,
            )

        # Validate numeric constraints
        if normalized_volume < 0:
            errors.append(f"Volume cannot be negative: {normalized_volume}")
        elif normalized_volume == 0 and not allow_zero:
            errors.append("Zero volume not allowed")
        elif normalized_volume < min_volume:
            warnings.append(f"Volume {normalized_volume} below minimum {min_volume}")
            confidence_score = max(
                0.0, 0.5 - float(min_volume - normalized_volume) / float(min_volume)
            )
        elif normalized_volume > max_volume:
            warnings.append(f"Volume {normalized_volume} exceeds maximum {max_volume}")
            confidence_score = max(
                0.0, 0.5 - float(normalized_volume - max_volume) / float(max_volume)
            )
        else:
            # Volume is within acceptable range
            confidence_score = 1.0

        # Check for precision issues
        if normalized_volume.as_tuple().exponent < -18:
            warnings.append(
                "Volume has excessive decimal precision, may cause rounding errors"
            )
            confidence_score = min(confidence_score, 0.8)

        # Check for suspiciously large volumes
        if normalized_volume > Decimal(100000000):  # 100M
            warnings.append("Volume is suspiciously large, verify data source")
            confidence_score = min(confidence_score, 0.7)

        is_valid = len(errors) == 0

        return VolumeValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_volume=normalized_volume,
            confidence_score=confidence_score,
        )

    except Exception as e:
        errors.append(f"Unexpected error during volume validation: {e}")
        return VolumeValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            normalized_volume=None,
            confidence_score=0.0,
        )


def validate_24h_volume(volume_24h: Any) -> VolumeValidationResult:
    """Validate 24-hour volume data specifically.

    Args:
        volume_24h: 24-hour volume to validate

    Returns:
        Validation result for 24h volume
    """
    return validate_volume_data(
        volume=volume_24h,
        min_volume=Decimal(0),
        max_volume=Decimal(10000000000),  # 10B max for 24h volume
        allow_zero=True,  # 24h volume can be zero for new/inactive markets
    )


def validate_orderbook_volume(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]]
) -> VolumeValidationResult:
    """Validate orderbook volume data.

    Args:
        bids: List of (price, volume) tuples for bids
        asks: List of (price, volume) tuples for asks

    Returns:
        Validation result for orderbook volumes
    """
    errors = []
    warnings = []
    total_volume = Decimal(0)
    confidence_score = 1.0

    try:
        # Validate bid volumes
        for i, (price, volume) in enumerate(bids):
            result = validate_volume_data(volume, allow_zero=True)
            if not result.is_valid:
                errors.extend([f"Bid {i}: {error}" for error in result.errors])
            if result.warnings:
                warnings.extend([f"Bid {i}: {warning}" for warning in result.warnings])
            if result.normalized_volume:
                total_volume += result.normalized_volume
            confidence_score = min(confidence_score, result.confidence_score)

        # Validate ask volumes
        for i, (price, volume) in enumerate(asks):
            result = validate_volume_data(volume, allow_zero=True)
            if not result.is_valid:
                errors.extend([f"Ask {i}: {error}" for error in result.errors])
            if result.warnings:
                warnings.extend([f"Ask {i}: {warning}" for warning in result.warnings])
            if result.normalized_volume:
                total_volume += result.normalized_volume
            confidence_score = min(confidence_score, result.confidence_score)

        # Check for orderbook balance
        bid_volume = sum(Decimal(str(volume)) for _, volume in bids)
        ask_volume = sum(Decimal(str(volume)) for _, volume in asks)

        if bid_volume == 0 and ask_volume == 0:
            errors.append("Orderbook has no volume on either side")
        elif bid_volume == 0:
            warnings.append("No bid volume in orderbook")
            confidence_score = min(confidence_score, 0.5)
        elif ask_volume == 0:
            warnings.append("No ask volume in orderbook")
            confidence_score = min(confidence_score, 0.5)
        else:
            # Check for extreme imbalance
            total_book_volume = bid_volume + ask_volume
            imbalance = abs(bid_volume - ask_volume) / total_book_volume
            if imbalance > Decimal("0.8"):  # More than 80% imbalance
                warnings.append(f"Extreme orderbook imbalance: {imbalance:.2%}")
                confidence_score = min(confidence_score, 0.6)

        is_valid = len(errors) == 0

        return VolumeValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_volume=total_volume,
            confidence_score=confidence_score,
        )

    except Exception as e:
        errors.append(f"Unexpected error during orderbook volume validation: {e}")
        return VolumeValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            normalized_volume=None,
            confidence_score=0.0,
        )


def validate_volume_series(
    volumes: list[Any], check_consistency: bool = True
) -> VolumeValidationResult:
    """Validate a series of volume data points.

    Args:
        volumes: List of volume values
        check_consistency: Whether to check for consistency across the series

    Returns:
        Validation result for the volume series
    """
    errors = []
    warnings = []
    confidence_score = 1.0
    normalized_volumes = []

    if not volumes:
        errors.append("Volume series is empty")
        return VolumeValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            normalized_volume=None,
            confidence_score=0.0,
        )

    # Validate each volume
    for i, volume in enumerate(volumes):
        result = validate_volume_data(volume, allow_zero=True)
        if not result.is_valid:
            errors.extend([f"Volume {i}: {error}" for error in result.errors])
        if result.warnings:
            warnings.extend([f"Volume {i}: {warning}" for warning in result.warnings])
        if result.normalized_volume is not None:
            normalized_volumes.append(result.normalized_volume)
        confidence_score = min(confidence_score, result.confidence_score)

    if check_consistency and len(normalized_volumes) > 2:
        # Check for consistency
        avg_volume = sum(normalized_volumes) / len(normalized_volumes)

        # Check for extreme outliers
        outliers = []
        for i, vol in enumerate(normalized_volumes):
            if vol > avg_volume * 10:  # Volume 10x average
                outliers.append(f"Volume {i} is {vol / avg_volume:.1f}x average")

        if outliers:
            warnings.extend(outliers)
            confidence_score = min(confidence_score, 0.7)

        # Check for zero volume percentage
        zero_count = sum(1 for vol in normalized_volumes if vol == 0)
        zero_percentage = zero_count / len(normalized_volumes)

        if zero_percentage > 0.5:  # More than 50% zero volumes
            warnings.append(f"{zero_percentage:.1%} of volumes are zero")
            confidence_score = min(confidence_score, 0.5)

    is_valid = len(errors) == 0
    total_volume = sum(normalized_volumes) if normalized_volumes else None

    return VolumeValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        normalized_volume=total_volume,
        confidence_score=confidence_score,
    )


def safe_volume_conversion(volume: Any, default: Decimal = Decimal(0)) -> Decimal:
    """Safely convert volume to Decimal with fallback.

    Args:
        volume: Volume value to convert
        default: Default value if conversion fails

    Returns:
        Converted volume or default value
    """
    try:
        result = validate_volume_data(volume, allow_zero=True)
        if result.is_valid and result.normalized_volume is not None:
            return result.normalized_volume
        logger.warning("Volume validation failed, using default: %s", result.errors)
        return default
    except Exception as e:
        logger.warning("Volume conversion failed, using default: %s", e)
        return default


# Import decimal for validation
import decimal
