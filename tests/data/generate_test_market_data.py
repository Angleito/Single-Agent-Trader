#!/usr/bin/env python3
"""
Test Market Data Generator for VuManChu E2E Testing.

This script generates realistic OHLCV market data for comprehensive VuManChu testing
across multiple market conditions and scenarios including:
- Trending markets (bullish/bearish)
- Ranging/sideways markets
- High volatility periods
- Low volume conditions
- Market gaps and extreme moves
- Normal trading conditions

Generated data includes proper OHLCV relationships and realistic volume patterns
for accurate indicator testing and signal validation.
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class MarketDataGenerator:
    """Generate realistic market data for testing scenarios."""

    def __init__(self, base_price: float = 50000.0, random_seed: int = 42):
        """Initialize generator with base parameters."""
        self.base_price = base_price
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Market parameters
        self.min_price = 0.01
        self.volume_base = 100.0
        self.volume_variance = 0.5

    def generate_default_data(
        self, periods: int = 2000, frequency: str = "1min"
    ) -> pd.DataFrame:
        """Generate default market data with normal volatility."""
        logger.info("Generating default market data: %s periods", periods)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        # Generate realistic returns with moderate volatility
        daily_vol = 0.015  # 1.5% daily volatility
        minute_vol = daily_vol / np.sqrt(24 * 60)  # Scale to minute

        returns = np.random.normal(0, minute_vol, periods)

        # Add some trend and mean reversion
        trend_component = np.sin(np.linspace(0, 4 * np.pi, periods)) * 0.0005
        returns += trend_component

        return self._create_ohlcv_data(timestamps, returns)

    def generate_trending_data(
        self,
        periods: int = 2000,
        trend_strength: float = 0.002,
        frequency: str = "1min",
    ) -> pd.DataFrame:
        """Generate trending market data (positive trend_strength = bullish)."""
        direction = "bullish" if trend_strength > 0 else "bearish"
        logger.info("Generating %s trending data: %s periods, strength %s", direction, periods, trend_strength)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        # Base volatility
        base_vol = 0.012

        # Trend component - linear with some curvature
        trend_factor = np.linspace(0, 1, periods)
        trend_returns = trend_strength * (1 + 0.5 * np.sin(trend_factor * 2 * np.pi))

        # Random component
        random_returns = np.random.normal(0, base_vol, periods)

        # Combine trend and random
        returns = trend_returns + random_returns

        return self._create_ohlcv_data(timestamps, returns)

    def generate_ranging_data(
        self, periods: int = 2000, range_size: float = 0.05, frequency: str = "1min"
    ) -> pd.DataFrame:
        """Generate ranging/sideways market data."""
        logger.info("Generating ranging market data: %s periods, range %s", periods, range_size)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        # Create bounded random walk
        base_vol = 0.008

        # Oscillating pattern within range
        center_drift = np.sin(np.linspace(0, 8 * np.pi, periods)) * range_size * 0.3
        random_component = np.random.normal(0, base_vol, periods)

        # Mean reversion force
        cumulative_drift = np.cumsum(center_drift + random_component)
        mean_reversion = -cumulative_drift * 0.01  # Pull back to center

        returns = center_drift + random_component + mean_reversion

        return self._create_ohlcv_data(timestamps, returns)

    def generate_volatile_data(
        self,
        periods: int = 2000,
        volatility_factor: float = 2.5,
        frequency: str = "1min",
    ) -> pd.DataFrame:
        """Generate high volatility market data."""
        logger.info("Generating volatile market data: %s periods, vol factor %s", periods, volatility_factor)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        base_vol = 0.015 * volatility_factor

        # Add volatility clustering (GARCH-like behavior)
        volatilities = np.zeros(periods)
        volatilities[0] = base_vol

        for i in range(1, periods):
            # Volatility persistence with random shocks
            volatilities[i] = (
                0.1 * base_vol
                + 0.8 * volatilities[i - 1]
                + 0.1 * abs(np.random.normal(0, base_vol))
            )

        # Generate returns with time-varying volatility
        returns = np.random.normal(0, volatilities, periods)

        # Add occasional volatility spikes
        spike_indices = np.random.choice(periods, size=periods // 20, replace=False)
        returns[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))

        return self._create_ohlcv_data(timestamps, returns)

    def generate_gap_data(
        self,
        periods: int = 2000,
        gap_frequency: int = 200,
        gap_size_range: tuple[float, float] = (0.02, 0.08),
        frequency: str = "1min",
    ) -> pd.DataFrame:
        """Generate data with gaps and extreme moves."""
        logger.info("Generating gap data: %s periods, gaps every %s periods", periods, gap_frequency)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        # Start with normal returns
        base_vol = 0.01
        returns = np.random.normal(0, base_vol, periods)

        # Add gaps at regular intervals
        gap_indices = list(range(gap_frequency, periods, gap_frequency))

        for gap_idx in gap_indices:
            if gap_idx < periods:
                # Random gap size and direction
                gap_size = np.random.uniform(gap_size_range[0], gap_size_range[1])
                gap_direction = np.random.choice([-1, 1])

                # Apply gap
                returns[gap_idx] = gap_size * gap_direction

                # Add some follow-through
                if gap_idx + 1 < periods:
                    returns[gap_idx + 1] = gap_size * gap_direction * 0.3

        # Add some extreme moves (flash crashes/spikes)
        extreme_indices = np.random.choice(periods, size=periods // 100, replace=False)
        for extreme_idx in extreme_indices:
            extreme_size = np.random.uniform(0.05, 0.15)
            extreme_direction = np.random.choice([-1, 1])
            returns[extreme_idx] = extreme_size * extreme_direction

        return self._create_ohlcv_data(timestamps, returns)

    def generate_low_volume_data(
        self, periods: int = 2000, volume_factor: float = 0.3, frequency: str = "1min"
    ) -> pd.DataFrame:
        """Generate market data with low volume conditions."""
        logger.info("Generating low volume data: %s periods, vol factor %s", periods, volume_factor)

        # Generate base market data
        data = self.generate_default_data(periods, frequency)

        # Reduce volume significantly
        data["volume"] *= volume_factor

        # Add volume clustering (low volume periods)
        low_volume_periods = np.random.choice(periods, size=periods // 5, replace=False)
        for period_start in low_volume_periods:
            period_end = min(period_start + 20, periods)  # 20-minute low volume periods
            data.iloc[period_start:period_end, data.columns.get_loc("volume")] *= 0.1

        return data

    def generate_crypto_weekend_data(
        self, periods: int = 2000, frequency: str = "1min"
    ) -> pd.DataFrame:
        """Generate data simulating crypto weekend trading patterns."""
        logger.info("Generating crypto weekend data: %s periods", periods)

        timestamps = pd.date_range("2024-01-01", periods=periods, freq=frequency)

        # Different volatility for weekends vs weekdays
        weekend_vol = 0.008  # Lower weekend volatility
        weekday_vol = 0.015  # Higher weekday volatility

        returns = np.zeros(periods)
        volumes = np.zeros(periods)

        for i, timestamp in enumerate(timestamps):
            is_weekend = timestamp.weekday() >= 5  # Saturday/Sunday

            if is_weekend:
                returns[i] = np.random.normal(0, weekend_vol)
                volumes[i] = self.volume_base * 0.6  # Lower weekend volume
            else:
                returns[i] = np.random.normal(0, weekday_vol)
                volumes[i] = self.volume_base * 1.2  # Higher weekday volume

        return self._create_ohlcv_data(timestamps, returns, custom_volumes=volumes)

    def _create_ohlcv_data(
        self,
        timestamps: pd.DatetimeIndex,
        returns: np.ndarray,
        custom_volumes: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Create realistic OHLCV data from returns."""
        # periods = len(timestamps)  # Commented out as it's not used

        # Generate price series
        prices = self.base_price * np.exp(np.cumsum(returns))
        prices = np.maximum(prices, self.min_price)  # Ensure positive prices

        # Generate OHLCV data
        data = []

        for i, (timestamp, close_price) in enumerate(
            zip(timestamps, prices, strict=True)
        ):
            # Calculate realistic OHLC from close and return
            return_vol = abs(returns[i]) if i < len(returns) else 0.01
            price_range = close_price * return_vol * np.random.uniform(0.5, 2.0)

            # Generate open price (previous close + some gap)
            if i == 0:
                open_price = close_price * (1 + np.random.normal(0, 0.001))
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, 0.002))

            # Generate high and low
            high_offset = np.random.exponential(price_range * 0.3)
            low_offset = np.random.exponential(price_range * 0.3)

            high = max(open_price, close_price) + high_offset
            low = min(open_price, close_price) - low_offset
            low = max(low, self.min_price)  # Ensure positive

            # Generate volume
            if custom_volumes is not None:
                volume = custom_volumes[i]
            else:
                # Volume correlated with price movement and randomness
                volume_factor = 1 + abs(returns[i]) * 10  # More volume on big moves
                base_volume = self.volume_base * volume_factor
                volume = np.random.gamma(
                    2, base_volume / 2
                )  # Gamma distribution for volume

            data.append(
                {
                    "timestamp": timestamp,
                    "open": max(open_price, self.min_price),
                    "high": max(high, self.min_price),
                    "low": max(low, self.min_price),
                    "close": max(close_price, self.min_price),
                    "volume": max(volume, 0.1),
                }
            )

        df = pd.DataFrame(data)
        df = df.set_index("timestamp")

        # Validate OHLC relationships
        self._validate_ohlc_data(df)

        return df

    def _validate_ohlc_data(self, df: pd.DataFrame) -> None:
        """Validate OHLC data relationships."""
        # Check that high >= max(open, close) and low <= min(open, close)
        high_valid = (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        low_valid = (df["low"] <= df[["open", "close"]].min(axis=1)).all()

        if not high_valid:
            logger.warning("Some high prices are below open/close - fixing")
            df["high"] = df[["high", "open", "close"]].max(axis=1)

        if not low_valid:
            logger.warning("Some low prices are above open/close - fixing")
            df["low"] = df[["low", "open", "close"]].min(axis=1)

        # Check for non-positive values
        for col in ["open", "high", "low", "close", "volume"]:
            if (df[col] <= 0).any():
                logger.warning("Non-positive values found in %s - fixing", col)
                df[col] = df[col].clip(lower=self.min_price if col != "volume" else 0.1)


class TestDataSuite:
    """Comprehensive test data suite generator."""

    def __init__(self, output_dir: Path, base_price: float = 50000.0):
        """Initialize test data suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = MarketDataGenerator(base_price=base_price)

        self.scenarios = {
            "default": "Default market conditions with normal volatility",
            "trending": "Strong trending market conditions",
            "ranging": "Sideways/ranging market conditions",
            "volatile": "High volatility market conditions",
            "gap_data": "Market data with gaps and extreme moves",
            "low_volume": "Low volume trading conditions",
            "crypto_weekend": "Crypto weekend trading patterns",
        }

    def generate_all_scenarios(
        self, sizes: list[int] | None = None
    ) -> dict[str, list[str]]:
        """Generate all test scenarios with multiple sizes."""
        if sizes is None:
            sizes = [1000, 5000, 10000]

        logger.info("Generating test data for %s scenarios", len(self.scenarios))
        logger.info("Data sizes: %s", sizes)

        generated_files = {}

        for scenario in self.scenarios:
            scenario_files = []

            for size in sizes:
                filename = f"market_data_{scenario}_{size}.csv"
                filepath = self.output_dir / filename

                logger.info("Generating %s data with %s periods", scenario, size)

                # Generate data based on scenario
                if scenario == "default":
                    data = self.generator.generate_default_data(size)
                elif scenario == "trending":
                    data = self.generator.generate_trending_data(
                        size, trend_strength=0.002
                    )
                elif scenario == "ranging":
                    data = self.generator.generate_ranging_data(size)
                elif scenario == "volatile":
                    data = self.generator.generate_volatile_data(size)
                elif scenario == "gap_data":
                    data = self.generator.generate_gap_data(size)
                elif scenario == "low_volume":
                    data = self.generator.generate_low_volume_data(size)
                elif scenario == "crypto_weekend":
                    data = self.generator.generate_crypto_weekend_data(size)
                else:
                    logger.warning("Unknown scenario: %s, using default", scenario)
                    data = self.generator.generate_default_data(size)

                # Save data
                data.to_csv(filepath)
                scenario_files.append(str(filepath))

                logger.info("Saved: %s (%s rows)", filepath, len(data))

            generated_files[scenario] = scenario_files

        # Also generate standard named files for compatibility
        self._generate_standard_files()

        return generated_files

    def _generate_standard_files(self):
        """Generate standard files with expected names."""
        standard_scenarios = {
            "market_data_default.csv": ("default", 2000),
            "market_data_trending.csv": ("trending", 2000),
            "market_data_ranging.csv": ("ranging", 2000),
            "market_data_volatile.csv": ("volatile", 2000),
            "market_data_gap_data.csv": ("gap_data", 2000),
        }

        for filename, (scenario, size) in standard_scenarios.items():
            filepath = self.output_dir / filename

            if scenario == "default":
                data = self.generator.generate_default_data(size)
            elif scenario == "trending":
                data = self.generator.generate_trending_data(size, trend_strength=0.002)
            elif scenario == "ranging":
                data = self.generator.generate_ranging_data(size)
            elif scenario == "volatile":
                data = self.generator.generate_volatile_data(size)
            elif scenario == "gap_data":
                data = self.generator.generate_gap_data(size)

            data.to_csv(filepath)
            logger.info("Generated standard file: %s", filepath)

    def generate_data_summary(self) -> dict:
        """Generate summary of all test data."""
        summary = {
            "generation_time": datetime.now().isoformat(),
            "output_directory": str(self.output_dir),
            "scenarios": self.scenarios,
            "files": [],
        }

        # List all generated files
        for csv_file in self.output_dir.glob("*.csv"):
            file_info = {
                "filename": csv_file.name,
                "filepath": str(csv_file),
                "size_bytes": csv_file.stat().st_size,
                "modified_time": datetime.fromtimestamp(
                    csv_file.stat().st_mtime
                ).isoformat(),
            }

            # Get row count if possible
            try:
                df = pd.read_csv(csv_file)
                file_info["row_count"] = len(df)
                file_info["columns"] = list(df.columns)
            except Exception as e:
                file_info["error"] = str(e)

            summary["files"].append(file_info)

        # Save summary
        summary_file = self.output_dir / "test_data_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Generated summary: %s", summary_file)
        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test market data for VuManChu E2E testing"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_data",
        help="Output directory for test data files",
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        default="default,trending,ranging,volatile,gap_data",
        help="Comma-separated list of scenarios to generate",
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,5000,10000",
        help="Comma-separated list of data sizes to generate",
    )

    parser.add_argument(
        "--base-price",
        type=float,
        default=50000.0,
        help="Base price for generated data",
    )

    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducible data"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse scenarios and sizes
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    logger.info("VuManChu Test Data Generator")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Scenarios: %s", scenarios)
    logger.info("Sizes: %s", sizes)
    logger.info("Base price: %s", args.base_price)
    logger.info("Random seed: %s", args.random_seed)

    # Initialize generator
    generator = MarketDataGenerator(
        base_price=args.base_price, random_seed=args.random_seed
    )

    # Initialize test suite
    test_suite = TestDataSuite(output_dir=args.output_dir, base_price=args.base_price)

    # Generate requested scenarios
    generated_files = {}

    for scenario in scenarios:
        if scenario in test_suite.scenarios:
            scenario_files = []

            for size in sizes:
                filename = f"market_data_{scenario}_{size}.csv"
                filepath = Path(args.output_dir) / filename

                logger.info("Generating %s data: %s periods", scenario, size)

                if scenario == "default":
                    data = generator.generate_default_data(size)
                elif scenario == "trending":
                    data = generator.generate_trending_data(size, trend_strength=0.002)
                elif scenario == "ranging":
                    data = generator.generate_ranging_data(size)
                elif scenario == "volatile":
                    data = generator.generate_volatile_data(size)
                elif scenario == "gap_data":
                    data = generator.generate_gap_data(size)
                elif scenario == "low_volume":
                    data = generator.generate_low_volume_data(size)
                elif scenario == "crypto_weekend":
                    data = generator.generate_crypto_weekend_data(size)

                # Save data
                data.to_csv(filepath)
                scenario_files.append(str(filepath))

                logger.info("Saved: %s (%s rows)", filepath, len(data))

            generated_files[scenario] = scenario_files
        else:
            logger.warning("Unknown scenario: %s", scenario)

    # Generate standard files
    test_suite._generate_standard_files()

    # Generate summary
    summary = test_suite.generate_data_summary()

    logger.info("Test data generation completed")
    logger.info("Generated %s files", len(summary['files']))
    logger.info("Total scenarios: %s", len(generated_files))

    return generated_files


if __name__ == "__main__":
    main()
