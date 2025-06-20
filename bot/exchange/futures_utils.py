"""Utilities for futures contract management."""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FuturesContractManager:
    """Manages futures contract selection and updates."""

    def __init__(self, client):
        """Initialize with Coinbase client."""
        self.client = client
        self._current_contract = None
        self._last_update = None

    async def get_active_futures_contract(
        self, base_currency: str = "ETH"
    ) -> str | None:
        """
        Get the current active futures contract for a given base currency.

        Args:
            base_currency: Base currency (e.g., "ETH", "BTC")

        Returns:
            Contract symbol (e.g., "ETH 27 JUN 25") or None if not found
        """
        try:
            # Get all products
            products_response = await self.client._retry_request(
                self.client._client.get_products
            )

            # Extract products list
            if hasattr(products_response, "products"):
                products = products_response.products
            else:
                products = products_response.get("products", [])

            # Find futures contracts for the base currency
            futures_contracts = []
            current_date = datetime.utcnow()

            for product in products:
                product_id = (
                    product.product_id
                    if hasattr(product, "product_id")
                    else product.get("product_id", "")
                )

                # Check if this is a futures contract (contains date pattern)
                if base_currency in product_id and any(
                    month in product_id
                    for month in [
                        "JAN",
                        "FEB",
                        "MAR",
                        "APR",
                        "MAY",
                        "JUN",
                        "JUL",
                        "AUG",
                        "SEP",
                        "OCT",
                        "NOV",
                        "DEC",
                    ]
                ):
                    # Parse expiry date from contract name (e.g., "ETH 27 JUN 25")
                    try:
                        # Extract date components
                        parts = product_id.split()
                        if len(parts) >= 4:
                            day = int(parts[1])
                            month_str = parts[2]
                            year = int(parts[3])

                            # Convert to full year
                            if year < 100:
                                year += 2000

                            # Month mapping
                            month_map = {
                                "JAN": 1,
                                "FEB": 2,
                                "MAR": 3,
                                "APR": 4,
                                "MAY": 5,
                                "JUN": 6,
                                "JUL": 7,
                                "AUG": 8,
                                "SEP": 9,
                                "OCT": 10,
                                "NOV": 11,
                                "DEC": 12,
                            }
                            month = month_map.get(month_str, 0)

                            if month > 0:
                                expiry_date = datetime(year, month, day)

                                # Only consider contracts that haven't expired
                                if expiry_date > current_date:
                                    futures_contracts.append(
                                        {
                                            "symbol": product_id,
                                            "expiry": expiry_date,
                                            "days_to_expiry": (
                                                expiry_date - current_date
                                            ).days,
                                        }
                                    )

                    except (ValueError, IndexError) as e:
                        logger.debug("Could not parse date from %s: %s", product_id, e)
                        continue

            if not futures_contracts:
                logger.warning(
                    "No active futures contracts found for %s", base_currency
                )
                return None

            # Sort by expiry date and get the nearest one (front month)
            futures_contracts.sort(key=lambda x: x["expiry"])
            selected_contract = futures_contracts[0]

            logger.info(
                "Selected %s futures contract: %s (expires in %s days)",
                base_currency,
                selected_contract["symbol"],
                selected_contract["days_to_expiry"],
            )

            # Log all available contracts for debugging
            if len(futures_contracts) > 1:
                logger.debug("Other available %s contracts:", base_currency)
                for contract in futures_contracts[1:]:
                    logger.debug(
                        "  - %s (expires in %s days)",
                        contract["symbol"],
                        contract["days_to_expiry"],
                    )

            self._current_contract = selected_contract["symbol"]
            self._last_update = datetime.utcnow()

            return selected_contract["symbol"]

        except Exception as e:
            logger.exception("Failed to get active futures contract: %s", e)
            return None

    def get_cached_contract(self) -> str | None:
        """Get the cached contract if still valid."""
        if self._current_contract and self._last_update:
            # Cache for 1 hour
            if (datetime.utcnow() - self._last_update).seconds < 3600:
                return self._current_contract
        return None
