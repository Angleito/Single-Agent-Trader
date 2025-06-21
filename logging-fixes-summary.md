# Logging Format Error Fixes Summary

## Overview
Fixed critical logging format errors that were causing TypeError exceptions with messages like:
- "TypeError: must be real number, not str"
- "TypeError: %d format: a real number is required, not str"

## Root Causes Identified

1. **Type Mismatches in Format Specifiers**
   - Using `%s` for numeric values that should use `%d` (integers) or `%f` (floats)
   - Passing string values to `%d` format specifiers
   - Not converting values to appropriate types before logging

2. **Common Problem Patterns**
   - `len()` results logged with `%s` instead of `%d`
   - Configuration values from `getattr()` potentially being strings
   - HTTP status codes that might be integers logged with `%s`
   - Numeric values not explicitly converted to int/float

## Files Fixed

### 1. `/Users/angel/Documents/Projects/cursorprod/bot/strategy/llm_cache.py`
- **Line 202-204**: Fixed initialization logging to use actual instance variables
- **Line 298-302**: Changed hits format from `%s` to `%d` and ensured float conversion
- **Line 321-323**: Added explicit float conversion for compute_time
- **Line 249**: Changed `%s` to `%d` for len(expired_keys)
- **Line 261**: Changed `%s` to `%d` for excess_count
- **Line 455**: Changed `%s` to `%d` for cleared_count

### 2. `/Users/angel/Documents/Projects/cursorprod/bot/order_manager.py`
- **Line 674**: Changed `%s` to `%d` for len(self._active_orders)
- **Line 705**: Changed `%s` to `%d` for len(self._order_history)
- **Line 728**: Changed `%s` to `%d` for removed_count

### 3. `/Users/angel/Documents/Projects/cursorprod/bot/command_consumer.py`
- **Line 146**: Added int() conversion for poll_interval
- **Line 213**: Changed `%s` to `%d` for len(pending_commands)

### 4. `/Users/angel/Documents/Projects/cursorprod/bot/validator.py`
- **Line 217**: Changed `%s` to `%d` and added int() conversion for leverage

### 5. `/Users/angel/Documents/Projects/cursorprod/bot/position_manager.py`
- **Line 882**: Changed `%s` to `%d` for len(self._positions)

### 6. `/Users/angel/Documents/Projects/cursorprod/bot/backtest/engine.py`
- **Line 191**: Changed `%s` to `%d` for len(filtered_data)

### 7. `/Users/angel/Documents/Projects/cursorprod/bot/config_utils.py`
- **Line 234**: Changed `%s` to `%d` for len(warning_issues)

### 8. `/Users/angel/Documents/Projects/cursorprod/bot/exchange/bluefin.py`
- **Line 464**: Changed `%s` to `%d` for len(self._contract_info)
- **Line 1054**: Changed `%s` to `%d` for len(positions)

### 9. `/Users/angel/Documents/Projects/cursorprod/bot/exchange/coinbase.py`
- **Line 1177**: Changed `%s` to `%d` for len(self._portfolios)
- **Line 3029**: Changed `%s` to `%d` for len(positions)
- **Line 3085**: Changed `%s` to `%d` for len(positions)

## Patterns Applied

1. **For integer counts (len(), count variables)**:
   ```python
   # Before
   logger.info("Found %s items", len(items))

   # After
   logger.info("Found %d items", len(items))
   ```

2. **For potentially string config values**:
   ```python
   # Before
   logger.info("TTL: %d seconds", ttl_seconds)  # ttl_seconds might be string

   # After
   logger.info("TTL: %d seconds", int(ttl_seconds))
   ```

3. **For float values**:
   ```python
   # Before
   logger.info("Time: %s", compute_time)

   # After
   logger.info("Time: %.2f", float(compute_time))
   ```

4. **Added validation before logging**:
   ```python
   if not isinstance(age_seconds, (int, float)) or age_seconds < 0:
       age_seconds = 0.0
       logger.warning("Invalid age_seconds calculated, using 0.0")
   ```

## Remaining Issues to Monitor

1. **HTTP Status Codes**: Some places log `response.status` with `%s`. These should be verified if the status is an integer and use `%d` if so.

2. **Float Formatting**: Several places use `%.2f` or `%.1f` format. Ensure the values are actually numeric.

3. **Dynamic Values**: Values from external APIs or user input should be validated before logging.

## Testing Recommendations

1. Run the bot with various configurations to ensure no TypeError exceptions
2. Check logs for proper formatting of numeric values
3. Test edge cases where values might be None or invalid types
4. Monitor for any new TypeError exceptions in production logs

## Prevention Guidelines

1. Always use appropriate format specifiers:
   - `%d` for integers
   - `%f` or `%.2f` for floats
   - `%s` for strings or when type is uncertain

2. Explicitly convert types when needed:
   - `int(value)` for integers
   - `float(value)` for floats
   - `str(value)` for strings

3. Validate values before logging, especially from external sources

4. Consider using f-strings or .format() for complex formatting to avoid type issues:
   ```python
   logger.info(f"Found {len(items)} items")  # Type-safe
   ```
