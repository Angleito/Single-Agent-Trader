# Bluefin Order Signature System

## Overview

The Bluefin Order Signature System provides cryptographic signing of orders for on-chain settlement on the Bluefin DEX. This system is **critical for live trading** - without proper order signatures, orders cannot be validated by smart contracts and will be rejected.

## 🔐 Key Features

- **ED25519 Cryptographic Signatures**: Industry-standard elliptic curve signatures
- **Order Hash Calculation**: Deterministic hashing of all order fields
- **Nonce Management**: Prevents replay attacks with unique nonces
- **Signature Verification**: Built-in verification for testing and validation
- **Error Handling**: Comprehensive error handling with detailed messages
- **Backward Compatibility**: Works seamlessly with existing paper trading

## 📁 Files

### Core Implementation
- `bot/exchange/bluefin_order_signature.py` - Main signature system implementation
- `bot/exchange/bluefin.py` - Updated to use signatures for live trading
- `bot/exchange/bluefin_client.py` - Updated to handle signed orders

### Testing
- `test_signature_standalone.py` - Standalone test suite
- `test_bluefin_signature_system.py` - Full integration test suite

### Dependencies
- `pyproject.toml` - Updated with cryptography dependency

## 🚀 Usage

### Automatic Integration

The signature system is automatically integrated into the Bluefin exchange:

```python
# For paper trading (dry_run=True)
# - Orders are unsigned
# - No signature validation required
# - Works as before

# For live trading (dry_run=False)
# - Orders are automatically signed with ED25519
# - Signatures include hash, nonce, timestamp
# - Orders rejected if signature invalid
```

### Environment Setup

1. **Install Dependencies**:
   ```bash
   poetry install  # Installs cryptography automatically
   ```

2. **Set Private Key**:
   ```bash
   # Required for live trading
   export EXCHANGE__BLUEFIN_PRIVATE_KEY="your_64_char_hex_private_key"
   ```

3. **Enable Live Trading**:
   ```bash
   export SYSTEM__DRY_RUN=false
   ```

### Manual Usage

```python
from bot.exchange.bluefin_order_signature import BluefinOrderSignatureManager

# Initialize with private key
manager = BluefinOrderSignatureManager("your_private_key_hex")

# Sign a market order
signed_order = manager.sign_market_order(
    symbol='SUI-PERP',
    side='BUY',
    quantity=Decimal('10.0'),
    estimated_fee=Decimal('0.01')
)

# Verify signature
is_valid = manager.verify_order_signature(signed_order)
```

## 🔧 Technical Details

### Order Hash Calculation

Orders are hashed using SHA-256 with the following fields in deterministic order:

```
symbol:SUI-PERP|side:BUY|quantity:10.0|price:3.45|orderType:LIMIT|timestamp:1703123456789|nonce:7519603463191271924879508216
```

### Signature Fields

Signed orders include these additional fields:

- `signature`: 128-character hex string (64 bytes)
- `publicKey`: 64-character hex string (32 bytes)  
- `orderHash`: 64-character hex string (32 bytes)
- `signatureType`: "ED25519"
- `nonce`: Unique integer identifier
- `timestamp`: Unix timestamp in milliseconds

### Nonce Generation

Nonces are generated using:
- High-precision timestamp (nanoseconds)
- 32-bit random component
- Uniqueness verification
- Automatic cleanup of old nonces

## 🧪 Testing

### Run Standalone Test

```bash
poetry run python test_signature_standalone.py
```

Expected output:
```
🚀 Starting Bluefin Order Signature System Test
🧪 Testing Bluefin Order Signature System...
✅ Signer initialized successfully
✅ Public key generated: 207a067892821e25...
✅ Order signed successfully
✅ Signature fields validated
✅ Signature verification passed
✅ Invalid signature detection works
✅ Nonce uniqueness verified
🎉 All signature system tests passed!
✅ Signature system is working correctly!
🔐 Orders can now be signed for on-chain settlement
```

### Run Full Integration Test

```bash
poetry run python test_bluefin_signature_system.py
```

## 🔒 Security Features

### Private Key Protection

- Private keys are stored securely in environment variables
- Keys are masked in logs and error messages
- Automatic validation of key format and length

### Replay Attack Prevention

- Unique nonces prevent order replay
- Timestamp validation ensures freshness
- Nonce tracking prevents reuse

### Signature Validation

- Full cryptographic verification
- Hash integrity checking
- Public key validation

## 📊 Logging

The system provides comprehensive logging:

```bash
# Paper Trading
📊 Paper trading mode - order signatures not required

# Live Trading - Successful
✅ Order signature system initialized for live trading
✍️ Market order signed - Hash: 1a2b3c4d..., Public Key: 5e6f7890...
🔐 Placing BUY market order: 10.0 SUI-PERP [SIGNED]

# Live Trading - Errors
❌ Failed to initialize order signature system: Invalid private key
❌ Failed to sign market order: Order hash calculation failed
```

## 🚨 Error Handling

### Common Errors

1. **Missing Cryptography Library**:
   ```
   ⚠️ Order signature system not available - install cryptography package
   ```
   **Solution**: `poetry install`

2. **Invalid Private Key**:
   ```
   ❌ Order signature initialization failed: Invalid private key length
   ```
   **Solution**: Ensure private key is 64 hex characters

3. **Missing Private Key**:
   ```
   Private key required for live trading. Set EXCHANGE__BLUEFIN_PRIVATE_KEY
   ```
   **Solution**: Set environment variable

### Error Recovery

- System falls back to paper trading if signature fails
- Detailed error messages for debugging
- Graceful degradation with warnings

## 🔄 Migration Guide

### From Unsigned to Signed Orders

1. **Update Dependencies**:
   ```bash
   poetry lock
   poetry install
   ```

2. **Set Environment Variables**:
   ```bash
   export EXCHANGE__BLUEFIN_PRIVATE_KEY="your_key"
   export SYSTEM__DRY_RUN=false
   ```

3. **Test Signature System**:
   ```bash
   poetry run python test_signature_standalone.py
   ```

4. **Deploy and Monitor**:
   - Check logs for signature confirmation
   - Monitor order success rates
   - Verify on-chain settlement

### Backward Compatibility

- Paper trading continues to work unchanged
- No breaking changes to existing APIs
- Optional signature verification in development

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│              Trading Bot                │
├─────────────────────────────────────────┤
│         Bluefin Exchange               │
│  ┌─────────────────────────────────┐    │
│  │    Order Placement Methods      │    │
│  │  - place_market_order()        │    │
│  │  - place_limit_order()         │    │
│  │  - cancel_order()               │    │
│  └─────────────────────────────────┘    │
│              │                          │
│              ▼                          │
│  ┌─────────────────────────────────┐    │
│  │  Order Signature Manager        │    │
│  │  - sign_market_order()          │    │
│  │  - sign_limit_order()           │    │
│  │  - verify_signature()           │    │
│  └─────────────────────────────────┘    │
│              │                          │
│              ▼                          │
│  ┌─────────────────────────────────┐    │
│  │     Bluefin Service Client      │    │
│  │  - place_order() [signed]       │    │
│  │  - cancel_order()               │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│              │                          │
│              ▼                          │
├─────────────────────────────────────────┤
│         Bluefin Smart Contract          │
│  ┌─────────────────────────────────┐    │
│  │    Signature Verification       │    │
│  │  - Validate ED25519 signature   │    │
│  │  - Check order hash             │    │
│  │  - Verify nonce uniqueness      │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## 🚀 Next Steps

### Immediate Actions

1. **Test in Development**:
   - Run signature tests
   - Verify paper trading still works
   - Test error conditions

2. **Configure Live Environment**:
   - Set private key environment variable
   - Enable live trading mode
   - Monitor initial orders

3. **Production Deployment**:
   - Deploy with signature system
   - Monitor order success rates
   - Verify on-chain settlement

### Future Enhancements

- **Batch Order Signatures**: Sign multiple orders in one operation
- **Hardware Security Module**: Support for HSM-based signing
- **Multi-Signature**: Support for multi-signature orders
- **Signature Caching**: Cache signatures for performance
- **Audit Logging**: Enhanced audit trail for signed orders

## 📞 Support

For issues with the signature system:

1. **Check Logs**: Look for signature-related error messages
2. **Run Tests**: Execute standalone signature tests
3. **Verify Environment**: Ensure private key and dependencies are set
4. **Fallback Mode**: Use paper trading to isolate signature issues

---

**⚠️ CRITICAL**: Live trading on Bluefin **requires** the signature system. Orders without valid signatures will be rejected by smart contracts.