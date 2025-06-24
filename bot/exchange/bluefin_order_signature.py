"""
Bluefin Order Signature System for On-Chain Settlement.

This module provides cryptographic order signature functionality required for
Bluefin DEX on-chain order settlement. Orders must be signed with ED25519 or
Secp256k1 signatures to be validated by smart contracts.
"""

import hashlib
import logging
import secrets
import time
from decimal import Decimal
from typing import Any, Literal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

logger = logging.getLogger(__name__)


class OrderSignatureError(Exception):
    """Exception raised when order signature operations fail."""


class NonceManager:
    """
    Manages nonces to prevent replay attacks.
    
    Each order must have a unique nonce to prevent the same order
    from being submitted multiple times.
    """
    
    def __init__(self):
        self._last_nonce = 0
        self._used_nonces: set[int] = set()
    
    def generate_nonce(self) -> int:
        """Generate a unique nonce for an order."""
        # Use high precision timestamp and random component for uniqueness
        timestamp_ns = int(time.time_ns())  # nanoseconds for higher precision
        random_component = secrets.randbits(32)
        
        # Combine timestamp with random component
        nonce = (timestamp_ns << 32) | random_component
        
        # Ensure uniqueness
        while nonce in self._used_nonces:
            nonce = (timestamp_ns << 32) | secrets.randbits(32)
        
        self._used_nonces.add(nonce)
        self._last_nonce = nonce
        
        # Clean up old nonces (keep only last 10000)
        if len(self._used_nonces) > 10000:
            sorted_nonces = sorted(self._used_nonces)
            self._used_nonces = set(sorted_nonces[-10000:])
        
        return nonce
    
    def is_nonce_valid(self, nonce: int) -> bool:
        """Check if a nonce is valid (not used and not too old)."""
        return nonce not in self._used_nonces and nonce > self._last_nonce - 1000000


class BluefinOrderSigner:
    """
    Handles cryptographic signing of orders for Bluefin DEX.
    
    This class provides ED25519 signature generation for orders,
    ensuring they can be validated by smart contracts on-chain.
    """
    
    def __init__(self, private_key_hex: str):
        """
        Initialize the order signer with a private key.
        
        Args:
            private_key_hex: Hexadecimal private key string
        """
        self.private_key_hex = private_key_hex
        self._private_key = self._load_private_key(private_key_hex)
        self.nonce_manager = NonceManager()
    
    def _load_private_key(self, private_key_hex: str) -> Ed25519PrivateKey:
        """Load ED25519 private key from hex string."""
        try:
            # Remove 0x prefix if present
            if private_key_hex.startswith('0x'):
                private_key_hex = private_key_hex[2:]
            
            # Convert hex to bytes (32 bytes for ED25519)
            if len(private_key_hex) != 64:  # 32 bytes * 2 hex chars
                raise OrderSignatureError(f"Invalid private key length: {len(private_key_hex)}, expected 64 hex characters")
            
            private_key_bytes = bytes.fromhex(private_key_hex)
            return Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            
        except Exception as e:
            raise OrderSignatureError(f"Failed to load private key: {e}") from e
    
    def get_public_key_hex(self) -> str:
        """Get the public key in hexadecimal format."""
        public_key = self._private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return public_key_bytes.hex()
    
    def calculate_order_hash(self, order_data: dict[str, Any]) -> bytes:
        """
        Calculate the hash of an order for signing.
        
        The hash includes all order fields that are critical for
        settlement: symbol, side, quantity, price, timestamp, nonce.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            32-byte hash of the order
        """
        try:
            # Extract required fields in deterministic order
            symbol = str(order_data.get('symbol', ''))
            side = str(order_data.get('side', ''))
            quantity = str(order_data.get('quantity', '0'))
            price = str(order_data.get('price', '0'))
            order_type = str(order_data.get('orderType', 'MARKET'))
            timestamp = str(order_data.get('timestamp', int(time.time() * 1000)))
            nonce = str(order_data.get('nonce', '0'))
            
            # Create deterministic message for hashing
            message_parts = [
                f"symbol:{symbol}",
                f"side:{side}",
                f"quantity:{quantity}",
                f"price:{price}",
                f"orderType:{order_type}",
                f"timestamp:{timestamp}",
                f"nonce:{nonce}"
            ]
            
            message = "|".join(message_parts)
            logger.debug("Order hash message: %s", message)
            
            # Calculate SHA-256 hash
            return hashlib.sha256(message.encode('utf-8')).digest()
            
        except Exception as e:
            raise OrderSignatureError(f"Failed to calculate order hash: {e}") from e
    
    def sign_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
        """
        Sign an order with ED25519 signature.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            Order data with signature fields added
        """
        try:
            # Generate nonce if not present
            if 'nonce' not in order_data:
                order_data['nonce'] = self.nonce_manager.generate_nonce()
            
            # Add timestamp if not present
            if 'timestamp' not in order_data:
                order_data['timestamp'] = int(time.time() * 1000)
            
            # Calculate order hash
            order_hash = self.calculate_order_hash(order_data)
            
            # Sign the hash
            signature_bytes = self._private_key.sign(order_hash)
            signature_hex = signature_bytes.hex()
            
            # Add signature fields to order
            signed_order = order_data.copy()
            signed_order.update({
                'signature': signature_hex,
                'publicKey': self.get_public_key_hex(),
                'orderHash': order_hash.hex(),
                'signatureType': 'ED25519'
            })
            
            logger.debug(
                "Order signed - Hash: %s, Signature: %s",
                order_hash.hex()[:16] + "...",
                signature_hex[:16] + "..."
            )
            
            return signed_order
            
        except Exception as e:
            raise OrderSignatureError(f"Failed to sign order: {e}") from e
    
    def verify_signature(self, order_data: dict[str, Any]) -> bool:
        """
        Verify an order signature.
        
        Args:
            order_data: Signed order data
            
        Returns:
            True if signature is valid
        """
        try:
            signature_hex = order_data.get('signature')
            public_key_hex = order_data.get('publicKey')
            
            if not signature_hex or not public_key_hex:
                return False
            
            # Recreate order hash
            order_data_copy = order_data.copy()
            # Remove signature fields for hash calculation
            for key in ['signature', 'publicKey', 'orderHash', 'signatureType']:
                order_data_copy.pop(key, None)
            
            order_hash = self.calculate_order_hash(order_data_copy)
            
            # Load public key and verify
            public_key_bytes = bytes.fromhex(public_key_hex)
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            signature_bytes = bytes.fromhex(signature_hex)
            public_key.verify(signature_bytes, order_hash)
            
            return True
            
        except Exception as e:
            logger.debug("Signature verification failed: %s", e)
            return False


class BluefinOrderSignatureManager:
    """
    High-level manager for order signature operations.
    
    Provides convenient methods for signing different types of orders
    (market, limit, stop) with proper error handling and logging.
    """
    
    def __init__(self, private_key_hex: str):
        """Initialize with private key."""
        self.signer = BluefinOrderSigner(private_key_hex)
    
    def sign_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        estimated_fee: Decimal = Decimal("0")
    ) -> dict[str, Any]:
        """Sign a market order."""
        order_data = {
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'orderType': 'MARKET',
            'estimated_fee': float(estimated_fee),
        }
        
        return self.signer.sign_order(order_data)
    
    def sign_limit_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        price: Decimal,
        estimated_fee: Decimal = Decimal("0")
    ) -> dict[str, Any]:
        """Sign a limit order."""
        order_data = {
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'price': float(price),
            'orderType': 'LIMIT',
            'estimated_fee': float(estimated_fee),
        }
        
        return self.signer.sign_order(order_data)
    
    def sign_stop_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: Decimal,
        stop_price: Decimal,
        estimated_fee: Decimal = Decimal("0")
    ) -> dict[str, Any]:
        """Sign a stop order."""
        order_data = {
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'stopPrice': float(stop_price),
            'orderType': 'STOP',
            'estimated_fee': float(estimated_fee),
        }
        
        return self.signer.sign_order(order_data)
    
    def verify_order_signature(self, signed_order: dict[str, Any]) -> bool:
        """Verify a signed order."""
        return self.signer.verify_signature(signed_order)
    
    def get_public_key(self) -> str:
        """Get the public key for this signer."""
        return self.signer.get_public_key_hex()