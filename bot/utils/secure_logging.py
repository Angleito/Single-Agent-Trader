"""
Secure logging utilities to prevent sensitive data exposure.
"""

import logging
import re
from typing import Any, Dict, List, Pattern


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that redacts sensitive information from log messages.
    """
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS: List[Pattern] = [
        # API Keys
        re.compile(r'(api[_-]?key|apikey|api_secret)[\s:=]+[\'""]?([A-Za-z0-9\-_]{20,})[\'""]?', re.IGNORECASE),
        re.compile(r'(sk-[A-Za-z0-9]{48,})', re.IGNORECASE),  # OpenAI keys
        re.compile(r'(pk-[A-Za-z0-9]{48,})', re.IGNORECASE),  # Public keys
        re.compile(r'(tvly-[A-Za-z0-9]{32,})', re.IGNORECASE),  # Tavily keys
        re.compile(r'(pplx-[A-Za-z0-9]{32,})', re.IGNORECASE),  # Perplexity keys
        re.compile(r'(jina_[A-Za-z0-9]{32,})', re.IGNORECASE),  # Jina keys
        re.compile(r'(fc-[A-Za-z0-9]{32,})', re.IGNORECASE),  # Firecrawl keys
        
        # Private Keys
        re.compile(r'(private[_-]?key)[\s:=]+[\'""]?([A-Za-z0-9\-_/+=]{32,})[\'""]?', re.IGNORECASE),
        re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----'),
        
        # Mnemonics and Seeds
        re.compile(r'(mnemonic|seed[_-]?phrase)[\s:=]+[\'""]?([a-z\s]{20,})[\'""]?', re.IGNORECASE),
        
        # Passwords
        re.compile(r'(password|passwd|pwd)[\s:=]+[\'""]?([^\s\'"",]+)[\'""]?', re.IGNORECASE),
        
        # Bearer Tokens
        re.compile(r'(bearer|token)[\s:]+([A-Za-z0-9\-_.]+)', re.IGNORECASE),
        
        # URLs with credentials
        re.compile(r'(https?://)([^:]+):([^@]+)@', re.IGNORECASE),
        
        # Base64 encoded potential secrets (min 40 chars)
        re.compile(r'[A-Za-z0-9+/]{40,}={0,2}'),
    ]
    
    # Safe placeholder for redacted content
    REDACTED = "[REDACTED]"
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and redact sensitive information from log records.
        """
        # Redact message
        if hasattr(record, 'msg'):
            record.msg = self._redact_sensitive_data(str(record.msg))
        
        # Redact args if present
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact_sensitive_data(str(v)) for k, v in record.args.items()}
            elif isinstance(record.args, (list, tuple)):
                record.args = type(record.args)(self._redact_sensitive_data(str(arg)) for arg in record.args)
        
        return True
    
    def _redact_sensitive_data(self, text: str) -> str:
        """
        Redact sensitive data from text using compiled patterns.
        """
        if not text:
            return text
            
        for pattern in self.SENSITIVE_PATTERNS:
            text = pattern.sub(self._replacement_func, text)
        
        return text
    
    def _replacement_func(self, match) -> str:
        """
        Replacement function for regex substitution.
        """
        # For patterns with groups, keep the key name but redact the value
        groups = match.groups()
        if len(groups) >= 2:
            return f"{groups[0]}={self.REDACTED}"
        return self.REDACTED


def setup_secure_logging(logger_name: str = None) -> logging.Logger:
    """
    Set up a logger with secure filtering.
    
    Args:
        logger_name: Name of the logger (None for root logger)
    
    Returns:
        Configured logger with security filter
    """
    logger = logging.getLogger(logger_name)
    
    # Add sensitive data filter
    secure_filter = SensitiveDataFilter()
    
    # Add filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(secure_filter)
    
    # Also add to the logger itself
    logger.addFilter(secure_filter)
    
    return logger


def create_secure_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Create a new secure logger with rotation support.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        max_bytes: Max size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured secure logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add secure filter
    secure_filter = SensitiveDataFilter()
    console_handler.addFilter(secure_filter)
    
    logger.addHandler(console_handler)
    
    # File handler with rotation if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(secure_filter)
        
        logger.addHandler(file_handler)
    
    return logger