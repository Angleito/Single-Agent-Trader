"""Type stubs for pandas_ta library."""

from typing import Any

# Use Any for pandas types to avoid import issues in stub files
DataFrame = Any
Series = Any

__all__ = [
    "adx",
    "atr",
    "bbands",
    "cci",
    "cmf",
    "cmo",
    "cti",
    "donchian",
    "ema",
    "entropy",
    "ha",
    "hma",
    "ichimoku",
    "kc",
    "kurtosis",
    "linreg",
    "macd",
    "mfi",
    "obv",
    "psar",
    "rma",
    "roc",
    "rsi",
    "skew",
    "sma",
    "stdev",
    "stoch",
    "stochrsi",
    "supertrend",
    "tsi",
    "ultimate",
    "variance",
    "volatility",
    "vwap",
    "willr",
    "wma",
    "zscore",
]

# RSI
def rsi(
    close: Any,
    length: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# EMA
def ema(
    close: Any, 
    length: int | None = None, 
    offset: int | None = None, 
    **kwargs: Any
) -> Any: ...

# SMA
def sma(
    close: Any, 
    length: int | None = None, 
    offset: int | None = None, 
    **kwargs: Any
) -> Any: ...

# RMA (Running Moving Average)
def rma(
    close: Any, 
    length: int | None = None, 
    offset: int | None = None, 
    **kwargs: Any
) -> Any: ...

# WMA
def wma(
    close: Any,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# HMA
def hma(
    close: Any,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Stochastic
def stoch(
    high: Any,
    low: Any,
    close: Any,
    k: int | None = None,
    d: int | None = None,
    smooth_k: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# Stochastic RSI
def stochrsi(
    close: Any,
    length: int | None = None,
    rsi_length: int | None = None,
    k: int | None = None,
    d: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# Bollinger Bands
def bbands(
    close: Any,
    length: int | None = None,
    std: float | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# ATR
def atr(
    high: Any,
    low: Any,
    close: Any,
    length: int | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# ADX
def adx(
    high: Any,
    low: Any,
    close: Any,
    length: int | None = None,
    lensig: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# MACD
def macd(
    close: Any,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# OBV
def obv(
    close: Any,
    volume: Any,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# MFI
def mfi(
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
    length: int | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# VWAP
def vwap(
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
    anchor: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Linear Regression
def linreg(
    close: Any,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Williams %R
def willr(
    high: Any,
    low: Any,
    close: Any,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# CCI
def cci(
    high: Any,
    low: Any,
    close: Any,
    length: int | None = None,
    c: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# CMF
def cmf(
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Standard Deviation
def stdev(
    close: Any,
    length: int | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Z-Score
def zscore(
    close: Any,
    length: int | None = None,
    std: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Any: ...

# Heikin Ashi
def ha(
    open_: Any,
    high: Any,
    low: Any,
    close: Any,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...
