"""Type stubs for pandas_ta library."""

from typing import Any, overload

from pandas import DataFrame, Series

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
@overload
def rsi(
    close: Series,
    length: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series: ...
@overload
def rsi(
    close: DataFrame,
    length: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# EMA
@overload
def ema(
    close: Series, length: int | None = None, offset: int | None = None, **kwargs: Any
) -> Series: ...
@overload
def ema(
    close: DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# SMA
@overload
def sma(
    close: Series, length: int | None = None, offset: int | None = None, **kwargs: Any
) -> Series: ...
@overload
def sma(
    close: DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# RMA (Running Moving Average)
@overload
def rma(
    close: Series, length: int | None = None, offset: int | None = None, **kwargs: Any
) -> Series: ...
@overload
def rma(
    close: DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# WMA
def wma(
    close: Series | DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# HMA
def hma(
    close: Series | DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Stochastic
def stoch(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    k: int | None = None,
    d: int | None = None,
    smooth_k: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# Stochastic RSI
def stochrsi(
    close: Series | DataFrame,
    length: int | None = None,
    rsi_length: int | None = None,
    k: int | None = None,
    d: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# Bollinger Bands
def bbands(
    close: Series | DataFrame,
    length: int | None = None,
    std: float | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# ATR
def atr(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    length: int | None = None,
    mamode: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# ADX
def adx(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    length: int | None = None,
    lensig: int | None = None,
    scalar: float | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# MACD
def macd(
    close: Series | DataFrame,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...

# OBV
def obv(
    close: Series | DataFrame,
    volume: Series | DataFrame,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# MFI
def mfi(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    volume: Series | DataFrame,
    length: int | None = None,
    drift: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# VWAP
def vwap(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    volume: Series | DataFrame,
    anchor: str | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Linear Regression
def linreg(
    close: Series | DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Williams %R
def willr(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# CCI
def cci(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    length: int | None = None,
    c: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# CMF
def cmf(
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    volume: Series | DataFrame,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Standard Deviation
def stdev(
    close: Series | DataFrame,
    length: int | None = None,
    ddof: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Z-Score
def zscore(
    close: Series | DataFrame,
    length: int | None = None,
    std: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | DataFrame: ...

# Heikin Ashi
def ha(
    open_: Series | DataFrame,
    high: Series | DataFrame,
    low: Series | DataFrame,
    close: Series | DataFrame,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame: ...
