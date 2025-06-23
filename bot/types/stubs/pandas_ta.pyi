"""Type stubs for pandas_ta library."""

from typing import Any, Optional, Union, overload
import pandas as pd
from pandas import DataFrame, Series

__all__ = [
    "rsi",
    "ema",
    "sma",
    "wma",
    "hma",
    "stoch",
    "stochrsi",
    "bbands",
    "atr",
    "adx",
    "macd",
    "obv",
    "mfi",
    "vwap",
    "kc",
    "donchian",
    "supertrend",
    "psar",
    "ichimoku",
    "ha",
    "rma",
    "linreg",
    "willr",
    "cci",
    "cmf",
    "cmo",
    "roc",
    "tsi",
    "ultimate",
    "volatility",
    "kurtosis",
    "skew",
    "stdev",
    "variance",
    "zscore",
    "entropy",
    "cti",
]

# RSI
@overload
def rsi(
    close: Series,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Series: ...

@overload
def rsi(
    close: DataFrame,
    length: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# EMA
@overload
def ema(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Series: ...

@overload
def ema(
    close: DataFrame,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# SMA
@overload
def sma(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Series: ...

@overload
def sma(
    close: DataFrame,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# RMA (Running Moving Average)
@overload
def rma(
    close: Series,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Series: ...

@overload
def rma(
    close: DataFrame,
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# WMA
def wma(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# HMA
def hma(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Stochastic
def stoch(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    k: Optional[int] = None,
    d: Optional[int] = None,
    smooth_k: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# Stochastic RSI
def stochrsi(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    rsi_length: Optional[int] = None,
    k: Optional[int] = None,
    d: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# Bollinger Bands
def bbands(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    std: Optional[float] = None,
    ddof: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# ATR
def atr(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    mamode: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# ADX
def adx(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    lensig: Optional[int] = None,
    scalar: Optional[float] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# MACD
def macd(
    close: Union[Series, DataFrame],
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...

# OBV
def obv(
    close: Union[Series, DataFrame],
    volume: Union[Series, DataFrame],
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# MFI
def mfi(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    volume: Union[Series, DataFrame],
    length: Optional[int] = None,
    drift: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# VWAP
def vwap(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    volume: Union[Series, DataFrame],
    anchor: Optional[str] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Linear Regression
def linreg(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Williams %R
def willr(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# CCI
def cci(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    c: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# CMF
def cmf(
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    volume: Union[Series, DataFrame],
    length: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Standard Deviation
def stdev(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    ddof: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Z-Score
def zscore(
    close: Union[Series, DataFrame],
    length: Optional[int] = None,
    std: Optional[float] = None,
    offset: Optional[int] = None,
    **kwargs: Any
) -> Union[Series, DataFrame]: ...

# Heikin Ashi
def ha(
    open_: Union[Series, DataFrame],
    high: Union[Series, DataFrame],
    low: Union[Series, DataFrame],
    close: Union[Series, DataFrame],
    offset: Optional[int] = None,
    **kwargs: Any
) -> DataFrame: ...