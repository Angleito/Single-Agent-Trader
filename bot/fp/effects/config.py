"""
Configuration Effects for Functional Trading Bot

This module provides functional effects for configuration loading,
validation, and hot reloading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from .io import IO, IOEither, from_try

A = TypeVar("A")


class ConfigSource(Enum):
    ENV = "environment"
    FILE = "file"
    REMOTE = "remote"


@dataclass
class Config:
    data: dict[str, Any]
    source: ConfigSource
    version: str


@dataclass
class ValidatedConfig:
    config: Config
    validated: bool
    errors: list


@dataclass
class Secret:
    value: str
    masked: bool = True


def load_config(
    source: ConfigSource, path: str | None = None
) -> IOEither[Exception, Config]:
    """Load configuration from source"""

    def load():
        if source == ConfigSource.ENV:
            import os

            return Config(data=dict(os.environ), source=source, version="1.0")
        if source == ConfigSource.FILE and path:
            # Simulate file loading
            return Config(
                data={"trading": {"symbol": "BTC-USD"}}, source=source, version="1.0"
            )
        raise ValueError(f"Invalid config source: {source}")

    return from_try(load)


def validate_config(config: Config) -> IO[ValidatedConfig]:
    """Validate configuration"""

    def validate():
        # Simulate validation
        return ValidatedConfig(config=config, validated=True, errors=[])

    return IO(validate)


def load_secret(key: str) -> IOEither[Exception, Secret]:
    """Load secret value"""

    def load():
        # Simulate secret loading
        return Secret(value="secret_value", masked=True)

    return from_try(load)


def merge_configs(configs: list[Config]) -> IO[Config]:
    """Merge multiple configurations"""

    def merge():
        merged_data = {}
        for config in configs:
            merged_data.update(config.data)

        return Config(
            data=merged_data,
            source=ConfigSource.ENV,  # Default
            version="merged",
        )

    return IO(merge)
