"""
Global State Management with STM for Functional Trading Bot

This module provides Software Transactional Memory (STM) for managing
global state in a functional way.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, TypeVar

from ..effects.io import IO

T = TypeVar("T")


@dataclass
class STMRef:
    """STM reference to a value"""

    _value: Any
    _lock: Lock = field(default_factory=Lock)
    _version: int = 0

    def read(self) -> IO[Any]:
        """Read the current value"""

        def read_value():
            with self._lock:
                return copy.deepcopy(self._value)

        return IO(read_value)

    def write(self, new_value: Any) -> IO[None]:
        """Write a new value"""

        def write_value():
            with self._lock:
                self._value = copy.deepcopy(new_value)
                self._version += 1

        return IO(write_value)

    def modify(self, f: Callable[[Any], Any]) -> IO[Any]:
        """Modify the value with a function"""

        def modify_value():
            with self._lock:
                old_value = copy.deepcopy(self._value)
                new_value = f(old_value)
                self._value = copy.deepcopy(new_value)
                self._version += 1
                return new_value

        return IO(modify_value)


class StateManager:
    """Global state manager"""

    def __init__(self):
        self._refs: dict[str, STMRef] = {}
        self._lock = Lock()

    def create_ref(self, key: str, initial_value: T) -> STMRef:
        """Create a new STM reference"""
        with self._lock:
            if key in self._refs:
                raise ValueError(f"State ref {key} already exists")

            ref = STMRef(initial_value)
            self._refs[key] = ref
            return ref

    def get_ref(self, key: str) -> STMRef | None:
        """Get an existing STM reference"""
        with self._lock:
            return self._refs.get(key)

    def delete_ref(self, key: str) -> bool:
        """Delete an STM reference"""
        with self._lock:
            if key in self._refs:
                del self._refs[key]
                return True
            return False

    def get_all_refs(self) -> dict[str, STMRef]:
        """Get all STM references"""
        with self._lock:
            return copy.copy(self._refs)


# Global state manager
_state_manager = StateManager()


def get_state_manager() -> StateManager:
    """Get the global state manager"""
    return _state_manager


def create_state(key: str, initial_value: T) -> STMRef:
    """Create global state"""
    return _state_manager.create_ref(key, initial_value)


def get_state(key: str) -> STMRef | None:
    """Get global state"""
    return _state_manager.get_ref(key)
