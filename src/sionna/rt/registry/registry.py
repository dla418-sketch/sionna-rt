#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for implementing registries"""

class Registry:
    """Base class for registries"""

    def __init__(self):
        self._registry = {}

    def register(self, obj=None, name=None):
        """Register an object with an optional name.
            Can be used as a decorator.
        """
        if obj is None:
            # If no object is provided, return the decorator
            def decorator(inner_obj):
                nonlocal name
                if name is None:
                    name = inner_obj.__name__
                self._registry[name] = inner_obj
                return inner_obj
            return decorator
        else:
            # Get name from object if none is provided
            if name is None:
                name = obj.__name__

            # Remove object from list if it already exists
            if name in self._registry:
                self.unregister(name)

            # Register object
            self._registry[name] = obj

    def get(self, name):
        """Retrieve an object by its name"""
        if name not in self._registry:
            raise KeyError(f"No object found with name: {name}")
        return self._registry.get(name)

    def unregister(self, name):
        """Remove an object from the registry"""
        if name in self._registry:
            del self._registry[name]

    def list(self):
        """List all registered names"""
        return list(self._registry.keys())
