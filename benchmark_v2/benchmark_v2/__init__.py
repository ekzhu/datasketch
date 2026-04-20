"""Benchmark v2 package."""

from importlib import import_module

__all__ = ["get_cli_app"]


def get_cli_app():
    """Import the Typer CLI lazily to avoid import-time side effects."""
    module = import_module("benchmark_v2.cli")
    return module.app
