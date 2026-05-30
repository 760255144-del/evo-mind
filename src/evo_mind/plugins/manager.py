"""PluginManager: discovers, loads, and manages lifecycle of plugins via pluggy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pluggy

from evo_mind.plugins.spec import EvoMindHooks

logger = logging.getLogger(__name__)

PROJECT_NAME = "evo_mind"


class PluginManager:
    """Discovers, loads, and manages lifecycle of evo-mind plugins."""

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT_NAME)
        self._pm.add_hookspecs(EvoMindHooks)
        self._plugins: dict[str, object] = {}
        self._loaded = False

    @property
    def hook(self) -> EvoMindHooks:
        """Access the hook caller proxy. Usage: pm.hook.on_memory_created(memory=mem)"""
        return self._pm.hook  # type: ignore[return-value]

    def discover(self, paths: list[Path] | None = None) -> int:
        """Discover plugins via setuptools entry points and optional paths."""
        count = 0

        # Load from setuptools entry points
        self._pm.load_setuptools_entrypoints(PROJECT_NAME)
        count += len(self._pm.get_plugins())

        # Load from explicit paths
        if paths:
            for path in paths:
                if path.is_file() and path.suffix == ".py":
                    self._register_from_file(path)
                    count += 1
                elif path.is_dir():
                    for py_file in path.glob("*.py"):
                        if py_file.name.startswith("_"):
                            continue
                        self._register_from_file(py_file)
                        count += 1

        logger.info("plugins_discovered", count=count)
        return count

    async def load_all(self) -> None:
        """Initialize all discovered plugins."""
        if self._loaded:
            return

        for name, plugin in self._pm.list_name_plugin():
            self._plugins[name] = plugin
            if hasattr(plugin, "on_load") and callable(plugin.on_load):
                try:
                    await plugin.on_load()  # type: ignore[union-attr]
                    logger.info("plugin_loaded", name=name)
                except Exception:
                    logger.exception("plugin_load_failed", name=name)

        self._loaded = True

    async def unload_all(self) -> None:
        """Shutdown all loaded plugins."""
        for name, plugin in list(self._plugins.items()):
            if hasattr(plugin, "on_unload") and callable(plugin.on_unload):
                try:
                    await plugin.on_unload()  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning("plugin_unload_failed", name=name, error=str(e))
        self._plugins.clear()
        self._loaded = False

    def get_plugin(self, name: str) -> object | None:
        """Get a specific loaded plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List names of loaded plugins."""
        return list(self._plugins.keys())

    def _register_from_file(self, path: Path) -> None:
        """Register a plugin from a Python file path."""
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location(
                f"evo_mind_plugin_{path.stem}", str(path)
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Find plugin class in module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and hasattr(attr, "name")
                        and attr_name.endswith("Plugin")
                    ):
                        self._pm.register(attr(), name=getattr(attr, "name"))
            else:
                logger.warning("plugin_import_failed", path=str(path))
        except Exception as e:
            logger.error("plugin_import_failed", path=str(path), error=str(e))
