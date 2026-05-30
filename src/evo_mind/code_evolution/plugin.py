"""CodeEvolutionPlugin — evo-mind plugin for automatic code optimization.

Implements on_evolution_step hook to trigger code fixes when new
correction_pattern rules are discovered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RuleType

logger = logging.getLogger(__name__)


class CodeEvolutionPlugin:
    """Evo-mind plugin: watches evolution rules and auto-fixes code.

    Plugin properties required by the Plugin Protocol:
    - name: str
    - version: str
    - on_load() / on_unload()
    """

    name: str = "code-evolution"
    version: str = "0.1.0"

    def __init__(
        self,
        project_root: Path | str = ".",
        auto_apply: bool = False,
        test_command: str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.auto_apply = auto_apply
        self.test_command = test_command
        self._store: MemoryStore | None = None
        self._loaded = False

    async def on_load(self) -> None:
        """Plugin loaded — register with hook system."""
        self._loaded = True
        logger.info("code_evolution_plugin_loaded",
                     project=str(self.project_root),
                     auto_apply=self.auto_apply)

    async def on_unload(self) -> None:
        """Plugin unloading — cleanup."""
        self._loaded = False
        logger.info("code_evolution_plugin_unloaded")

    # ---- Hook implementations ----

    async def on_evolution_step(self, rules: list) -> None:
        """Called when new evolution rules are discovered.

        If correction_pattern rules are found, trigger code analysis.
        """
        correction_rules = [r for r in rules if getattr(r, 'rule_type', None) == RuleType.CORRECTION_PATTERN]
        if not correction_rules:
            return

        logger.info("code_evolution_triggered", correction_rules=len(correction_rules))

        if not self._store:
            logger.warning("no_store_available")
            return

        # Run optimization using discovered rules
        from evo_mind.code_evolution.engine import CodeEvolutionEngine

        engine = CodeEvolutionEngine(
            self._store,
            self.project_root,
            test_command=self.test_command,
            max_fixes_per_run=5,
            auto_apply=self.auto_apply,
        )

        try:
            result = await engine.optimize()
            logger.info(
                "code_evolution_completed",
                issues=result.issues_found,
                fixes=result.fixes_applied,
                succeeded=result.fixes_succeeded,
                failed=result.fixes_failed,
            )
        except Exception as e:
            logger.error("code_evolution_failed", error=str(e))

    async def on_memory_created(self, memory) -> None:
        """Track memories of type 'code_issue' for pattern analysis."""
        pass  # Implement if needed

    async def on_retrieval_post_search(self, results, query) -> None:
        """Inject code-evolution context into search results."""
        pass  # Implement if needed

    # ---- Integration ----

    def set_store(self, store: MemoryStore) -> None:
        """Inject MemoryStore reference (called by application)."""
        self._store = store


# Re-export for discovery
__all__ = ["CodeEvolutionPlugin"]
