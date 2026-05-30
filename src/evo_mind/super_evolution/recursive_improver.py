"""Recursive Improver — the system modifies and improves its own source code.

This is the key to "super" evolution: the ability to safely read, analyze,
modify, test, and hot-reload its own implementation. Changes are validated
in isolation before being committed.

Safety guarantees:
- All changes are written to temp files first
- Syntax validation before applying
- Test suite must pass before accepting changes
- Rollback on failure
- Changes recorded as memories for audit trail
"""

from __future__ import annotations

import ast
import difflib
import importlib
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


# ---- Types ----

@dataclass
class CodeChange:
    """A single proposed code change."""
    file_path: str
    original_lines: list[str]
    new_lines: list[str]
    line_start: int
    line_end: int
    reason: str
    risk: str  # "safe" | "moderate" | "risky"
    applied: bool = False
    validated: bool = False
    rollback: str | None = None  # Backup content for rollback


@dataclass
class ModificationResult:
    """Outcome of a self-modification attempt."""
    changes_proposed: int
    changes_applied: int
    changes_validated: int
    changes_rolled_back: int
    tests_passed_before: bool
    tests_passed_after: bool
    improvements: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ---- Improver ----

class RecursiveImprover:
    """Self-modifying system: analyzes own code, proposes improvements, safely applies them.

    The system examines its own source directory, identifies potential
    improvements using learned rules and static analysis, generates changes,
    validates them, and applies only those that pass all checks.
    """

    def __init__(
        self,
        store: MemoryStore,
        source_root: Path | None = None,
        test_command: str = "python -m pytest tests/ -x --tb=short 2>&1",
        auto_apply: bool = False,
        backup_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.source_root = Path(source_root or Path(__file__).parent.parent)
        self.test_command = test_command
        self.auto_apply = auto_apply
        self.backup_dir = backup_dir or Path(tempfile.mkdtemp(prefix="evo_backup_"))

    async def improve(self, max_changes: int = 5) -> dict[str, Any]:
        """Run the recursive self-improvement pipeline.

        Returns:
            dict with changes_made, tests_passed, improvements, errors
        """
        result = ModificationResult(
            changes_proposed=0,
            changes_applied=0,
            changes_validated=0,
            changes_rolled_back=0,
            tests_passed_before=False,
            tests_passed_after=False,
        )

        # 1. Run tests before
        result.tests_passed_before = await self._run_tests()
        logger.info("pre_tests: passed=%s", result.tests_passed_before)

        # 2. Identify improvement opportunities
        opportunities = await self._analyze_self()
        result.changes_proposed = len(opportunities)

        if not opportunities:
            logger.info("no_improvements_found")
            return result.to_dict()

        # 3. Generate code changes
        changes = await self._generate_changes(opportunities[:max_changes])

        # 4. Validate each change
        for change in changes:
            try:
                # Write to temp file and validate syntax
                tmp_path = Path(tempfile.mktemp(suffix=".py"))
                new_content = "\n".join(change.new_lines)
                tmp_path.write_text(new_content)

                # Syntax check
                is_valid = await self._validate_syntax(tmp_path)
                if not is_valid:
                    change.validated = False
                    result.errors.append(f"Syntax error in {change.file_path}")
                    continue

                change.validated = True
                result.changes_validated += 1

                # Apply if safe or auto-apply enabled
                if change.risk == "safe" or self.auto_apply:
                    success = await self._apply_change(change)
                    if success:
                        change.applied = True
                        result.changes_applied += 1
                        result.improvements.append(
                            f"{change.file_path}: L{change.line_start}-{change.line_end}: {change.reason}"
                        )
                    else:
                        result.changes_rolled_back += 1
            except Exception as e:
                result.errors.append(str(e))
                logger.warning("change_failed: %s", e)

        # 5. Run tests after
        result.tests_passed_after = await self._run_tests()

        # 6. Rollback all changes if tests fail
        if not result.tests_passed_after and result.changes_applied > 0:
            await self._rollback_all(changes)
            result.changes_rolled_back = result.changes_applied
            result.changes_applied = 0
            logger.warning("changes_rolled_back: tests_failed")

        # 7. Record results
        await self._record_modification(result)

        logger.info(
            "self_modification_complete: applied=%s validated=%s tests_pass=%s",
            result.changes_applied, result.changes_validated, result.tests_passed_after,
        )

        return result.to_dict()

    async def _analyze_self(self) -> list[dict[str, Any]]:
        """Analyze own source code for improvement opportunities."""
        opportunities: list[dict[str, Any]] = []

        # Scan all Python files in own source
        py_files = list(self.source_root.rglob("*.py"))
        py_files = [
            f for f in py_files
            if "__pycache__" not in str(f)
            and not f.name.startswith("test_")
            and "super_evolution" not in str(f)  # Don't modify ourselves during initial run
        ]

        for file_path in py_files[:20]:  # Limit scope
            try:
                content = file_path.read_text()
                issues = self._find_code_improvements(str(file_path), content)
                opportunities.extend(issues)
            except Exception:
                pass

        # Also check learned evolution rules for applicable patterns
        try:
            rows = await self.store.db.fetch_all(
                """SELECT * FROM evolution_rules
                   WHERE rule_type IN ('correction_pattern', 'strategy_heuristic')
                     AND confidence >= 0.6 AND status = 'active'
                   ORDER BY confidence DESC LIMIT 10"""
            )
            for row in rows:
                condition = __import__('json').loads(row["condition_json"])
                action = __import__('json').loads(row["action_json"])
                opportunities.append({
                    "type": "evolution_rule",
                    "rule_id": row["id"],
                    "confidence": row["confidence"],
                    "condition": condition,
                    "action": action,
                    "label": row["label"],
                })
        except Exception:
            pass

        return opportunities

    def _find_code_improvements(
        self, file_path: str, content: str
    ) -> list[dict[str, Any]]:
        """Find specific code improvements in a file."""
        issues: list[dict[str, Any]] = []
        lines = content.split("\n")

        import re
        for i, line in enumerate(lines, 1):
            # Bare except
            if re.match(r'^\s*except\s*:\s*$', line):
                issues.append({
                    "type": "bare_except",
                    "file": file_path, "line": i,
                    "severity": "warning",
                    "fix": f"{line[:line.index('except')]}except Exception as e:",
                })

            # Missing type hints on public functions
            if re.match(r'^\s*def\s+\w+\s*\([^)]*\)\s*:\s*$', line):
                if not line.rstrip().endswith("->"):
                    # Heuristic: suggest adding type hints
                    pass  # Too noisy, skip for now

            # Debugging imports left in
            if re.match(r'^\s*import\s+pdb\b', line) or re.match(r'^\s*from\s+pdb\b', line):
                issues.append({
                    "type": "debug_import",
                    "file": file_path, "line": i,
                    "severity": "warning",
                    "fix": f"# {line}  # Removed debug import",
                })

            # Hard-coded magic numbers
            numbers = re.findall(r'(?<![.\w])(\d{4,})(?![.\w])', line)
            for num in numbers:
                if num not in ("1000", "10000", "100000"):  # Common round numbers
                    issues.append({
                        "type": "magic_number",
                        "file": file_path, "line": i,
                        "severity": "suggestion",
                        "fix": f"# TODO: Consider extracting magic number {num} to a named constant",
                    })

        return issues

    async def _generate_changes(
        self, opportunities: list[dict[str, Any]]
    ) -> list[CodeChange]:
        """Generate concrete code changes from improvement opportunities."""
        changes: list[CodeChange] = []

        for opp in opportunities:
            opp_type = opp.get("type", "")

            if opp_type in ("bare_except", "debug_import", "magic_number"):
                file_path = opp["file"]
                try:
                    path = Path(file_path)
                    if not path.exists():
                        continue

                    original_content = path.read_text()
                    original_lines = original_content.split("\n")
                    line_num = opp["line"]

                    if line_num < 1 or line_num > len(original_lines):
                        continue

                    new_lines = list(original_lines)
                    if opp_type == "bare_except":
                        new_lines[line_num - 1] = opp["fix"]
                        reason = f"Specify exception type at line {line_num}"
                        risk = "safe"
                    elif opp_type == "debug_import":
                        new_lines[line_num - 1] = opp["fix"]
                        reason = f"Remove debug import at line {line_num}"
                        risk = "safe"
                    elif opp_type == "magic_number":
                        # Comment-based suggestion, not actually changing code
                        reason = f"Flag magic number at line {line_num}"
                        risk = "moderate"
                        continue  # Skip actual changes for suggestions
                    else:
                        continue

                    changes.append(CodeChange(
                        file_path=file_path,
                        original_lines=original_lines,
                        new_lines=new_lines,
                        line_start=line_num,
                        line_end=line_num,
                        reason=reason,
                        risk=risk,
                    ))

                except Exception as e:
                    logger.debug("change_generation_failed: file=%s error=%s", file_path, e)

            elif opp_type == "evolution_rule":
                # Apply evolution rules as code changes
                action = opp.get("action", {})
                condition = opp.get("condition", {})
                changes.append(CodeChange(
                    file_path=str(self.source_root / "evolution_applied.py"),
                    original_lines=["# Rule application placeholder"],
                    new_lines=[
                        f"# Applied evolution rule: {opp.get('label', 'unknown')}",
                        f"# Confidence: {opp.get('confidence', 0):.2f}",
                        f"# Condition: {condition}",
                        f"# Action: {action}",
                    ],
                    line_start=1,
                    line_end=1,
                    reason=f"Apply learned rule: {opp.get('label', 'unknown')}",
                    risk="moderate",
                ))

        return changes

    async def _validate_syntax(self, file_path: Path) -> bool:
        """Validate that a file has valid Python syntax."""
        try:
            source = file_path.read_text()
            ast.parse(source)
            return True
        except SyntaxError as e:
            logger.debug("syntax_validation_failed: path=%s error=%s", file_path, e)
            return False

    async def _apply_change(self, change: CodeChange) -> bool:
        """Apply a code change, with backup for rollback."""
        try:
            path = Path(change.file_path)

            # Backup original
            change.rollback = path.read_text()

            # Apply change
            new_content = "\n".join(change.new_lines)
            path.write_text(new_content)

            logger.info("change_applied: file=%s reason=%s", change.file_path, change.reason)
            return True

        except Exception as e:
            logger.error("apply_failed: %s", e)
            return False

    async def _rollback_all(self, changes: list[CodeChange]) -> None:
        """Rollback all applied changes."""
        for change in changes:
            if change.applied and change.rollback is not None:
                try:
                    Path(change.file_path).write_text(change.rollback)
                    logger.info("change_rolled_back: file=%s", change.file_path)
                except Exception as e:
                    logger.error("rollback_failed: file=%s error=%s", change.file_path, e)

    async def _run_tests(self) -> bool:
        """Run the test suite. Returns True ONLY if tests actually pass.

        Safety-critical: does NOT assume OK on failure. A real test suite
        must exist and pass for self-modification to be accepted.
        """
        try:
            result = subprocess.run(
                self.test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.source_root.parent.parent),
            )
            # Explicitly require tests to pass — no "assume OK" fallback
            if result.returncode != 0:
                logger.warning("tests_failed: returncode=%s stderr=%s", result.returncode,
                               result.stderr[:200] if result.stderr else "")
                return False
            # If there are no tests at all, that's also a failure
            output = (result.stdout or "") + (result.stderr or "")
            if "no tests" in output.lower() or "0 items" in output.lower():
                logger.warning("no_tests_found: output=%s", output[:200])
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.warning("tests_timed_out")
            return False
        except Exception as e:
            logger.error("test_runner_failed: %s", e)
            return False  # Never assume OK on failure

    async def _record_modification(self, result: ModificationResult) -> None:
        """Record self-modification results as memories."""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.PROCEDURAL,
            content={
                "type": "self_modification",
                "changes_applied": result.changes_applied,
                "changes_validated": result.changes_validated,
                "changes_rolled_back": result.changes_rolled_back,
                "tests_passed": result.tests_passed_after,
                "improvements": result.improvements,
                "errors": result.errors,
            },
            importance=1.0,
            source="plugin",
            tags=["super-evolution", "self-modification", "recursive"],
        ))

    async def hot_reload(self, module_name: str) -> bool:
        """Attempt to hot-reload a modified module."""
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                logger.info("module_reloaded: name=%s", module_name)
                return True
        except Exception as e:
            logger.error("reload_failed: name=%s error=%s", module_name, e)
        return False
