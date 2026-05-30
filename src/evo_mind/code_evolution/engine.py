"""Code Evolution Engine — self-optimizing code pipeline.

Pipeline: Analyze → Diagnose → Fix → Test → Learn
All steps recorded as memories in the shared MemoryStore.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RelationType

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Data Types ----

@dataclass(slots=True)
class CodeIssue:
    """A detected code issue."""
    file_path: str
    line: int
    severity: str  # "error" | "warning" | "suggestion"
    category: str  # "bug" | "perf" | "style" | "security" | "complexity"
    title: str
    description: str
    suggestion: str | None = None
    rule_id: str | None = None  # EvolutionRule ID that found this


@dataclass(slots=True)
class CodeFix:
    """A proposed or applied code fix."""
    issue: CodeIssue
    original_code: str
    fixed_code: str
    diff: str
    applied: bool = False
    success: bool | None = None  # None = not tested yet
    test_output: str | None = None


@dataclass(slots=True)
class OptimizationResult:
    """Result of a code optimization run."""
    files_scanned: int
    issues_found: int
    fixes_generated: int
    fixes_applied: int
    fixes_succeeded: int
    fixes_failed: int
    memories_recorded: int
    duration_seconds: float


# ---- Engine ----

class CodeEvolutionEngine:
    """Self-optimizing code pipeline integrated with evo-mind memory.

    Watches evolution rules (correction_patterns), applies them to code,
    runs tests, and records outcomes as feedback memories for future learning.
    """

    def __init__(
        self,
        store: MemoryStore,
        project_root: Path | str,
        test_command: str | None = None,
        max_fixes_per_run: int = 5,
        auto_apply: bool = False,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root)
        self.test_command = test_command or "pytest -x --tb=short"
        self.max_fixes_per_run = max_fixes_per_run
        self.auto_apply = auto_apply

    async def optimize(self) -> OptimizationResult:
        """Run full code optimization pipeline."""
        start_time = asyncio.get_event_loop().time()

        session_id = await self.store.start_session({
            "phase": "code_optimization",
            "started_at": _now(),
        })

        try:
            # 1. Analyze: find issues using evolution rules + static analysis
            issues = await self._analyze()
            logger.info("issues_found", count=len(issues))

            # 2. Diagnose: prioritize and filter issues
            prioritized = self._prioritize(issues)[:self.max_fixes_per_run]
            logger.info("issues_prioritized", count=len(prioritized))

            # 3. Fix: generate and apply fixes
            fixes = await self._generate_fixes(prioritized)

            # 4. Test: run tests after each fix
            for fix in fixes:
                if fix.applied:
                    success, output = await self._run_tests()
                    fix.success = success
                    fix.test_output = output

            # 5. Learn: record everything as memories
            memories_count = await self._record_results(fixes, session_id)

            duration = asyncio.get_event_loop().time() - start_time

            result = OptimizationResult(
                files_scanned=len(set(i.file_path for i in issues)),
                issues_found=len(issues),
                fixes_generated=len(fixes),
                fixes_applied=sum(1 for f in fixes if f.applied),
                fixes_succeeded=sum(1 for f in fixes if f.success is True),
                fixes_failed=sum(1 for f in fixes if f.success is False),
                memories_recorded=memories_count,
                duration_seconds=round(duration, 2),
            )

            from dataclasses import asdict
            await self.store.end_session(session_id, json.dumps(asdict(result)))
            return result

        except Exception as e:
            logger.exception("optimization_failed")
            await self.store.end_session(session_id, f"Failed: {e}")
            raise

    async def _analyze(self) -> list[CodeIssue]:
        """Run analysis: evolution rules + static patterns."""
        issues: list[CodeIssue] = []

        # 1. Get active correction_pattern rules from evolution
        from evo_mind.evolution.engine import EvolutionEngine
        from evo_mind.types import RuleType

        evolution = EvolutionEngine(self.store, self.store.db)
        rules = await evolution.get_rules(
            rule_type=RuleType.CORRECTION_PATTERN,
            min_confidence=0.5,
        )

        # 2. Scan project files
        py_files = list(self.project_root.rglob("*.py"))
        # Exclude common patterns
        py_files = [
            f for f in py_files
            if "__pycache__" not in str(f)
            and ".venv" not in str(f)
            and "site-packages" not in str(f)
            and not f.name.startswith("test_")
        ]

        # 3. Analyze each file
        for file_path in py_files:
            try:
                content = file_path.read_text()
                file_issues = self._analyze_file(str(file_path), content, rules)
                issues.extend(file_issues)
            except Exception:
                logger.debug("file_analysis_failed", path=str(file_path))

        # 4. Add static analysis patterns
        for file_path in py_files:
            try:
                content = file_path.read_text()
                static_issues = self._static_analysis(str(file_path), content)
                issues.extend(static_issues)
            except Exception:
                pass

        return issues

    def _analyze_file(
        self,
        file_path: str,
        content: str,
        rules: list,
    ) -> list[CodeIssue]:
        """Apply evolution rules to find issues in a file."""
        issues: list[CodeIssue] = []

        for rule in rules:
            condition = rule.condition
            error_pattern = condition.get("error_pattern", "")

            if not error_pattern:
                continue

            # Simple pattern matching: check if the error pattern keywords
            # appear in the code or relate to common issues
            keywords = self._extract_keywords(error_pattern)
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                matches = sum(1 for kw in keywords if kw.lower() in line.lower())
                if matches >= 2:  # At least 2 keyword matches
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line=i,
                        severity="warning",
                        category="bug",
                        title=f"Rule: {rule.label or error_pattern[:60]}",
                        description=f"Pattern match for '{error_pattern[:100]}'",
                        suggestion=rule.action.get("correction_strategy", ""),
                        rule_id=rule.id,
                    ))

        return issues

    def _static_analysis(self, file_path: str, content: str) -> list[CodeIssue]:
        """Basic static analysis for common issues."""
        issues: list[CodeIssue] = []
        lines = content.split("\n")

        patterns = [
            (r"except\s*:", "bare_except", "error",
             "Bare except clause", "Specify exception type: 'except Exception as e:'"),
            (r"except\s+Exception\s*:", "broad_except", "warning",
             "Broad Exception catch", "Catch more specific exception type"),
            (r"\.close\(\)(?!\s*#)", "manual_close", "suggestion",
             "Manual resource close", "Use 'with' statement or context manager"),
            (r"time\.sleep\(", "time_sleep", "suggestion",
             "Blocking time.sleep in async code", "Use 'await asyncio.sleep()' instead"),
            (r"open\([^)]+\)(?!\s*as\b)", "unclosed_file", "warning",
             "File opened without context manager", "Use 'with open(...) as f:'"),
        ]

        import re
        for i, line in enumerate(lines, 1):
            for pattern, rule_name, severity, title, suggestion in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line=i,
                        severity=severity,
                        category="bug" if severity == "error" else "style",
                        title=title,
                        description=f"Line {i}: {line.strip()[:100]}",
                        suggestion=suggestion,
                    ))

        return issues

    async def _generate_fixes(self, issues: list[CodeIssue]) -> list[CodeFix]:
        """Generate and optionally apply fixes for issues."""
        fixes: list[CodeFix] = []

        for issue in issues:
            try:
                file_path = Path(issue.file_path)
                if not file_path.exists():
                    continue

                content = file_path.read_text()
                lines = content.split("\n")

                if issue.line < 1 or issue.line > len(lines):
                    continue

                original_line = lines[issue.line - 1]
                fixed_line = self._suggest_fix(original_line, issue)

                if fixed_line == original_line:
                    continue

                # Compute diff
                old_lines = content.split("\n")
                new_lines = old_lines.copy()
                new_lines[issue.line - 1] = fixed_line
                diff = self._compute_diff(old_lines, new_lines, issue.line)

                fix = CodeFix(
                    issue=issue,
                    original_code=original_line,
                    fixed_code=fixed_line,
                    diff=diff,
                )

                if self.auto_apply:
                    try:
                        new_content = "\n".join(new_lines)
                        file_path.write_text(new_content)
                        fix.applied = True
                        logger.info("fix_applied", file=issue.file_path, line=issue.line)
                    except Exception as e:
                        logger.warning("fix_apply_failed", error=str(e))

                fixes.append(fix)

            except Exception as e:
                logger.debug("fix_generation_failed", error=str(e))

        return fixes

    def _suggest_fix(self, line: str, issue: CodeIssue) -> str:
        """Generate a fix suggestion for a single line."""
        indent = len(line) - len(line.lstrip())
        prefix = " " * indent

        if "bare_except" in (issue.suggestion or ""):
            return f"{prefix}except Exception as e:"
        if "except Exception" in line:
            return line  # Already specific enough in many cases
        if "time.sleep" in line:
            return line.replace("time.sleep", "await asyncio.sleep")
        if ".close()" in line and "with" not in line:
            return line  # Keep as-is, flag for manual review

        # Default: apply suggestion if it's a concrete replacement
        suggestion = issue.suggestion or ""
        if suggestion and "use" in suggestion.lower():
            # Extract suggestion between quotes
            import re
            quoted = re.findall(r"'([^']*)'", suggestion)
            if quoted:
                return f"{prefix}{quoted[0]}"

        return line

    async def _run_tests(self) -> tuple[bool, str]:
        """Run the test suite and return (success, output)."""
        try:
            proc = await asyncio.create_subprocess_shell(
                f"cd {self.project_root} && {self.test_command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )
            output = stdout.decode()[:2000]
            if stderr:
                output += "\n" + stderr.decode()[:500]
            return proc.returncode == 0, output
        except asyncio.TimeoutError:
            return False, "Test run timed out (60s)"
        except Exception as e:
            return False, str(e)

    async def _record_results(
        self, fixes: list[CodeFix], session_id: str
    ) -> int:
        """Record fix results as memories for future learning."""
        count = 0

        for fix in fixes:
            # Record the issue as a feedback memory
            issue_mem = await self.store.record(MemoryCreate(
                memory_type=MemoryType.FEEDBACK,
                content={
                    "type": "code_issue",
                    "file": fix.issue.file_path,
                    "line": fix.issue.line,
                    "category": fix.issue.category,
                    "title": fix.issue.title,
                    "severity": fix.issue.severity,
                    "original_code": fix.original_code,
                },
                importance=0.7 if fix.issue.severity == "error" else 0.4,
                session_id=session_id,
                source="plugin",
                tags=["code-evolution", "issue", fix.issue.category],
            ))
            count += 1

            # Record the fix as a procedural memory
            if fix.applied:
                fix_mem = await self.store.record(MemoryCreate(
                    memory_type=MemoryType.PROCEDURAL,
                    content={
                        "type": "code_fix",
                        "pattern": "automated_fix",
                        "original": fix.original_code.strip(),
                        "fixed": fix.fixed_code.strip(),
                        "success": fix.success,
                        "test_output": fix.test_output[:500] if fix.test_output else None,
                    },
                    importance=0.6,
                    session_id=session_id,
                    source="plugin",
                    tags=["code-evolution", "fix", "success" if fix.success else "failed"],
                ))
                count += 1

                # Link issue and fix
                await self.store.relate(
                    fix_mem.id, issue_mem.id,
                    RelationType.CORRECTS,
                    strength=0.8 if fix.success else 0.3,
                )

        return count

    def _prioritize(self, issues: list[CodeIssue]) -> list[CodeIssue]:
        """Prioritize issues: errors first, then warnings, then suggestions."""
        severity_order = {"error": 0, "warning": 1, "suggestion": 2}
        return sorted(issues, key=lambda i: severity_order.get(i.severity, 99))

    @staticmethod
    def _compute_diff(
        old_lines: list[str], new_lines: list[str], line_num: int
    ) -> str:
        """Compute a simple inline diff."""
        if line_num < 1 or line_num > len(old_lines):
            return ""
        return (
            f"- {old_lines[line_num - 1]}\n"
            f"+ {new_lines[line_num - 1]}"
        )

    @staticmethod
    def _extract_keywords(text: str, n: int = 8) -> list[str]:
        """Extract significant keywords from text."""
        import re
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "to", "of", "in",
            "on", "at", "for", "with", "by", "from", "and", "or", "but",
            "not", "no", "this", "that", "it", "its",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        from collections import Counter
        return [w for w, _ in Counter(filtered).most_common(n)]
