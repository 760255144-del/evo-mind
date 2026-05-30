#!/usr/bin/env python3
"""自愈引导脚本 — 确保系统始终处于可运行状态。

在每次进化前自动运行:
  1. 检查数据目录完整性
  2. 验证数据库可访问
  3. 检查关键模块可导入
  4. 修复常见问题
  5. 报告系统健康状态
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _now():
    return datetime.now(timezone.utc).isoformat()


class SelfHealer:
    """自愈检查器 — 检测并修复系统问题"""

    def __init__(self):
        self.home = Path.home() / ".evo_mind"
        self.data_dir = self.home / "data"
        self.log_dir = self.home / "daily_logs"
        self.project_root = Path(__file__).parent
        self.issues_found = []
        self.issues_fixed = []

    async def heal(self) -> dict:
        """运行全面自愈检查"""
        print(f"\n{'='*50}")
        print(f"  🏥 自愈检查 — {_now()}")
        print(f"{'='*50}\n")

        checks = [
            ("目录结构", self._check_directories),
            ("数据完整性", self._check_data_integrity),
            ("模块导入", self._check_imports),
            ("数据库健康", self._check_database),
            ("向量存储健康", self._check_vector_store),
            ("进化日志", self._check_evolution_logs),
            ("权限", self._check_permissions),
            ("磁盘空间", self._check_disk_space),
        ]

        for name, check_fn in checks:
            try:
                ok = await check_fn() if asyncio.iscoroutinefunction(check_fn) else check_fn()
                status = "✅" if ok else "⚠️"
                print(f"  {status} {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                self.issues_found.append(f"{name}: {e}")

        print(f"\n  发现: {len(self.issues_found)} 个问题")
        print(f"  修复: {len(self.issues_fixed)} 个问题")

        # 保存健康报告
        report = {
            "timestamp": _now(),
            "issues_found": self.issues_found,
            "issues_fixed": self.issues_fixed,
            "healthy": len(self.issues_found) == 0,
        }
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "health_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

        return report

    def _check_directories(self) -> bool:
        ok = True
        for d in [self.home, self.data_dir, self.log_dir]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                self.issues_fixed.append(f"Created {d}")
                ok = False
        return ok

    def _check_data_integrity(self) -> bool:
        db_path = self.data_dir / "evo_mind.db"
        if not db_path.exists():
            self.issues_found.append("Database file missing")
            return False
        if db_path.stat().st_size == 0:
            self.issues_found.append("Database file is empty (0 bytes)")
            return False
        return True

    def _check_imports(self) -> bool:
        ok = True
        modules = [
            "evo_mind.config",
            "evo_mind.types",
            "evo_mind.core.models",
            "evo_mind.core.store",
            "evo_mind.persistence.database",
            "evo_mind.persistence.memory_repo",
            "evo_mind.training",
            "evo_mind.super_evolution",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except Exception as e:
                self.issues_found.append(f"Import failed: {mod} — {e}")
                ok = False
        return ok

    async def _check_database(self) -> bool:
        try:
            from evo_mind.persistence.database import Database
            db = Database(self.data_dir / "evo_mind.db", pool_size=2)
            await db.initialize()

            # 检查关键表
            tables = await db.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            table_names = [r["name"] for r in tables]
            expected = ["memories", "sessions", "evolution_rules", "tags"]
            for t in expected:
                if t not in table_names:
                    self.issues_found.append(f"Missing table: {t}")

            await db.close()
            return True
        except Exception as e:
            self.issues_found.append(f"Database check failed: {e}")
            return False

    async def _check_vector_store(self) -> bool:
        """Check ChromaDB vector store health."""
        try:
            from evo_mind.persistence.vector_store import ChromaVectorStore
            chroma_path = self.data_dir / "chroma"
            vs = ChromaVectorStore(chroma_path)
            await vs.initialize()
            count = await vs.count()
            await vs.close()
            return True  # Accessible, even if empty
        except Exception as e:
            self.issues_found.append(f"Vector store check failed: {e}")
            return False

    def _check_evolution_logs(self) -> bool:
        logs = list(self.log_dir.glob("*.json"))
        if not logs:
            self.issues_found.append("No evolution logs found (first run?)")
        return True  # Not having logs is OK for first run

    def _check_permissions(self) -> bool:
        ok = True
        for d in [self.home, self.data_dir, self.log_dir]:
            if d.exists() and not os.access(d, os.W_OK):
                self.issues_found.append(f"No write permission: {d}")
                ok = False
        return ok

    def _check_disk_space(self) -> bool:
        import shutil
        stat = shutil.disk_usage(self.home)
        free_gb = stat.free / (1024**3)
        if free_gb < 0.1:
            self.issues_found.append(f"Low disk space: {free_gb:.1f}GB free")
            return False
        return True


async def main():
    healer = SelfHealer()
    report = await healer.heal()

    health = "✅ 健康" if report["healthy"] else "⚠️ 需要关注"
    print(f"\n  系统状态: {health}")
    print(f"  报告: {healer.home / 'health_report.json'}\n")


if __name__ == "__main__":
    asyncio.run(main())
