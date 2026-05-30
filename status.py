#!/usr/bin/env python3
"""状态仪表盘 — 一键查看系统运行状态和进化进度。

用法: python3 status.py
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def fmt_pct(v, total):
    if total == 0: return "N/A"
    return f"{v/total:.0%}"


def bar(value, max_val=1.0, width=20):
    filled = int(min(value, max_val) / max_val * width)
    return "█" * filled + "░" * (width - filled)


async def show_dashboard():
    home = Path.home() / ".evo_mind"
    data_dir = home / "data"
    log_dir = home / "daily_logs"

    print(f"\n{'='*60}")
    print(f"  🧬 evo-mind 状态仪表盘")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # ---- 数据库状态 ----
    print(f"\n  📦 数据层")
    try:
        from evo_mind.persistence.database import Database
        db = Database(data_dir / "evo_mind.db", pool_size=1)
        await db.initialize()

        r = await db.fetch_one("SELECT COUNT(*) as c FROM memories WHERE deleted_at IS NULL")
        total_mem = r["c"] if r else 0

        r = await db.fetch_one("SELECT COUNT(*) as c FROM memories WHERE status='consolidated'")
        consolidated = r["c"] if r else 0

        r = await db.fetch_one("SELECT COUNT(*) as c FROM memories WHERE status='active'")
        active = r["c"] if r else 0

        r = await db.fetch_one("SELECT COUNT(*) as c FROM evolution_rules WHERE status='active'")
        rules = r["c"] if r else 0

        r = await db.fetch_one("SELECT AVG(confidence) as a FROM evolution_rules WHERE status='active'")
        avg_conf = r["a"] if r and r["a"] else 0.0

        r = await db.fetch_one("SELECT COUNT(*) as c FROM memory_relationships")
        relationships = r["c"] if r else 0

        r = await db.fetch_one("SELECT COUNT(*) as c FROM tags")
        tags = r["c"] if r else 0

        print(f"  总记忆: {total_mem}  │  活跃: {active}  │  已整合: {consolidated}")
        print(f"  整合率: {bar(consolidated/(max(total_mem,1)), 1.0)} {fmt_pct(consolidated, total_mem)}")
        print(f"  进化规则: {rules}  │  平均置信度: {avg_conf:.2f}")
        print(f"  关系: {relationships}  │  标签: {tags}")
        await db.close()
    except Exception as e:
        print(f"  ❌ 数据库不可访问: {e}")

    # ---- 进化日志 ----
    print(f"\n  📊 进化进度")
    logs = sorted(log_dir.glob("2*.json")) if log_dir.exists() else []
    if logs:
        latest = json.loads(logs[-1].read_text()) if logs else {}
        goal = latest.get("goal", {})
        print(f"  最近进化: {latest.get('date', '?')}")
        print(f"  目标: {goal.get('title', '?')}")
        print(f"  完成: {'✅' if goal.get('completed') else '❌'}")
        print(f"  适应度变化: {latest.get('fitness_delta', 0):+.4f}")
        print(f"  SOP提取: {latest.get('sops_extracted', 0)}")
        print(f"  耗时: {latest.get('duration_seconds', 0):.0f}s")

        # 趋势
        if len(logs) >= 2:
            deltas = []
            for lf in logs[-7:]:
                try:
                    d = json.loads(lf.read_text())
                    deltas.append(d.get("fitness_delta", 0))
                except: pass
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            trend = "📈 上升" if avg_delta > 0.01 else ("📉 下降" if avg_delta < -0.01 else "➡️ 持平")
            print(f"  7日趋势: {trend} ({avg_delta:+.4f}/天)")
    else:
        print(f"  暂无进化记录")

    # ---- 模块状态 ----
    print(f"\n  🔧 模块")
    modules = [
        ("记忆 (Phase 1)", "evo_mind.core.store", "MemoryStore"),
        ("嵌入 (Phase 1)", "evo_mind.embedding.cache", "EmbeddingCache"),
        ("代码进化 (Phase 2)", "evo_mind.code_evolution.engine", "CodeEvolutionEngine"),
        ("Agent群 (Phase 3)", "evo_mind.agent_swarm.coordinator", "SwarmCoordinator"),
        ("进化算法 (Phase 4)", "evo_mind.evolutionary.genome", "Genome"),
        ("超进化 (Phase 5)", "evo_mind.super_evolution.meta_engine", "MetaEngine"),
        ("训练系统", "evo_mind.training.orchestrator", "TrainingOrchestrator"),
    ]
    for name, module, cls in modules:
        try:
            m = __import__(module, fromlist=[cls])
            getattr(m, cls)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    # ---- 健康报告 ----
    health_file = home / "health_report.json"
    if health_file.exists():
        report = json.loads(health_file.read_text())
        status = "✅ 健康" if report.get("healthy") else "⚠️ 需关注"
        print(f"\n  🏥 系统: {status}")
        issues = report.get("issues_found", [])
        if issues:
            print(f"  待解决问题: {len(issues)}")
            for i in issues[:3]:
                print(f"    - {i}")

    # ---- 定时任务 ----
    print(f"\n  ⏰ 定时")
    import subprocess
    r = subprocess.run(["launchctl", "list", "com.evo-mind.daily-evolve"],
                       capture_output=True, text=True)
    if r.returncode == 0 and "PID" in r.stdout:
        print(f"  ✅ launchd: 已加载 (每天 8:57)")
    else:
        print(f"  ⚠️ launchd: 未加载")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(show_dashboard())
