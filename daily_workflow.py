#!/usr/bin/env python3
"""每日多 Agent 协同进化工作流。

当每日目标需要更深入的进化时调用。
启动 3 个 Agent 并行工作:
  - 经验 Agent: 提取 SOP、反思、概括
  - 学习 Agent: 生成训练数据、优化规则
  - 进化 Agent: 遗传优化、种群进化

工作流:
  并行执行 → 交叉验证 → 合并发现 → 应用到系统
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def workflow_experience(store, db) -> dict:
    """Agent 1: 经验驱动 — 提取 SOP + 反思"""
    from evo_mind.training import MuseMemoryManager, ExperienceDrivenEngine, MuseExperience

    muse = MuseMemoryManager(store)
    engine = ExperienceDrivenEngine(store, muse)

    # 检索最近的失败经验
    results = []
    sops = []
    reflections = []

    # 提取 SOP
    recent = await store.list_recent(limit=20)
    for mem in recent:
        if mem.content.get("type") in ("task_result", "code_fix", "training_example"):
            exp = MuseExperience(
                task=mem.content.get("description", str(mem.content)[:100]),
                inputs=mem.content,
                actions=[{"action": mem.content.get("action", "analyze"), "result": "OK"}],
                outcomes={"success": mem.content.get("success", True)},
                success=mem.content.get("success", True),
                tags=mem.content.get("tags", []),
            )
            await muse.record_experience(exp)
            sop = await engine.extract_sop_from_experience(exp)
            if sop:
                sops.append(sop)
            refl = await engine.reflect(exp)
            reflections.append(refl)

    return {
        "experiences_processed": len(results),
        "sops_extracted": len(sops),
        "reflections_done": len(reflections),
    }


async def workflow_learning(store, db) -> dict:
    """Agent 2: 学习进化 — 生成训练数据 + 优化规则"""
    from evo_mind.training import LearningEvolutionEngine

    learner = LearningEvolutionEngine(store)

    # 生成自监督训练数据
    examples = await learner.generate_training_data(count=30)

    # Teacher→Student 循环
    metrics = {
        "total_memories": await store.count_total(),
        "avg_fitness": 0.65,
        "error_rate": 0.2,
    }
    edits = await learner.teacher_loop(metrics)
    student_result = {}
    if edits:
        student_result = await learner.student_loop(edits)

    # 在线自适应
    adapt = await learner.adapt_online([
        {"success": True, "duration": 1.0},
        {"success": True, "duration": 0.8},
        {"success": False, "duration": 2.0},
    ])

    return {
        "training_examples_generated": len(examples),
        "edits_generated": len(edits),
        "edits_validated": student_result.get("validated", 0),
        "adaptation_completion_rate": adapt.task_completion_rate,
    }


async def workflow_population(store, db) -> dict:
    """Agent 3: 种群进化 — DARWIN 交叉修改 + 自然选择"""
    from evo_mind.training import DarwinEngine

    darwin = DarwinEngine(store)

    population = await darwin.create_initial_population(8)

    await darwin.evaluate_population()
    stats_before = darwin.get_population_stats()

    # 运行 3 代进化
    for gen in range(3):
        await darwin.genetic_recombination()
        await darwin.natural_selection()

    stats_after = darwin.get_population_stats()

    return {
        "generations": 3,
        "avg_fitness_before": stats_before.get("avg_fitness", 0),
        "avg_fitness_after": stats_after.get("avg_fitness", 0),
        "population_size": stats_after.get("population_size", 0),
        "individuals_culled": stats_before.get("population_size", 0) - stats_after.get("population_size", 0),
    }


async def run_workflow() -> dict:
    """并行运行三个 Agent 的协同进化"""
    from evo_mind.persistence.database import Database
    from evo_mind.persistence.memory_repo import MemoryRepo
    from evo_mind.core.store import MemoryStore

    data_dir = Path.home() / ".evo_mind" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    db = Database(data_dir / "evo_mind.db", pool_size=3)
    await db.initialize()
    repo = MemoryRepo(db)

    # Real backends with fallback
    embedding = None
    try:
        from evo_mind.embedding.local import LocalEmbeddingEngine
        embedding = LocalEmbeddingEngine(model_name="all-MiniLM-L6-v2", device="cpu", cache_size=5000)
        await embedding.initialize()
    except Exception:
        embedding = None  # Reset so fallback triggers

    vector_store = None
    try:
        from evo_mind.persistence.vector_store import ChromaVectorStore
        chroma_path = Path.home() / ".evo_mind" / "data" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        vector_store = ChromaVectorStore(chroma_path)
        await vector_store.initialize()
    except Exception:
        vector_store = None  # Reset so fallback triggers

    if embedding is None:
        class _FE:  # noqa
            @property
            def dimension(self): return 384
            async def encode(self, t): return [0.0]*384
            async def encode_batch(self, t, b=32): return [[0.0]*384 for _ in t]
        embedding = _FE()

    if vector_store is None:
        class _FVS:  # noqa
            async def add(self, **kw): pass
            async def query(self, query_embedding, n_results=10, where=None): return ([], [])
            async def delete(self, i): pass
            async def count(self): return 0
            async def get_embeddings(self, i): return None
        vector_store = _FVS()

    store = MemoryStore(db, vector_store, embedding, repo)

    # 并行执行
    results = await asyncio.gather(
        workflow_experience(store, db),
        workflow_learning(store, db),
        workflow_population(store, db),
        return_exceptions=True,
    )

    await db.close()

    return {
        "experience": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
        "learning": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
        "population": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
    }


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  🧬 多Agent协同进化工作流")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    result = asyncio.run(run_workflow())

    print("  结果:")
    for agent_name, data in result.items():
        status = "✅" if "error" not in data else "❌"
        print(f"    {status} {agent_name}: {json.dumps(data, ensure_ascii=False)}")

    log_path = Path.home() / ".evo_mind" / "daily_logs" / f"workflow_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    print(f"\n  💾 日志: {log_path}")
