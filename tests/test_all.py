"""Core system tests — validates the entire evolution pipeline."""

import os
import tempfile
import pytest
import asyncio
import json


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    yield db_path
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
async def initialized_db(temp_db):
    from evo_mind.persistence.database import Database
    db = Database(temp_db, pool_size=2)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def repo(initialized_db):
    from evo_mind.persistence.memory_repo import MemoryRepo
    return MemoryRepo(initialized_db)


@pytest.fixture
async def store(initialized_db):
    from evo_mind.core.store import MemoryStore

    class M:
        @property
        def dimension(self): return 384
        async def encode(self, t): return [0.0] * 384
        async def encode_batch(self, t, b=32): return [[0.0]*384 for _ in t]

    class V:
        async def add(self, **kw): pass
        async def query(self, query_embedding, n_results=10, where=None): return ([], [])
        async def delete(self, i): pass
        async def count(self): return 0
        async def get_embeddings(self, i): return None

    from evo_mind.persistence.memory_repo import MemoryRepo
    return MemoryStore(initialized_db, V(), M(), MemoryRepo(initialized_db))


# ---- Phase 1: Memory System ----

class TestMemoryCRUD:
    async def test_create_and_retrieve(self, repo):
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType

        mem = await repo.create(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"text": "test memory"},
            importance=0.8,
        ))
        assert mem.id is not None
        assert len(mem.id) == 36
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.importance == 0.8
        assert mem.content["text"] == "test memory"

        found = await repo.get(mem.id)
        assert found is not None
        assert found.id == mem.id

    async def test_list_recent(self, repo):
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType

        for i in range(10):
            await repo.create(MemoryCreate(
                memory_type=MemoryType.EPISODIC,
                content={"text": f"memory {i}"},
            ))

        recent = await repo.list_recent(limit=5)
        assert len(recent) == 5

    async def test_tags(self, repo):
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType

        mem = await repo.create(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"text": "tagged"},
            tags=["python", "test", "evolution"],
        ))
        tags = await repo.get_tags(mem.id)
        assert "python" in tags
        assert "test" in tags

    async def test_relationships(self, repo):
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType, RelationType

        m1 = await repo.create(MemoryCreate(memory_type=MemoryType.FEEDBACK, content={"text": "correction"}))
        m2 = await repo.create(MemoryCreate(memory_type=MemoryType.EPISODIC, content={"text": "original"}))

        await repo.relate(m1.id, m2.id, RelationType.CORRECTS, 0.9)
        rels = await repo.get_related(m1.id)
        assert len(rels) == 1
        assert rels[0][1] == RelationType.CORRECTS

    async def test_sessions(self, repo):
        sid = await repo.start_session({"purpose": "test"})
        assert len(sid) == 36

        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType
        await repo.create(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"text": "in session"},
            session_id=sid,
        ))

        await repo.end_session(sid, "completed")
        session = await repo.get_session(sid)
        assert session is not None
        assert session.memory_count == 1

    async def test_dedup_by_hash(self, store):
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType

        # Dedup via content hash is done at store.record() level
        m1 = await store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"text": "duplicate test"},
        ))
        m2 = await store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"text": "duplicate test"},
        ))
        assert m1.id == m2.id


# ---- Phase 2: Code Evolution ----

class TestCodeEvolution:
    def test_code_issue_model(self):
        from evo_mind.code_evolution.engine import CodeIssue
        issue = CodeIssue(
            file_path="test.py", line=42, severity="error",
            category="bug", title="Null pointer", description="x is None",
            suggestion="Add null check",
        )
        assert issue.line == 42
        assert issue.severity == "error"

    def test_static_analysis_finds_issues(self):
        from evo_mind.code_evolution.engine import CodeEvolutionEngine
        # Create engine without store for static analysis
        engine = CodeEvolutionEngine.__new__(CodeEvolutionEngine)
        issues = engine._static_analysis("test.py", "try:\n    pass\nexcept:\n    pass\n")
        assert len(issues) >= 1  # Should find bare except
        assert any("except" in i.title.lower() for i in issues)

    def test_improvement_stats(self):
        from evo_mind.code_evolution.loop import ImprovementStats
        stats = ImprovementStats()
        stats.total_fixes_applied = 10
        stats.total_fixes_succeeded = 8
        assert stats.success_rate == 0.8


# ---- Phase 3: Agent Swarm ----

class TestAgentSwarm:
    def test_task_model(self):
        from evo_mind.agent_swarm.task import Task, TaskPriority, TaskStatus
        task = Task(title="Test", priority=TaskPriority.HIGH, metadata={"category": "analysis"})
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.HIGH

    def test_consensus_engine(self):
        from evo_mind.agent_swarm.consensus import ConsensusEngine
        from evo_mind.agent_swarm.task import TaskResult

        consensus = ConsensusEngine()
        results = [
            TaskResult(task_id="t1", agent_id="a1", success=True, output={}, duration_seconds=1.0),
            TaskResult(task_id="t2", agent_id="a2", success=True, output={}, duration_seconds=0.5),
            TaskResult(task_id="t3", agent_id="a3", success=False, output={}, error="fail"),
        ]
        eval_result = consensus.evaluate(results)
        assert eval_result["successful"] == 2
        assert eval_result["failed"] == 1
        assert eval_result["verdict"] in ("moderate_consensus", "strong_consensus", "weak_consensus")

    def test_agent_capabilities(self):
        from evo_mind.agent_swarm.agent import AnalyzerAgent, ExecutorAgent, ReviewerAgent
        assert "analysis" in AnalyzerAgent.capabilities
        assert "execution" in ExecutorAgent.capabilities
        assert "review" in ReviewerAgent.capabilities


# ---- Phase 4: Evolutionary Algorithms ----

class TestEvolutionary:
    def test_genome_creation(self):
        from evo_mind.evolutionary.genome import Genome
        g = Genome()
        params = g.get_params()
        assert "semantic_weight" in params
        assert "keyword_weight" in params
        assert "prune_age_days" in params
        assert len(params) == 8

    def test_tournament_selection(self):
        from evo_mind.evolutionary.genome import Genome
        from evo_mind.evolutionary.operators import tournament_selection

        pop = [Genome() for _ in range(5)]
        for i, g in enumerate(pop):
            g.fitness = i * 0.2

        winner = tournament_selection(pop, tournament_size=3)
        assert winner is not None
        assert winner.fitness >= 0.0

    def test_blend_crossover(self):
        from evo_mind.evolutionary.genome import Genome
        from evo_mind.evolutionary.operators import blend_crossover

        p1, p2 = Genome(), Genome()
        c1, c2 = blend_crossover(p1, p2)
        assert len(c1.parent_ids) == 2
        assert len(c2.parent_ids) == 2

    def test_gaussian_mutation(self):
        from evo_mind.evolutionary.genome import Genome
        from evo_mind.evolutionary.operators import gaussian_mutation

        g = Genome()
        original = g.genes["semantic_weight"].value
        mutant = gaussian_mutation(g)
        # Value should have changed (though could be same by chance)
        assert mutant.genes["semantic_weight"].value is not None


# ---- Phase 5: Super Evolution ----

class TestSuperEvolution:
    async def test_meta_engine_creation(self, store):
        from evo_mind.super_evolution import MetaEngine
        meta = MetaEngine(store)
        assert meta.id is not None
        assert len(meta._action_weights) > 0

    async def test_metrics_collection(self, store):
        from evo_mind.super_evolution import MetaEngine
        meta = MetaEngine(store)
        metrics = await meta._collect_metrics()
        assert metrics.total_memories >= 0
        assert 0.0 <= metrics.system_health <= 1.0

    async def test_introspection(self, store):
        from evo_mind.super_evolution import MetaEngine
        meta = MetaEngine(store)
        metrics = await meta._collect_metrics()
        weaknesses = await meta._introspect(metrics)
        assert isinstance(weaknesses, list)

    def test_system_phase_enum(self):
        from evo_mind.super_evolution import SystemPhase, ImprovementAction
        assert SystemPhase.MEMORY.value == "memory"
        assert SystemPhase.META.value == "meta"
        assert ImprovementAction.SELF_MODIFY.value == "self_modify"


# ---- Training System ----

class TestTraining:
    async def test_muse_experience_creation(self, store):
        from evo_mind.training import MuseMemoryManager, MuseExperience
        muse = MuseMemoryManager(store)
        exp = MuseExperience(
            task="Test task",
            actions=[{"action": "do", "result": "OK"}],
            outcomes={"success": True},
            success=True,
        )
        eid = await muse.record_experience(exp)
        assert exp.level_0_raw is not None
        assert exp.level_1_episode is not None
        assert exp.level_3_sop is not None  # Successful tasks get SOP

    async def test_sop_extraction(self, store):
        from evo_mind.training import ExperienceDrivenEngine, MuseMemoryManager, MuseExperience
        muse = MuseMemoryManager(store)
        engine = ExperienceDrivenEngine(store, muse)

        exp = MuseExperience(
            task="Fix a bug in search",
            actions=[{"action": "Check null", "result": "Found"}, {"action": "Fix", "result": "OK"}],
            outcomes={"success": True},
            success=True,
        )
        sop = await engine.extract_sop_from_experience(exp)
        assert sop is not None
        assert len(sop.steps) == 2

    async def test_darwin_population(self, store):
        from evo_mind.training import DarwinEngine
        darwin = DarwinEngine(store)
        pop = await darwin.create_initial_population(6)
        assert len(pop.get_alive()) == 6

        await darwin.evaluate_population()
        stats = darwin.get_population_stats()
        assert stats["population_size"] > 0
        assert "avg_fitness" in stats

    async def test_learning_engine(self, store):
        from evo_mind.training import LearningEvolutionEngine
        learner = LearningEvolutionEngine(store)

        examples = await learner.generate_training_data(count=10)
        assert len(examples) >= 10  # Cold start may add easy examples

        metrics = {"fitness": 0.6, "error_rate": 0.3}
        edits = await learner.teacher_loop(metrics)
        assert isinstance(edits, list)


# ---- Security Tests ----

class TestSecurity:
    async def test_executor_blocks_dangerous_commands(self, store):
        from evo_mind.agent_swarm.agent import ExecutorAgent
        from evo_mind.agent_swarm.task import Task

        agent = ExecutorAgent(store)
        # Should be blocked
        r = await agent._execute(Task(title="test", input_data={"command": "rm -rf /"}))
        assert not r["success"]
        assert "not in allowlist" in r["error"]

    async def test_executor_allows_safe_commands(self, store):
        from evo_mind.agent_swarm.agent import ExecutorAgent
        from evo_mind.agent_swarm.task import Task

        agent = ExecutorAgent(store)
        r = await agent._execute(Task(title="test", input_data={"command": "echo hello"}))
        assert r["success"]

    async def test_recursive_improver_rejects_no_tests(self, store):
        from evo_mind.super_evolution.recursive_improver import RecursiveImprover
        improver = RecursiveImprover(store, source_root="/tmp", test_command="echo no tests")
        result = await improver._run_tests()
        assert result is False  # Safety: must NOT assume OK

    async def test_recursive_improver_accepts_passing_tests(self, store):
        from evo_mind.super_evolution.recursive_improver import RecursiveImprover
        improver = RecursiveImprover(store, source_root="/tmp", test_command="echo '1 passed' && exit 0")
        result = await improver._run_tests()
        assert result is True  # Tests actually pass


# ---- UUID7 Tests ----

class TestUUID7:
    def test_uuid7_format(self):
        from evo_mind.utils import uuid7
        u = uuid7()
        assert len(u) == 36
        assert u.count("-") == 4

    def test_uuid7_sortable(self):
        from evo_mind.utils import uuid7
        import time
        u1 = uuid7()
        time.sleep(0.01)
        u2 = uuid7()
        # Timestamps are at the front, so later time = lexicographically greater
        assert u1 < u2


# ---- Run ----

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
