"""沙盒探索者 — 在开放世界中持续学习与环境交互。

核心循环 (终生):
  Perceive → Decide → Act → Observe → Learn → Remember → Repeat

每个 episode:
  1. 选择环境 (好奇心驱动: 优先探索不熟悉的环境)
  2. 探索交互
  3. 学习技能
  4. 积累经验
  5. 整合到 MUSE 记忆
  6. 更新进化目标
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.sandbox.curiosity import CuriosityEngine, CuriositySignal, LifelongLearner, Skill
from evo_mind.sandbox.environment import (
    ActionSpace,
    BaseEnvironment,
    EnvironmentType,
    StepResult,
    State,
    Action,
    create_environment,
)
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 探索结果 ----

@dataclass
class ExplorationEpisode:
    """一个探索 episode"""
    id: str = field(default_factory=uuid7)
    env_type: EnvironmentType = EnvironmentType.PUZZLE
    steps_taken: int = 0
    total_reward: float = 0.0
    total_intrinsic_reward: float = 0.0
    curiosity_avg: float = 0.0
    terminal: bool = False
    solved: bool = False
    skills_acquired: int = 0
    skills_improved: int = 0
    experiences_recorded: int = 0
    timestamp: str = field(default_factory=_now)


@dataclass
class DailyExplorationResult:
    """每日探索结果"""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    episodes_completed: int = 0
    episodes_solved: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    total_intrinsic: float = 0.0
    skills_total: int = 0
    skills_mastered: int = 0  # proficiency >= 0.8
    experiences_recorded: int = 0
    env_coverage: dict[str, int] = field(default_factory=dict)
    top_discovery: str = ""
    summary: str = ""


# ---- 探索者 ----

class SandboxExplorer:
    """沙盒探索者 — 在虚拟世界中持续学习和进化。

    终生学习循环:
      while alive:
        env = select_environment(curiosity)    # 好奇心驱动选择
        for step in range(max_steps):
          action = decide(state, curiosity)     # ε-greedy + 好奇心
          result = env.step(action)
          curiosity = compute_curiosity(result)  # 更新好奇心
          skill = learn(result)                  # 提取技能
          remember(result, skill)                # 存入MUSE
          if result.terminal: break
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.curiosity = CuriosityEngine()
        self.learner = LifelongLearner()

        # 每个环境类型的表现追踪
        self._env_performance: dict[str, list[float]] = defaultdict(list)
        self._env_visit_count: dict[str, int] = defaultdict(int)

        # 全局统计
        self._total_steps = 0
        self._total_episodes = 0

    # ---- 主探索循环 ----

    async def explore(
        self,
        episodes: int = 8,
        max_steps_per_episode: int = 100,
        difficulty: float | None = None,
    ) -> DailyExplorationResult:
        """执行今日探索"""
        result = DailyExplorationResult()
        print(f"\n  🌍 开放世界探索")

        for ep_idx in range(episodes):
            # 1. 选择环境 (好奇心驱动)
            env_type = self._select_environment()
            difficulty = difficulty or self._adaptive_difficulty(env_type)
            env = create_environment(env_type, max_steps_per_episode)

            # 2. 运行一个 episode
            episode = await self._run_episode(env, difficulty)
            result.episodes_completed += 1
            if episode.solved:
                result.episodes_solved += 1
            result.total_steps += episode.steps_taken
            result.total_reward += episode.total_reward
            result.total_intrinsic += episode.total_intrinsic_reward
            result.experiences_recorded += episode.experiences_recorded
            result.env_coverage[env_type.value] = result.env_coverage.get(env_type.value, 0) + 1

            # 3. 显示进度
            icon = "🏆" if episode.solved else "✅" if episode.terminal else "🔄"
            print(f"    {icon} {env_type.value}: {episode.steps_taken}步, "
                  f"奖励={episode.total_reward:.1f}, 好奇={episode.curiosity_avg:.3f}")

        # 4. 尝试技能合成
        await self._compose_skills()

        # 5. 学习进度
        skills = self.learner.get_learning_progress()
        result.skills_total = skills["total_skills"]
        result.skills_mastered = skills["proficiency_distribution"].get("master (0.8-1.0)", 0)

        # 6. 记录到 MUSE
        await self._record_exploration_result(result)

        result.summary = (
            f"{result.episodes_solved}/{result.episodes_completed} episodes solved, "
            f"{result.skills_total} skills ({result.skills_mastered} mastered), "
            f"{result.total_steps} steps"
        )
        print(f"    📊 {result.summary}")

        return result

    async def _run_episode(
        self, env: BaseEnvironment, difficulty: float
    ) -> ExplorationEpisode:
        """运行一个探索 episode"""
        episode = ExplorationEpisode(env_type=env.env_type)
        state = env.reset(difficulty)
        curiosity_signals: list[CuriositySignal] = []

        for step in range(env.max_steps):
            # 选择动作 (ε-greedy + 好奇心偏置)
            action = self._select_action(env, state)

            # 执行动作
            result = env.step(action)

            # 计算好奇心
            signal = self.curiosity.compute_curiosity(
                state.features, action.action_id, result.state.features
            )
            curiosity_signals.append(signal)

            # 组合奖励 = 外在 + 内在
            combined_reward = result.reward + signal.intrinsic_reward
            episode.total_reward += combined_reward
            episode.total_intrinsic_reward += signal.intrinsic_reward
            episode.steps_taken += 1
            self._total_steps += 1

            # 学习技能
            skill_name = f"{env.env_type.value}_{self._extract_skill_name(result)}"
            skill = self.learner.acquire_skill(
                skill_name,
                env.env_type.value,
                success=result.reward > 0,
                parameters={"difficulty": difficulty, "step": step},
            )
            if skill.times_used == 1:
                episode.skills_acquired += 1
            else:
                episode.skills_improved += 1

            # 记录经验
            self.learner.add_experience({
                "env_type": env.env_type.value,
                "state": state.features,
                "action": action.action_id,
                "reward": result.reward,
                "intrinsic_reward": signal.intrinsic_reward,
                "next_state": result.state.features,
                "terminal": result.terminal,
                "curiosity": signal.intrinsic_reward,
            })
            episode.experiences_recorded += 1

            # 存入 MUSE 记忆 (每10步批量)
            if step % 10 == 0 or result.terminal:
                await self._store_interaction(state, action, result, signal)

            state = result.state
            if result.terminal:
                episode.terminal = True
                # 判断是否真正解决
                if result.reward > 5.0:
                    episode.solved = True
                break

        episode.curiosity_avg = (
            sum(s.intrinsic_reward for s in curiosity_signals) / len(curiosity_signals)
            if curiosity_signals else 0.0
        )

        # 更新环境表现
        self._env_performance[env.env_type.value].append(episode.total_reward)
        self._env_visit_count[env.env_type.value] += 1
        self._total_episodes += 1

        return episode

    # ---- 动作选择 ----

    def _select_action(self, env: BaseEnvironment, state: State) -> Action:
        """ε-greedy + 好奇心偏置的动作选择"""
        n_actions = env.get_action_space_size()

        # ε-greedy: 探索率随步数衰减
        epsilon = max(0.05, 0.3 * math.exp(-state.step * 0.02))

        if random.random() < epsilon:
            # 探索: 随机动作
            action_id = random.randint(0, n_actions - 1)
        else:
            # 利用: 选预测误差最大的动作 (最不确定的 = 最好奇的)
            action_id = self._select_curious_action(state, n_actions)

        # Map discrete action IDs to meaningful direction values
        if n_actions == 3:
            direction_map = {0: -1.0, 1: 0.0, 2: 1.0}
            value = direction_map.get(action_id, 0.0)
        elif n_actions == 1:
            value = random.uniform(-1, 1)
        else:
            value = float(action_id)

        return Action(
            action_id=action_id,
            action_type=ActionSpace.DISCRETE if n_actions > 1 else ActionSpace.CONTINUOUS,
            value=value,
        )

    def _select_curious_action(
        self, state: State, n_actions: int
    ) -> int:
        """选择最不确定的动作 (最大预测误差)"""
        best_action = 0
        best_error = -1.0

        for a in range(n_actions):
            key = (self.curiosity._hash_features(state.features), a)
            if key not in self.curiosity._forward_model:
                # 从未尝试过的动作 → 最高好奇心
                return a
            # 已尝试过 → 检查是否有改善空间
            error = 1.0 / (1.0 + self.curiosity._visit_counts.get(
                self.curiosity._hash_features(state.features), 1
            ))
            if error > best_error:
                best_error = error
                best_action = a

        return best_action

    # ---- 环境选择 ----

    def _select_environment(self) -> EnvironmentType:
        """好奇心驱动的环境选择: 优先探索不熟悉的环境"""
        env_types = list(EnvironmentType)

        # 计算每个环境的好奇心得分
        scores = []
        for et in env_types:
            visits = self._env_visit_count.get(et.value, 0)
            avg_reward = (
                sum(self._env_performance.get(et.value, [0])) /
                max(len(self._env_performance.get(et.value, [1])), 1)
            )
            # 访问少的 + 奖励不确定的 = 高分
            novelty = 1.0 / math.sqrt(visits + 1)
            uncertainty = 1.0 / (abs(avg_reward) + 1.0)
            scores.append(novelty + uncertainty * 0.5)

        # 加权随机选择
        total = sum(scores)
        if total == 0:
            return random.choice(env_types)

        r = random.uniform(0, total)
        cumulative = 0.0
        for et, score in zip(env_types, scores):
            cumulative += score
            if cumulative >= r:
                return et
        return env_types[-1]

    def _adaptive_difficulty(self, env_type: EnvironmentType) -> float:
        """根据历史表现自适应难度"""
        perf = self._env_performance.get(env_type.value, [])
        if len(perf) < 3:
            return 0.3  # 初始难度
        avg_reward = sum(perf[-5:]) / len(perf[-5:])
        # 表现好 → 加难度
        if avg_reward > 5.0:
            return min(1.0, 0.5 + (avg_reward - 5.0) * 0.05)
        return max(0.1, 0.5 - (5.0 - avg_reward) * 0.05)

    # ---- 技能合成 ----

    async def _compose_skills(self) -> None:
        """尝试合成新技能"""
        best = self.learner.get_best_skills(top_k=10)
        if len(best) < 2:
            return

        # 尝试合成两个高熟练度的不同领域技能
        for i in range(min(5, len(best))):
            for j in range(i + 1, min(6, len(best))):
                if best[i].domain != best[j].domain:
                    composed = self.learner.compose_skill(
                        best[i].id, best[j].id,
                        f"composed_{best[i].name}_{best[j].name}",
                    )
                    if composed:
                        logger.debug("skill_composed", name=composed.name)
                        return  # 一次只合成一个

    # ---- 存储到 MUSE ----

    async def _store_interaction(
        self, state: State, action: Action, result: StepResult, signal: CuriositySignal
    ) -> None:
        """将交互存储为 MUSE 记忆"""
        try:
            await self.store.record(MemoryCreate(
                memory_type=MemoryType.EPISODIC,
                content={
                    "type": "sandbox_interaction",
                    "env": result.state.env_type.value,
                    "step": state.step,
                    "state_features": state.features,
                    "action": action.action_id,
                    "reward": result.reward,
                    "intrinsic_reward": signal.intrinsic_reward,
                    "prediction_error": signal.prediction_error,
                    "novelty": signal.novelty_bonus,
                    "terminal": result.terminal,
                },
                importance=0.3 + signal.intrinsic_reward * 0.3,
                source="plugin",
                tags=["sandbox", "exploration", result.state.env_type.value],
            ))
        except Exception as e:
            logger.warning("Failed to store interaction: %s", e)

    async def _record_exploration_result(self, result: DailyExplorationResult) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "daily_exploration",
                "date": result.date,
                "episodes": result.episodes_completed,
                "solved": result.episodes_solved,
                "total_steps": result.total_steps,
                "total_reward": result.total_reward,
                "skills": result.skills_total,
                "skills_mastered": result.skills_mastered,
                "env_coverage": result.env_coverage,
                "curiosity_stats": self.curiosity.get_exploration_stats(),
                "learning_progress": self.learner.get_learning_progress(),
            },
            importance=0.7,
            source="plugin",
            tags=["sandbox", "daily", "exploration"],
        ))

    @staticmethod
    def _extract_skill_name(result: StepResult) -> str:
        """从交互中提取技能名"""
        if result.reward > 0:
            return "positive_action"
        if result.terminal:
            return "terminal_handling"
        return "exploration_move"

    def get_status(self) -> dict[str, Any]:
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "curiosity": self.curiosity.get_exploration_stats(),
            "learning": self.learner.get_learning_progress(),
            "env_performance": {
                et.value: (
                    sum(self._env_performance.get(et.value, [0])) /
                    max(len(self._env_performance.get(et.value, [1])), 1)
                )
                for et in EnvironmentType
            },
        }
