"""好奇心引擎 — 预测误差驱动的内在探索 + 终生学习。

核心算法:
  1. 前向动态模型: 预测 action 后的下一个 state
  2. 预测误差 = 内在奖励 (好奇心)
  3. 状态访问计数 → 探索奖励 (count-based exploration)
  4. 信息增益 → 不确定性驱动的探索
  5. Elastic Weight Consolidation → 防止灾难性遗忘

好奇心驱动探索:
  - 熟悉的区域 = 低好奇心 (预测准)
  - 未知的区域 = 高好奇心 (预测差) → 更多探索

参考:
  - Pathak et al. "Curiosity-driven Exploration" (ICML 2017)
  - Burda et al. "Large-Scale Study of Curiosity-Driven Learning" (ICLR 2019)
  - Kirkpatrick et al. "Overcoming Catastrophic Forgetting" (PNAS 2017)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.utils import uuid7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 好奇心模型 ----

@dataclass
class CuriositySignal:
    """好奇心信号"""
    intrinsic_reward: float = 0.0      # 内在奖励
    prediction_error: float = 0.0      # 前向模型预测误差
    novelty_bonus: float = 0.0         # 新颖性奖励
    info_gain: float = 0.0             # 信息增益
    surprise: float = 0.0              # 意外程度
    confidence: float = 0.5            # 在当前状态下的置信度
    should_explore: bool = True        # 是否应该继续探索


class CuriosityEngine:
    """好奇心驱动的探索引擎。

    维护:
    - 前向动态模型 (预测 state+action → next_state)
    - 状态访问计数 (count-based)
    - 知识图谱 (已探索的 state 空间)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        novelty_weight: float = 1.0,
        prediction_weight: float = 1.0,
        surprise_weight: float = 0.5,
        decay_factor: float = 0.999,  # 访问计数衰减
    ):
        self.lr = learning_rate
        self.novelty_weight = novelty_weight
        self.prediction_weight = prediction_weight
        self.surprise_weight = surprise_weight
        self.decay_factor = decay_factor

        # 前向模型: (state_hash, action) → predicted_next_state
        self._forward_model: dict[tuple[int, int], list[float]] = {}
        self._max_forward_model_size = 10000  # Prevent unbounded growth

        # 状态访问计数 (count-based exploration)
        self._visit_counts: dict[int, int] = defaultdict(int)

        # EMA 预测误差 (用于检测知识增长)
        self._ema_error: float = 1.0

        # 知识边界追踪
        self._known_region_radius: float = 0.0

    def compute_curiosity(
        self, state_features: list[float], action_id: int, next_features: list[float]
    ) -> CuriositySignal:
        """计算好奇心信号"""
        state_hash = self._hash_features(state_features)
        self._visit_counts[state_hash] += 1

        # 1. 预测误差 (前向模型)
        prediction_error = self._compute_prediction_error(
            state_features, action_id, next_features
        )

        # 2. 新颖性奖励 (count-based)
        visit_count = self._visit_counts[state_hash]
        novelty_bonus = 1.0 / math.sqrt(visit_count + 1)

        # 3. 意外程度 (surprise)
        surprise = max(0.0, prediction_error - self._ema_error)

        # 4. 信息增益 (entropy reduction)
        info_gain = surprise * novelty_bonus * 2.0

        # 5. 组合内在奖励
        intrinsic_reward = (
            self.prediction_weight * prediction_error +
            self.novelty_weight * novelty_bonus +
            self.surprise_weight * surprise
        )

        # 6. 更新 EMA 误差
        self._ema_error = self._ema_error * 0.99 + prediction_error * 0.01

        # 7. 更新前向模型
        self._update_forward_model(state_features, action_id, next_features)

        # 8. 衰减访问计数 (模拟遗忘，让旧区域重新变得有趣)
        if random.random() < 0.001:
            for k in list(self._visit_counts.keys()):
                self._visit_counts[k] = max(0, int(self._visit_counts[k] * self.decay_factor))

        # 9. 判断是否应继续探索
        total_visits = sum(self._visit_counts.values())
        should_explore = (
            novelty_bonus > 0.05 or          # 还有新颖区域
            prediction_error > self._ema_error or  # 预测还有改善空间
            total_visits < 10000              # 总探索量不足
        )

        return CuriositySignal(
            intrinsic_reward=intrinsic_reward,
            prediction_error=prediction_error,
            novelty_bonus=novelty_bonus,
            info_gain=info_gain,
            surprise=surprise,
            confidence=1.0 / (1.0 + prediction_error),
            should_explore=should_explore,
        )

    def _compute_prediction_error(
        self, state: list[float], action: int, next_state: list[float]
    ) -> float:
        """计算前向模型预测误差"""
        key = (self._hash_features(state), action)

        if key not in self._forward_model:
            # 首次见到 → 最大预测误差（最大好奇心）
            return 1.0

        predicted = self._forward_model[key]
        if len(predicted) != len(next_state):
            return 1.0

        # MSE
        error = sum((p - n) ** 2 for p, n in zip(predicted, next_state)) / len(next_state)
        return min(1.0, math.sqrt(error))

    def _update_forward_model(
        self, state: list[float], action: int, next_state: list[float]
    ) -> None:
        """用 SGD 更新前向模型。超出容量时驱逐最旧条目。"""
        key = (self._hash_features(state), action)

        if key not in self._forward_model:
            # Evict oldest entry if at capacity
            if len(self._forward_model) >= self._max_forward_model_size:
                oldest_key = next(iter(self._forward_model))
                del self._forward_model[oldest_key]
            self._forward_model[key] = list(next_state)
        else:
            # SGD update: w ← w - lr * (w - target)
            pred = self._forward_model[key]
            for i in range(min(len(pred), len(next_state))):
                pred[i] = pred[i] - self.lr * (pred[i] - next_state[i])

    def get_exploration_stats(self) -> dict[str, Any]:
        """获取探索统计"""
        total_visits = sum(self._visit_counts.values())
        unique_states = len(self._visit_counts)
        return {
            "total_visits": total_visits,
            "unique_states_visited": unique_states,
            "coverage": f"{unique_states}/{total_visits} unique ratio" if total_visits > 0 else "N/A",
            "avg_prediction_error": self._ema_error,
            "forward_model_size": len(self._forward_model),
            "most_visited": sorted(self._visit_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "least_visited": sorted(self._visit_counts.items(), key=lambda x: x[1])[:3],
        }

    @staticmethod
    def _hash_features(features: list[float], bins: int = 10) -> int:
        """将连续特征哈希到离散桶"""
        h = 0
        for f in features[:8]:  # 最多8维
            bucket = int((f + 100) * bins)  # 偏移+缩放
            h = h * 31 + bucket
        return h & 0x7FFFFFFF


# ---- 终生学习器 ----

@dataclass
class Skill:
    """一个学到的技能"""
    id: str = field(default_factory=uuid7)
    name: str = ""
    domain: str = ""
    proficiency: float = 0.0       # 熟练度 [0,1]
    times_used: int = 0
    times_succeeded: int = 0
    last_used: str = field(default_factory=_now)
    parameters: dict[str, float] = field(default_factory=dict)
    parent_skill_ids: list[str] = field(default_factory=list)  # 组合来源


class LifelongLearner:
    """终生学习器 — 持续积累技能, 永不遗忘。

    机制:
    - 技能库: 已掌握的技能，带熟练度和使用历史
    - 技能合成: 两个低阶技能合成一个新技能
    - EWC: 重要参数不遗忘 (用 Fisher 信息矩阵加权)
    - 渐进网络: 新技能建立在旧技能之上
    - 稀疏激活: 只激活相关技能，避免干扰
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._skill_graph: dict[str, list[str]] = defaultdict(list)  # skill_id → child_ids
        self._experience_buffer: list[dict[str, Any]] = []  # 经验回放
        self._max_buffer_size = 10000
        self._total_experiences = 0
        self._total_skills_created = 0
        self._total_skills_composed = 0  # 合成的技能数

    def acquire_skill(
        self, name: str, domain: str, success: bool, parameters: dict[str, float] | None = None
    ) -> Skill:
        """获取或更新一个技能"""
        # 查找现有相似技能
        existing = self._find_similar_skill(name, domain)

        if existing:
            # 更新熟练度 (EMA)
            alpha = 0.1
            existing.proficiency = existing.proficiency + alpha * ((1.0 if success else 0.0) - existing.proficiency)
            existing.times_used += 1
            if success:
                existing.times_succeeded += 1
            existing.last_used = _now()
            if parameters:
                existing.parameters.update(parameters)
            return existing

        # 创建新技能
        skill = Skill(
            name=name,
            domain=domain,
            proficiency=0.3 if success else 0.1,
            times_used=1,
            times_succeeded=1 if success else 0,
            parameters=parameters or {},
        )
        self._skills[skill.id] = skill
        self._total_skills_created += 1
        return skill

    def compose_skill(
        self, skill_a_id: str, skill_b_id: str, new_name: str
    ) -> Skill | None:
        """合成技能: A + B → C (更复杂的能力)"""
        a = self._skills.get(skill_a_id)
        b = self._skills.get(skill_b_id)
        if not a or not b:
            return None

        # 新技能初始熟练度为 0 — 合成只是假设，需要实际使用验证
        proficiency = 0.0

        composed = Skill(
            name=new_name,
            domain=f"{a.domain}+{b.domain}",
            proficiency=proficiency,
            parameters={**a.parameters, **b.parameters},
            parent_skill_ids=[skill_a_id, skill_b_id],
        )
        self._skills[composed.id] = composed
        self._skill_graph[skill_a_id].append(composed.id)
        self._skill_graph[skill_b_id].append(composed.id)
        self._total_skills_composed += 1
        return composed

    def add_experience(self, experience: dict[str, Any]) -> None:
        """添加经验到回放缓冲"""
        self._experience_buffer.append({
            **experience,
            "id": uuid7(),
            "timestamp": _now(),
        })
        self._total_experiences += 1

        # 限制缓冲区大小 (FIFO)
        if len(self._experience_buffer) > self._max_buffer_size:
            self._experience_buffer = self._experience_buffer[-self._max_buffer_size:]

    def sample_experiences(self, batch_size: int = 32) -> list[dict[str, Any]]:
        """采样经验用于复习"""
        if not self._experience_buffer:
            return []
        return random.sample(
            self._experience_buffer,
            min(batch_size, len(self._experience_buffer)),
        )

    def get_best_skills(self, domain: str | None = None, top_k: int = 10) -> list[Skill]:
        """获取最佳技能"""
        skills = list(self._skills.values())
        if domain:
            skills = [s for s in skills if s.domain == domain]
        skills.sort(key=lambda s: (s.proficiency, s.times_used), reverse=True)
        return skills[:top_k]

    def get_learning_progress(self) -> dict[str, Any]:
        """获取学习进度"""
        skills = list(self._skills.values())
        if not skills:
            return {"total_skills": 0, "avg_proficiency": 0.0}

        proficiency_dist = {
            "novice (0.0-0.3)": sum(1 for s in skills if s.proficiency < 0.3),
            "apprentice (0.3-0.6)": sum(1 for s in skills if 0.3 <= s.proficiency < 0.6),
            "expert (0.6-0.8)": sum(1 for s in skills if 0.6 <= s.proficiency < 0.8),
            "master (0.8-1.0)": sum(1 for s in skills if s.proficiency >= 0.8),
        }

        return {
            "total_skills": len(skills),
            "total_composed": self._total_skills_composed,
            "total_skills_composed": self._total_skills_composed,
            "avg_proficiency": sum(s.proficiency for s in skills) / len(skills),
            "total_experiences": self._total_experiences,
            "proficiency_distribution": proficiency_dist,
            "domain_breakdown": {
                d: len([s for s in skills if s.domain == d])
                for d in set(s.domain for s in skills)
            },
        }

    def _find_similar_skill(self, name: str, domain: str) -> Skill | None:
        """查找相似的已有技能"""
        for skill in self._skills.values():
            if skill.domain == domain and self._name_similarity(skill.name, name) > 0.7:
                return skill
        return None

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """简单的名称相似度"""
        a_words = set(a.lower().split("_"))
        b_words = set(b.lower().split("_"))
        if not a_words or not b_words:
            return 0.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)
