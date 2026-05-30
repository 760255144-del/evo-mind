"""虚拟沙盒环境 — 开放世界任务生成 + 安全隔离执行。

环境类型:
  puzzle       — 逻辑谜题 (路径寻找、约束满足)
  optimization — 参数优化 (找到最优配置)
  prediction   — 序列预测 (预测下一个状态)
  exploration  — 未知地图探索
  creation     — 创造性任务 (生成代码/文本/策略)
  survival     — 资源管理 (预算约束下的最优化)

安全机制:
  - 所有代码在沙盒进程中执行
  - 资源限制 (CPU时间、内存)
  - 失败不回滚系统状态
  - 毒性检测
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from evo_mind.utils import uuid7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 环境类型 ----

class EnvironmentType(StrEnum):
    PUZZLE = "puzzle"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    EXPLORATION = "exploration"
    CREATION = "creation"
    SURVIVAL = "survival"


class ActionSpace(StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


# ---- 环境基类 ----

@dataclass
class State:
    """环境状态"""
    id: str = field(default_factory=uuid7)
    env_type: EnvironmentType = EnvironmentType.PUZZLE
    features: list[float] = field(default_factory=list)
    observation: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    terminal: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """环境动作"""
    action_id: int = 0
    action_type: ActionSpace = ActionSpace.DISCRETE
    value: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """单步交互结果"""
    state: State
    reward: float
    intrinsic_reward: float = 0.0  # 好奇心奖励
    terminal: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """沙盒环境基类"""

    def __init__(self, env_type: EnvironmentType, max_steps: int = 100):
        self.env_type = env_type
        self.max_steps = max_steps
        self._state = State(env_type=env_type)

    @abstractmethod
    def reset(self, difficulty: float = 0.5) -> State:
        """重置环境到初始状态"""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """执行一步交互"""

    @abstractmethod
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""

    @property
    def state(self) -> State:
        return self._state


# ---- 具体环境实现 ----

class PuzzleEnvironment(BaseEnvironment):
    """逻辑谜题环境 — 路径寻找、排序、模式补全"""

    def __init__(self, max_steps: int = 50):
        super().__init__(EnvironmentType.PUZZLE, max_steps)
        self._grid_size = 0
        self._target = (0, 0)
        self._position = (0, 0)
        self._solution_steps = 0

    def reset(self, difficulty: float = 0.5) -> State:
        size = 3 + int(difficulty * 7)  # 3x3 to 10x10
        self._grid_size = size
        self._position = (0, 0)
        self._target = (size - 1, size - 1)
        self._solution_steps = (size - 1) * 2  # Manhattan distance
        self._state = State(
            env_type=self.env_type,
            features=[float(size), 0.0, 0.0],
            observation={
                "type": "grid_navigation",
                "grid_size": size,
                "position": list(self._position),
                "target": list(self._target),
                "obstacles": [],
            },
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        x, y = self._position
        # 0=up, 1=right, 2=down, 3=left
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = moves.get(action.action_id, (0, 0))
        nx, ny = x + dx, y + dy

        # 边界检查
        if 0 <= nx < self._grid_size and 0 <= ny < self._grid_size:
            self._position = (nx, ny)

        # 计算奖励
        dist_before = abs(x - self._target[0]) + abs(y - self._target[1])
        dist_after = abs(nx - self._target[0]) + abs(ny - self._target[1])
        reward = (dist_before - dist_after) * 0.5  # 接近目标=正奖励

        terminal = self._position == self._target or self._state.step >= self.max_steps
        if terminal and self._position == self._target:
            reward += 10.0  # 达成目标

        self._state.step += 1
        self._state.features = [float(self._grid_size), float(self._position[0]), float(self._position[1])]
        self._state.observation.update({
            "position": list(self._position),
            "target": list(self._target),
        })
        self._state.terminal = terminal

        return StepResult(state=self._state, reward=reward, terminal=terminal)

    def get_action_space_size(self) -> int:
        return 4


class OptimizationEnvironment(BaseEnvironment):
    """参数优化环境 — 找函数最优值"""

    def __init__(self, max_steps: int = 100):
        super().__init__(EnvironmentType.OPTIMIZATION, max_steps)
        self._func = lambda x: x**2
        self._x = 0.0
        self._x_min = 0.0
        self._x_range = 5.0

    def reset(self, difficulty: float = 0.5) -> State:
        a = random.uniform(1, 5); b = random.uniform(-3, 3)
        self._x_min = -b / (2 * a) if a != 0 else 0
        self._func = lambda x: a * x**2 + b * x
        self._x = random.uniform(-self._x_range, self._x_range)
        self._state = State(
            env_type=self.env_type,
            features=[self._x, 0.0],
            observation={
                "type": "function_optimization",
                "x": self._x,
                "x_range": [-self._x_range, self._x_range],
                "hint": f"Find minimum of a quadratic function",
            },
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        # 连续动作: 向 action.value 方向移动
        old_x = self._x
        step_size = 0.1 + abs(action.value) * self._x_range * 0.2
        self._x += action.value * step_size
        self._x = max(-self._x_range, min(self._x_range, self._x))

        old_val = self._func(old_x)
        new_val = self._func(self._x)
        optimal_val = self._func(self._x_min)

        # 奖励: 越接近最优值越好
        reward = (old_val - new_val) / max(abs(optimal_val) + 0.01, 1.0)

        terminal = (abs(self._x - self._x_min) < 0.05 or self._state.step >= self.max_steps)
        if terminal and abs(self._x - self._x_min) < 0.05:
            reward += 10.0

        self._state.step += 1
        self._state.features = [self._x, self._func(self._x)]
        self._state.observation.update({"x": self._x, "f(x)": new_val})
        self._state.terminal = terminal

        return StepResult(state=self._state, reward=reward, terminal=terminal)

    def get_action_space_size(self) -> int:
        return 3  # left, stay, right


class PredictionEnvironment(BaseEnvironment):
    """序列预测环境 — 预测下一个数字/模式"""

    def __init__(self, max_steps: int = 30):
        super().__init__(EnvironmentType.PREDICTION, max_steps)
        self._sequence = []
        self._position = 0
        self._rule = lambda i: i

    def reset(self, difficulty: float = 0.5) -> State:
        length = 5 + int(difficulty * 10)
        # 生成模式: 等差/等比/斐波那契/多项式
        rule_type = random.choice(["arithmetic", "geometric", "fibonacci", "polynomial"])
        if rule_type == "arithmetic":
            d = random.randint(1, 5)
            self._rule = lambda i: i * d
            hint = f"Arithmetic: x_n = n * {d}"
        elif rule_type == "geometric":
            r = random.randint(2, 3)
            self._rule = lambda i: r ** i
            hint = f"Geometric: x_n = {r}^n"
        elif rule_type == "fibonacci":
            self._rule = lambda i: self._fib(i)
            hint = "Fibonacci-like sequence"
        else:
            a = random.randint(1, 3)
            self._rule = lambda i: i ** 2 + a
            hint = f"Polynomial: x_n = n² + {a}"

        self._sequence = [self._rule(i) for i in range(length)]
        self._position = 0
        self._state = State(
            env_type=self.env_type,
            features=[float(length), 0.0],
            observation={
                "type": "sequence_prediction",
                "seen": self._sequence[:3],
                "length": length,
                "hint": hint,
            },
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        true_value = self._sequence[self._position]
        predicted = action.value

        error = abs(true_value - predicted)
        reward = max(0.0, 1.0 - error / max(abs(true_value) + 1, 1.0)) * 2.0

        self._position += 1
        terminal = self._position >= len(self._sequence)
        if terminal:
            reward += 5.0 * (1.0 - min(error, 5.0) / 5.0)

        self._state.step += 1
        self._state.observation["seen"] = self._sequence[:self._position + 3]
        self._state.terminal = terminal

        return StepResult(state=self._state, reward=reward, terminal=terminal)

    def get_action_space_size(self) -> int:
        return 1  # Continuous prediction

    @staticmethod
    def _fib(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a


class ExplorationEnvironment(BaseEnvironment):
    """未知地图探索 — 在陌生环境中最大化信息增益"""

    def __init__(self, max_steps: int = 80):
        super().__init__(EnvironmentType.EXPLORATION, max_steps)
        self._grid = []
        self._size = 0
        self._pos = (0, 0)
        self._visited = set()
        self._total_cells = 0

    def reset(self, difficulty: float = 0.5) -> State:
        size = 5 + int(difficulty * 10)
        self._size = size
        self._grid = [[random.uniform(-1, 1) for _ in range(size)] for _ in range(size)]
        self._pos = (size // 2, size // 2)
        self._visited = {self._pos}
        self._total_cells = size * size
        self._state = State(
            env_type=self.env_type,
            features=[0.0, float(size)],
            observation={
                "type": "grid_exploration",
                "grid_size": size,
                "position": list(self._pos),
                "visited_count": 1,
                "map_visible": "?",
            },
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        x, y = self._pos
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = moves.get(action.action_id, (0, 0))
        nx, ny = x + dx, y + dy

        if 0 <= nx < self._size and 0 <= ny < self._size:
            self._pos = (nx, ny)

        # 奖励: 发现新格子
        is_new = self._pos not in self._visited
        self._visited.add(self._pos)
        reward = 1.0 if is_new else -0.1
        # 内容奖励 (发现高价值格)
        cell_value = self._grid[self._pos[0]][self._pos[1]]
        if is_new and cell_value > 0.5:
            reward += cell_value

        coverage = len(self._visited) / self._total_cells
        terminal = coverage >= 0.95 or self._state.step >= self.max_steps
        if terminal:
            reward += coverage * 10.0

        self._state.step += 1
        self._state.features = [coverage, float(self._size)]
        self._state.observation.update({
            "position": list(self._pos),
            "visited_count": len(self._visited),
            "coverage": f"{coverage:.0%}",
            "cell_value": cell_value,
        })
        self._state.terminal = terminal

        # 内在奖励: 覆盖率提升 = 好奇心满足
        prev_coverage = self._state.observation.get("coverage", "0%")
        intrinsic = 0.5 if is_new else 0.0

        return StepResult(state=self._state, reward=reward, intrinsic_reward=intrinsic, terminal=terminal)

    def get_action_space_size(self) -> int:
        return 4


# ---- 环境工厂 ----

def create_environment(env_type: EnvironmentType, max_steps: int | None = None) -> BaseEnvironment:
    """创建环境实例"""
    factories = {
        EnvironmentType.PUZZLE: lambda: PuzzleEnvironment(max_steps or 50),
        EnvironmentType.OPTIMIZATION: lambda: OptimizationEnvironment(max_steps or 100),
        EnvironmentType.PREDICTION: lambda: PredictionEnvironment(max_steps or 30),
        EnvironmentType.EXPLORATION: lambda: ExplorationEnvironment(max_steps or 80),
    }
    return factories.get(env_type, lambda: PuzzleEnvironment())()
