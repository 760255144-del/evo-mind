"""三大研究方向的知识库 — 关键论文 + 开源项目。

每天扫描新进展，提取可应用的洞察。

方向:
  1. 元学习 (Meta-Learning) — 学会如何学习
  2. 世界模型 (World Models) — 内部表征与预测
  3. 自动对齐 (Auto-Alignment) — 自我价值校准

每篇论文记录: 核心思想、可应用的技术、对 evo-mind 的改进建议。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 研究方向 ----

@dataclass
class Paper:
    """一篇研究论文"""
    id: str = ""
    title: str = ""
    authors: str = ""
    year: int = 2024
    venue: str = ""           # arXiv | NeurIPS | ICML | ICLR | ...
    url: str = ""
    direction: str = ""       # meta_learning | world_models | auto_alignment
    tags: list[str] = field(default_factory=list)

    # 核心贡献
    core_idea: str = ""       # 一句话核心思想
    key_technique: str = ""   # 关键技术方法
    breakthrough: str = ""    # 突破性贡献

    # 对 evo-mind 的应用
    applicable_to: list[str] = field(default_factory=list)  # 适用模块
    improvement_suggestion: str = ""  # 具体改进建议
    implementation_difficulty: str = "medium"  # easy|medium|hard
    expected_impact: str = "medium"  # low|medium|high|breakthrough

    # 状态
    read: bool = False
    applied: bool = False
    date_added: str = field(default_factory=_now)
    date_read: str | None = None
    date_applied: str | None = None


@dataclass
class Project:
    """一个开源项目"""
    id: str = ""
    name: str = ""
    repo_url: str = ""
    stars: int = 0
    direction: str = ""
    description: str = ""
    language: str = "Python"

    # 可用性
    can_integrate: bool = False
    integration_plan: str = ""
    license_compatible: bool = True

    # 状态
    evaluated: bool = False
    integrated: bool = False
    date_added: str = field(default_factory=_now)


@dataclass
class ResearchInsight:
    """从论文/项目中提取的可执行洞察"""
    id: str = ""
    source_type: str = ""     # paper | project
    source_title: str = ""
    direction: str = ""
    insight: str = ""         # 核心洞察
    action: str = ""          # 可执行的动作
    target_module: str = ""   # 目标模块
    priority: str = "medium"  # critical|high|medium|low
    applied: bool = False
    date_extracted: str = field(default_factory=_now)


# ============================================================
# 知识库
# ============================================================

# ---- 元学习 (Meta-Learning) ----

META_LEARNING_PAPERS: list[dict] = [
    {
        "title": "MAML: Model-Agnostic Meta-Learning",
        "authors": "Finn et al.",
        "year": 2017, "venue": "ICML",
        "core_idea": "学习一个初始化参数，使模型能在少量梯度步内适应新任务",
        "key_technique": "二阶梯度优化 + 任务分布采样",
        "breakthrough": "首次证明梯度下降本身可以被学习",
        "applicable_to": ["evolutionary/engine.py", "training/orchestrator.py"],
        "improvement_suggestion": "用MAML思想优化GA的初始种群: 不从随机开始，而是学习一个好的初始化基因组，使种群在少数几代内就能收敛",
        "implementation_difficulty": "hard",
        "expected_impact": "high",
    },
    {
        "title": "Reptile: A Scalable Metalearning Algorithm",
        "authors": "Nichol et al.",
        "year": 2018, "venue": "arXiv",
        "core_idea": "一阶元学习: 参数向任务最优方向移动，比MAML更简单高效",
        "key_technique": "SGD-based meta-update without second derivatives",
        "breakthrough": "证明一阶近似可以达到二阶的性能",
        "applicable_to": ["evolutionary/operators.py", "training/learning_evolution.py"],
        "improvement_suggestion": "在LearningEvolutionEngine中用Reptile替代Teacher→Student固定步：每个编辑尝试都是元学习的一个episode，梯度方向累积",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
    {
        "title": "Learning to Learn by Gradient Descent by Gradient Descent",
        "authors": "Andrychowicz et al.",
        "year": 2016, "venue": "NeurIPS",
        "core_idea": "用LSTM学习一个优化器，该优化器输出参数更新",
        "key_technique": "Learner LSTM + coordinatewise processing",
        "breakthrough": "学习到的优化器可以泛化到未见过的任务",
        "applicable_to": ["evolutionary/engine.py", "super_evolution/meta_engine.py"],
        "improvement_suggestion": "MetaEngine的action_weight更新可以用学习的优化器替代EMA：根据历史action→outcome序列预测最佳权重调整",
        "implementation_difficulty": "hard",
        "expected_impact": "high",
    },
    {
        "title": "Meta-Learning with Implicit Gradients",
        "authors": "Rajeswaran et al. (iMAML)",
        "year": 2019, "venue": "NeurIPS",
        "core_idea": "隐式微分用于元学习，避免MAML的高内存消耗",
        "key_technique": "Implicit differentiation + conjugate gradient",
        "breakthrough": "O(1)内存元学习",
        "applicable_to": ["evolutionary/engine.py"],
        "improvement_suggestion": "GA适应度评估时不需要存储整个种群的计算图，用隐式梯度降低内存",
        "implementation_difficulty": "hard",
        "expected_impact": "medium",
    },
    {
        "title": "Self-Supervised Meta-Learning",
        "authors": "Hsu et al.",
        "year": 2019, "venue": "ICLR",
        "core_idea": "从无标签数据自动构造元学习任务",
        "key_technique": "Self-supervised task proposal + meta-training",
        "breakthrough": "无需人工标注即可进行元学习",
        "applicable_to": ["training/learning_evolution.py"],
        "improvement_suggestion": "用自监督任务提案替代LearningEvolutionEngine的template-based生成：系统自己发现可改进的子任务",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
    {
        "title": "MUSE: Multi-Unit Supervised Evolution",
        "authors": "2024",
        "year": 2024, "venue": "arXiv",
        "core_idea": "多层记忆抽象 + 进化优化 = 持续自我提升",
        "key_technique": "分层记忆 (原始→情节→语义→程序→策略)",
        "breakthrough": "五级抽象的经验驱动进化",
        "applicable_to": ["training/muse_memory.py"],
        "improvement_suggestion": "已在系统中实现! 持续优化MUSE层级间的压缩率和信息保留度",
        "implementation_difficulty": "easy",
        "expected_impact": "breakthrough",
    },
]

# ---- 世界模型 (World Models) ----

WORLD_MODELS_PAPERS: list[dict] = [
    {
        "title": "World Models",
        "authors": "Ha & Schmidhuber",
        "year": 2018, "venue": "NeurIPS",
        "core_idea": "用VAE+MDN-RNN构建环境的内部表征，在表征空间中训练控制器",
        "key_technique": "VAE (视觉) + MDN-RNN (时序) + Controller (决策)",
        "breakthrough": "证明了在梦想中学习比在现实中学习更高效",
        "applicable_to": ["core/store.py", "retrieval/engine.py"],
        "improvement_suggestion": "构建MemoryStore的'世界模型': 预测哪些记忆会被访问、哪些规则会成功，在内部模拟中预验证修改",
        "implementation_difficulty": "hard",
        "expected_impact": "breakthrough",
    },
    {
        "title": "DreamerV3: Mastering Diverse Domains with World Models",
        "authors": "Hafner et al.",
        "year": 2023, "venue": "arXiv",
        "core_idea": "通用的世界模型架构，在150+任务上无需调参即可工作",
        "key_technique": "RSSM + actor-critic in latent space + symlog predictions",
        "breakthrough": "单一算法、固定超参数、150+任务",
        "applicable_to": ["super_evolution/meta_engine.py"],
        "improvement_suggestion": "MetaEngine内省不是简单查指标，而是构建系统行为的预测模型: 给定action→预测outcome→选择最优action",
        "implementation_difficulty": "hard",
        "expected_impact": "high",
    },
    {
        "title": "JEPA: Joint Embedding Predictive Architecture",
        "authors": "LeCun",
        "year": 2022, "venue": "arXiv",
        "core_idea": "不预测原始像素，而是预测抽象表征空间中的状态变化",
        "key_technique": "Context encoder + target encoder + predictor in latent space",
        "breakthrough": "更接近人类推理的预测方式",
        "applicable_to": ["retrieval/semantic.py", "consolidation/engine.py"],
        "improvement_suggestion": "Consolidation不要合并原始文本，而是在embedding空间中预测记忆之间的演化关系",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
    {
        "title": "Spatial-Temporal Predictive Models",
        "authors": "Watter et al. (Embed to Control)",
        "year": 2015, "venue": "NeurIPS",
        "core_idea": "学习状态空间的局部线性模型用于控制",
        "key_technique": "Locally linear dynamics in learned latent space",
        "applicable_to": ["super_evolution/unified_loop.py"],
        "improvement_suggestion": "用局部线性模型预测系统健康度变化趋势，提前触发纠正动作",
        "implementation_difficulty": "medium",
        "expected_impact": "medium",
    },
    {
        "title": "Generative World Explorer",
        "authors": "2024",
        "year": 2024, "venue": "arXiv",
        "core_idea": "用生成模型探索可能的世界状态，而不需要实际交互",
        "key_technique": "Generative planning in latent space",
        "applicable_to": ["training/population_evolution.py"],
        "improvement_suggestion": "在生成模型空间中探索基因组可能性，而不是在真实种群中做代价高昂的评估",
        "implementation_difficulty": "hard",
        "expected_impact": "high",
    },
    {
        "title": "DayDreamer: World Models for Physical Robots",
        "authors": "Wu et al.",
        "year": 2023, "venue": "CoRL",
        "core_idea": "Dreamer应用到真实机器人，在线学习世界模型",
        "key_technique": "Online world model learning on real hardware",
        "applicable_to": ["agent_swarm/agent.py"],
        "improvement_suggestion": "Agent在Swarm中维护自己的'世界模型'，预测其他Agent的行为，实现更高效的协作",
        "implementation_difficulty": "medium",
        "expected_impact": "medium",
    },
]

# ---- 自动对齐 (Auto-Alignment) ----

AUTO_ALIGNMENT_PAPERS: list[dict] = [
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai et al. (Anthropic)",
        "year": 2022, "venue": "arXiv",
        "core_idea": "用一组宪法原则指导AI自我改进，无需人类反馈",
        "key_technique": "Self-critique + self-revision guided by principles",
        "breakthrough": "证明AI可以通过自我批评来对齐，大幅减少对人类标注的依赖",
        "applicable_to": ["super_evolution/recursive_improver.py", "training/experience_driven.py"],
        "improvement_suggestion": "为RecursiveImprover添加'宪法原则'列表：自修改必须满足安全性、性能、正确性约束，修改前先自我批评",
        "implementation_difficulty": "medium",
        "expected_impact": "breakthrough",
    },
    {
        "title": "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback",
        "authors": "Lee et al.",
        "year": 2023, "venue": "arXiv",
        "core_idea": "用LLM替代人类标注者提供RLHF中的反馈信号",
        "key_technique": "LLM-as-judge + chain-of-thought evaluation",
        "breakthrough": "AI反馈可以达到人类反馈的97%性能",
        "applicable_to": ["training/learning_evolution.py", "agent_swarm/consensus.py"],
        "improvement_suggestion": "ConsensusEngine用AI反馈替代固定投票规则: 每个Agent给评分时附带CoT理由，加权聚合",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
    {
        "title": "Self-Rewarding Language Models",
        "authors": "Yuan et al. (Meta)",
        "year": 2024, "venue": "arXiv",
        "core_idea": "LLM自我生成训练数据、自我评估质量、自我迭代提升",
        "key_technique": "LLM-as-a-Judge + self-play training loop",
        "breakthrough": "模型可以在没有外部监督的情况下持续自我提升",
        "applicable_to": ["training/learning_evolution.py", "training/orchestrator.py"],
        "improvement_suggestion": "实现自奖励循环: TrainingOrchestrator生成训练样本→评估质量→用高质量样本重新训练",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
    {
        "title": "Direct Preference Optimization (DPO)",
        "authors": "Rafailov et al.",
        "year": 2023, "venue": "NeurIPS",
        "core_idea": "直接从偏好数据优化策略，无需显式奖励模型",
        "key_technique": "Implicit reward from policy ratio, closed-form optimal policy",
        "breakthrough": "RLHF可以简化为分类问题",
        "applicable_to": ["evolution/engine.py", "evolutionary/engine.py"],
        "improvement_suggestion": "EvolutionEngine的规则评估用DPO替代: 比较规则应用前后的结果，直接优化规则库",
        "implementation_difficulty": "hard",
        "expected_impact": "high",
    },
    {
        "title": "Weak-to-Strong Generalization",
        "authors": "Burns et al. (OpenAI)",
        "year": 2024, "venue": "arXiv",
        "core_idea": "用弱监督训练强模型，强模型泛化超越弱监督",
        "key_technique": "Auxiliary confidence loss + bootstrapping",
        "breakthrough": "揭示了超人类对齐的可行性路径",
        "applicable_to": ["training/orchestrator.py"],
        "improvement_suggestion": "用当前(较弱)系统的输出来训练下一代(更强)的参数，让进化产生超线性增益",
        "implementation_difficulty": "hard",
        "expected_impact": "breakthrough",
    },
    {
        "title": "Scalable Oversight via Debate",
        "authors": "Irving et al. (DeepMind)",
        "year": 2018, "venue": "arXiv",
        "core_idea": "两个AI辩论，人类裁判判断胜负 → 从辩论中学习对齐",
        "key_technique": "Adversarial debate + sparse human judgment",
        "applicable_to": ["agent_swarm/consensus.py", "agent_swarm/coordinator.py"],
        "improvement_suggestion": "ConsensusEngine改为辩论模式: 两个Agent争论一个改进方案，第三个Agent裁判，辩论结果作为训练信号",
        "implementation_difficulty": "medium",
        "expected_impact": "high",
    },
]

# ---- 开源项目 ----

OPEN_SOURCE_PROJECTS: list[dict] = [
    # 元学习
    {
        "name": "learn2learn", "repo_url": "https://github.com/learnables/learn2learn",
        "stars": 2700, "direction": "meta_learning",
        "description": "PyTorch元学习库，实现MAML/Reptile/ProtoNet等",
        "can_integrate": True,
        "integration_plan": "将learn2learn的元学习优化器集成到EvolutionaryEngine中",
    },
    {
        "name": "higher", "repo_url": "https://github.com/facebookresearch/higher",
        "stars": 1600, "direction": "meta_learning",
        "description": "Facebook的区分通过梯度计算的元学习库",
        "can_integrate": True,
        "integration_plan": "用higher实现二阶梯度优化GA适应度",
    },
    # 世界模型
    {
        "name": "dreamerv3", "repo_url": "https://github.com/danijar/dreamerv3",
        "stars": 2100, "direction": "world_models",
        "description": "DreamerV3官方实现，150+任务通用世界模型",
        "can_integrate": False,
        "integration_plan": "抽取RSSM模块用于MemoryStore的预测模型",
    },
    {
        "name": "iris", "repo_url": "https://github.com/eloialonso/iris",
        "stars": 1100, "direction": "world_models",
        "description": "基于Transformer的世界模型用于Atari游戏",
        "can_integrate": False,
        "integration_plan": "参考Transformer-based dynamics用于系统状态预测",
    },
    # 自动对齐
    {
        "name": "trl", "repo_url": "https://github.com/huggingface/trl",
        "stars": 10000, "direction": "auto_alignment",
        "description": "HuggingFace的Transformer强化学习库 (DPO/RLHF)",
        "can_integrate": True,
        "integration_plan": "用TRL的DPOTrainer替代EvolutionEngine的规则评估",
    },
    {
        "name": "Safe-RLHF", "repo_url": "https://github.com/PKU-Alignment/safe-rlhf",
        "stars": 1300, "direction": "auto_alignment",
        "description": "北大对齐团队的安全RLHF实现",
        "can_integrate": True,
        "integration_plan": "参考安全约束机制用于RecursiveImprover的自修改防护",
    },
]


def get_papers_by_direction(direction: str) -> list[dict]:
    """获取某个方向的所有论文"""
    mapping = {
        "meta_learning": META_LEARNING_PAPERS,
        "world_models": WORLD_MODELS_PAPERS,
        "auto_alignment": AUTO_ALIGNMENT_PAPERS,
    }
    return mapping.get(direction, [])


def get_all_papers() -> list[dict]:
    """获取所有论文"""
    return META_LEARNING_PAPERS + WORLD_MODELS_PAPERS + AUTO_ALIGNMENT_PAPERS


def get_projects_by_direction(direction: str) -> list[dict]:
    """获取某个方向的开源项目"""
    return [p for p in OPEN_SOURCE_PROJECTS if p["direction"] == direction]
