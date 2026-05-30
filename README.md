# evo-mind: Self-Evolution Memory System

A production-grade memory system that enables AI to **evolve across sessions** by recording experiences, consolidating knowledge, retrieving relevant context, and detecting patterns to generate adaptive rules.

## Architecture

```
Record → Embed → Store → Retrieve → Consolidate → Evolve
  ↑______________________________________________|
                     feedback loop
```

- **Memory Types**: episodic, semantic, procedural, feedback
- **Storage**: SQLite (structured) + ChromaDB (vector embeddings)
- **Retrieval**: semantic + keyword + temporal → Reciprocal Rank Fusion
- **Consolidation**: clustering → summarization → deduplication → pruning
- **Evolution**: pattern detection → rule learning → strategy optimization

## Quick Start

```bash
pip install -e .
evo-mind --help
```

## Commands

```bash
evo-mind record "Found a bug in authentication" --type feedback --tags bug,auth
evo-mind search "authentication error" -n 10
evo-mind list --type episodic
evo-mind show <memory-id>
evo-mind consolidate
evo-mind evolve
evo-mind rules
evo-mind stats
```
