"""CLI entry point for evo-mind using Typer and Rich."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from evo_mind.config import EvoMindConfig
from evo_mind.core.models import MemoryCreate, SearchQuery
from evo_mind.core.store import MemoryStore
from evo_mind.embedding.local import LocalEmbeddingEngine
from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore
from evo_mind.retrieval.engine import RetrievalEngine
from evo_mind.types import MemoryType

app = typer.Typer(
    name="evo-mind",
    help="Self-Evolution Memory System",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger("evo_mind")


# Global state (lazily initialized)
_state: dict = {}


async def _get_store() -> MemoryStore:
    """Lazily initialize and return the MemoryStore singleton."""
    if "store" not in _state:
        config = EvoMindConfig()

        db = Database(config.database.path, config.database.pool_size,
                      config.database.busy_timeout_ms)
        await db.initialize()

        vector_store = ChromaVectorStore(
            config.chroma.path,
            config.chroma.collection_name,
            config.chroma.distance_metric,
        )
        await vector_store.initialize()

        embedding = LocalEmbeddingEngine(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            cache_size=config.embedding.cache_size,
            normalize=config.embedding.normalize,
        )
        await embedding.initialize()

        _state["store"] = MemoryStore(db, vector_store, embedding)
        _state["db"] = db
        _state["vector_store"] = vector_store
        _state["embedding"] = embedding

    return _state["store"]


async def _get_retrieval() -> RetrievalEngine:
    store = await _get_store()
    if "retrieval" not in _state:
        _state["retrieval"] = RetrievalEngine(
            _state["db"], _state["vector_store"], _state["embedding"]
        )
    return _state["retrieval"]


async def _cleanup() -> None:
    """Clean up resources."""
    if "vector_store" in _state:
        await _state["vector_store"].close()
    if "db" in _state:
        await _state["db"].close()
    _state.clear()


# ---- Commands ----

@app.command()
def record(
    text: Annotated[str, typer.Argument(help="Memory content text")],
    type: Annotated[str, typer.Option("--type", "-t", help="Memory type")] = "episodic",
    importance: Annotated[float, typer.Option("--importance", "-i",
                                               help="Importance [0.0-1.0]")] = 0.5,
    tags: Annotated[Optional[str], typer.Option("--tags",
                                                 help="Comma-separated tags")] = None,
):
    """Record a new memory."""
    async def _record():
        store = await _get_store()
        tag_list = [t.strip() for t in tags.split(",")] if tags else []

        mem = await store.record(MemoryCreate(
            memory_type=MemoryType(type),
            content={"text": text},
            importance=importance,
            tags=tag_list,
        ))
        console.print(f"[green]✓[/green] Memory recorded: [bold]{mem.id}[/bold]")
        console.print(Panel(
            json.dumps(mem.content, indent=2, ensure_ascii=False),
            title=f"[{mem.memory_type.value}] importance={mem.importance:.2f}",
            border_style="blue",
        ))
        await _cleanup()

    asyncio.run(_record())


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 10,
    semantic: Annotated[float, typer.Option("--semantic", help="Semantic weight")] = 1.0,
    keyword: Annotated[float, typer.Option("--keyword", help="Keyword weight")] = 0.5,
    temporal: Annotated[float, typer.Option("--temporal", help="Temporal weight")] = 0.3,
):
    """Search memories across all strategies."""
    async def _search():
        retrieval = await _get_retrieval()
        results = await retrieval.search(SearchQuery(
            query_text=query,
            max_results=limit,
            semantic_weight=semantic,
            keyword_weight=keyword,
            temporal_weight=temporal,
        ))

        table = Table(title=f"Search results for: '{query}'")
        table.add_column("#", style="dim")
        table.add_column("Score", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Content", style="white")
        table.add_column("Strategies", style="yellow")

        for i, r in enumerate(results, 1):
            preview = (r.memory.content_plain or str(r.memory.content))[:100]
            strategies = ", ".join(r.score_breakdown.keys())
            table.add_row(
                str(i),
                f"{r.score:.4f}",
                r.memory.memory_type.value,
                preview,
                strategies,
            )

        console.print(table)
        console.print(f"[dim]{len(results)} results[/dim]")
        await _cleanup()

    asyncio.run(_search())


@app.command()
def list_memories(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    type: Annotated[Optional[str], typer.Option("--type", "-t")] = None,
):
    """List recent memories."""
    async def _list():
        store = await _get_store()
        mem_type = MemoryType(type) if type else None
        memories = await store.list_recent(limit=limit, memory_type=mem_type)

        table = Table(title="Recent Memories")
        table.add_column("ID", style="dim")
        table.add_column("Type", style="magenta")
        table.add_column("Importance", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Preview")

        for mem in memories:
            table.add_row(
                mem.id[:12] + "...",
                mem.memory_type.value,
                f"{mem.importance:.2f}",
                mem.created_at.strftime("%Y-%m-%d %H:%M"),
                (mem.content_plain or "")[:80],
            )

        console.print(table)
        await _cleanup()

    asyncio.run(_list())


@app.command()
def show(
    memory_id: Annotated[str, typer.Argument(help="Memory ID")],
):
    """Show full details of a memory."""
    async def _show():
        store = await _get_store()
        mem = await store.get(memory_id)
        if not mem:
            console.print(f"[red]Memory not found: {memory_id}[/red]")
            await _cleanup()
            return

        console.print(Panel(
            Syntax(json.dumps(mem.content, indent=2, ensure_ascii=False), "json"),
            title=f"[{mem.memory_type.value}] {mem.id}",
            border_style="blue",
        ))
        console.print(f"Importance: {mem.importance:.2f} | "
                       f"Accesses: {mem.access_count} | "
                       f"Status: {mem.status.value} | "
                       f"Created: {mem.created_at.isoformat()}")

        # Show tags
        tags = await store.get_tags(memory_id)
        if tags:
            console.print(f"Tags: [yellow]{', '.join(tags)}[/yellow]")

        # Show relationships
        relations = await store.get_related(memory_id)
        if relations:
            console.print("\n[bold]Relationships:[/bold]")
            for related_mem, rel_type, strength in relations:
                console.print(f"  {rel_type.value} → {related_mem.id[:12]}... "
                              f"(strength: {strength:.2f})")

        await _cleanup()

    asyncio.run(_show())


@app.command()
def consolidate(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without executing")] = False,
):
    """Run consolidation pipeline."""
    async def _run():
        from evo_mind.consolidation.engine import ConsolidationEngine
        from evo_mind.types import ConsolidationTrigger
        from evo_mind.config import EvoMindConfig

        await _get_store()
        config = EvoMindConfig()

        retrieval = await _get_retrieval()
        engine = ConsolidationEngine(
            _state["store"],
            retrieval,
            _state["embedding"],
            _state["vector_store"],
            _state["db"],
            config={
                "similarity_threshold": config.consolidation.similarity_threshold,
                "dedup_threshold": config.consolidation.dedup_threshold,
                "max_total_memories": config.consolidation.max_total_memories,
                "min_importance_to_keep": config.consolidation.min_importance_to_keep,
                "default_max_age_days": config.consolidation.default_max_age_days,
            },
        )

        pending = await engine.get_pending_count()
        console.print(f"Pending memories: [cyan]{pending}[/cyan]")

        if dry_run:
            console.print("[yellow]Dry run — no changes made[/yellow]")
            await _cleanup()
            return

        result = await engine.consolidate(trigger=ConsolidationTrigger.MANUAL)
        console.print(Panel(
            f"Groups formed: {result.groups_formed}\n"
            f"Summaries generated: {result.summaries_generated}\n"
            f"Duplicates merged: {result.duplicates_merged}\n"
            f"Memories pruned: {result.memories_pruned}\n"
            f"Compression ratio: {result.compression_ratio:.2f}",
            title="Consolidation Complete",
            border_style="green",
        ))
        await _cleanup()

    asyncio.run(_run())


@app.command()
def evolve():
    """Run evolution pipeline to detect patterns and learn rules."""
    async def _run():
        from evo_mind.evolution.engine import EvolutionEngine
        from evo_mind.config import EvoMindConfig

        await _get_store()
        config = EvoMindConfig()

        engine = EvolutionEngine(
            _state["store"],
            _state["db"],
            config={
                "min_support": config.evolution.min_support,
                "min_confidence": config.evolution.min_confidence,
                "evaluation_interval_hours": config.evolution.evaluation_interval_hours,
            },
        )

        rules = await engine.evolve()

        if rules:
            table = Table(title="Evolution Rules Discovered")
            table.add_column("Type", style="magenta")
            table.add_column("Label", style="white")
            table.add_column("Confidence", style="cyan")
            table.add_column("Support", style="green")

            for r in rules:
                table.add_row(
                    r.rule_type.value,
                    r.label or "(no label)",
                    f"{r.confidence:.2f}",
                    str(r.support_count),
                )
            console.print(table)
        else:
            console.print("[yellow]No new rules discovered[/yellow]")

        await _cleanup()

    asyncio.run(_run())


@app.command()
def rules(
    type: Annotated[Optional[str], typer.Option("--type", "-t",
                                                  help="Filter by rule type")] = None,
):
    """List learned evolution rules."""
    async def _run():
        from evo_mind.evolution.engine import EvolutionEngine
        from evo_mind.types import RuleType

        await _get_store()
        engine = EvolutionEngine(_state["store"], _state["db"])

        rule_type = RuleType(type) if type else None
        rules = await engine.get_rules(rule_type=rule_type, min_confidence=0.0)

        if rules:
            table = Table(title="Learned Rules")
            table.add_column("ID", style="dim")
            table.add_column("Type", style="magenta")
            table.add_column("Label")
            table.add_column("Conf", style="cyan")
            table.add_column("Support", style="green")
            table.add_column("Status")

            for r in rules:
                table.add_row(
                    r.id[:12] + "...",
                    r.rule_type.value,
                    r.label or "-",
                    f"{r.confidence:.2f}",
                    str(r.support_count),
                    r.status,
                )
            console.print(table)
        else:
            console.print("[yellow]No rules found[/yellow]")

        await _cleanup()

    asyncio.run(_run())


@app.command()
def stats():
    """Show system statistics."""
    async def _run():
        from evo_mind.types import MemoryStatus

        store = await _get_store()

        total = await store.count_total()
        active = await store.count_by_status(MemoryStatus.ACTIVE)
        consolidated = await store.count_by_status(MemoryStatus.CONSOLIDATED)

        table = Table(title="System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total memories", str(total))
        table.add_row("Active (unconsolidated)", str(active))
        table.add_row("Consolidated", str(consolidated))
        table.add_row("Consolidation ratio",
                      f"{consolidated/total:.1%}" if total > 0 else "N/A")

        console.print(table)
        await _cleanup()

    asyncio.run(_run())


def main() -> None:
    """Entry point for `evo-mind` CLI."""
    app()
