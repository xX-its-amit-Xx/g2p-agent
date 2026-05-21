"""``g2p-agent`` command-line interface (Typer)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings

app = typer.Typer(add_completion=False, help="Retrieval-augmented Claude agent over G2P portal data.")
console = Console()


@app.command()
def ingest(
    genes: str | None = typer.Option(None, help="Comma-separated gene symbols (default: baseline set)."),
    reset: bool = typer.Option(True, help="Drop and rebuild the Chroma collection."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force re-fetch from the G2P API."),
):
    """Pull G2P data, chunk, embed, and write to Chroma."""
    from .ingest import ingest as run_ingest

    gene_list = [g.strip() for g in genes.split(",")] if genes else None
    manifest = run_ingest(
        gene_list, reset=reset, use_cache=not no_cache,
        progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )
    console.print(Panel.fit(
        f"[bold green]Ingested {manifest['n_chunks']} chunks[/bold green] from "
        f"{len(manifest['genes'])} genes\nembedder=[cyan]{manifest['embedder']}[/cyan]  "
        f"collection=[cyan]{manifest['collection']}[/cyan]",
        title="ingest complete",
    ))


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language variant question."),
    json_out: bool = typer.Option(False, "--json", help="Emit raw AgentResponse JSON."),
):
    """Ask the agent a question."""
    from .agent import Agent

    response = Agent().ask(question)
    if json_out:
        console.print_json(response.model_dump_json())
        return

    badge = {"high": "green", "medium": "yellow", "low": "red"}[response.confidence.value]
    console.print(Panel(response.answer, title="answer", border_style=badge))
    console.print(f"[bold]confidence:[/bold] [{badge}]{response.confidence.value}[/{badge}] — "
                  f"{response.confidence_reasoning}")
    if response.claims:
        table = Table(title="claims & citations", show_lines=False)
        table.add_column("claim", overflow="fold")
        table.add_column("cites", style="cyan")
        for c in response.claims:
            table.add_row(c.text, ", ".join(ct.chunk_id for ct in c.citations) or "[red]NONE[/red]")
        console.print(table)
    console.print(f"[dim]backend={response.debug.get('llm_backend')} "
                  f"turns={response.debug.get('turns')} "
                  f"retrieved={response.debug.get('n_retrieved')}[/dim]")


@app.command(name="eval")
def eval_cmd(
    benchmark: Path = typer.Option(Path("eval/benchmark.jsonl"), help="Path to benchmark JSONL."),
    out: Path = typer.Option(Path("eval/results/latest.json"), help="Where to write the JSON report."),
    limit: int | None = typer.Option(None, help="Evaluate only the first N items."),
):
    """Run the evaluation harness and write a report."""
    from .eval import load_benchmark, run_eval

    items = load_benchmark(benchmark)
    if limit:
        items = items[:limit]
    report = run_eval(items, progress=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    m = report["metrics"]
    table = Table(title=f"eval ({report['n_items']} items)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in m.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[dim]backends: {report['backends']}[/dim]")
    console.print(f"[green]report written to {out}[/green]")


@app.command()
def info():
    """Show resolved backends and storage paths."""
    console.print(Panel.fit(
        f"embedder (resolved): [cyan]{settings.resolve_embedder()}[/cyan]\n"
        f"llm (resolved):      [cyan]{settings.resolve_llm()}[/cyan]\n"
        f"agent_model:         {settings.agent_model}\n"
        f"judge_model:         {settings.judge_model}\n"
        f"chroma_dir:          {settings.chroma_dir}\n"
        f"collection:          {settings.collection}\n"
        f"top_k / candidate_k: {settings.top_k} / {settings.candidate_k}",
        title="g2p-agent config",
    ))


if __name__ == "__main__":
    app()
