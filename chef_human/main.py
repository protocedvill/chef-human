from __future__ import annotations

import asyncio
import logging

import click

from chef_human.agent import create_context_assembler
from chef_human.agent.planner import Planner
from chef_human.agent.react_loop import AgentResult, ReActConfig, ReActLoop
from chef_human.llm import create_backend
from chef_human.tools import create_tool_registry
from chef_human.ui.debug_tui import DebugTUI
from chef_human.ui.protocol import NoopUI


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("task", required=False)
@click.option("--debug-tui/--no-debug-tui", default=True, help="Enable/disable debug TUI")
@click.option("--max-steps", type=int, default=25, help="Max agent steps")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def run(
    task: str | None,
    debug_tui: bool,
    max_steps: int,
    workspace: str | None,
    no_stream: bool,
) -> None:
    """Run chef-human on a task."""
    if not task:
        task = click.prompt("Task", default="")
        if not task:
            click.echo("No task provided. Exiting.")
            return

    result = asyncio.run(
        _execute_task(
            task=task,
            debug_tui=debug_tui,
            max_steps=max_steps,
            workspace=workspace,
            stream=not no_stream,
        )
    )

    click.echo(f"\n{'=' * 40}")
    click.echo(f"Result: {'SUCCESS' if result.success else 'FAILURE'}")
    click.echo(f"Steps: {result.steps_taken}")
    click.echo(f"Message: {result.message}")
    if not result.success:
        raise SystemExit(1)


async def _execute_task(
    task: str,
    debug_tui: bool = True,
    max_steps: int = 25,
    workspace: str | None = None,
    stream: bool = True,
) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)

    context = create_context_assembler(workspace_root=workspace)
    backend = create_backend()
    tool_registry = create_tool_registry(context.workspace)
    planner = Planner(llm_backend=backend)
    config = ReActConfig(max_steps=max_steps, stream=stream)
    ui = DebugTUI() if debug_tui else NoopUI()

    loop = ReActLoop(
        llm_backend=backend,
        tool_registry=tool_registry,
        context_assembler=context,
        planner=planner,
        config=config,
        ui=ui,
    )

    return await loop.run(task)


if __name__ == "__main__":
    cli()
