from __future__ import annotations

import asyncio
import json
import logging

import click

from chef_human.agent import create_context_assembler
from chef_human.agent.context import ContextManager
from chef_human.agent.persistence import (
    delete_session,
    list_sessions,
    load_session_data,
)
from chef_human.agent.planner import Planner
from chef_human.agent.react_loop import AgentResult, ReActConfig, ReActLoop
from chef_human.llm import create_backend
from chef_human.tools import create_tool_registry
from chef_human.ui.debug_tui import DebugTUI
from chef_human.ui.protocol import NoopUI


@click.group()
def cli() -> None:
    pass


@cli.group()
def session() -> None:
    """Manage saved sessions."""


@session.command("list")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
def session_list(save_dir: str | None) -> None:
    """List all saved sessions."""
    sessions = list_sessions(save_dir=save_dir or ".")
    if not sessions:
        click.echo("No sessions found.")
        return
    for s in sessions:
        click.echo(f"{s['session_id']:12}  {s.get('task', '?')[:60]}")


@session.command("show")
@click.argument("session_id")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
def session_show(session_id: str, save_dir: str | None) -> None:
    """Show details of a saved session."""
    data = load_session_data(session_id, save_dir=save_dir or ".")
    if data is None:
        click.echo(f"Session '{session_id}' not found.", err=True)
        raise SystemExit(1)
    click.echo(f"Session ID: {data.get('session_id')}")
    click.echo(f"Task: {data.get('task', 'N/A')}")
    conv = data.get("conversation", {})
    messages = conv.get("messages", [])
    click.echo(f"Messages: {len(messages)}")
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:80]
        click.echo(f"  [{role}] {content}")


@session.command("delete")
@click.argument("session_id")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
def session_delete(session_id: str, save_dir: str | None) -> None:
    """Delete a saved session."""
    if not delete_session(session_id, save_dir=save_dir or "."):
        click.echo(f"Session '{session_id}' not found.", err=True)
        raise SystemExit(1)
    click.echo(f"Session '{session_id}' deleted.")


@session.command("export")
@click.argument("session_id")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", help="Export format")
def session_export(session_id: str, save_dir: str | None, fmt: str) -> None:
    """Export a session in JSON or Markdown format."""
    data = load_session_data(session_id, save_dir=save_dir or ".")
    if data is None:
        click.echo(f"Session '{session_id}' not found.", err=True)
        raise SystemExit(1)
    if fmt == "json":
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"# Session: {session_id}")
        click.echo(f"**Task:** {data.get('task', 'N/A')}")
        click.echo()
        conv = data.get("conversation", {})
        for msg in conv.get("messages", []):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            click.echo(f"### {role}")
            click.echo(content)
            click.echo()


@cli.command()
@click.argument("task", required=False)
@click.option("--debug-tui/--no-debug-tui", default=True, help="Enable/disable debug TUI")
@click.option("--max-steps", type=int, default=25, help="Max agent steps")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.option("--headless", is_flag=True, help="Run without TUI, output JSON result")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--save-dir", default=None, type=click.Path(), help="Directory for saving sessions")
def run(
    task: str | None,
    debug_tui: bool,
    max_steps: int,
    workspace: str | None,
    no_stream: bool,
    headless: bool,
    resume: str | None,
    save_dir: str | None,
) -> None:
    """Run chef-human on a task."""
    if headless:
        debug_tui = False

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or ".")
        if session_data is None:
            click.echo(f"Session '{resume}' not found.", err=True)
            raise SystemExit(1)
        task = task or session_data.get("task", "")

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
            headless=headless,
            resume=resume,
            save_dir=save_dir,
        )
    )

    if headless:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
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
    headless: bool = False,
    resume: str | None = None,
    save_dir: str | None = None,
) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)

    context = create_context_assembler(workspace_root=workspace)

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or ".")
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                context.conversation.messages = loaded.messages

    backend = create_backend()
    tool_registry = create_tool_registry(context.workspace)
    planner = Planner(llm_backend=backend)
    config = ReActConfig(
        max_steps=max_steps,
        stream=stream,
        save_dir=save_dir,
    )
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
