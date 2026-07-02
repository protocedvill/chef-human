from __future__ import annotations

import asyncio
import json
import logging
import sys

import click

from chef_human.agent import create_agent
from chef_human.agent.context import ContextManager
from chef_human.agent.persistence import (
    DEFAULT_SAVE_DIR,
    delete_session,
    list_sessions,
    load_session_data,
    save_conversation,
)
from chef_human.agent.react_loop import AgentResult
from chef_human.config import settings

import re
from rich.console import Console
from rich.syntax import Syntax


def _resolve_settings(
    model: str | None,
    temperature: float | None,
    config_path: str | None,
) -> "Settings":
    from chef_human.config import Settings, load_settings
    import dataclasses

    if config_path:
        return load_settings(config_path=config_path)

    overrides: dict[str, object] = {}
    if model is not None:
        overrides["ollama_model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature

    if overrides:
        return dataclasses.replace(settings, **overrides)
    return settings


def _print_message_with_code_highlighting(console: Console, message: str) -> None:
    pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)
    last_end = 0
    for m in pattern.finditer(message):
        before = message[last_end : m.start()]
        if before.strip():
            for line in before.strip().split("\n"):
                console.print(f"  {line}")
        lang = m.group(1) or "text"
        code = m.group(2)
        try:
            syntax = Syntax(code, lang, theme="monokai", word_wrap=True)
            console.print(syntax)
        except Exception:
            console.print(f"  ```{lang}")
            for line in code.split("\n"):
                console.print(f"  {line}")
            console.print("  ```")
        last_end = m.end()
    remaining = message[last_end:]
    if remaining.strip():
        for line in remaining.strip().split("\n"):
            console.print(f"  {line}")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
def show_config(config_path: str | None) -> None:
    """Display the effective configuration."""
    from rich.console import Console
    from rich.table import Table
    cfg = _resolve_settings(model=None, temperature=None, config_path=config_path)
    table = Table(title="chef-human Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    import dataclasses
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        table.add_row(field.name, str(value))
    Console().print(table)


@cli.group()
def session() -> None:
    """Manage saved sessions."""


@session.command("list")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
def session_list(save_dir: str | None) -> None:
    """List all saved sessions."""
    from datetime import datetime
    sessions = list_sessions(save_dir=save_dir or DEFAULT_SAVE_DIR)
    if not sessions:
        click.echo("No sessions found.")
        return
    from rich.table import Table
    table = Table(title="Saved Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Date", style="green")
    table.add_column("Task", style="white")
    for s in sessions:
        sid = s.get("session_id", "?")[:12]
        ts = s.get("created", 0)
        if isinstance(ts, (int, float)):
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(ts)[:16]
        task = (s.get("task") or "?")[:80]
        table.add_row(sid, date_str, task)
    from rich.console import Console
    Console().print(table)


@session.command("show")
@click.argument("session_id")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
def session_show(session_id: str, save_dir: str | None) -> None:
    """Show details of a saved session."""
    data = load_session_data(session_id, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
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
    if not delete_session(session_id, save_dir=save_dir or str(DEFAULT_SAVE_DIR)):
        click.echo(f"Session '{session_id}' not found.", err=True)
        raise SystemExit(1)
    click.echo(f"Session '{session_id}' deleted.")


@session.command("export")
@click.argument("session_id")
@click.option("--save-dir", default=None, type=click.Path(), help="Session save directory")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", help="Export format")
def session_export(session_id: str, save_dir: str | None, fmt: str) -> None:
    """Export a session in JSON or Markdown format."""
    data = load_session_data(session_id, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
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
@click.option("--json", "json_output", is_flag=True, help="Output final result as JSON after streaming")
@click.option("--quiet", is_flag=True, help="Suppress all output except final result")
@click.option("--model", default=None, help="LLM model (overrides config)")
@click.option("--temperature", type=float, default=None, help="Model temperature")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--continue", "resume_from", default=None, type=str, help="Continue a previous session (alias for --resume)")
@click.option("--save-dir", default=None, type=click.Path(), help="Directory for saving sessions")
def run(
    task: str | None,
    debug_tui: bool,
    max_steps: int,
    workspace: str | None,
    no_stream: bool,
    headless: bool,
    json_output: bool,
    quiet: bool,
    model: str | None,
    temperature: float | None,
    config_path: str | None,
    resume: str | None,
    resume_from: str | None,
    save_dir: str | None,
) -> None:
    """Run chef-human on a task."""
    if headless:
        debug_tui = False

    resume = resume or resume_from

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
        if session_data is None:
            click.echo(f"Session '{resume}' not found.", err=True)
            raise SystemExit(1)
        task = task or session_data.get("task", "")

    if not task and sys.stdin.isatty():
        task = click.prompt("Task", default="")
        if not task:
            click.echo("No task provided. Exiting.")
            return

    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()

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
            quiet=quiet,
            model=model,
            temperature=temperature,
            config_path=config_path,
            resume=resume,
            save_dir=save_dir,
        )
    )

    if headless:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        from rich.console import Console
        console = Console()
        status = "[bold green]✓ SUCCESS[/]" if result.success else "[bold red]✗ FAILURE[/]"
        console.print(f"\n[bold]Result:[/] {status}")
        console.print(f"  [dim]Steps:[/] {result.steps_taken}")
        if result.message:
            console.print(f"\n[bold]Message:[/]")
            _print_message_with_code_highlighting(console, result.message)
        if result.total_prompt_tokens or result.total_completion_tokens:
            p = f"{result.total_prompt_tokens:,}" if result.total_prompt_tokens else "?"
            c = f"{result.total_completion_tokens:,}" if result.total_completion_tokens else "?"
            console.print(f"\n[dim]Tokens:[/] {p} prompt / {c} completion")
        if json_output:
            console.print()
            console.print(json.dumps(result.to_dict(), indent=2))
    if not result.success:
        raise SystemExit(1)


async def _execute_task(
    task: str,
    debug_tui: bool = True,
    max_steps: int = 25,
    workspace: str | None = None,
    stream: bool = True,
    headless: bool = False,
    quiet: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)

    import chef_human.config as config_module
    import dataclasses

    old_settings = config_module.settings
    if any((model, temperature, config_path)):
        config_module.settings = _resolve_settings(model, temperature, config_path)

    try:
        loop, _ = create_agent(
            workspace_root=workspace,
            max_steps=max_steps,
            debug_tui=debug_tui,
        )
    finally:
        config_module.settings = old_settings

    loop._config.stream = stream
    loop._config.save_dir = save_dir

    from chef_human.ui.streaming import StreamingUI

    if not headless and not debug_tui:
        loop._ui = StreamingUI(quiet=quiet)

    if resume:
        from chef_human.agent.persistence import load_session_data
        session_data = load_session_data(resume, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                loop._context.conversation.messages = loaded.messages

    return await loop.run(task)


@cli.command()
@click.option("--max-steps", type=int, default=25, help="Max agent steps per turn")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--model", default=None, help="LLM model (overrides config)")
@click.option("--temperature", type=float, default=None, help="Model temperature")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--continue", "resume_from", default=None, type=str, help="Continue a previous session (alias for --resume)")
@click.option("--save-dir", default=None, type=click.Path(), help="Directory for saving sessions")
def repl(
    max_steps: int,
    workspace: str | None,
    model: str | None,
    temperature: float | None,
    config_path: str | None,
    resume: str | None,
    resume_from: str | None,
    save_dir: str | None,
) -> None:
    """Start an interactive REPL session."""
    resume = resume or resume_from
    asyncio.run(_run_repl(
        max_steps=max_steps,
        workspace=workspace,
        model=model,
        temperature=temperature,
        config_path=config_path,
        resume=resume,
        save_dir=save_dir,
    ))


async def _run_repl(
    max_steps: int,
    workspace: str | None,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
) -> None:
    import chef_human.config as config_module
    import dataclasses

    logging.basicConfig(level=logging.WARNING)

    old_settings = config_module.settings
    if any((model, temperature, config_path)):
        config_module.settings = _resolve_settings(model, temperature, config_path)

    try:
        from chef_human.agent import create_context_assembler
        context = create_context_assembler(workspace_root=workspace)
    finally:
        config_module.settings = old_settings

    from chef_human.agent.planner import Planner
    from chef_human.agent.react_loop import ReActConfig, ReActLoop
    from chef_human.config import settings as current_settings
    from chef_human.llm import create_backend
    from chef_human.tools import create_tool_registry
    from chef_human.ui.repl import ReplUI

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                context.conversation.messages = loaded.messages

    backend = create_backend()
    tool_registry = create_tool_registry(
        workspace=context.workspace,
        symbol_index=context.symbol_index,
        file_context=context.file_context,
        dep_graph=context.dep_graph,
    )
    planner = Planner(llm_backend=backend)
    ui = ReplUI()

    total_prompt_tokens = 0
    total_completion_tokens = 0
    conversation_saved = False

    ui._console.print("[bold cyan]chef-human interactive mode[/]")
    ui._console.print("Type [bold]/help[/] for commands. [dim]Ctrl+C or /exit to quit.[/]")

    while True:
        text = ui.read_input()
        if text is None:
            break

        if text.startswith("/"):
            cmd = text[1:].lower().strip()

            if cmd == "clear":
                context.conversation.messages.clear()
                ui._console.print("[dim]Conversation history cleared.[/]")
                continue

            if cmd == "save":
                conv = context.conversation.to_dict()
                path = save_conversation(
                    conv,
                    task="repl-session",
                    save_dir=save_dir or str(DEFAULT_SAVE_DIR),
                )
                ui._console.print(f"[dim]Session saved: {path}[/]")
                conversation_saved = True
                continue

            if cmd == "tokens":
                ui._console.print(
                    f"[dim]Tokens: {total_prompt_tokens:,} prompt / "
                    f"{total_completion_tokens:,} completion[/]"
                )
                continue

            if cmd == "history":
                messages = context.conversation.messages
                if not messages:
                    ui._console.print("[dim]No messages yet.[/]")
                else:
                    for m in messages[-20:]:
                        role = m.role.value if hasattr(m.role, 'value') else str(m.role)
                        content = (m.content or "")[:120].replace("\n", " ")
                        ui._console.print(f"  [{role}] {content}")
                continue

            if cmd == "undo":
                tool = tool_registry.get("undo")
                if tool is None:
                    ui._console.print("[red]Undo tool is not available.[/]")
                else:
                    from chef_human.tools.registry import ToolResult
                    result = await tool.run()
                    ui._console.print(f"[dim]{result.output or result.error}[/]")
                continue

            if cmd == "redo":
                tool = tool_registry.get("redo")
                if tool is None:
                    ui._console.print("[red]Redo tool is not available.[/]")
                else:
                    from chef_human.tools.registry import ToolResult
                    result = await tool.run()
                    ui._console.print(f"[dim]{result.output or result.error}[/]")
                continue

            continue

        if not text:
            continue

        config = ReActConfig(
            max_steps=max_steps,
            tool_timeout=settings.tool_timeout,
            stream=True,
            save_sessions=False,
        )

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=tool_registry,
            context_assembler=context,
            planner=planner,
            config=config,
            ui=ui,
        )

        result = await loop.run(text)

        total_prompt_tokens += result.total_prompt_tokens
        total_completion_tokens += result.total_completion_tokens

        ui.display_result(result)

    if not conversation_saved:
        conv = context.conversation.to_dict()
        save_conversation(
            conv,
            task="repl-session",
            save_dir=save_dir or str(DEFAULT_SAVE_DIR),
        )

    ui._console.print("\n[bold cyan]Goodbye![/]")


@cli.command()
@click.option("--max-steps", type=int, default=25, help="Max agent steps per turn")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--model", default=None, help="LLM model (overrides config)")
@click.option("--temperature", type=float, default=None, help="Model temperature")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--continue", "resume_from", default=None, type=str, help="Continue a previous session (alias for --resume)")
@click.option("--save-dir", default=None, type=click.Path(), help="Directory for saving sessions")
def tui(
    max_steps: int,
    workspace: str | None,
    model: str | None,
    temperature: float | None,
    config_path: str | None,
    resume: str | None,
    resume_from: str | None,
    save_dir: str | None,
) -> None:
    """Start the split-pane Textual TUI (file tree, chat/log, diff preview)."""
    resume = resume or resume_from
    asyncio.run(_run_tui(
        max_steps=max_steps,
        workspace=workspace,
        model=model,
        temperature=temperature,
        config_path=config_path,
        resume=resume,
        save_dir=save_dir,
    ))


async def _run_tui(
    max_steps: int,
    workspace: str | None,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
) -> None:
    import chef_human.config as config_module

    logging.basicConfig(level=logging.WARNING)

    old_settings = config_module.settings
    if any((model, temperature, config_path)):
        config_module.settings = _resolve_settings(model, temperature, config_path)

    try:
        from chef_human.agent import create_context_assembler
        context = create_context_assembler(workspace_root=workspace)
    finally:
        config_module.settings = old_settings

    from chef_human.agent.planner import Planner
    from chef_human.agent.react_loop import ReActConfig, ReActLoop
    from chef_human.llm import create_backend
    from chef_human.tools import create_tool_registry
    from chef_human.ui.textual_tui import ChefHumanTUI

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or str(DEFAULT_SAVE_DIR))
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                context.conversation.messages = loaded.messages

    backend = create_backend()
    tool_registry = create_tool_registry(
        workspace=context.workspace,
        symbol_index=context.symbol_index,
        file_context=context.file_context,
        dep_graph=context.dep_graph,
    )
    planner = Planner(llm_backend=backend)

    app: ChefHumanTUI

    async def handle_task(text: str) -> None:
        config = ReActConfig(
            max_steps=max_steps,
            tool_timeout=settings.tool_timeout,
            stream=True,
            save_sessions=False,
        )
        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=tool_registry,
            context_assembler=context,
            planner=planner,
            config=config,
            ui=app.tui_ui,
        )
        result = await loop.run(text)
        app.tui_ui.display_result(result)

    app = ChefHumanTUI(workspace_root=context.workspace.root, on_submit=handle_task)
    try:
        await app.run_async()
    finally:
        conv = context.conversation.to_dict()
        save_conversation(
            conv,
            task="tui-session",
            save_dir=save_dir or str(DEFAULT_SAVE_DIR),
        )


def main() -> None:
    subcommands = {"run", "repl", "session", "tui"}
    if len(sys.argv) > 1 and sys.argv[1] not in subcommands and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    cli()


if __name__ == "__main__":
    main()
