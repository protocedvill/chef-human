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


def _configure_logging(log_file: str | None) -> None:
    """Wire up logging for a run. With no --log-file, behavior is unchanged
    (WARNING to stderr). With --log-file, everything chef-human logs --
    including per-LLM-call and per-tool-call DEBUG entries emitted by
    ReActLoop -- goes to that file instead, at DEBUG level, so a stuck or
    misbehaving run can be diagnosed after the fact from the last few lines
    written rather than guessed at from the TUI transcript alone."""
    if log_file:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_file,
            filemode="a",
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)


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
    console = Console()
    console.print(table)

    from chef_human.agent.model_advisor import recommend_model
    recommendation = recommend_model(cfg.ollama_model)
    if recommendation is not None:
        console.print(
            f"\n[bold yellow]Tip:[/] {recommendation.reason}\n"
            f"  Run [bold]chef-human recommend-model[/] for details, or "
            f"[bold]ollama pull {recommendation.suggested.ollama_tag}[/] to try it."
        )


@cli.command("recommend-model")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
def recommend_model_cmd(config_path: str | None) -> None:
    """Suggest a better model if your hardware can comfortably run one."""
    from rich.console import Console
    from rich.table import Table

    from chef_human.agent.hardware import detect_hardware
    from chef_human.agent.model_advisor import find_model_spec, recommend_model

    console = Console()
    cfg = _resolve_settings(model=None, temperature=None, config_path=config_path)
    hardware = detect_hardware()

    if hardware.capacity_gb() is None:
        console.print(
            "[yellow]Could not detect available RAM or VRAM on this machine, "
            "so no recommendation can be made.[/]"
        )
        return

    table = Table(title="Detected Hardware")
    table.add_column("Resource", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("System RAM", f"{hardware.ram_gb:.1f} GB" if hardware.ram_gb else "unknown")
    table.add_row("GPU VRAM", f"{hardware.vram_gb:.1f} GB" if hardware.vram_gb else "not detected")
    console.print(table)

    current_spec = find_model_spec(cfg.ollama_model)
    console.print(f"\n[bold]Current model:[/] {cfg.ollama_model}", end="")
    if current_spec is not None:
        console.print(
            f" ([dim]{current_spec.display_name}, {current_spec.size_b:.0f}B, "
            f"quality {'★' * current_spec.quality}{'☆' * (5 - current_spec.quality)}[/])"
        )
    else:
        console.print(" [dim](not in the known model catalog)[/]")

    recommendation = recommend_model(cfg.ollama_model, hardware=hardware)
    if recommendation is None:
        console.print(
            "\n[bold green]You're already using the best-fit model this machine "
            "can comfortably run.[/]"
        )
        return

    best = recommendation.suggested
    console.print(
        f"\n[bold yellow]Recommendation:[/] {best.display_name} "
        f"([bold]{best.ollama_tag}[/])"
    )
    console.print(
        f"  {best.size_b:.0f}B parameters, needs ~{best.min_ram_gb:.0f} GB, "
        f"quality {'★' * best.quality}{'☆' * (5 - best.quality)}, {best.license} license"
    )
    console.print(f"  {recommendation.reason}")
    console.print(
        f"\n  To try it: [bold]ollama pull {best.ollama_tag}[/] then "
        f"[bold]chef-human run --model {best.ollama_tag} \"...\"[/]\n"
        f"  Or make it the default by setting [bold]ollama_model = \"{best.ollama_tag}\"[/] "
        "in config.toml."
    )


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
@click.option(
    "--debug-tui/--no-debug-tui",
    default=True,
    help="Use the split-pane TUI (default) vs. plain streaming output",
)
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
@click.option(
    "--log-file",
    default=None,
    type=click.Path(),
    help="Write DEBUG-level logs (LLM calls, tool calls, guard rejections, retries/replans, "
    "with timing) to this file for diagnosing stuck or misbehaving runs",
)
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
    log_file: str | None,
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
            log_file=log_file,
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
    log_file: str | None = None,
) -> AgentResult:
    _configure_logging(log_file)

    if not headless and debug_tui:
        return await _run_task_in_tui(
            task=task,
            max_steps=max_steps,
            workspace=workspace,
            stream=stream,
            model=model,
            temperature=temperature,
            config_path=config_path,
            resume=resume,
            save_dir=save_dir,
        )

    import chef_human.config as config_module

    old_settings = config_module.settings
    if any((model, temperature, config_path)):
        config_module.settings = _resolve_settings(model, temperature, config_path)

    try:
        loop, _ = create_agent(
            workspace_root=workspace,
            max_steps=max_steps,
        )
    finally:
        config_module.settings = old_settings

    loop._config.stream = stream
    loop._config.save_dir = save_dir

    from chef_human.ui.streaming import StreamingUI

    if not headless:
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


async def _run_task_in_tui(
    task: str,
    max_steps: int,
    workspace: str | None,
    stream: bool,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
) -> AgentResult:
    """Run a single task inside the split-pane Textual TUI (the default for
    `chef-human run` / `chef-human "task"`), auto-submitting `task` on
    launch and exiting once it completes."""
    import chef_human.config as config_module

    old_settings = config_module.settings
    if any((model, temperature, config_path)):
        config_module.settings = _resolve_settings(model, temperature, config_path)

    try:
        from chef_human.agent import create_context_assembler
        context = create_context_assembler(workspace_root=workspace)
    finally:
        config_module.settings = old_settings

    from chef_human.agent.planner import Plan, Planner
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
    result_holder: list[AgentResult] = []

    async def handle_task(text: str) -> None:
        config = ReActConfig(
            max_steps=max_steps,
            tool_timeout=settings.tool_timeout,
            stream=stream,
            save_dir=save_dir,
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
        result_holder.append(result)
        app.tui_ui.display_result(result)

    app = ChefHumanTUI(
        workspace_root=context.workspace.root,
        on_submit=handle_task,
        initial_task=task,
        auto_exit_after_initial_task=True,
    )
    await app.run_async()

    if result_holder:
        return result_holder[-1]
    return AgentResult(
        plan=Plan(goal=task, steps=[]),
        steps_taken=0,
        message="Exited before the task finished.",
        success=False,
    )


@cli.command()
@click.option("--max-steps", type=int, default=25, help="Max agent steps per turn")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--model", default=None, help="LLM model (overrides config)")
@click.option("--temperature", type=float, default=None, help="Model temperature")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--continue", "resume_from", default=None, type=str, help="Continue a previous session (alias for --resume)")
@click.option("--save-dir", default=None, type=click.Path(), help="Directory for saving sessions")
@click.option(
    "--log-file",
    default=None,
    type=click.Path(),
    help="Write DEBUG-level logs (LLM calls, tool calls, guard rejections, retries/replans, "
    "with timing) to this file for diagnosing stuck or misbehaving runs",
)
def repl(
    max_steps: int,
    workspace: str | None,
    model: str | None,
    temperature: float | None,
    config_path: str | None,
    resume: str | None,
    resume_from: str | None,
    save_dir: str | None,
    log_file: str | None,
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
        log_file=log_file,
    ))


async def _run_repl(
    max_steps: int,
    workspace: str | None,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
    log_file: str | None = None,
) -> None:
    import chef_human.config as config_module
    import dataclasses

    _configure_logging(log_file)

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
@click.option(
    "--log-file",
    default=None,
    type=click.Path(),
    help="Write DEBUG-level logs (LLM calls, tool calls, guard rejections, retries/replans, "
    "with timing) to this file for diagnosing stuck or misbehaving runs",
)
def tui(
    max_steps: int,
    workspace: str | None,
    model: str | None,
    temperature: float | None,
    config_path: str | None,
    resume: str | None,
    resume_from: str | None,
    save_dir: str | None,
    log_file: str | None,
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
        log_file=log_file,
    ))


async def _run_tui(
    max_steps: int,
    workspace: str | None,
    model: str | None = None,
    temperature: float | None = None,
    config_path: str | None = None,
    resume: str | None = None,
    save_dir: str | None = None,
    log_file: str | None = None,
) -> None:
    import chef_human.config as config_module

    _configure_logging(log_file)

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
    # Derived from the actual registered commands (rather than a hand-maintained
    # list) so adding a new @cli.command() can't silently make `chef-human
    # <that-name>` get misrouted into `run` with the command name as the task.
    subcommands = set(cli.commands.keys())
    if len(sys.argv) > 1 and sys.argv[1] not in subcommands and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    cli()


if __name__ == "__main__":
    main()
