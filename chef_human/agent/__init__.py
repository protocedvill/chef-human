from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.planner import Plan, PlanStep, Planner, StepStatus
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import RegexExtractor, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.config import settings
from chef_human.llm.tokenizer import create_tokenizer


def create_context_assembler() -> ContextAssembler:
    tokenizer = create_tokenizer(settings.ollama_model)
    workspace = WorkspaceManager(root=settings.workspace or None)
    config = ContextConfig(
        max_tokens=settings.max_context_tokens,
        max_response_tokens=settings.max_response_tokens,
    )
    conversation = ContextManager(config=config, tokenizer=tokenizer)
    file_ctx = FileContextManager(
        workspace=workspace,
        tokenizer=tokenizer,
    )
    repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
    )


__all__ = [
    "ContextAssembler",
    "ContextConfig",
    "ContextManager",
    "FileContextManager",
    "Plan",
    "PlanStep",
    "Planner",
    "RepoMap",
    "StepStatus",
    "WorkspaceManager",
    "create_context_assembler",
]
