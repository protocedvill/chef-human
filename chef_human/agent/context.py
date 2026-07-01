from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chef_human.llm.backend import Message, Role
from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.repo_map import RepoMap
    from chef_human.agent.workspace import WorkspaceManager


@dataclass
class ContextConfig:
    max_tokens: int = 32768
    max_response_tokens: int = 4096
    summary_tokens: int = 512
    repo_map_tokens: int = 2000
    file_context_tokens: int = 10000


class ContextManager:
    def __init__(
        self,
        config: ContextConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer or create_tokenizer()
        self.messages: list[Message] = []
        self._summary: str = ""

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self._trim_if_needed()

    def get_messages(self) -> list[Message]:
        return self.messages

    def token_count(self) -> int:
        return sum(self.tokenizer.count(m.content) for m in self.messages)

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "tool_calls": m.tool_calls,
                    "tool_call_id": m.tool_call_id,
                }
                for m in self.messages
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        config: ContextConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> ContextManager:
        cm = cls(config=config, tokenizer=tokenizer)
        cm.messages = [
            Message(
                role=Role(msg["role"]),
                content=msg["content"],
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
            for msg in data["messages"]
        ]
        return cm

    def _trim_if_needed(self) -> None:
        budget = self.config.max_tokens - self.config.max_response_tokens - self.config.summary_tokens
        while self.token_count() > budget and len(self.messages) > 1:
            if not self._summary and len(self.messages) > 3:
                old = self.messages[:2]
                self._summary = f"[Previous conversation: {len(old)} messages trimmed]"
                self.messages = self.messages[2:]
            elif len(self.messages) > 2:
                self.messages.pop(0)
            else:
                break


class ContextAssembler:
    def __init__(
        self,
        conversation: ContextManager,
        workspace: WorkspaceManager,
        file_context: FileContextManager,
        repo_map: RepoMap,
    ) -> None:
        self._conversation = conversation
        self._workspace = workspace
        self._file_context = file_context
        self._repo_map = repo_map

    @property
    def conversation(self) -> ContextManager:
        return self._conversation

    @property
    def workspace(self) -> WorkspaceManager:
        return self._workspace

    def assemble(
        self,
        system_prompt: str,
        tool_definitions: str = "",
    ) -> list[Message]:
        system_content = system_prompt
        if tool_definitions:
            system_content += "\n\n" + tool_definitions

        system_tokens = self._conversation.tokenizer.count(system_content)
        remaining = (
            self._conversation.config.max_tokens
            - self._conversation.config.max_response_tokens
            - system_tokens
        )

        conversation_messages = self._conversation.get_messages()

        repo_map_text = ""
        repo_budget = min(
            self._conversation.config.repo_map_tokens,
            int(remaining * 0.15),
        )
        if repo_budget > 100:
            repo_map_text = self._repo_map.generate(max_tokens=repo_budget)
            remaining -= self._conversation.tokenizer.count(repo_map_text)

        file_text = self._build_file_context()
        file_tokens = self._conversation.tokenizer.count(file_text)
        file_budget = min(
            self._conversation.config.file_context_tokens,
            remaining,
        )
        if file_tokens > file_budget:
            file_text = self._truncate_file_context(file_text, file_budget)

        messages: list[Message] = []
        messages.append(Message(role=Role.system, content=system_content))

        if repo_map_text:
            messages.append(
                Message(role=Role.system, content=f"## Repository Structure\n\n{repo_map_text}")
            )

        if file_text:
            messages.append(
                Message(role=Role.system, content=f"## File Context\n\n{file_text}")
            )

        messages.extend(conversation_messages)
        return messages

    def _build_file_context(self) -> str:
        sections: list[str] = []
        for path in self._file_context.cached_files():
            content = self._file_context.get(path)
            if content is not None:
                resolved = path if path.is_absolute() else self._workspace.resolve(path)
                rel = resolved.relative_to(self._workspace.root)
                lines_list = content.splitlines()
                sections.append(f"File: {rel} ({len(lines_list)} lines)")
                sections.append("```")
                sections.append(content)
                sections.append("```")
                sections.append("")
        return "\n".join(sections)

    def _truncate_file_context(self, text: str, max_tokens: int) -> str:
        sections = text.split("\nFile: ")
        kept: list[str] = []
        remaining = max_tokens
        for sec in sections:
            if not sec.strip():
                continue
            entry_tokens = self._conversation.tokenizer.count(sec)
            if entry_tokens <= remaining:
                kept.append(sec)
                remaining -= entry_tokens
            else:
                break
        return "\nFile: ".join(kept)
