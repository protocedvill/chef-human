from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chef_human.llm.backend import Message, Role
from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.rag.retriever import RAGRetriever
    from chef_human.agent.repo_map import RepoMap
    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.symbols.retriever import SymbolRetriever
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
        # Never pop past the last user (or system) message: the Qwen3.5/3.8
        # chat template raises 'No user query found in messages' if every
        # user message has been consumed, so at least one user (or the
        # opening system prompt) must survive any trim pass.
        while self.token_count() > budget and len(self.messages) > 1:
            idx = self._last_non_user_index()
            if idx < 1:
                break
            if not self._summary and idx > 2:
                old = self.messages[1 : idx + 1]
                self._summary = (
                    f"[Previous conversation: {len(old)} messages trimmed]"
                )
                self.messages = [self.messages[0]] + self.messages[idx + 1 :]
            else:
                self.messages.pop(1)

    def _last_non_user_index(self) -> int:
        """Index of the last message with a role other than `user`, or -1 if
        none exists (i.e. only user messages remain). Popping past this
        position would leave only user messages -- which is the shape the
        Qwen3.5/3.8 template rejects -- so trims must stop at or before it.
        The leading system message at index 0 is always preserved (idx>=1
        above), which also keeps the guard happy since the template only
        needs a single non-tool user or system message to render."""
        last = -1
        for i, m in enumerate(self.messages):
            if m.role.value != "user":
                last = i
        return last


class ContextAssembler:
    def __init__(
        self,
        conversation: ContextManager,
        workspace: WorkspaceManager,
        file_context: FileContextManager,
        repo_map: RepoMap,
        symbol_index: SymbolIndex | None = None,
        dep_graph: DependencyGraph | None = None,
        symbol_retriever: SymbolRetriever | None = None,
        rag_retriever: RAGRetriever | None = None,
    ) -> None:
        self._conversation = conversation
        self._workspace = workspace
        self._file_context = file_context
        self._repo_map = repo_map
        self._symbol_index = symbol_index
        self._dep_graph = dep_graph
        self._symbol_retriever = symbol_retriever
        self._rag_retriever = rag_retriever
        # generate() re-walks the whole workspace tree and re-reads files
        # for the symbol map -- assemble() runs once per ReAct turn (up to
        # max_steps times per task), and the repo layout is unchanged
        # between turns unless a file-mutating tool actually ran. Cache by
        # budget (the only thing that varies call-to-call) and let the
        # caller invalidate explicitly after a mutation.
        self._repo_map_cache: dict[int, str] = {}

    def invalidate_repo_map_cache(self) -> None:
        self._repo_map_cache.clear()

    @property
    def conversation(self) -> ContextManager:
        return self._conversation

    @property
    def workspace(self) -> WorkspaceManager:
        return self._workspace

    @property
    def symbol_index(self) -> SymbolIndex | None:
        return self._symbol_index

    @property
    def file_context(self) -> FileContextManager:
        return self._file_context

    @property
    def dep_graph(self) -> DependencyGraph | None:
        return self._dep_graph

    def assemble(
        self,
        system_prompt: str,
    ) -> list[Message]:
        system_content = system_prompt

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
            if repo_budget not in self._repo_map_cache:
                self._repo_map_cache[repo_budget] = self._repo_map.generate(max_tokens=repo_budget)
            repo_map_text = self._repo_map_cache[repo_budget]
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

        if self._rag_retriever and conversation_messages and remaining > 500:
            rag_text = self._build_rag_context(conversation_messages, remaining)
            if rag_text:
                messages.append(
                    Message(role=Role.system, content=f"## Related Code\n\n{rag_text}")
                )
        elif self._symbol_retriever and conversation_messages and remaining > 500:
            symbol_text = self._build_symbol_context(conversation_messages, remaining)
            if symbol_text:
                messages.append(
                    Message(role=Role.system, content=f"## Related Symbols\n\n{symbol_text}")
                )

        messages.extend(conversation_messages)
        return messages

    def _build_symbol_context(
        self, conversation_messages: list[Message], budget: int
    ) -> str:
        retriever = self._symbol_retriever
        assert retriever is not None
        recent = " ".join(
            m.content for m in conversation_messages[-4:] if m.role != Role.system
        )
        names = retriever.detect_symbol_references(recent)

        sections: list[str] = []
        for name in names[:5]:
            defn = retriever.retrieve(name)
            if defn:
                tokens = self._conversation.tokenizer.count(defn)
                if tokens <= budget:
                    sections.append(defn)
                    budget -= tokens
        return "\n\n".join(sections)

    def _build_rag_context(
        self, conversation_messages: list[Message], budget: int
    ) -> str:
        retriever = self._rag_retriever
        assert retriever is not None
        recent = " ".join(
            m.content for m in conversation_messages[-4:] if m.role != Role.system
        )
        chunks = retriever.retrieve(recent, top_k=5)
        if not chunks:
            return ""
        return retriever.format_for_prompt(chunks, budget)

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
