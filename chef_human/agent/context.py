from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.llm.backend import Message


@dataclass
class ContextConfig:
    max_tokens: int = 32768
    max_response_tokens: int = 4096
    summary_tokens: int = 512


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
