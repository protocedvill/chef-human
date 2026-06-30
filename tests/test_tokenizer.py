from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from chef_human.llm.tokenizer import (
    ApproxTokenizer,
    create_tokenizer,
)
from chef_human.agent.context import ContextConfig, ContextManager


# Minimal Message for testing
@dataclass
class FakeMessage:
    role: str = "user"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@pytest.fixture
def fake_message():
    return lambda content: FakeMessage(content=content)  # type: ignore[return-value]


class TestApproxTokenizer:
    def test_empty_string(self):
        assert ApproxTokenizer().count("") == 1

    def test_short_string(self):
        assert ApproxTokenizer().count("hi") == 1

    def test_four_chars_is_one_token(self):
        assert ApproxTokenizer().count("aaaa") == 1

    def test_five_chars_is_one_token(self):
        assert ApproxTokenizer().count("aaaaa") == 1

    def test_eight_chars_is_two_tokens(self):
        assert ApproxTokenizer().count("aaaaaaaa") == 2


class TestCreateTokenizer:
    def test_fallback_to_approx_when_tiktoken_missing(self):
        tokenizer = create_tokenizer()
        assert isinstance(tokenizer, ApproxTokenizer)

    def test_accepts_model_name(self):
        tokenizer = create_tokenizer("qwen2.5-coder:7b")
        assert isinstance(tokenizer, ApproxTokenizer)


class TestTiktokenTokenizer:
    def test_raises_importerror_when_not_installed(self):
        with pytest.raises(ImportError):
            from chef_human.llm.tokenizer import TiktokenTokenizer

            TiktokenTokenizer()


class TestContextConfig:
    def test_defaults(self):
        cfg = ContextConfig()
        assert cfg.max_tokens == 32768
        assert cfg.max_response_tokens == 4096
        assert cfg.summary_tokens == 512

    def test_custom_values(self):
        cfg = ContextConfig(max_tokens=16384, max_response_tokens=2048, summary_tokens=256)
        assert cfg.max_tokens == 16384
        assert cfg.max_response_tokens == 2048


class TestContextManager:
    def test_initially_empty(self):
        cm = ContextManager()
        assert cm.get_messages() == []

    def test_token_count_zero_when_empty(self):
        cm = ContextManager()
        assert cm.token_count() == 0

    def test_add_message_appends(self):
        cm = ContextManager()
        msg = FakeMessage(content="hello")
        cm.add_message(msg)  # type: ignore[arg-type]
        assert len(cm.get_messages()) == 1

    def test_token_count_after_add(self):
        cm = ContextManager()
        cm.add_message(FakeMessage(content="hello world"))  # type: ignore[arg-type]
        # ApproxTokenizer: 11 chars // 4 = 2
        assert cm.token_count() == 2

    def test_does_not_trim_under_budget(self):
        cm = ContextManager(config=ContextConfig(max_tokens=100, max_response_tokens=0, summary_tokens=0))
        for _ in range(5):
            cm.add_message(FakeMessage(content="a" * 16))  # type: ignore[arg-type] # 4 tokens each, 20 total
        assert len(cm.get_messages()) == 5

    def test_trims_oldest_when_over_budget(self):
        cm = ContextManager(config=ContextConfig(max_tokens=20, max_response_tokens=0, summary_tokens=0))
        for i in range(10):
            cm.add_message(FakeMessage(content="a" * 16))  # type: ignore[arg-type] # 4 tokens each
        assert len(cm.get_messages()) < 10

    def test_protects_last_message_when_only_two_remain(self):
        cm = ContextManager(config=ContextConfig(max_tokens=8, max_response_tokens=0, summary_tokens=0))
        cm.add_message(FakeMessage(content="a" * 16))  # type: ignore[arg-type] # 4 tokens
        cm.add_message(FakeMessage(content="b" * 16))  # type: ignore[arg-type] # 4 tokens
        # Both fit under 8? 4+4=8, so no trim yet
        cm.add_message(FakeMessage(content="c" * 16))  # type: ignore[arg-type] # 12 tokens > 8
        # Should trim to keep at least 1 (but not 0)
        assert len(cm.get_messages()) >= 1

    def test_does_not_trim_when_within_budget_with_response_reserve(self):
        cm = ContextManager(config=ContextConfig(max_tokens=100, max_response_tokens=50, summary_tokens=10))
        for _ in range(5):
            cm.add_message(FakeMessage(content="a" * 16))  # type: ignore[arg-type] # 4 tokens each, 20 total
        assert len(cm.get_messages()) == 5

    def test_custom_tokenizer(self):
        class FixedTokenizer:
            def count(self, text: str) -> int:
                return 42

        cm = ContextManager(tokenizer=FixedTokenizer())
        cm.add_message(FakeMessage(content="anything"))  # type: ignore[arg-type]
        assert cm.token_count() == 42
