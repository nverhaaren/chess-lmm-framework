"""Tests for llm_agent module using a mock Anthropic client."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chess_lmm.llm_agent import CHESS_TOOLS, _build_position_context, llm_turn
from chess_lmm.mock_server import MockChessServer
from chess_lmm.recording import LlmInteractionLogger

# --- Mock Anthropic client ---


class MockContentBlock:
    """Mock for an Anthropic content block."""

    def __init__(self, block_type: str, **kwargs: Any) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockResponse:
    """Mock for an Anthropic Messages response."""

    def __init__(self, content: list[MockContentBlock]) -> None:
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for b in self.content:
            d: dict[str, Any] = {"type": b.type}
            if b.type == "thinking":
                d["thinking"] = getattr(b, "thinking", "")
                d["signature"] = getattr(b, "signature", "")
            elif b.type == "text":
                d["text"] = getattr(b, "text", "")
            elif b.type == "tool_use":
                d["id"] = getattr(b, "id", "")
                d["name"] = getattr(b, "name", "")
                d["input"] = getattr(b, "input", {})
            blocks.append(d)
        return {"content": blocks}


def make_tool_use_response(
    tool_name: str, tool_input: dict[str, Any], tool_id: str = "tool_1"
) -> MockResponse:
    """Create a mock response with a single tool_use block."""
    return MockResponse(
        [MockContentBlock("tool_use", id=tool_id, name=tool_name, input=tool_input)]
    )


def make_text_response(text: str) -> MockResponse:
    """Create a mock response with only text."""
    return MockResponse([MockContentBlock("text", text=text)])


def make_thinking_tool_response(
    thinking_text: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    tool_id: str = "tool_1",
    signature: str = "test-signature-abc123",
) -> MockResponse:
    """Create a mock response with thinking + tool_use blocks."""
    return MockResponse(
        [
            MockContentBlock(
                "thinking",
                thinking=thinking_text,
                signature=signature,
            ),
            MockContentBlock(
                "tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input,
            ),
        ]
    )


# --- Fixtures ---


async def _setup_game(
    server: MockChessServer,
) -> tuple:
    """Create game and join two players."""
    white = await server.create_session()
    black = await server.create_session()
    await white.create_game()
    await white.join_game("white")
    await black.join_game("black")
    return white, black


@pytest.fixture
def server() -> MockChessServer:
    return MockChessServer()


# --- Tests ---


class TestBuildPositionContext:
    def test_basic_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "white",
            "fullmove_number": 1,
            "is_check": False,
            "draw_offered": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        moves: dict[str, Any] = {
            "moves": [{"san": "e4"}, {"san": "d4"}],
        }
        ctx = _build_position_context(status, "startpos", moves)
        assert "white" in ctx
        assert "e4" in ctx
        assert "d4" in ctx

    def test_check_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "black",
            "fullmove_number": 5,
            "is_check": True,
            "draw_offered": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        ctx = _build_position_context(status, "some_fen", {"moves": []})
        assert "CHECK" in ctx

    def test_draw_offered_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "white",
            "fullmove_number": 10,
            "is_check": False,
            "draw_offered": True,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        ctx = _build_position_context(status, "some_fen", {"moves": []})
        assert "accept_draw" in ctx


class TestLlmTurn:
    async def test_makes_move(self, server: MockChessServer) -> None:
        """LLM makes a valid move via tool_use."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result is True
        # Verify the move was made
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_handles_text_then_move(self, server: MockChessServer) -> None:
        """LLM first returns text, then makes a move on reprompt."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_text_response("Let me think..."),
            make_tool_use_response("make_move", {"move": "d4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result is True
        assert mock_anthropic.messages.create.call_count == 2

    async def test_handles_illegal_move_error(self, server: MockChessServer) -> None:
        """LLM tries an illegal move, gets error, then makes a valid move."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_tool_use_response("make_move", {"move": "e2e5"}),
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result is True
        assert mock_anthropic.messages.create.call_count == 2

    async def test_resign(self, server: MockChessServer) -> None:
        """LLM resigns the game."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "resign", {}
        )

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result is False

    async def test_game_already_over(self, server: MockChessServer) -> None:
        """Returns False when game is already over."""
        white, black = await _setup_game(server)
        await white.resign()

        mock_anthropic = MagicMock()
        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result is False
        mock_anthropic.messages.create.assert_not_called()

    async def test_logs_to_llm_logger(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Verify LLM interactions are logged."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        await llm_turn(white, mock_anthropic, "test-model", llm_logger=llm_logger)

        log_content = (tmp_path / "llm.jsonl").read_text()
        assert "api_request" in log_content
        assert "api_response" in log_content


class TestThinking:
    """Tests for extended thinking support."""

    async def test_makes_move_with_thinking(self, server: MockChessServer) -> None:
        """Thinking block + tool_use: move succeeds normally."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "I should play e4 to control the center.",
            "make_move",
            {"move": "e4"},
        )

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking_budget=2048,
        )

        assert result is True
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_thinking_budget_in_request(self, server: MockChessServer) -> None:
        """Verify thinking param and adjusted max_tokens in API request."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Analysis...", "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking_budget=10000,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }
        assert call_kwargs["max_tokens"] == 11024

    async def test_thinking_disabled_by_default(self, server: MockChessServer) -> None:
        """Without thinking_budget, no thinking param and max_tokens=1024."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(white, mock_anthropic, "test-model")

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert "thinking" not in call_kwargs
        assert call_kwargs["max_tokens"] == 1024

    async def test_thinking_blocks_preserved_in_history(
        self, server: MockChessServer
    ) -> None:
        """Thinking blocks (with signature) are preserved in messages
        passed to the API on tool-result continuation."""
        white, black = await _setup_game(server)

        # First response: thinking + illegal move (triggers retry)
        # Second response: thinking + valid move
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response(
                "Let me try e5.",
                "make_move",
                {"move": "e2e5"},
                signature="sig-first",
            ),
            make_thinking_tool_response(
                "e5 was illegal, try e4.",
                "make_move",
                {"move": "e4"},
                signature="sig-second",
            ),
        ]

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking_budget=2048,
        )

        assert result is True
        assert mock_anthropic.messages.create.call_count == 2

        # Check the second API call's messages contain the thinking
        # block from the first response
        second_call_kwargs = mock_anthropic.messages.create.call_args_list[1][1]
        second_messages = second_call_kwargs["messages"]

        # Find the assistant message with the first thinking block
        assistant_msgs = [m for m in second_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        first_assistant = assistant_msgs[0]["content"]

        # The thinking block should be the SDK object with signature
        thinking_blocks = [
            b for b in first_assistant if getattr(b, "type", None) == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].signature == "sig-first"

    async def test_thinking_with_illegal_move_retry(
        self, server: MockChessServer
    ) -> None:
        """Thinking enabled, illegal move then valid move: both succeed."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response(
                "Try Ke9.",
                "make_move",
                {"move": "Ke9"},
            ),
            make_thinking_tool_response(
                "That was wrong. Play d4.",
                "make_move",
                {"move": "d4"},
            ),
        ]

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking_budget=2048,
        )

        assert result is True
        assert mock_anthropic.messages.create.call_count == 2
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_thinking_budget_too_small(self, server: MockChessServer) -> None:
        """thinking_budget < 1024 raises ValueError."""
        white, black = await _setup_game(server)
        mock_anthropic = MagicMock()

        with pytest.raises(ValueError, match="thinking_budget must be"):
            await llm_turn(
                white,
                mock_anthropic,
                "test-model",
                thinking_budget=500,
            )

    async def test_thinking_logged(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Thinking content appears in LLM interaction log."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Deep analysis of the position.",
            "make_move",
            {"move": "e4"},
        )

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            llm_logger=llm_logger,
            thinking_budget=2048,
        )

        log_content = (tmp_path / "llm.jsonl").read_text()
        assert "api_response" in log_content
        assert "Deep analysis of the position" in log_content

    async def test_thinking_logged_on_retry(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Logging doesn't crash when thinking blocks are in retry messages.

        Thinking blocks are SDK objects (not dicts) in the messages list.
        The logger must handle serializing them when logging the request
        payload on the second iteration of the tool-use loop.
        """
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response("Try e5.", "make_move", {"move": "e2e5"}),
            make_thinking_tool_response("Try e4.", "make_move", {"move": "e4"}),
        ]

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            llm_logger=llm_logger,
            thinking_budget=2048,
        )

        assert result is True
        # Should have logged 4 entries: request, response, request, response
        log_lines = (tmp_path / "llm.jsonl").read_text().strip().split("\n")
        assert len(log_lines) == 4


class TestChessTools:
    def test_tool_definitions_structure(self) -> None:
        """Verify tool definitions have required fields."""
        for tool in CHESS_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_make_move_has_required_param(self) -> None:
        make_move = next(t for t in CHESS_TOOLS if t["name"] == "make_move")
        assert "move" in make_move["input_schema"]["properties"]
        assert "move" in make_move["input_schema"]["required"]
