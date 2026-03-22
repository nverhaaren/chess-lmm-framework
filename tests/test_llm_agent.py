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
        return {"content": [{"type": b.type} for b in self.content]}


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
