"""Tests for orchestrator module."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from chess_lmm.orchestrator import parse_args, run_game


class MockContentBlock:
    def __init__(self, block_type: str, **kwargs: Any) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockResponse:
    def __init__(self, content: list[MockContentBlock]) -> None:
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {"content": []}


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.color == "white"
        assert args.model == "claude-sonnet-4-6"
        assert args.fen is None
        assert args.log_dir == Path("./game-logs")
        assert args.thinking_budget is None

    def test_custom_args(self) -> None:
        args = parse_args(
            [
                "--color",
                "black",
                "--model",
                "claude-opus-4-6",
                "--fen",
                "8/8/8/8/8/8/8/8 w - - 0 1",
                "--log-dir",
                "/tmp/logs",
            ]
        )
        assert args.color == "black"
        assert args.model == "claude-opus-4-6"
        assert args.fen == "8/8/8/8/8/8/8/8 w - - 0 1"
        assert args.log_dir == Path("/tmp/logs")

    def test_thinking_budget(self) -> None:
        args = parse_args(["--thinking-budget", "10000"])
        assert args.thinking_budget == 10000

    def test_thinking_budget_default_none(self) -> None:
        args = parse_args([])
        assert args.thinking_budget is None


class TestRunGame:
    async def test_human_resigns_immediately(self, tmp_path: Path) -> None:
        """Integration test: human resigns on first move."""
        args = parse_args(["--color", "white", "--log-dir", str(tmp_path)])

        mock_anthropic = MagicMock()
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        output = stdout.getvalue()
        assert "white" in output.lower()
        assert "logs saved" in output.lower() or "Game logs" in output

        # Check that log files were created
        assert (tmp_path / "mcp_recording.jsonl").exists()

    async def test_llm_moves_then_human_resigns(self, tmp_path: Path) -> None:
        """Integration test: LLM plays as white, human as black, human resigns."""
        args = parse_args(["--color", "black", "--log-dir", str(tmp_path)])

        # LLM plays e4
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = MockResponse(
            [
                MockContentBlock(
                    "tool_use", id="t1", name="make_move", input={"move": "e4"}
                )
            ]
        )

        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        output = stdout.getvalue()
        assert "Claude played: e4" in output
