"""Tests for human_player module using mock stdin/stdout."""

from __future__ import annotations

import io

import pytest

from chess_lmm.human_player import human_turn
from chess_lmm.mock_server import MockChessServer


async def _setup_game(
    server: MockChessServer,
    *,
    fen: str | None = None,
) -> tuple:
    """Create game and join two players."""
    white = server.create_session()
    black = server.create_session()
    await white.create_game(fen=fen)
    await white.join_game("white")
    await black.join_game("black")
    return white, black


class TestHumanPlayer:
    @pytest.fixture
    def server(self) -> MockChessServer:
        return MockChessServer()

    async def test_make_move(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("e4\n")
        stdout = io.StringIO()

        result = await human_turn(white, input_stream=stdin, output_stream=stdout)

        assert result is True
        output = stdout.getvalue()
        assert "Played: e4" in output

    async def test_invalid_move(self, server: MockChessServer) -> None:
        """Invalid move shows error, then a valid move succeeds."""
        white, black = await _setup_game(server)
        stdin = io.StringIO("xyz\ne4\n")
        stdout = io.StringIO()

        result = await human_turn(white, input_stream=stdin, output_stream=stdout)

        assert result is True
        output = stdout.getvalue()
        assert "Error:" in output
        assert "Played: e4" in output

    async def test_help_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/help\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "/moves" in output
        assert "/board" in output

    async def test_board_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/board\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "R  N  B  Q  K  B  N  R" in output

    async def test_status_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/status\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "Status: ongoing" in output
        assert "Turn: white" in output

    async def test_moves_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/moves\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "Legal moves (20)" in output

    async def test_moves_with_square(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/moves e2\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "Legal moves (2)" in output

    async def test_history_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await black.make_move("e5")

        stdin = io.StringIO("/history\nNf3\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "1. e4 e5" in output

    async def test_resign_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        result = await human_turn(white, input_stream=stdin, output_stream=stdout)

        assert result is False
        output = stdout.getvalue()
        assert "Resigned" in output

    async def test_draw_offer_and_decline(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/draw offer\ne4\n")
        stdout = io.StringIO()

        result = await human_turn(white, input_stream=stdin, output_stream=stdout)

        assert result is True
        output = stdout.getvalue()
        assert "Draw offered" in output

    async def test_eof_returns_false(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("")  # EOF immediately
        stdout = io.StringIO()

        result = await human_turn(white, input_stream=stdin, output_stream=stdout)

        assert result is False

    async def test_unknown_command(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        stdin = io.StringIO("/unknown\ne4\n")
        stdout = io.StringIO()

        await human_turn(white, input_stream=stdin, output_stream=stdout)

        output = stdout.getvalue()
        assert "Unknown command" in output
