"""CLI human player for chess games.

Reads commands from stdin, displays board and moves to stdout.
Commands:
  e4, Nf3, e2e4   → make_move
  /moves [square]  → get_legal_moves
  /board           → get_board (re-display)
  /status          → get_status
  /history         → get_history
  /draw offer      → offer_draw
  /draw claim      → claim_draw
  /draw accept     → accept_draw
  /draw decline    → decline_draw
  /resign          → resign
  /help            → show commands
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any, TextIO

from chess_lmm.mcp_interface import ChessSessionClient
from chess_lmm.recording import render_board
from chess_lmm.types import McpError

HELP_TEXT = """Commands:
  <move>           Make a move (e.g., e4, Nf3, e2e4, O-O)
  /moves [square]  Show legal moves (optionally for a specific square)
  /board           Re-display the board
  /status          Show game status
  /history         Show move history
  /draw offer      Offer a draw
  /draw claim      Claim a draw (fifty-move or threefold repetition)
  /draw accept     Accept a pending draw offer
  /draw decline    Decline a pending draw offer
  /resign          Resign the game
  /help            Show this help message
"""


async def human_turn(
    client: ChessSessionClient,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> bool:
    """Handle one turn for the human player.

    Reads commands from input_stream until a move is made or the game ends.
    Returns True if the game is still ongoing, False if it ended.
    """

    def write(text: str) -> None:
        output_stream.write(text + "\n")
        output_stream.flush()

    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, input_stream.readline
            )
        except EOFError:
            return False

        if not line:
            return False

        line = line.strip()
        if not line:
            continue

        try:
            if line == "/help":
                write(HELP_TEXT)
                continue

            if line == "/board":
                board = await client.get_board()
                write(render_board(board["fen"]))
                continue

            if line == "/status":
                status = await client.get_status()
                _display_status(dict(status), write)
                continue

            if line.startswith("/moves"):
                parts = line.split()
                square = parts[1] if len(parts) > 1 else None
                moves = await client.get_legal_moves(square=square)
                _display_moves(dict(moves), write)
                continue

            if line == "/history":
                history = await client.get_history()
                _display_history(dict(history), write)
                continue

            if line == "/draw offer":
                await client.offer_draw()
                write("Draw offered.")
                continue

            if line == "/draw claim":
                result = await client.claim_draw()
                write(f"Draw claimed: {result.get('termination_reason', 'draw')}")
                return False

            if line == "/draw accept":
                result = await client.accept_draw()
                write(f"Draw accepted: {result.get('termination_reason', 'draw')}")
                return False

            if line == "/draw decline":
                await client.decline_draw()
                write("Draw offer declined.")
                continue

            if line == "/resign":
                result = await client.resign()
                write(f"Resigned: {result.get('termination_reason', 'resigned')}")
                return False

            if line.startswith("/"):
                write(f"Unknown command: {line}")
                write("Type /help for available commands.")
                continue

            # Treat as a move
            result = await client.make_move(line)
            san = result["move_played"].get("san", line)
            write(f"Played: {san}")
            fen = result.get("fen") or ""
            write(render_board(fen))

            # Check for game over
            if result.get("server_state") == "game_over":
                reason = result.get("termination_reason", "Game over")
                write(f"\n{reason}")
                return False

            return True

        except McpError as e:
            write(f"Error: {e.message}")
            continue


def _display_status(status: dict[str, Any], write: Callable[[str], None]) -> None:
    """Display game status information."""
    write(f"Status: {status.get('game_status', '?')}")
    if status.get("turn"):
        write(f"Turn: {status['turn']}")
    if status.get("is_check"):
        write("CHECK!")
    if status.get("draw_offered"):
        write("Your opponent has offered a draw. Use /draw accept or /draw decline.")
    if status.get("can_claim_draw"):
        cd = status["can_claim_draw"]
        if cd.get("fifty_move"):
            write("You can claim a draw under the fifty-move rule (/draw claim)")
        if cd.get("repetition"):
            write("You can claim a draw by threefold repetition (/draw claim)")


def _display_moves(moves: dict[str, Any], write: Callable[[str], None]) -> None:
    """Display legal moves."""
    write(f"Legal moves ({moves['count']}):")
    move_strs = []
    for m in moves["moves"]:
        san = m.get("san", "")
        lan = m.get("lan", "")
        if san and lan:
            move_strs.append(f"{san} ({lan})")
        elif san:
            move_strs.append(san)
        else:
            move_strs.append(lan)
    # Display in rows of 8
    for i in range(0, len(move_strs), 8):
        write("  " + "  ".join(move_strs[i : i + 8]))


def _display_history(history: dict[str, Any], write: Callable[[str], None]) -> None:
    """Display move history."""
    if not history["moves"]:
        write("No moves played yet.")
        return
    for entry in history["moves"]:
        num = entry["move_number"]
        white = entry.get("white")
        black = entry.get("black")
        w_str = white["san"] if white else "..."
        b_str = black["san"] if black else ""
        if b_str:
            write(f"  {num}. {w_str} {b_str}")
        else:
            write(f"  {num}. {w_str}")
