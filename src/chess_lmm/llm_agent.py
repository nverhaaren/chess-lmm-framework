"""Claude agentic loop for playing chess.

Uses the Anthropic SDK directly with a manual loop:
1. Get status + board + legal moves via MCP (query calls)
2. Build Claude API request with tool definitions for actions only
3. Claude responds with tool_use block
4. Execute the tool via MCP client
5. Feed tool_result back, repeat if needed
6. Log everything

Query tools are NOT exposed as Claude tools — board state and legal moves
are injected into the user message to reduce API calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from chess_lmm.mcp_interface import ChessSessionClient
from chess_lmm.recording import LlmInteractionLogger
from chess_lmm.types import McpError

logger = logging.getLogger(__name__)

# Tool definitions exposed to Claude (action tools only)
CHESS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "make_move",
        "description": (
            "Submit a chess move in SAN (e.g. 'e4', 'Nf3', 'O-O') "
            "or LAN (e.g. 'e2e4') notation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "The move to play.",
                }
            },
            "required": ["move"],
        },
    },
    {
        "name": "offer_draw",
        "description": "Offer a draw to the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "claim_draw",
        "description": (
            "Claim a draw under the fifty-move rule or threefold "
            "repetition. Only valid when can_claim_draw is true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "accept_draw",
        "description": "Accept a pending draw offer from the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "decline_draw",
        "description": "Decline a pending draw offer from the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "resign",
        "description": "Resign the game. The opponent wins.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _build_position_context(
    status: dict[str, Any],
    board_fen: str,
    legal_moves: dict[str, Any],
) -> str:
    """Build a context string describing the current position for Claude."""
    parts: list[str] = []

    parts.append(f"You are playing as {status.get('turn', '?')}.")
    parts.append(f"Current FEN: {board_fen}")
    parts.append(f"Move number: {status.get('fullmove_number', '?')}")

    if status.get("is_check"):
        parts.append("You are in CHECK! You must get out of check.")

    if status.get("draw_offered"):
        parts.append(
            "Your opponent has offered a draw. "
            "You must accept_draw or decline_draw before making a move."
        )

    claim = status.get("can_claim_draw", {})
    if claim.get("fifty_move"):
        parts.append("You can claim a draw under the fifty-move rule.")
    if claim.get("repetition"):
        parts.append("You can claim a draw by threefold repetition.")

    if status.get("insufficient_material"):
        parts.append("Insufficient material — the position is a theoretical draw.")

    # Legal moves
    moves = legal_moves.get("moves", [])
    move_strs = [m.get("san", m.get("lan", "?")) for m in moves]
    parts.append(f"\nLegal moves ({len(move_strs)}): {', '.join(move_strs)}")

    return "\n".join(parts)


async def llm_turn(
    client: ChessSessionClient,
    anthropic_client: Any,
    model: str,
    *,
    llm_logger: LlmInteractionLogger | None = None,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> bool:
    """Handle one turn for the LLM agent.

    Returns True if the game is still ongoing, False if it ended.
    """
    # 1. Query current state
    status = await client.get_status()
    board = await client.get_board()
    legal_moves = await client.get_legal_moves(format="san")

    # Check if game already ended
    if status.get("server_state") == "game_over":
        return False

    # 2. Build the context message
    position_context = _build_position_context(
        dict(status), board["fen"], dict(legal_moves)
    )

    if system_prompt is None:
        system_prompt = (
            "You are playing a game of chess. Analyze the position and make "
            "your best move. Use the make_move tool to submit your move in "
            "SAN notation. Think carefully about tactics and strategy."
        )

    messages: list[dict[str, Any]] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": position_context})

    # 3. Call Claude with tools
    max_iterations = 5  # Safety limit for tool-use loop
    for _ in range(max_iterations):
        request_payload = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": messages,
            "tools": CHESS_TOOLS,
        }

        if llm_logger:
            llm_logger.log({"type": "api_request", "payload": request_payload})

        response = anthropic_client.messages.create(**request_payload)

        if llm_logger:
            llm_logger.log(
                {
                    "type": "api_response",
                    "response": _serialize_response(response),
                }
            )

        # Process response
        assistant_content = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                logger.info("LLM says: %s", block.text)
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                tool_use_blocks.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_use_blocks:
            # No tool call — Claude just talked. Prompt again.
            messages.append(
                {
                    "role": "user",
                    "content": "Please use one of the tools to take an action.",
                }
            )
            continue

        # Execute tool calls
        tool_results: list[dict[str, Any]] = []
        game_ended = False

        for tool_block in tool_use_blocks:
            tool_result = await _execute_tool(client, tool_block.name, tool_block.input)

            if tool_result.get("is_error"):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "is_error": True,
                        "content": json.dumps(tool_result.get("error", {})),
                    }
                )
            else:
                result_data = tool_result.get("result", {})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": json.dumps(result_data),
                    }
                )
                # Check if game ended
                if result_data.get("server_state") == "game_over":
                    game_ended = True

        messages.append({"role": "user", "content": tool_results})

        if game_ended:
            return False

        # If a move was successfully made, we're done
        for tr in tool_results:
            if not tr.get("is_error"):
                return True

    logger.warning("LLM turn exceeded max iterations")
    return True


async def _execute_tool(
    client: ChessSessionClient,
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Execute a chess tool call via the MCP client."""
    try:
        tool_result: Any
        if tool_name == "make_move":
            tool_result = await client.make_move(tool_input["move"])
        elif tool_name == "offer_draw":
            tool_result = await client.offer_draw()
        elif tool_name == "claim_draw":
            tool_result = await client.claim_draw()
        elif tool_name == "accept_draw":
            tool_result = await client.accept_draw()
        elif tool_name == "decline_draw":
            tool_result = await client.decline_draw()
        elif tool_name == "resign":
            tool_result = await client.resign()
        else:
            return {
                "is_error": True,
                "error": {
                    "error": "unknown_tool",
                    "message": f"Unknown tool: {tool_name}",
                },
            }
        return {"result": dict(tool_result)}
    except McpError as e:
        return {"is_error": True, "error": e.to_dict()}


def _serialize_response(response: Any) -> dict[str, Any]:
    """Serialize an Anthropic API response for logging."""
    try:
        return response.model_dump()  # type: ignore[no-any-return]
    except AttributeError:
        return {"raw": str(response)}
