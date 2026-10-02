"""Entrypoint: read the step input, run the graph, return what was posted.

Input (all optional), from Studio or `trase-os-sdk run-workflow --input '{...}'`:
    {"repo": "TraseSystems/trase-os-sdk", "slack_channel": "#ansh-test",
     "github_connection": "github-mcp-trase",
     "user_message": "anything blocking the next release?"}
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

# Exactly one graph exported from this module, for build-time topology inspection.
from agent.graph import issues_graph  # noqa: F401

log = logging.getLogger(__name__)

DEFAULT_REPO = "TraseSystems/trase-os-sdk"
DEFAULT_CHANNEL = "#ansh-test"  # the channel the platform's Slack e2e posts to
# A connection you create (upstream https://api.githubcopilot.com) whose token can read
# TraseSystems repos. Pass "github-mcp" to use the platform's own GitHub MCP connection.
DEFAULT_GITHUB_CONNECTION = "github-mcp-trase"


def _read_input() -> dict[str, Any]:
    try:
        from trase_os_sdk.sandbox import NoInputError, NotInASandboxError, read_input
    except ImportError:
        return {}
    try:
        value = read_input()
    except (NoInputError, NotInASandboxError):
        return {}
    if isinstance(value, str):
        return {"user_message": value}
    return value if isinstance(value, dict) else {}


def run() -> dict[str, Any]:
    """Called by the platform with no arguments. The return value is the step output."""
    payload = _read_input()
    state = {
        "repo": payload.get("repo") or DEFAULT_REPO,
        "channel": payload.get("slack_channel") or DEFAULT_CHANNEL,
        "github_connection": payload.get("github_connection") or DEFAULT_GITHUB_CONNECTION,
        "question": (payload.get("user_message") or "").strip(),
    }
    result = asyncio.run(issues_graph.ainvoke(state))
    return {
        "repo": result["repo"],
        "open_issues": len(result.get("issues", [])),
        "error": result.get("error"),
        "slack": result.get("slack"),
        "message": result.get("summary"),
    }
