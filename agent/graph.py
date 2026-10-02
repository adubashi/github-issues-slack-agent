"""GitHub issues -> summary -> Slack, as a LangGraph graph.

    fetch_issues  (GitHub MCP server: list_issues)
        -> summarize  (LLM)
        -> notify     (Slack chat.postMessage)

Three governed connections, one run, and the sandbox holds none of their credentials:

- `github-mcp`: the GitHub MCP server, at the platform-injected TRASE_GITHUB_MCP_URL.
- `openai`: the model, at TRASE_OPENAI_BASE_URL.
- `slack`: Slack's Web API, at ${TRASE_EGRESS_GATEWAY_URL}/slack/api/.

Each call presents the run credential as its bearer token; the gateway checks policy and
swaps in the real GitHub token, OpenAI key or Slack bot token upstream.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

log = logging.getLogger(__name__)

# Graphs are built at import, which the build-time topology inspector also does,
# without a sandbox environment; this placeholder lets the model be constructed there.
_INSPECTION_ONLY = "topology-inspection-only"


def _credential() -> str:
    return os.environ.get("TRASE_RUN_CREDENTIAL", _INSPECTION_ONLY)


model = ChatOpenAI(
    model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
    base_url=os.environ.get("TRASE_OPENAI_BASE_URL"),
    api_key=_credential(),
    temperature=0,
)


class State(TypedDict, total=False):
    repo: str  # "owner/name"
    channel: str  # Slack channel, e.g. "#ansh-test"
    question: str
    issues: list[dict[str, Any]]
    mcp_tools: list[str]
    error: str
    summary: str
    slack: dict[str, Any]


def _issue_rows(raw: Any, repo: str) -> list[dict[str, Any]]:
    """Normalize the MCP list_issues result to a list of issue dicts."""
    if isinstance(raw, list):  # list of content blocks
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):  # GitHub's MCP server wraps the list
        data = data.get("issues") or data.get("items") or data.get("nodes") or []
    rows = []
    for issue in data or []:
        labels = issue.get("labels") or []
        rows.append(
            {
                "number": issue.get("number"),
                "title": issue.get("title"),
                # GitHub's MCP server returns no link; build the web URL from repo + number.
                "url": issue.get("html_url")
                or issue.get("url")
                or f"https://github.com/{repo}/issues/{issue.get('number')}",
                "author": (issue.get("user") or issue.get("author") or {}).get("login"),
                "labels": [lb.get("name") if isinstance(lb, dict) else lb for lb in labels],
                "created_at": issue.get("created_at") or issue.get("createdAt"),
                "comments": issue.get("comments"),
            }
        )
    return rows


async def fetch_issues(state: State) -> State:
    """Ask the GitHub MCP server for the repo's open issues."""
    owner, name = state["repo"].split("/", 1)
    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "streamable_http",
                "url": os.environ["TRASE_GITHUB_MCP_URL"],
                "headers": {"Authorization": f"Bearer {_credential()}"},
            }
        }
    )
    tools = {t.name: t for t in await client.get_tools()}
    log.info("GitHub MCP offered %d tools through the gateway", len(tools))
    if "list_issues" not in tools:
        # GitHub's MCP server hides tools the brokered token can't use.
        return {"mcp_tools": sorted(tools), "error": "the GitHub MCP server offered no list_issues tool"}
    try:
        raw = await tools["list_issues"].ainvoke(
            {"owner": owner, "repo": name, "state": "OPEN", "perPage": 20}
        )
    except Exception as exc:  # noqa: BLE001 - reported in the summary, not raised
        log.warning("list_issues on %s failed: %s", state["repo"], exc)
        return {"mcp_tools": sorted(tools), "error": f"list_issues failed: {exc}"[:500]}
    issues = _issue_rows(raw, state["repo"])
    log.info("list_issues returned %d open issue(s) for %s", len(issues), state["repo"])
    return {"mcp_tools": sorted(tools), "issues": issues}


SUMMARY_PROMPT = """You write short Slack updates for an engineering team.

Write a Slack message (Slack mrkdwn, under 120 words) about the open GitHub issues in
{repo}. Start with one line giving the count. Then list each issue as
"• <url|#number title>" with labels in brackets if any, at most 8 issues. If there is an
error instead of issues, say plainly that the issues couldn't be fetched and why.
{question}
Data:
{data}"""


def summarize(state: State) -> State:
    data = {"issues": state.get("issues", []), "error": state.get("error")}
    question = f"The user asked: {state['question']}" if state.get("question") else ""
    prompt = SUMMARY_PROMPT.format(repo=state["repo"], question=question, data=json.dumps(data))
    return {"summary": model.invoke(prompt).content}


def notify(state: State) -> State:
    """Post the summary to Slack through the gateway's slack connection."""
    response = httpx.post(
        f"{os.environ['TRASE_EGRESS_GATEWAY_URL'].rstrip('/')}/slack/api/chat.postMessage",
        headers={"Authorization": f"Bearer {_credential()}"},
        json={"channel": state["channel"], "text": state["summary"], "unfurl_links": False},
        timeout=30,
    )
    body = response.json() if response.content else {}
    if not body.get("ok"):
        log.warning("Slack refused the post (HTTP %s): %s", response.status_code, body)
        return {"slack": {"ok": False, "error": body.get("error") or f"HTTP {response.status_code}"}}
    log.info("posted to %s (ts=%s)", body.get("channel"), body.get("ts"))
    return {"slack": {"ok": True, "channel": state["channel"], "ts": body.get("ts")}}


def build() -> Any:
    graph = StateGraph(State)
    graph.add_node("fetch_issues", fetch_issues)
    graph.add_node("summarize", summarize)
    graph.add_node("notify", notify)
    graph.add_edge(START, "fetch_issues")
    graph.add_edge("fetch_issues", "summarize")
    graph.add_edge("summarize", "notify")
    graph.add_edge("notify", END)
    return graph.compile()


# Built at import: the topology inspector imports agent.main and reads this graph.
issues_graph = build()
