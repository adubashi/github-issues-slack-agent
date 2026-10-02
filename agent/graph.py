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
import re
from datetime import UTC, datetime, timedelta
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
    github_connection: str  # connection handle fronting the GitHub MCP server
    hours: int  # report window: issues opened, updated or closed in the last N hours
    since: str  # ISO 8601 start of the window
    question: str
    issues: list[dict[str, Any]]
    counts: dict[str, int]
    mcp_tools: list[str]
    error: str
    summary: str
    slack: dict[str, Any]


def _activity(issue: dict[str, Any], since: str) -> str:
    """What happened to an issue inside the window: new, closed, or updated."""
    if (issue.get("created_at") or "") >= since:
        return "new"
    if str(issue.get("state", "")).lower() == "closed":
        return "closed"
    return "updated"


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
                "state": issue.get("state"),
                "created_at": issue.get("created_at") or issue.get("createdAt"),
                "updated_at": issue.get("updated_at") or issue.get("updatedAt"),
                "comments": issue.get("comments"),
            }
        )
    return rows


def _mcp_url(handle: str) -> str:
    """Gateway URL of the GitHub MCP server behind connection `handle`.

    `github-mcp` is the platform's own connection, injected as TRASE_GITHUB_MCP_URL. Any other
    handle is a connection you created with upstream https://api.githubcopilot.com, reached at
    the gateway under its handle plus the server's /mcp/ path, e.g. a connection whose token
    can read a repository the platform's token can't.
    """
    if handle == "github-mcp" and os.environ.get("TRASE_GITHUB_MCP_URL"):
        return os.environ["TRASE_GITHUB_MCP_URL"]
    return f"{os.environ['TRASE_EGRESS_GATEWAY_URL'].rstrip('/')}/{handle}/mcp/"


async def fetch_issues(state: State) -> State:
    """Ask the GitHub MCP server for issues opened, updated or closed inside the window."""
    owner, name = state["repo"].split("/", 1)
    since = (datetime.now(UTC) - timedelta(hours=state.get("hours") or 24)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    handle = state.get("github_connection") or "github-mcp"
    log.info("using GitHub MCP connection %r", handle)
    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "streamable_http",
                "url": _mcp_url(handle),
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
        # No state filter: open and closed both count as activity. `since` filters on the
        # issue's last update, so it catches new, updated and just-closed issues.
        raw = await tools["list_issues"].ainvoke(
            {"owner": owner, "repo": name, "since": since, "perPage": 100}
        )
    except Exception as exc:  # noqa: BLE001 - reported in the summary, not raised
        log.warning("list_issues on %s failed: %s", state["repo"], exc)
        return {"mcp_tools": sorted(tools), "error": f"list_issues failed: {exc}"[:500]}
    try:
        issues = _issue_rows(raw, state["repo"])
    except ValueError:
        # GitHub's MCP server reports failures (e.g. a repo the token can't see) as plain text.
        text = raw if isinstance(raw, str) else json.dumps(raw)
        log.warning("list_issues on %s returned an error: %s", state["repo"], text[:300])
        return {"mcp_tools": sorted(tools), "since": since, "error": f"GitHub MCP said: {text[:400]}"}
    for issue in issues:
        issue["activity"] = _activity(issue, since)
    # Newest activity first, new issues ahead of updates.
    order = {"new": 0, "closed": 1, "updated": 2}
    issues.sort(key=lambda i: (order[i["activity"]], -(i.get("number") or 0)))
    counts = {a: sum(i["activity"] == a for i in issues) for a in ("new", "updated", "closed")}
    log.info("%s since %s: %s", state["repo"], since, counts)
    return {"mcp_tools": sorted(tools), "since": since, "issues": issues, "counts": counts}


SUMMARY_PROMPT = """You write short Slack updates for an engineering team.

Write a Slack message in Slack mrkdwn (NOT a code block, no ``` fences), under 150 words,
about GitHub issue activity in {repo} in the last {hours} hours.

First line, exactly: *{repo} · last {hours}h:* {new} new · {updated} updated · {closed} closed
Then one line per issue, at most 10, as "• <url|#number title>" followed by the activity
in italics (_new_, _updated_ or _closed_) and labels in brackets if any. If there are more
than 10, end with "…and N more". Use the counts above as given; do not recount.
If there was no activity, say so in one line. If there is an error instead of issues, say
plainly that the issues couldn't be fetched and why.
{question}
Data:
{data}"""


def _strip_fences(text: str) -> str:
    """Models sometimes wrap Slack text in a ``` block, which Slack shows literally."""
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())


def summarize(state: State) -> State:
    counts = state.get("counts") or {"new": 0, "updated": 0, "closed": 0}
    data = {"issues": state.get("issues", []), "error": state.get("error")}
    question = f"The user asked: {state['question']}" if state.get("question") else ""
    prompt = SUMMARY_PROMPT.format(
        repo=state["repo"],
        hours=state.get("hours") or 24,
        question=question,
        data=json.dumps(data),
        **counts,
    )
    return {"summary": _strip_fences(model.invoke(prompt).content)}


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
