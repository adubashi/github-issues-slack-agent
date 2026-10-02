# github-issues-slack-agent

A Trase OS third-party agent built with LangGraph. It reads a repository's open issues from the
**GitHub MCP server**, has the model write a short update, and **posts it to Slack**: a read
and a governed write in one run.

```
fetch_issues (GitHub MCP: list_issues) -> summarize (OpenAI) -> notify (Slack chat.postMessage)
```

## Connections (all governed; the agent holds none of their credentials)

| Connection | Reached at | Used for |
| --- | --- | --- |
| `github-mcp-trase` (default; you create it) | `${TRASE_EGRESS_GATEWAY_URL}/github-mcp-trase/mcp/` | `list_issues` over MCP, with a token that can read TraseSystems repos |
| `github-mcp` (platform's own; pass `"github_connection": "github-mcp"`) | `TRASE_GITHUB_MCP_URL` (platform-injected) | same, with the platform's token |
| `openai` | `TRASE_OPENAI_BASE_URL` | the summary |
| `slack` | `${TRASE_EGRESS_GATEWAY_URL}/slack/api/` | `chat.postMessage` |

Each call sends `Authorization: Bearer ${TRASE_RUN_CREDENTIAL}`. The gateway checks policy and
replaces it upstream with the real GitHub token, OpenAI key or Slack bot token. All three
connections must be granted to this agent, with policy activated.

GitHub's MCP server only shows the tools, and only returns the repositories, that the brokered
GitHub token allows. If the token can't see the repository, the agent reports that in Slack
instead of failing.

## Input (all optional)

```json
{"repo": "TraseSystems/trase-os-sdk", "slack_channel": "#ansh-test",
 "user_message": "anything blocking the next release?"}
```

Defaults: `TraseSystems/trase-os-sdk`, `#ansh-test`, connection `github-mcp-trase`. The Slack
bot must be a member of the channel.

### Creating `github-mcp-trase`

Kind **MCP**, host `https://api.githubcopilot.com`, credential: a bearer token, a GitHub
fine-grained token (resource owner TraseSystems, only `trase-os-sdk`, Issues: read-only).
Publish it, grant it to this agent, and activate policy.

## Output

`{"repo", "open_issues", "error", "slack": {"ok", "channel", "ts"}, "message"}`
