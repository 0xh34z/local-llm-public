# Pi web search

Tavily-backed `web_search` and `web_read` for Pi 1.4.0. Uses Pi's native extension/tool API with built-in Node fetch; no Pi fork, MCP server, npm install or Docker service is required. Repository source lives here; the installed runtime files live in `~/.pi/agent/extensions/local-llm-websearch/`.

## Setup

Run locally in PowerShell (do not paste your key into a chat):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kasper\Documents\GitHub\local-llm\outputs\pi-extensions\websearch\Setup-Tavily.ps1"
```

The prompt hides typing and writes the key to `C:\Users\kasper\.pi\agent\websearch.json`, a plaintext credential file outside this repository. Keep it private and out of Git. Alternatively, supply `TAVILY_API_KEY` in Pi's process environment; it takes precedence. The extension reads configuration per call, so changing the file does not require restarting Pi. Never add the key to a model prompt or tool arguments.

After installation, use `/reload` or restart Pi, then `/websearch-status`. That command checks local configuration without network access, spending credits, or displaying the key. It does not validate the key against Tavily. For a live end-to-end check, ask Pi: **Use web_search to find the official Python asyncio documentation, then use web_read on the relevant source and cite it.** This makes paid API calls under your Tavily plan.

If an existing session has a restricted tool list, enable `web_search` and `web_read` through its tool settings. Do not also load the repository copy with `--extension` or a settings entry, as that would duplicate the installed copy.

## Behavior

- `web_search`: required query; optional max_results (1–8, default 5), depth (basic/advanced), domains, topic (general/news), time_range (day/week/month/year). Basic is the default; automatic depth escalation, answer generation, images and full raw page content are disabled. Excerpts are limited to 1500 characters per source. URLs, titles and provider-estimated publication/update dates accompany each excerpt. News defaults to a week when no time_range is supplied and enables strict provider date filtering, which excludes undated sources. Date estimates can reflect updates rather than original publication; inspect sources before reporting news.
- At each user turn, `before_agent_start` appends the current calendar date and OS timezone to the existing system prompt, with news verification/citation guidance. Search results also show the retrieval date and actual filters. No clock time is injected, avoiding a changing prompt prefix on every turn within the same day. This guides the model but cannot guarantee that it follows the instructions or cites accurately.
- `web_read`: one HTTP(S) URL, optional focus query. Requests basic Tavily extraction as Markdown. Returns at most 16000 characters with an explicit truncation marker; a focus query asks Tavily for relevant chunks. It is not a browser session and cannot use local authenticated cookies.
- Requests go only to `https://api.tavily.com/search` or `/extract`; key in the Authorization header. Queries and requested page URLs are shared with Tavily. Retrieved snippets/page text enter Pi's conversation and may be saved by Pi's normal session persistence.
- 45-second timeout, cancellation support, 2 MB response cap, no automatic retries or hidden fallback. HTTP error bodies and request headers are not returned to the model. No background requests or startup API calls. No new search cache or transcript log is written by this extension.
- Returned web content is labeled untrusted. Tool instructions tell the model to use it as evidence and cite URLs; this reduces ambiguity but does not guarantee resistance to prompt injection or factual accuracy.
- Search and extraction consume Tavily credits; advanced search costs more. Check your dashboard for current plan limits. There is no local spending cap; each model tool call can make one API request.

## Backend choice

Start with Tavily because the user already has a key and it provides both search excerpts and readable extraction through documented APIs. This is an operational choice, not a measured ranking of relevance or latency. SearXNG remains an option if avoiding a paid search API or controlling the service matters more. Its JSON search format must be enabled; an instance can otherwise return 403. It would require its own deployed endpoint, maintenance and a separate page-reading approach. No SearXNG instance or silent multi-provider fallback is deployed here.

Primary references checked on 2026-10-29: [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search), [Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract), [SearXNG Search API](https://docs.searxng.org/dev/search_api.html). Pi declarations and tool examples were checked in the installed 1.4.0 package.

## Verification

```powershell
node --test .\outputs\pi-extensions\websearch\websearch.test.mjs
```

Six tests passed using mocked HTTP responses and the actual installed Pi extension loader: request defaults/filters, schema bounds, source formatting/truncation, extraction focus/URL validation, HTTP errors/no retries, invalid JSON shape, response cap, cancellation, credential redaction, both tool registrations, status command, date hook preserving the original prompt, timezone date rollover, news filters and date formatting. No real key or paid Tavily request was used by these tests.

The user's first live transcript confirms successful search execution, but the model queried `latest tech news today 2025`, omitted source citations and did not read sources before its news summary. The date hook, news topic/filtering and stronger citation guidance were added in response. The pasted transcript truncates most search results, so it does not establish which individual headlines were supported; their factual accuracy was not audited. Live model compliance with the revised guidance remains unverified.
