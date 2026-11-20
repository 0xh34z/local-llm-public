import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { pathToFileURL } from "node:url";
import { Type } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
import { createTools, request, localDate, dateContext } from "./websearch.mjs";

const agent = path.join(process.env.USERPROFILE || os.homedir(), ".pi/agent");
let version = null;
try {
  version = fs.readFileSync(path.join(agent, "install/current-version"), "utf8").trim();
} catch {
  // Pi is optional for the unit tests; only the loader integration test needs it.
}
const piLoader = version
  ? path.join(agent, "install/releases", version, "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js")
  : null;
const piAvailable = piLoader && fs.existsSync(piLoader);
const json = data => new Response(JSON.stringify(data));
const key = async () => "tvly-test-not-real";

test("search sends bounded basic request and formats sources without full response or secret", async () => {
  let sent;
  const tools = createTools(Type, { readKey: key, fetch: async (url, options) => {
    sent = { url, ...options, body: JSON.parse(options.body) };
    return json({ results: [{ title: "Docs", url: "https://example.org/docs", content: "a".repeat(5000) }, { url: "javascript:bad" }], private_metadata: "not echoed" });
  } });
  const result = await tools[0].execute("1", { query: "test", domains: ["example.org"], time_range: "month" });
  assert.equal(sent.url, "https://api.tavily.com/search");
  assert.equal(sent.headers.Authorization, "Bearer tvly-test-not-real");
  assert.equal(sent.body.search_depth, "basic"); assert.equal(sent.body.max_results, 5);
  assert.equal(sent.body.auto_parameters, false); assert.equal(sent.body.include_raw_content, false);
  assert.deepEqual(sent.body.include_domains, ["example.org"]);
  assert.equal(sent.body.time_range, "month");
  assert.match(result.content[0].text, /https:\/\/example.org\/docs/);
  assert.match(result.content[0].text, /truncated/);
  assert(result.content[0].text.length < 2400);
  assert(!JSON.stringify(result).includes("tvly-")); assert.equal(result.details.results, 1);
  assert(!Value.Check(tools[0].parameters, { query: "test", max_results: 999 }));
});

test("extraction is bounded and focus forwarded", async () => {
  let body;
  const tools = createTools(Type, { readKey: key, fetch: async (_url, options) => {
    body = JSON.parse(options.body);
    return json({ results: [{ url: "https://example.org/", raw_content: "x".repeat(20000) }] });
  } });
  const result = await tools[1].execute("1", { url: "https://example.org", query: "install" });
  assert.equal(body.query, "install"); assert.equal(body.chunks_per_source, 3);
  assert.equal(result.details.truncated, true); assert(result.content[0].text.length < 16500);
  await assert.rejects(tools[1].execute("1", { url: "file:///etc/passwd" }), /HTTP/);
  await assert.rejects(tools[1].execute("1", { url: "https://user:password@example.org" }), /credentials/);
});

test("HTTP errors do not echo provider bodies or retry", async () => {
  let calls = 0;
  await assert.rejects(request("search", {}, undefined, { readKey: key, fetch: async () => {
    calls++; return new Response("secret response", { status: 429 });
  } }), /Tavily HTTP 429/);
  assert.equal(calls, 1);
  await assert.rejects(request("search", {}, undefined, { readKey: key, fetch: async () => json({ bad: true }) }), /results array/);
  await assert.rejects(request("search", {}, undefined, { readKey: key, fetch: async () => { throw new Error("tvly-test-not-real"); } }), error => !error.message.includes("tvly-"));
});

test("size limit, cancellation and credential redaction", async () => {
  await assert.rejects(request("search", {}, undefined, { readKey: key, fetch: async () => new Response("x".repeat(2000001)) }), /2 MB/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(request("search", {}, controller.signal, { readKey: key, fetch: async (_url, options) => { options.signal.throwIfAborted(); } }), /cancelled/);
  const data = await request("search", {}, undefined, { readKey: key, fetch: async () => json({ results: [{ content: "tvly-test-not-real" }] }) });
  assert.equal(data.results[0].content, "[REDACTED]");
});

test("actual installed Pi loader registers both tools and status command",
  { skip: piAvailable ? false : "Pi agent is not installed" }, async () => {
  const { loadExtensions } = await import(pathToFileURL(piLoader));
  const entry = new URL("./index.js", import.meta.url);
  const loaded = await loadExtensions([decodeURIComponent(entry.pathname).replace(/^\/(\w:)/, "$1")], process.cwd());
  assert.deepEqual(loaded.errors, []); assert.equal(loaded.extensions.length, 1);
  const extension = loaded.extensions[0];
  assert.deepEqual([...extension.tools.keys()], ["web_search", "web_read"]);
  assert(extension.commands.has("websearch-status"));
  const dateHook = extension.handlers.get("before_agent_start")[0];
  const patched = await dateHook({ systemPrompt: "Original instructions" });
  assert(patched.systemPrompt.startsWith("Original instructions\n\n"));
  assert.match(patched.systemPrompt, /Current local date: \d{4}-\d{2}-\d{2}/);
});

test("date follows local calendar and news requests retain dates with a bounded default range", async () => {
  assert.equal(localDate(new Date("2026-10-29T23:30:00Z"), "Europe/Amsterdam"), "2026-10-30");
  assert.match(dateContext(new Date("2026-10-29T12:00:00Z"), "Europe/Amsterdam"), /2026-10-29 \(Europe\/Amsterdam\)/);
  let body;
  const tools = createTools(Type, { readKey: key, fetch: async (_url, options) => {
    body = JSON.parse(options.body);
    return json({ results: [{ title: "News", url: "https://example.org/", content: "Evidence", published_date: "2026-10-28" }] });
  } });
  const response = await tools[0].execute("1", { query: "tech news", topic: "news" });
  assert.equal(body.topic, "news"); assert.equal(body.time_range, "week");
  assert.equal(body.filter_by_published_date, true); assert.equal(body.include_published_date, true);
  assert.match(response.content[0].text, /2026-10-28/);
  assert.match(response.content[0].text, /provider estimate/);
  await tools[0].execute("2", { query: "today's news", topic: "news", time_range: "day" });
  assert.equal(body.time_range, "day");
});
