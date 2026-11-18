import fs from "node:fs";
import { hasOutput } from "./pi-extensions/performance/performance.mjs";

// Explicitly loaded only by the isolated Pi benchmark.
export default function install(pi) {
  let start, first, timings, usage, request = 0;
  pi.on("before_provider_request", (event) => {
    request++;
    if (request === 1 && process.env.PI_PREFIX_GUIDANCE) {
      const guidance = fs.readFileSync(process.env.PI_PREFIX_GUIDANCE, "utf8").trim();
      const loaded = event.payload.messages.some(m =>
        m.role === "system" && typeof m.content === "string" && m.content.includes(guidance));
      fs.writeFileSync(process.env.PI_PREFIX_GUIDANCE_AUDIT, JSON.stringify({
        loaded, tools: (event.payload.tools ?? []).map(t => t.function?.name)
      }));
    }
    if (request > 8) throw new Error("Unexpected tool loop");
    start = performance.now();
    first = null;
    timings = usage = null;
    const progress = process.env.PI_PREFIX_LONG === "1" ? { return_progress: true } : {};
    return { ...event.payload, seed: 42, temperature: 0, timings_per_token: true, ...progress };
  });
  pi.on("provider_stream_event", (event) => {
    const chunk = event.data;
    if (hasOutput(chunk) && first === null) first = performance.now();
    if (chunk.timings) timings = chunk.timings;
    if (chunk.usage) usage = chunk.usage;
  });
  pi.on("message_end", ({message}) => {
    if (message.role !== "assistant") return;
    const row = { request, first_ms: first === null ? null : first - start,
      total_ms: performance.now() - start, timings, usage, stop: message.stopReason,
      calls: message.content.filter(x => x.type === "toolCall").map(x => x.name) };
    fs.appendFileSync(process.env.PI_PREFIX_EVIDENCE, JSON.stringify(row) + "\n");
  });
}
