const finite = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
const text = (v) => typeof v === "string" && v.length > 0;

export function hasOutput(chunk) {
  if (chunk?.prompt_progress || !Array.isArray(chunk?.choices)) return false;
  return chunk.choices.some((choice) => {
    const delta = choice?.delta;
    return delta && (
    text(delta.content) || text(delta.reasoning_content) || text(delta.reasoning) ||
    (Array.isArray(delta.tool_calls) && delta.tool_calls.some((call) => text(call?.function?.name) || text(call?.function?.arguments)))
    );
  });
}

export class Meter {
  constructor(start, approximate = false) {
    this.start = start;
    this.approximate = approximate;
    this.first = null;
    this.end = null;
    this.tokens = null;
    this.tps = null;
    this.phase = "waiting";
  }
  output(now) {
    if (this.first === null) this.first = now;
    this.phase = "streaming";
  }
  observe(chunk, now) {
    if (hasOutput(chunk)) this.output(now);
    const timings = chunk?.timings;
    if (finite(timings?.predicted_n)) {
      this.tokens = timings.predicted_n;
      if (timings.predicted_n > 1 && finite(timings.predicted_per_second) && timings.predicted_per_second > 0) {
        this.tps = timings.predicted_per_second;
      }
    } else if (finite(chunk?.usage?.completion_tokens)) {
      this.tokens = chunk.usage.completion_tokens;
    }
  }
  finish(message, now) {
    this.end = now;
    if (this.tokens === null && finite(message?.usage?.output)) this.tokens = message.usage.output;
    this.phase = message?.stopReason === "aborted" ? "aborted" : message?.stopReason === "error" ? "error" : "done";
  }
  render(now) {
    if (this.phase === "waiting" && this.first === null && this.tps === null) return "Perf | TTFT waiting | TPS --";
    const ttft = this.first === null ? "--" : `${this.approximate ? "~" : ""}${((this.first-this.start)/1000).toFixed(2)}s`;
    const rate = this.tps === null ? "--" : `${this.tps.toFixed(1)} tok/s srv`;
    const count = this.tokens === null ? "" : ` | ${this.tokens} out`;
    const elapsed = Math.max(0, (this.end ?? now)-this.start)/1000;
    return `Perf | TTFT ${ttft} | ${rate}${count} | ${elapsed.toFixed(1)}s ${this.phase}`;
  }
}

function localModel(ctx) {
  if (ctx.model?.provider !== "llama_cpp" || ctx.model?.api !== "openai-completions") return false;
  try {
    return ["localhost", "127.0.0.1", "[::1]"].includes(new URL(ctx.model.baseUrl).hostname);
  } catch { return false; }
}

export default function install(pi, { now = () => performance.now() } = {}) {
  const key = "local-llm-performance";
  let meter = null;
  let enabled = true;
  let lastRender = -Infinity;
  const draw = (ctx, force = false) => {
    if (!ctx.hasUI) return;
    const time = now();
    if (!force && time-lastRender < 250) return;
    lastRender = time;
    ctx.ui.setStatus(key, enabled && meter ? meter.render(time) : undefined);
  };
  const reset = (_event, ctx) => { meter = null; draw(ctx, true); };
  pi.on("session_start", reset);
  pi.on("session_switch", reset);
  pi.on("model_select", reset);
  pi.on("session_shutdown", reset);

  pi.on("before_provider_request", (event, ctx) => {
    if (!enabled || !localModel(ctx)) { meter = null; draw(ctx, true); return; }
    meter = new Meter(now());
    draw(ctx, true);
    if (event.payload && typeof event.payload === "object" && !Array.isArray(event.payload)) {
      return { ...event.payload, timings_per_token: true, return_progress: true };
    }
  });
  pi.on("message_start", (event, ctx) => {
    if (!enabled || !localModel(ctx) || event.message?.role !== "assistant") return;
    if (!meter || meter.end !== null) meter = new Meter(now(), true);
    draw(ctx, true);
  });
  pi.on("provider_stream_event", (event, ctx) => {
    if (!enabled || !meter || meter.end !== null || !localModel(ctx) ||
        event.provider !== "llama_cpp" || event.model !== ctx.model.id) return;
    meter.observe(event.data, now());
    const chunk = event.data;
    const terminal = Array.isArray(chunk?.choices) && (
      (chunk.choices.length === 0 && finite(chunk.usage?.completion_tokens)) ||
      (chunk.timings && chunk.choices.some((choice) => text(choice?.finish_reason)))
    );
    if (terminal) meter.finish({usage:{output:chunk.usage?.completion_tokens},stopReason:"stop"},now());
    draw(ctx, Boolean(terminal));
  });
  pi.on("message_update", (event, ctx) => {
    if (!enabled || !meter || meter.end !== null || !localModel(ctx)) return;
    const update = event.assistantMessageEvent;
    if (["text_delta", "thinking_delta", "toolcall_delta"].includes(update?.type) && text(update.delta)) {
      meter.output(now());
    }
    draw(ctx);
  });
  pi.on("message_end", (event, ctx) => {
    if (!meter || meter.end !== null || event.message?.role !== "assistant" || !localModel(ctx)) return;
    meter.finish(event.message, now());
    draw(ctx, true);
  });
  pi.registerCommand("perf", {
    description: "Show local llama.cpp performance; /perf on, off, or reset",
    handler: async (args, ctx) => {
      const action = args.trim();
      if (action === "off") enabled = false;
      else if (action === "on") enabled = true;
      else if (action === "reset") meter = null;
      else if (action) { ctx.ui.notify("Usage: /perf [on|off|reset]", "info"); return; }
      if (!action && meter) ctx.ui.notify(meter.render(now()), "info");
      else if (!action) ctx.ui.notify("Performance display is ready for the next local model response.", "info");
      draw(ctx, true);
    },
  });
}
