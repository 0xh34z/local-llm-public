import { Type } from "typebox";
import { createTools, dateContext } from "./websearch.mjs";

export default function (pi) {
  for (const tool of createTools(Type)) pi.registerTool(tool);
  pi.on("before_agent_start", event => ({
    systemPrompt: `${event.systemPrompt}\n\n${dateContext()}`,
  }));
  pi.registerCommand("websearch-status", {
    description: "Check web search configuration without making a paid request",
    handler: async (_args, ctx) => {
      const { readKey } = await import("./websearch.mjs");
      try { await readKey(); ctx.ui.notify("Tavily key configured. web_search and web_read are available; no request made.", "info"); }
      catch (error) { ctx.ui.notify(error.message, "warning"); }
    },
  });
}
