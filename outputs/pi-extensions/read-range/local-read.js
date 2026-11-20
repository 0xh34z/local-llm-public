// Pi applies tool_call edits before invoking its built-in reader.
export default function localRead(pi) {
  pi.on("tool_call", (event, ctx) => {
    if (ctx.model?.provider === "llama_cpp" && !event.parentToolCallId &&
        event.toolName === "read" && event.input.limit === undefined) {
      event.input.limit = 40;
    }
  });
  pi.on("tool_result", (event, ctx) => {
    if (ctx.model?.provider !== "llama_cpp" || event.toolName !== "read" || event.parentToolCallId ||
        event.isError || event.content.length !== 1 || event.content[0].type !== "text") return;
    const text = event.content[0].text;
    if (!/\n\n\[\d+ more lines in file\. Use offset=\d+ to continue\.\]$/.test(text)) return;
    const tools = pi.getActiveTools();
    const search = tools.includes("grep") ? "grep" : tools.find(t => t === "powershell" || t === "bash");
    if (!search) return;
    const hint = search === "grep"
      ? `For a lookup, call grep on ${JSON.stringify(event.input.path)} with a keyword from the question and context 3 to find the relevant lines.`
      : `For a lookup, use ${search} to search this file for a keyword from the question and show nearby lines.`;
    const result = text + "\n" + hint;
    return {content: [{type: "text", text: result}], structuredContent: result};
  });
}
