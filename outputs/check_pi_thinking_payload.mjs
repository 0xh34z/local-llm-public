// Inspect the installed provider's request construction; abort before any network call.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const agent = path.join(process.env.USERPROFILE, '.pi', 'agent');
const version = fs.readFileSync(path.join(agent,'install','current-version'),'utf8').trim();
const modulePath = path.join(agent,'install','releases',version,'node_modules','@earendil-works','pi-ai','dist','api','openai-completions.js');
const { streamSimple } = await import(pathToFileURL(modulePath));
const config = JSON.parse(fs.readFileSync(path.join(root,'outputs','pi-models-recommended.json'),'utf8'));
const provider = config.providers.llama_cpp;
const model = { ...provider.models[0], provider:'llama_cpp', api:provider.api, baseUrl:provider.baseUrl,
  name:provider.models[0].id, input:['text'], cost:{input:0,output:0,cacheRead:0,cacheWrite:0} };
const context = { messages:[{role:'system',content:'You are a coding assistant.',timestamp:0},
  {role:'user',content:'Say hello.',timestamp:0}] };
let networkCalls = 0;
const results=[];
for (const level of ['off','low','medium']) {
  let payload;
  const stream = streamSimple(model,context,{apiKey:'dummy',reasoning:level,
    maxTokens:16384,fetch:async()=>{networkCalls++;throw new Error('Network disabled');},
    onPayload:body=>{payload=body;throw new Error('Intentional payload inspection stop');}});
  for await (const event of stream) { /* Expected error event after the inspection stop. */ }
  if (!payload) throw new Error('Provider did not construct payload');
  const enabled=payload.chat_template_kwargs?.enable_thinking;
  const budget=payload.thinking_budget_tokens;
  if (enabled !== (level!=='off')) throw new Error('Wrong thinking switch');
  if (payload.messages[0]?.role !== 'system') throw new Error('Unexpected system role');
  if (level==='low' && budget!==2048) throw new Error('Wrong low budget');
  if (level==='medium' && budget!==8192) throw new Error('Wrong medium budget');
  results.push({level,chat_template_kwargs:payload.chat_template_kwargs,
    thinking_budget_tokens:budget??null,max_tokens:payload.max_tokens,first_role:payload.messages[0]?.role});
}
if (networkCalls!==0) throw new Error('Unexpected network call');
const result={pi_version:version,network_calls:networkCalls,results,
  compaction_threshold:163840-61440,scope:'Client payload construction only; no live inference test'};
fs.writeFileSync(path.join(root,'outputs','pi-thinking-payload-check.json'),JSON.stringify(result,null,2));
process.stdout.write(JSON.stringify(result,null,2)+'\n');
