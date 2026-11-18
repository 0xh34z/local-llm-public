import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { pathToFileURL, fileURLToPath } from "node:url";
import install from "./performance.mjs";

const directory=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(directory,"../../..");
const agent=path.join(process.env.USERPROFILE,".pi","agent");
const version=fs.readFileSync(path.join(agent,"install/current-version"),"utf8").trim();
const packages=path.join(agent,"install/releases",version,"node_modules/@earendil-works");
const {loadExtensions}=await import(pathToFileURL(path.join(packages,"pi-coding-agent/dist/core/extensions/loader.js")));
const loaded=await loadExtensions([path.join(directory,"index.js")],root);
assert.deepEqual(loaded.errors,[]);
assert.equal(loaded.extensions.length,1);
assert(loaded.extensions[0].handlers.has("provider_stream_event"));

const {streamSimple}=await import(pathToFileURL(path.join(packages,"pi-ai/dist/api/openai-completions.js")));
let clock=0;
const handlers=new Map(); const displays=[];
install({on:(name,handler)=>handlers.set(name,handler),registerCommand:()=>{}},{now:()=>clock});
const model={provider:"llama_cpp",api:"openai-completions",id:"fixture-model",name:"fixture-model",
  baseUrl:"http://localhost:9931/v1",reasoning:true,input:["text"],contextWindow:163840,maxTokens:16384,
  cost:{input:0,output:0,cacheRead:0,cacheWrite:0},compat:{maxTokensField:"max_tokens",supportsDeveloperRole:false}};
const ctx={hasUI:true,model,ui:{setStatus:(_key,value)=>displays.push(value),notify:()=>{}}};
const frame=(delta,ms,extra={})=>({id:"fixture",object:"chat.completion.chunk",created:0,model:model.id,
  choices:[{index:0,delta,finish_reason:null}],fixture_ms:ms,...extra});
const chunks=[frame({role:"assistant",content:null},100,{prompt_progress:{processed:20},timings:{predicted_n:0}}),
  frame({reasoning_content:"Consider the request."},750),frame({content:"Hello."},1000),
  frame({tool_calls:[{index:0,id:"call_1",type:"function",function:{name:"read",arguments:'{"path":"example.txt"}'}}]},2000),
  frame({},4900,{choices:[{index:0,delta:{},finish_reason:"tool_calls"}]}),
  frame({},5000,{choices:[],usage:{prompt_tokens:40,completion_tokens:80,total_tokens:120,
    completion_tokens_details:{reasoning_tokens:20}},timings:{predicted_n:80,predicted_per_second:17.6}})];
let transportCalls=0;
let requestBody;
const stream=streamSimple(model,{messages:[{role:"user",content:"Say hello then read example.txt.",timestamp:0}]},{
  apiKey:"dummy",reasoning:"low",maxTokens:128,
  onPayload:body=>handlers.get("before_provider_request")({payload:body},ctx),
  onProviderStreamEvent:(data)=>{clock=data.fixture_ms;handlers.get("provider_stream_event")({provider:model.provider,model:model.id,data},ctx);},
  fetch:async(_url,options)=>{
    transportCalls++; requestBody=JSON.parse(options.body);
    const body=chunks.map(chunk=>`data: ${JSON.stringify(chunk)}\n\n`).join("")+"data: [DONE]\n\n";
    return new Response(body,{status:200,headers:{"content-type":"text/event-stream"}});
  },
});
for await (const event of stream) {
  if(event.type==="start") handlers.get("message_start")({message:{role:"assistant"}},ctx);
  else if(event.type==="done") handlers.get("message_end")({message:event.message},ctx);
  else if(event.type==="error") throw new Error(JSON.stringify(event));
  else handlers.get("message_update")({assistantMessageEvent:event},ctx);
}
assert.equal(transportCalls,1);
assert.equal(requestBody.timings_per_token,true); assert.equal(requestBody.return_progress,true);
assert.match(displays.at(-1),/TTFT 0.75s \| 17.6 tok\/s srv \| 80 out \| 5.0s done/);
const result={pi_version:version,extension_loader_errors:loaded.errors,
  registered_events:[...loaded.extensions[0].handlers.keys()],transport:"mocked SSE; no inference server/network used",
  fixture_includes:["empty role/progress","thinking","text","tool call","final server timings"],
  final_status:displays.at(-1),status_updates:displays.length};
fs.writeFileSync(path.join(directory,"integration-result.json"),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
