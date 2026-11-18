import test from "node:test";
import assert from "node:assert/strict";
import install, { Meter, hasOutput } from "./performance.mjs";

test("TTFT ignores role/progress frames, counts thinking, and uses server TPS", () => {
  const meter = new Meter(1000);
  meter.observe({ choices:[{delta:{role:"assistant",content:null}}] },1100);
  meter.observe({ prompt_progress:{processed:10},choices:[{delta:{content:""}}],timings:{predicted_n:0} },1200);
  assert.equal(meter.first,null);
  meter.observe({ choices:[{delta:{reasoning_content:"Consider this"}}] },1750);
  meter.observe({ timings:{predicted_n:80,predicted_per_second:17.6} },6000);
  meter.finish({usage:{output:80,reasoning:20},stopReason:"toolUse"},6200);
  assert.equal(meter.first,1750);
  assert.equal(meter.tokens,80);
  assert.match(meter.render(10000),/TTFT 0.75s \| 17.6 tok\/s srv \| 80 out \| 5.2s done/);
});

test("tool output counts but tool ID metadata alone does not", () => {
  assert.equal(hasOutput({choices:{}}),false);
  assert.equal(hasOutput({choices:[null,{delta:{tool_calls:{}}}]}),false);
  assert.equal(hasOutput({choices:[{delta:{tool_calls:[{id:"call_1"}]}}]}),false);
  assert.equal(hasOutput({choices:[{delta:{tool_calls:[{function:{arguments:'{"path"'}}]}}]}),true);
});

test("no invented TPS when server timings are missing or invalid", () => {
  const meter=new Meter(0,true);
  meter.observe({choices:[{delta:{content:"hello"}}],timings:{predicted_n:1,predicted_per_second:Infinity}},500);
  meter.observe({usage:{completion_tokens:10}},1000);
  meter.finish({usage:{output:10},stopReason:"aborted"},1100);
  assert.equal(meter.tps,null);
  assert.equal(meter.tokens,10);
  assert.match(meter.render(1200),/TTFT ~0.50s \| -- \| 10 out \| 1.1s aborted/);
});

test("request reset excludes tool execution; extension preserves payload and other providers", async () => {
  const handlers=new Map(); const statuses=[]; let clock=0;
  const pi={on:(event,handler)=>handlers.set(event,handler),registerCommand:()=>{}};
  const ctx={hasUI:true,model:{provider:"llama_cpp",api:"openai-completions",id:"qwen",baseUrl:"http://localhost:9931/v1"},
    ui:{setStatus:(_key,value)=>statuses.push(value),notify:()=>{}}};
  install(pi,{now:()=>clock});
  const payload={model:"qwen",temperature:0.6,messages:[]};
  const request=handlers.get("before_provider_request")({payload},ctx);
  assert.equal(request.timings_per_token,true); assert.equal(request.return_progress,true);
  assert.equal(request.messages,payload.messages); assert.equal(payload.timings_per_token,undefined);
  clock=750;
  handlers.get("message_update")({assistantMessageEvent:{type:"thinking_delta",delta:"think"}},ctx);
  clock=2000;
  handlers.get("provider_stream_event")({provider:"llama_cpp",model:"qwen",data:{timings:{predicted_n:25,predicted_per_second:19}}},ctx);
  handlers.get("message_end")({message:{role:"assistant",usage:{output:25},stopReason:"toolUse"}},ctx);
  assert.match(statuses.at(-1),/TTFT 0.75s/);
  clock=10000;
  handlers.get("before_provider_request")({payload},ctx);
  clock=10500;
  handlers.get("message_update")({assistantMessageEvent:{type:"toolcall_delta",delta:'{"name"'}},ctx);
  assert.match(statuses.at(-1),/TTFT 0.50s/);
  const other={...ctx,model:{...ctx.model,provider:"other"}};
  assert.equal(handlers.get("before_provider_request")({payload},other),undefined);
  assert.equal(statuses.at(-1),undefined);
});

test("final raw usage completes requests even without assistant lifecycle events", () => {
  const handlers=new Map(); const statuses=[]; let clock=0;
  const ctx={hasUI:true,model:{provider:"llama_cpp",api:"openai-completions",id:"qwen",baseUrl:"http://127.0.0.1:9931/v1"},
    ui:{setStatus:(_key,value)=>statuses.push(value)}};
  install({on:(event,handler)=>handlers.set(event,handler),registerCommand:()=>{}},{now:()=>clock});
  handlers.get("before_provider_request")({payload:{model:"qwen"}},ctx);
  clock=500;
  handlers.get("provider_stream_event")({provider:"llama_cpp",model:"qwen",data:{choices:[{delta:{content:"hello"}}]}},ctx);
  clock=1500;
  handlers.get("provider_stream_event")({provider:"llama_cpp",model:"qwen",data:{choices:[],
    usage:{completion_tokens:20},timings:{predicted_n:20,predicted_per_second:19}}},ctx);
  assert.match(statuses.at(-1),/TTFT 0.50s \| 19.0 tok\/s srv \| 20 out \| 1.5s done/);
});
