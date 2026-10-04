// Local DOM fixtures only; no browser network or cloud requests.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
function node(tag){return {tag,children:[],value:'',disabled:false,textContent:'',append(...a){this.children.push(...a)},replaceChildren(...a){this.children=a},focus(){},addEventListener(n,h){this['on'+n]=h},querySelector(){return this.option||(this.option={})},set innerHTML(v){throw Error('plain text required')}}}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,node(id));return elements.get(id)};
get('vision-mode').value='legacy';get('runtime-engine').value='native';get('mode').value='scripted';get('agent-mode').value='adaptive';
let sent,response;
const ctx=vm.createContext({document:{getElementById:get,createElement:node,querySelectorAll:()=>[]},fetch:async(url,opts)=>{sent=JSON.parse(opts.body);return {ok:true,json:async()=>response}},setInterval:()=>1,clearInterval(){}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/conversation.js'),'utf8'),ctx);
const text=n=>[n.textContent,...n.children.map(text)].join('\n');
const result={execution_mode:'scripted',execution_status:'completed',answer:'fixture',latency_ms:1,request_attempts:0,model_calls:4,tool_calls:1,evidence:[],citations:[],image_refs:{},trace:[]};
(async()=>{
 vm.runInContext("sessionId='fixture'",ctx);get('message').value='曝光';response={...result,runtime_engine:'native',graph_trace:[]};
 await get('chat-form').onsubmit({preventDefault(){}});assert.equal(sent.runtime_engine,'native');assert.ok(text(get('messages')).includes('运行器：native'));assert.equal(get('runtime-engine').disabled,true);
 get('new-session').onclick();assert.equal(get('runtime-engine').disabled,false);
 get('runtime-engine').value='langgraph';get('runtime-engine').onchange();assert.equal(get('agent-mode').value,'adaptive');assert.equal(get('agent-mode').option.disabled,true);
 vm.runInContext("sessionId='graph-fixture'",ctx);get('message').value='曝光';response={...result,runtime_engine:'langgraph',graph_trace:[{node:'validate',next:'repair',reason:'missing <script>',status:'validation_error',latency_ms:1},{node:'repair',next:'choose_action',reason:'修正证据',latency_ms:1}]};
 await get('chat-form').onsubmit({preventDefault(){}});assert.equal(sent.runtime_engine,'langgraph');const rendered=text(get('messages'));assert.ok(rendered.includes('图节点执行记录（2 次；模型 4 次）'));assert.ok(rendered.includes('完成校验 → 纠错'));assert.ok(rendered.includes('missing <script>'));
 console.log('PASS: 4 LangGraph UI scenarios; native default, engine pin/reset, adaptive guard, actual node trace and plain text');
})().catch(e=>{console.error(e);process.exitCode=1});
