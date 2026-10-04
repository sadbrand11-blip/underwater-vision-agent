// Additional presentation contracts; fixtures only, no network.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const repo=process.argv[2]||path.resolve(__dirname,'..');
function node(tag){return {tag,children:[],textContent:'',value:'',append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},focus(){},addEventListener(name,handler){this['on'+name]=handler},querySelector(){return {disabled:false}},set innerHTML(_){throw Error('Plain text only')}}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,node(id));return elements.get(id)};
get('vision-mode').value='legacy';get('mode').value='scripted';get('runtime-engine').value='native';
let response;
const context=vm.createContext({document:{getElementById:get,createElement:node,querySelectorAll:()=>[]},fetch:async()=>({ok:true,json:async()=>response}),setInterval:()=>1,clearInterval(){}});
vm.runInContext(fs.readFileSync(path.join(repo,'static/conversation.js'),'utf8'),context);vm.runInContext("sessionId='fixture'",context);
const text=n=>[n.textContent,...n.children.map(text)].join('\n');const descend=n=>[n,...n.children.flatMap(descend)];
async function ask(data){get('messages').replaceChildren();get('message').value='只检查曝光';response={execution_mode:'scripted',execution_status:'completed',agent_mode:'adaptive',runtime_engine:'native',answer:'Actual fixture result',latency_ms:1,request_attempts:0,model_calls:4,tool_calls:1,trace:[],evidence:[],citations:[],image_refs:{original:'/original',candidate_gamma:'/candidate',adaptive_heat_original:'/heat'},selected_image_id:'original',task_status:'completed',task_contract:{task_name:'检查曝光',task_type:'quality',image_id:'original',requirements:['质量结果']},task_validation:{passed:true,missing:[]},initial_plan:[],plan_revisions:[],...data};await get('chat-form').onsubmit({preventDefault(){}});return get('messages').children[0]}
(async()=>{
const html=fs.readFileSync(path.join(repo,'templates/conversation.html'),'utf8');assert.match(html,/<details id="advanced-settings"[^>]*>/);assert.ok(!/<details id="advanced-settings"[^>]*\bopen/.test(html));
let article=await ask({target_count:null,vision_status:null});assert.ok(!text(article).includes('候选目标：0'));assert.ok(article.children.findIndex(n=>n.className==='answer')<article.children.findIndex(n=>n.className==='task-details'));
let details=descend(article).find(n=>n.tag==='details'&&n.children[0]?.textContent==='其他候选、热图与结果图片');assert.equal(details.open,false);assert.ok(descend(details).filter(n=>n.tag==='img').every(n=>!n.src));details.open=true;details.ontoggle();assert.ok(descend(details).filter(n=>n.tag==='img').every(n=>n.src));
article=await ask({target_count:0,vision_status:'quality_failure',vision_result:{reasons:['全黑，不能恢复真实纹理']}});assert.ok(text(article).includes('质量失败'));assert.ok(text(article).includes('全黑，不能恢复真实纹理'));assert.ok(text(article).includes('候选目标：0'));
article=await ask({memory_context:{enabled:true,available:true,preferences:{display_detail:'expanded'},save_status:'saved'}});details=descend(article).find(n=>n.tag==='details'&&n.children[0]?.textContent==='其他候选、热图与结果图片');assert.ok(details.open);assert.ok(descend(details).filter(n=>n.tag==='img').every(n=>n.src));
article=await ask({execution_status:'task_incomplete',task_status:'incomplete',task_validation:{passed:false,missing:[{requirement:'actual comparison',reason:'missing'}]},error_details:{stage:'task_completion',cause:'missing evidence'}});assert.ok(article.children.find(n=>n.className==='error-details').open);assert.ok(article.children.find(n=>n.className==='task-details').open);
console.log('PASS: 5 compact UI contracts; collapsed settings, answer first, deferred images, real zero vs unmeasured, preference/error visibility');
})().catch(error=>{console.error(error);process.exitCode=1});
