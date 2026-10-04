// Local UI fixtures only; no model or network calls.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const elements=new Map();
function node(tag){return {tag,children:[],textContent:'',value:'',checked:false,options:[],append(...x){this.children.push(...x);},replaceChildren(...x){this.children=x;},get selectedOptions(){return this.options.filter(x=>x.selected);},set innerHTML(_){throw Error('Use plain text for memory data');}};}
const get=id=>{if(!elements.has(id))elements.set(id,node(id));return elements.get(id);};
get('default-targets').options=['pipe_type2','pipe','qr_codes'].map(value=>({value,selected:false}));
let prefs={enabled:true,target_classes:['pipe_type2','pipe'],display_detail:'compact'},history=[{memory_id:'memory:fixture',created_at:'2026-10-03T00:00:00Z',summary:'历史 <script>候选不等于可靠识别</script>',facts:{selected_candidate_count:0},versions:{agent_mode:'adaptive'}}],fail=false;
const calls=[];
const context=vm.createContext({document:{getElementById:get,createElement:node},confirm:()=>true,fetch:async(url,options={})=>{calls.push([url,options]);let data={};if(fail)return {ok:false,json:async()=>({error:'记忆不可用；数据库读取失败'})};if(url==='/api/memory/preferences'){if(options.method==='PUT')prefs=JSON.parse(options.body);data={preferences:prefs};}else if(options.method==='DELETE'){history=[];data={deleted:1};}else data={history,count:history.length};return {ok:true,json:async()=>data};}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/memory.js'),'utf8'),context);
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
const text=n=>[n.textContent,...n.children.map(text)].join('\n');
(async()=>{await tick();assert.equal(get('memory-enabled').checked,true);assert.deepEqual(get('default-targets').selectedOptions.map(x=>x.value),['pipe_type2','pipe']);assert.ok(text(get('history-list')).includes('历史 <script>'));assert.ok(text(get('history-status')).includes('不是当前图片证据'));
 get('default-targets').options.forEach(x=>x.selected=x.value==='qr_codes');get('memory-enabled').checked=false;get('display-detail').value='expanded';await get('preferences-form').onsubmit({preventDefault(){}});assert.deepEqual(prefs,{enabled:false,target_classes:['qr_codes'],display_detail:'expanded'});assert.equal(get('save-preferences').disabled,false);assert.ok(text(get('preference-status')).includes('下一轮'));
 get('history-query').value='管道&<';get('history-search').onsubmit({preventDefault(){}});await tick();assert.ok(calls.some(([url])=>url==='/api/memory/history?q='+encodeURIComponent('管道&<')));
 const article=get('history-list').children[0];await article.children.at(-1).onclick();assert.equal(get('history-list').children.length,0);assert.ok(calls.some(([url,options])=>url==='/api/memory/history/memory%3Afixture'&&options.method==='DELETE'));
 await get('clear-history').onclick();assert.ok(calls.some(([url,options])=>url==='/api/memory/history'&&options.method==='DELETE'));
 fail=true;await context.loadPreferences();await context.loadHistory();assert.ok(text(get('preference-status')).includes('记忆不可用'));assert.ok(text(get('history-status')).includes('记忆不可用'));
 console.log('PASS: 6 memory UI scenarios; load, explicit preferences, search encoding, deletion, clear, unavailable; plain text rendering');
})().catch(error=>{console.error(error);process.exitCode=1;});
