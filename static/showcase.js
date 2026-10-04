const el = id => document.getElementById(id);
const prompts = {quality:'只检查曝光',correction:'只校正曝光',analysis:'检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'};
function text(id, value){el(id).textContent=value;}
async function json(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data;}
fetch('/health').then(r=>r.json()).then(d=>text('model',d.detector_available?'SODD six-class checkpoint and detector dependencies available. Detection returns candidates.':'Detection unavailable. Exposure assessment/correction work without weights. See --with-detector.'));
el('run').onclick=async()=>{el('run').disabled=true;text('status','Executing actual local tools…');el('actions').replaceChildren();try{
 const form=new FormData();form.append('rag_mode','tfidf');form.append('vision_mode','legacy');
 if(el('upload').files.length)form.append('image',el('upload').files[0]);else{form.append('demo','true');form.append('demo_variant',el('variant').value);}
 const session=await json('/api/sessions',{method:'POST',body:form});el('before').src=session.image_refs.original;
 text('provenance',session.input_provenance.kind==='simulated_exposure_perturbation'?'SIMULATED exposure perturbation of a licensed SODD frame.':'Original uploaded or licensed SODD image.');
 const result=await json('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session.session_id,message:prompts[el('task').value],mode:'scripted',agent_mode:'adaptive',runtime_engine:'native'})});
 window.lastPublicResult=result;
 const refs=result.image_refs||{};const keys=Object.keys(refs);const selected=result.selected_image_id||'original';
 const visual=keys.find(k=>k.includes(selected)&&(k.includes('detections')||k.includes('boxes')))||keys.find(k=>k==='candidate_gamma')||keys.find(k=>k==='candidate_local')||selected;
 el('after').src=refs[visual]||session.image_refs.original;text('choice',`Selected image: ${selected}. Display: ${visual}.`);
 const evidence=result.evidence||result.observations||{};
 const quality=findQuality(result);const g=quality?.global||{};el('metrics').replaceChildren();
 for(const [label,value] of [['Exposure',quality?.exposure_state||'not measured'],['Dark ratio',g.dark_fraction],['Bright ratio',g.bright_fraction],['Candidates',result.target_count],['Tools',result.tool_calls],['Cloud requests',result.request_attempts??0]]){let p=document.createElement('p'),b=document.createElement('b');b.textContent=typeof value==='number'?value.toFixed(label.includes('ratio')?3:0):String(value??'—');p.append(b,document.createTextNode(label));el('metrics').append(p);}
 const goal=result.goal_contract||{};text('scope',`Goal: ${goal.task_type||'unknown'} · classes: ${(goal.target_classes||[]).join(', ')||'not requested'} · correction: ${goal.correction_policy||'not requested'}`);
 text('status',`Task: ${result.task_status||result.status} · actual tools: ${result.tool_calls??0} · cloud requests: ${result.request_attempts??0}`);
 text('conclusion',`Execution: ${result.execution_status||result.task_status}. Reliability: ${findReliability(result)||'not assessed for this task'}.`);
 text('reasons',result.answer||result.error||'');
 for(const event of result.trace||[]){if(event.type!=='tool'&&!event.tool)continue;const li=document.createElement('li');li.textContent=`${event.tool} ${JSON.stringify(event.arguments||{})}: ${event.ok?'completed':'failed'}${event.cached?' (cached)':''}${event.error_category?' · '+event.error_category:''}`;el('actions').append(li);}
 text('json',JSON.stringify(result,null,2));
 }catch(e){text('status',e.message);}finally{el('run').disabled=false;}};
function findQuality(value){if(!value||typeof value!=='object')return null;if(value.exposure_state&&value.global)return value;for(const child of Object.values(value)){const found=findQuality(child);if(found)return found;}return null;}
function findReliability(value){if(!value||typeof value!=='object')return null;if(value.target_conclusion)return value.status;for(const child of Object.values(value)){const found=findReliability(child);if(found)return found;}return null;}
