let sessionId=null, busy=false;
const byId=id=>document.getElementById(id);
function syncVisionScheduler(){const experimental=byId('vision-mode').value!=='legacy'||byId('runtime-engine').value==='langgraph';byId('agent-mode').querySelector('option[value="legacy"]').disabled=experimental;if(experimental)byId('agent-mode').value='adaptive';}
byId('vision-mode').addEventListener('change',syncVisionScheduler);
byId('runtime-engine').addEventListener('change',syncVisionScheduler);
syncVisionScheduler();
function element(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function expanded(data){return !!(data.memory_context?.enabled&&data.memory_context?.available&&data.memory_context.preferences?.display_detail==='expanded');}
function detailPanel(title,data,className){const panel=element('details',undefined,className);panel.open=expanded(data);panel.append(element('summary',title));return panel;}
function appendTaskDetails(turn,data){
 const task=data.task_contract, validation=data.task_validation;if(!('task_status' in data))return;
 const labels={completed:'已完成',incomplete:'未完成',needs_clarification:'需要澄清',unsupported:'不支持'};
 const panel=detailPanel('任务理解与完成校验',data,'task-details');panel.open=panel.open||data.task_status!=='completed';
 const image=task?({original:'原图',corrected:'校正图'}[task.image_id]||task.image_id):'';
 panel.append(element('p',`理解的任务：${task?task.task_name+'（'+(task.task_type==='comparison'||task.task_type==='explanation'?'原图与校正图':image)+'）':'尚未完成任务理解'}`));
 panel.append(element('p',`必要结果：${task&&task.requirements.length?task.requirements.join('；'):'等待明确任务后确定'}`));
 panel.append(element('p',`任务状态：${labels[data.task_status]||data.task_status} · ${validation&&validation.passed===true?'校验通过':validation&&validation.passed===null?'尚未进入执行':'校验未通过'}`));
 if(validation&&validation.missing.length)for(const item of validation.missing)panel.append(element('p',`缺项：${item.requirement} — ${item.reason}`,'small'));
 const vision={reliable:'通过当前内部可靠性规则',unreliable:'识别结论不可靠',quality_failure:'图像质量失败',quality_unassessable:'无法评估原始曝光'};
 panel.append(element('p',`视觉可靠性：${vision[data.vision_status]||'本任务未交付完整可靠性结论'}`,'small'));
 turn.append(panel);
}
function appendAdaptiveDetails(turn,data){
 if(data.agent_mode!=='adaptive')return;
 const panel=detailPanel('计划、行动依据与候选比较',data,'task-details');
 if(data.goal_contract){panel.append(element('p',`目标类别：${data.goal_contract.target_classes.join('、')}`));panel.append(element('p',`校正策略：${data.goal_contract.correction_policy} · 采用图片：${data.selected_image_id}`));}
 if(data.initial_plan&&data.initial_plan.length){panel.append(element('p','初始计划（拟议动作，实际执行见工具记录）'));for(const step of data.initial_plan)panel.append(element('p',`${step.tool}：${step.purpose}`,'small'));}
 for(const revision of data.plan_revisions||[]){panel.append(element('p',`计划修订 ${revision.revision}：${revision.reason}`));panel.append(element('p',`依据观察：${revision.observation_ids.join('、')}`,'small'));for(const step of revision.steps)panel.append(element('p',`${step.tool}：${step.purpose}`,'small'));}
 for(const evidence of data.evidence||[]){if(evidence.type==='comparison')for(const report of evidence.data.reports){const before=report.quality_before,after=report.quality_after;panel.append(element('p',`候选比较：${report.candidate_image_id} · 采用 ${report.selected_image} · ${report.status}${report.effect_degraded?' · 候选证据下降':''}`));panel.append(element('p',`亮度中位数 ${before.global.median.toFixed(3)} → ${after.global.median.toFixed(3)} · 暗部占比 ${(before.global.dark_fraction*100).toFixed(1)}% → ${(after.global.dark_fraction*100).toFixed(1)}% · 指定类别候选 ${report.detections_original.length} → ${report.detections_corrected.length}`,'small'));panel.append(element('p','候选数量或分数上升不代表准确率提升。','small'));}}
 panel.append(element('p',`停止状态：${data.execution_status}`,'small'));
 if(data.computation_counts)panel.append(element('p',`实际计算：质量 ${data.computation_counts.quality} 次 · 检测 ${data.computation_counts.detection} 次 · 校正 ${data.computation_counts.correction} 次 · 比较 ${data.computation_counts.comparison} 次`,'small'));
 turn.append(panel);
}
function appendGraphDetails(turn,data){
 turn.append(element('p','运行器：'+(data.runtime_engine||'native'),'small'));
 if(data.runtime_engine!=='langgraph')return;
 const panel=element('details');panel.append(element('summary',`图节点执行记录（${(data.graph_trace||[]).length} 次；模型 ${data.model_calls} 次）`));
 const names={initialize:'初始化',understand_goal:'理解目标',plan:'生成计划',choose_action:'选择行动',execute_tools:'执行工具',validate:'完成校验',repair:'纠错',output:'输出结果',exception_handler:'异常处理'};
 for(const item of data.graph_trace||[])panel.append(element('p',`${names[item.node]||item.node} → ${names[item.next]||item.next||'结束'} · ${item.reason||item.status} · ${item.latency_ms||0} ms`,'small'));
 turn.append(panel);
}
function appendErrorDetails(turn,data){
 const info=data.error_details;if(!info)return;
 const stages={configuration:'本地配置',connection:'网络连接',authentication:'账户认证',account_balance:'账户余额',api_request:'云端API请求',response_decode:'云端响应解析',model_request:'模型请求',model_response:'模型响应',task_understanding:'任务理解',task_completion:'任务完成校验',model_tool_validation:'工具调用格式校验',model_finish_validation:'模型结束结果与证据校验',tool_execution:'图像或知识工具执行',runtime_budget:'运行上限',runtime:'本地程序运行'};
 const panel=element('details',undefined,'error-details');panel.open=true;panel.append(element('summary','错误详情'));
 const http=data.execution_mode==='cloud'?(info.http_status==null?'未收到HTTP响应':`HTTP ${info.http_status}`):'离线模式，未调用云端API';
 panel.append(element('p',`云端响应：${http}`));panel.append(element('p',`失败位置：${stages[info.stage]||info.stage}`));
 if(info.cause)panel.append(element('p',`具体原因：${info.cause}`));
 if(info.suggestion)panel.append(element('p',`处理建议：${info.suggestion}`));
 if(info.finish_reason)panel.append(element('p',`模型结束原因：${info.finish_reason}`));
 if(info.request_id)panel.append(element('p',`请求编号：${info.request_id}`));
 if(info.response_preview){const preview=element('details');preview.append(element('summary','查看脱敏后的模型返回片段'));preview.append(element('pre',info.response_preview));panel.append(preview);}
 turn.append(panel);
}
async function readResponse(response){const data=await response.json();if(!response.ok)throw new Error(data.error||'请求失败');return data;}
async function createSession(demo){
 if(busy)return; const form=new FormData();
 form.append('vision_mode',byId('vision-mode').value);
 form.append('rag_mode',byId('rag-mode').value);
 if(demo){form.append('demo','true');form.append('demo_variant',byId('demo-variant').value||'normal');}else{const file=byId('image-file').files[0];if(!file){byId('session-status').textContent='请先选择图片';return;}form.append('image',file);}
 busy=true;byId('session-status').textContent='正在加载图片和本地模型…';byId('upload').disabled=byId('demo').disabled=true;byId('vision-mode').disabled=byId('rag-mode').disabled=true;
 try{const data=await readResponse(await fetch('/api/sessions',{method:'POST',body:form}));sessionId=data.session_id;
 byId('vision-mode').value=data.vision_mode;syncVisionScheduler();byId('vision-mode').disabled=byId('rag-mode').disabled=true;
 byId('original-preview').src=data.image_refs.original;byId('original-preview').hidden=false;if(typeof byId('original-preview').decode==='function')await byId('original-preview').decode();byId('session-status').textContent='图片已就绪，可提出任务；闲置30分钟后过期'+(data.rag_metadata?`。检索：${data.rag_metadata.mode}，版本 ${data.rag_metadata.corpus_hash?data.rag_metadata.corpus_hash.slice(0,12):'旧库'}${data.rag_metadata.fallback_reason?'；已降级：'+data.rag_metadata.fallback_reason:''}`:'');byId('messages').replaceChildren();}
 catch(error){byId('session-status').textContent=error.message;if(!sessionId)byId('vision-mode').disabled=byId('rag-mode').disabled=false;}finally{busy=false;byId('upload').disabled=byId('demo').disabled=false;}
}
byId('upload').onclick=()=>createSession(false);byId('demo').onclick=()=>createSession(true);
byId('new-session').onclick=()=>{if(busy)return;sessionId=null;byId('runtime-engine').disabled=false;byId('vision-mode').disabled=byId('rag-mode').disabled=false;byId('original-preview').hidden=true;byId('messages').replaceChildren();byId('session-status').textContent='请选择配置并创建新会话';syncVisionScheduler();};
document.querySelectorAll('.prompt').forEach(button=>button.onclick=()=>{byId('message').value=button.textContent;byId('message').focus();});
byId('chat-form').onsubmit=async event=>{
 event.preventDefault();if(busy)return;if(!sessionId){byId('running').textContent='请先选择图片';return;}
 const message=byId('message').value.trim();if(!message)return;busy=true;byId('send').disabled=true;const started=Date.now();
 const timer=setInterval(()=>byId('running').textContent=`正在分析，已用 ${Math.floor((Date.now()-started)/1000)} 秒…`,1000);
 const turn=element('article',undefined,'turn');turn.append(element('p',message,'query'));byId('messages').append(turn);
 try{const data=await readResponse(await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,message,mode:byId('mode').value,agent_mode:byId('agent-mode').value||'legacy',runtime_engine:byId('runtime-engine').value||'native'})}));
 byId('runtime-engine').disabled=true;
 const label=data.execution_mode==='scripted'?'离线规则演示':'云端模型调度';
 const statuses={graph_error:"图执行异常",graph_step_limit:"图步骤达到上限",completed:'已返回结果',task_incomplete:'任务未完成',needs_clarification:'需要澄清',unsupported:'不支持此任务',missing_api_key:'尚未配置密钥',authentication_error:'密钥或账户错误',insufficient_balance:'账户余额不足',network_error:'连接失败',invalid_response:'响应无效',tool_errors:'工具执行停止',budget_exceeded:'达到运行上限'};
 turn.append(element('p',`${label} · ${statuses[data.execution_status]||data.execution_status} · ${Math.round(data.latency_ms)} ms`,'badge'));

 turn.append(element('p',data.answer,'answer'));
 const visionLabels={reliable:'通过当前程序规则',unreliable:'不可靠',quality_failure:'质量失败',quality_unassessable:'质量不可评估'};
 turn.append(element('p',`视觉结论：${visionLabels[data.vision_status]||'尚无完整可靠性结论'}`,data.vision_status&&data.vision_status!=='reliable'?'error':'small'));
 if(typeof data.target_count==='number')turn.append(element('p',`本轮实际候选目标：${data.target_count} 个；候选数量不代表识别准确率。`,'small'));
 if(data.goal_contract)turn.append(element('p',`目标类别：${data.goal_contract.target_classes.join('、')||'本任务无需目标检测'} · 采用图片：${data.selected_image_id}`,'small'));
 if(data.vision_result?.reasons?.length)turn.append(element('p',data.vision_result.reasons.join('；'),'error'));
 appendTaskDetails(turn,data);
 appendAdaptiveDetails(turn,data);
 appendGraphDetails(turn,data);
  if(data.rag_metadata)turn.append(element('p',`知识检索：${data.rag_metadata.mode} · ${data.rag_metadata.corpus||'旧库'}${data.rag_metadata.fallback_reason?' · 降级原因：'+data.rag_metadata.fallback_reason:''}`,'small'));
  const memory=data.memory_context;
  if(memory){const section=element('details');section.open=!memory.available;section.append(element('summary',`长期记忆：${!memory.available?'不可用':memory.enabled?'已启用':'已关闭'} · ${({saved:'已保存摘要',reused:'复用已保存摘要',not_completed:'未完成，不保存',disabled:'不保存',unavailable:'存储不可用'})[memory.save_status]||memory.save_status}`));
   section.append(element('p','历史只能辅助规划，不能代替当前图片证据。','small'));
   if(memory.applied_preferences?.target_classes)section.append(element('p','本轮默认关注：'+memory.applied_preferences.target_classes.join('、')));
   if(memory.warning)section.append(element('p',memory.warning,'error'));
   for(const item of memory.history_refs||[])section.append(element('p',`${item.memory_id} · ${item.summary}`));
   if(memory.saved_memory_id)section.append(element('p','本轮摘要：'+memory.saved_memory_id));
   const link=element('a','查看历史 / 修改偏好');link.href='/memory';section.append(link);turn.append(section);}
 if(data.vision_mode&&data.vision_mode!=='legacy'){turn.append(element('p','当前为 0.4.0 视觉实验配置，机器人不细分 AUV/ROV；保留测试误报偏多，仅用于实验观察。候选数量变化不等于识别准确率提高。','small'));for(const [name,model] of Object.entries(data.vision_models.branches||{}))turn.append(element('p',`${name==='robot'?'机器人（UIIS10K）':'设施六类（SODD）'}：${model.available?'可用':'模型缺失，不能解释为无目标'}；校准 ${model.calibration?model.calibration.status:'不足'}；版本 ${(model.model_sha256||'').slice(0,12)}`,'small'));}
 if(data.input_provenance&&data.input_provenance.kind==='simulated_exposure_perturbation')turn.append(element('p',`输入为模拟曝光扰动（${data.input_provenance.variant}），不是实际相机重拍。`,'small'));
 turn.append(element('p',`${data.execution_mode==='cloud'?'API请求':'调度'}尝试 ${data.request_attempts??data.model_calls} 次 · 成功模型响应 ${data.model_calls} 次 · 工具执行 ${data.tool_calls} 次 · 结果纠错 ${data.format_repairs||0} 次`,'small'));
 appendErrorDetails(turn,data);
 if(data.tool_calls===0)turn.append(element('p',data.evidence.length?'本轮没有执行图像工具，已展示结果来自当前会话缓存。':'本轮尚未执行图像工具，没有产生新的曝光或检测结果。','small'));
 const views=element('div',undefined,'views');
 const imageLabels={original:'原图',candidate_gamma:'Gamma候选',candidate_local:'局部校正候选',adaptive_heat_original:'原图质量热图',adaptive_heat_candidate_gamma:'Gamma候选质量热图',adaptive_heat_candidate_local:'局部候选质量热图',corrected:'校正候选',original_quality_heatmap:'原图质量热图',corrected_quality_heatmap:'校正图质量热图',original_detections:'原图候选框',corrected_detections:'校正图候选框',report_detections_selected:'最终选择的检测证据'};
  const selected=data.selected_image_id||'original';
  const annotated=Object.keys(data.image_refs).find(id=>id.startsWith('adaptive_boxes_'+selected+'_'))||('report_detections_selected' in data.image_refs?'report_detections_selected':null);
  const primary=new Set(['original',annotated||selected]);
  const extra=element('div',undefined,'views'),deferred=[];
  for(const [id,url] of Object.entries(data.image_refs)){
   const frame=element('div');const label=imageLabels[id]||(id.startsWith('adaptive_boxes_')?'实际候选框':id);frame.append(element('p',label));
   const img=element('img');img.alt=label;frame.append(img);
   if(primary.has(id)){img.src=url;views.append(frame);}else{deferred.push([img,url]);extra.append(frame);}
  }
  turn.append(views);
  if(deferred.length){const other=detailPanel('其他候选、热图与结果图片',data);other.append(extra);
   const load=()=>{if(other.open)for(const [img,url] of deferred)if(!img.src)img.src=url;};other.addEventListener('toggle',load);load();turn.append(other);}
 if(data.citations.length){const section=element('details');section.append(element('summary','查看依据来源'));for(const citation of data.citations){const p=element('p');const a=element('a',`[${citation.citation_id}] ${citation.title}`);a.href=citation.source;a.target='_blank';a.rel='noopener noreferrer';p.append(a);if(citation.version_warning)p.append(element('p',citation.version_warning,'small'));section.append(p);}turn.append(section);}
 const trace=detailPanel('执行记录',data);trace.append(element('summary',`查看执行过程（请求尝试 ${data.request_attempts??data.model_calls} 次，成功响应 ${data.model_calls} 次，工具 ${data.tool_calls} 次）`));
 for(const item of data.trace){if(item.type==='task_understanding')trace.append(element('p',`任务理解：${item.contract.task_name}`));if(item.type==='tool')trace.append(element('p',`工具执行：${item.tool} · ${item.ok?'成功':'失败'}${item.cached?'（复用缓存）':''}`));if(item.type==='task_validation')trace.append(element('p',`完成校验：${item.passed?'通过':'未通过'}${item.missing.length?' · '+item.missing.map(x=>x.requirement).join('、'):''}`));}
  trace.open=!!(memory?.enabled&&memory.available&&memory.preferences?.display_detail==='expanded');
  trace.append(element('pre',JSON.stringify(data.trace,null,2)));turn.append(trace);
 byId('message').value='';}
 catch(error){turn.append(element('p',error.message,'error'));}finally{clearInterval(timer);busy=false;byId('send').disabled=false;byId('running').textContent='';}
};
