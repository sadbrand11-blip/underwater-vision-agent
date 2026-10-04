// Run with Node. All responses and DOM objects are local fixtures; no network is used.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

function node(tag) {
  return {tag, children: [], textContent: '', value: '', append(...items) {this.children.push(...items);},
    replaceChildren(...items) {this.children = items;}, focus() {},
    addEventListener(name,handler){this['on'+name]=handler;}, querySelector(){return {disabled:false};},
    set innerHTML(_) {throw new Error('Diagnostics must be rendered as plain text');}};
}
const elements = new Map();
const get = id => {if (!elements.has(id)) elements.set(id, node(id)); return elements.get(id);};
get('vision-mode').value='legacy';
let response;
const context = vm.createContext({document: {getElementById: get, createElement: node, querySelectorAll: () => []},
  fetch: async () => ({ok: true, json: async () => response}), setInterval: () => 1, clearInterval() {}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/conversation.js'), 'utf8'), context);
vm.runInContext("sessionId='fixture-session'", context);
get('mode').value = 'cloud';
const text = item => [item.textContent, ...item.children.map(text)].join('\n');

(async () => {
  const cases = [
    [401, 'authentication', 'HTTP 401', '账户认证'],
    [402, 'account_balance', 'HTTP 402', '账户余额'],
    [200, 'model_finish_validation', 'HTTP 200', '模型结束结果与证据校验'],
    [null, 'connection', '未收到HTTP响应', '网络连接']
  ];
  for (const [http, stage, expectedCode, expectedStage] of cases) {
    get('messages').replaceChildren(); get('message').value = '只检查曝光';
    response = {execution_mode: 'cloud', execution_status: 'invalid_response', answer: 'fixture failure',
      latency_ms: 1, request_attempts: 1, model_calls: http === 200 ? 1 : 0, tool_calls: 0,
      format_repairs: 0, evidence: [], image_refs: {}, citations: [], trace: [],
      error_details: {http_status: http, stage, cause: 'fixture', response_preview: '<img src=x onerror=alert(1)> [REDACTED]'}};
    await get('chat-form').onsubmit({preventDefault() {}});
    const article = get('messages').children[0];
    assert.ok(text(article).includes(expectedCode));
    assert.ok(text(article).includes(expectedStage));
    assert.ok(text(article).includes('API请求尝试 1 次'));
    assert.ok(text(article).includes('本轮尚未执行图像工具'));
    assert.ok(article.children.some(child => child.className === 'error-details' && child.open));
  }
  const taskCases = [
    ['completed', true, 'quality_failure', '任务状态：已完成', '图像质量失败'],
    ['incomplete', false, null, '任务状态：未完成', 'comparison:original'],
    ['needs_clarification', null, null, '任务状态：需要澄清', '尚未进入执行'],
    ['unsupported', null, null, '任务状态：不支持', '尚未进入执行']
  ];
  for (const [status, passed, vision, expectedStatus, expectedDetail] of taskCases) {
    get('messages').replaceChildren(); get('message').value = '比较前后';
    response = {execution_mode: 'scripted', execution_status: status, answer: 'fixture', latency_ms: 1,
      request_attempts: 2, model_calls: 2, tool_calls: 0, evidence: [], image_refs: {}, citations: [], trace: [],
      task_status: status, vision_status: vision, task_contract: {task_type: 'comparison', task_name: '比较校正前后',
        image_id: 'original', requirements: ['完整前后比较证据']}, task_validation: {passed,
        missing: passed === false ? [{requirement: 'comparison:original', reason: '尚未产生实际证据'}] : []}};
    await get('chat-form').onsubmit({preventDefault() {}});
    const rendered = text(get('messages').children[0]);
    assert.ok(rendered.includes('理解的任务：比较校正前后（原图与校正图）'));
    assert.ok(rendered.includes('必要结果：完整前后比较证据'));
    assert.ok(rendered.includes(expectedStatus));
    assert.ok(rendered.includes(expectedDetail));
  }
  get('messages').replaceChildren(); get('message').value = '条件分析';
  response = {execution_mode:'scripted',execution_status:'completed',agent_mode:'adaptive',answer:'实测结果',latency_ms:1,
    request_attempts:0,model_calls:8,tool_calls:9,evidence:[],image_refs:{},citations:[],trace:[],
    goal_contract:{target_classes:['pipe','pipe_type2'],correction_policy:'if_needed'},selected_image_id:'original',
    initial_plan:[{tool:'assess_image_quality',purpose:'先检查质量'}],
    plan_revisions:[{revision:1,reason:'候选检测退化 <script>',observation_ids:['comparison:real'],steps:[{tool:'compare_candidates',purpose:'保留原图'}]}],
    computation_counts:{quality:3,detection:3,correction:2,comparison:2}};
  await get('chat-form').onsubmit({preventDefault(){}});
  const adaptive = text(get('messages').children[0]);
  assert.ok(adaptive.includes('目标类别：pipe、pipe_type2'));
  assert.ok(adaptive.includes('初始计划（拟议动作'));
  assert.ok(adaptive.includes('计划修订 1'));
  assert.ok(adaptive.includes('依据观察：comparison:real'));
  assert.ok(adaptive.includes('校正 2 次'));
  for(const [enabled,available,save_status,expected] of [[true,true,'saved','已保存摘要'],[false,true,'disabled','不保存'],[true,false,'unavailable','存储不可用']]){
    get('messages').replaceChildren();get('message').value='只检查曝光';
    response={execution_mode:'scripted',execution_status:'completed',answer:'本图工具结果',latency_ms:1,request_attempts:0,model_calls:4,tool_calls:1,evidence:[],image_refs:{},citations:[],trace:[],memory_context:{enabled,available,save_status,preferences:{display_detail:'expanded'},applied_preferences:{target_classes:['pipe']},history_refs:[{memory_id:'memory:real',summary:'历史 <script>仅为参考</script>'}],warning:available?null:'记忆不可用'}};
    await get('chat-form').onsubmit({preventDefault(){}});
    const rendered=text(get('messages').children[0]);assert.ok(rendered.includes(expected));assert.ok(rendered.includes('历史只能辅助规划'));assert.ok(rendered.includes('本轮默认关注：pipe'));assert.ok(rendered.includes('memory:real'));
  }
  console.log('PASS: 12 conversation UI cases; HTTP/task/adaptive diagnostics and memory states; plain text rendering');
})().catch(error => {console.error(error); process.exitCode = 1;});
