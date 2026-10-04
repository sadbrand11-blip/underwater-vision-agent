"""Regression against independently captured pre-refactor behavior."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import test_adaptive as fixture
from test_tasks import WireQueue
from optical_agent.adaptive_runtime import AdaptiveRuntime
from optical_agent.langgraph_runtime import LangGraphRuntime
from optical_agent.adaptive_fixture import AdaptiveScriptedClient
from optical_agent.runtime_compare import core, without_timings
from optical_agent.paths import PROJECT_ROOT, code_files
from optical_agent.experiments import __main__ as cli

BASELINE = json.loads((Path(__file__).parent/'fixtures/runtime_v054_baseline.json').read_text(encoding='utf8'))
CASES = {
    'exposure': (100,['只检查曝光'],False),
    'correction': (45,['只校正曝光'],False),
    'detection': (100,['识别目标'],False),
    'normal_condition': (100,['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'],False),
    'dark_condition': (45,['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'],False),
    'black_condition': (0,['检查这张水下图，必要时校正曝光，然后只识别管道，说明结果是否可靠。'],False),
    'followup_cache': (45,['比较校正前后','识别校正后的图','只检查曝光'],False),
    'candidate_decline': (45,['比较校正前后','识别校正后的图'],True),
}

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Recorder(AdaptiveScriptedClient):
    def __init__(self):self.requests=[]
    def complete(self,messages,tools=None,timeout=30,**kwargs):
        self.requests.append(copy.deepcopy((messages,tools,kwargs)))
        return super().complete(messages,tools,timeout,**kwargs)

class QueueRecorder(WireQueue):
    def __init__(self,replies):
        super().__init__(copy.deepcopy(replies));self.requests=[]
    def complete(self,messages,tools=None,timeout=30,**kwargs):
        self.requests.append(copy.deepcopy((messages,tools,kwargs)))
        return super().complete(messages,tools,timeout,**kwargs)

@pytest.mark.parametrize('expected',BASELINE,ids=lambda r:r['engine']+'-'+r['case'])
def test_frozen_native_and_graph_requests_and_core(expected):
    engine=AdaptiveRuntime if expected['engine']=='native' else LangGraphRuntime
    if expected['engine']=='langgraph':pytest.importorskip('langgraph')
    name=expected['case'];options={}
    if name in CASES:
        value,messages,decline=CASES[name]
        session=fixture.session(np.full((64,64,3),value,np.uint8),fixture.Detector(decline=decline))
        client=Recorder()
    else:
        session=fixture.session();messages=['只检查曝光']
        final={'selected_image_id':'original','evidence_ids':['quality:original'],'citation_ids':[]}
        prefix=[fixture.response(fixture.goal('quality',[],'never',reliable=False)),fixture.PLAN]
        if name=='premature_finish':
            replies=prefix+[fixture.response(final),fixture.actions(('assess_image_quality',{'image_id':'original'})),fixture.response(final)]
        elif name=='fabrication':
            replies=prefix+[fixture.response(dict(final,evidence_ids=['made_up']))]*3
        elif name=='format_failure':replies=prefix+[{'message':{'content':'not json'}}]*3
        else:
            replies=None;options={'max_model_calls':2} if name=='model_budget' else {'max_tool_calls':1}
        client=QueueRecorder(replies) if replies else Recorder()
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No cloud in regression')):
        actual=[core(engine(client,**options).run(session,message)) for message in messages]
    assert digest(actual)==expected['core_sha256']
    assert digest(without_timings(client.requests,parse_json=True))==expected['requests_sha256']
    assert not session.lock.locked()

def test_dispatch_registry_is_fixed_lazy_and_preserves_arguments():
    files={p.stem for p in (PROJECT_ROOT/'optical_agent/experiments').glob('*.py') if p.stem not in {'__init__','__main__'}}
    assert files==set(cli.COMMANDS) and len(files)==32
    for command,module in cli.COMMANDS.items():
        if command in cli.NO_ARGUMENT_COMMANDS:continue
        observed=[]
        with patch.object(cli.runpy,'run_module',side_effect=lambda name,**kw:observed.append((name,kw,list(sys.argv)))):
            original=sys.argv
            cli.main([command,'--fixture','literal value'])
            assert sys.argv is original
        assert observed==[(module,{'run_name':'__main__'},[command,'--fixture','literal value'])]
    with patch.object(cli.runpy,'run_module') as execute:
        with pytest.raises(SystemExit) as stop:cli.main(['../not_registered'])
        assert stop.value.code==2 and not execute.called

def test_global_help_and_legacy_help_cannot_start_work(capsys):
    with patch.object(cli.runpy,'run_module',side_effect=AssertionError('No work from global help')):
        cli.main(['--help'])
        cli.main(['--list'])
        for name in cli.NO_ARGUMENT_COMMANDS:cli.main([name,'--help'])
        for name in cli.NO_ARGUMENT_COMMANDS:
            with pytest.raises(SystemExit):cli.main([name,'--unsupported'])
    assert 'Run one explicitly selected experiment' in capsys.readouterr().out
    code="import builtins,runpy,sys; original=builtins.__import__; blocked={'torch','torchvision','transformers','langgraph','mcp'}; builtins.__import__=lambda name,*a,**kw: (_ for _ in ()).throw(AssertionError(name)) if name.split('.')[0] in blocked else original(name,*a,**kw); sys.argv=['experiments','--help']; runpy.run_module('optical_agent.experiments',run_name='__main__')"
    completed=subprocess.run([sys.executable,'-B','-c',code],cwd=PROJECT_ROOT,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr

def test_recursive_freeze_covers_shared_and_nested_modules(tmp_path):
    files=set(code_files())
    assert PROJECT_ROOT/'optical_agent/adaptive_common.py' in files
    assert PROJECT_ROOT/'optical_agent/experiments/evaluate_adaptive.py' in files
    (tmp_path/'optical_agent/experiments').mkdir(parents=True)
    nested=tmp_path/'optical_agent/experiments/only.py';nested.write_text('first',encoding='utf8')
    from optical_agent.independent_eval import version_metadata
    first=version_metadata(tmp_path)['source_sha256']
    nested.write_text('changed',encoding='utf8')
    second=version_metadata(tmp_path)['source_sha256']
    assert first['experiments/only.py']!=second['experiments/only.py']

def test_navigation_and_knowledge_do_not_depend_on_homepage():
    assert len(list(PROJECT_ROOT.glob('*.py')))<=12
    for name in ['sources.json','sources_v2.json']:
        sources=json.loads((PROJECT_ROOT/'knowledge'/name).read_text(encoding='utf8'))
        assert all(item['path']!='README.md' for item in sources)
        assert all((PROJECT_ROOT/item['path']).is_file() for item in sources)
    config=json.loads((PROJECT_ROOT/'knowledge/rag_release.json').read_text(encoding='utf8'))
    if config.get('corpus')=='public-sanitized-v2':
        from optical_agent.rag_engine import build_chunks,fingerprint
        assert config['corpus_hash']==fingerprint(build_chunks(PROJECT_ROOT))
        assert 350<=len((PROJECT_ROOT/'README.md').read_text(encoding='utf8').split())<=450

def test_capability_probe_is_local_and_never_loads_models(tmp_path):
    from optical_agent import ui_capabilities
    with patch.object(ui_capabilities,'find_spec',return_value=None),patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
        result=ui_capabilities.probe_capabilities(tmp_path/'missing.pt',tmp_path/'embedding',experimental=False)
    assert not result['detector']['available']
    assert not result['embedding']['available']
    assert all(not mode['available'] and mode['reason'] for mode in result['vision_modes'].values())
