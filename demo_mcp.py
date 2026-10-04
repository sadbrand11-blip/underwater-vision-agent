"""Real stdio discovery/calls, compared with direct adaptive visual execution."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from optical_agent import __version__
from optical_agent.diagnostics import redact
from optical_agent.mcp_adapter import DATA_ROOT, MCP_ROOT

TOOLS = {'assess_image_quality', 'generate_exposure_candidate', 'detect_objects'}


def direct_results(path, method):
    from data import read_rgb
    from optical_agent.adaptive_goals import CLASSES, Goal
    from optical_agent.adaptive_tools import AdaptiveState, execute
    from optical_agent.mcp_adapter import load_default_detector
    from optical_agent.tools import ToolContext
    detector = load_default_detector()
    if not getattr(detector, 'model_available', True):
        raise RuntimeError('检测模型不可用，请检查 models/sodd_detector.pt。')
    state = AdaptiveState(ToolContext(detector, read_rgb(path)))

    def run(kind, image_id, name, args):
        state.begin(Goal(kind, image_id, CLASSES, 'always' if kind == 'correction' else 'never', False))
        result = execute(state, name, args)
        if not result['ok']:
            raise RuntimeError('直接视觉对照未完成：' + result.get('error_category', 'unknown'))
        return result['data']['data']

    before = run('quality', 'original', 'assess_image_quality', {'image_id': 'original'})
    correction = run('correction', 'original', 'generate_exposure_candidate', {'image_id': 'original', 'method': method})
    candidate = correction['image_id']
    after = run('quality', candidate, 'assess_image_quality', {'image_id': candidate})
    raw = run('detection', 'original', 'detect_objects', {'image_id': 'original'})
    fixed = run('detection', candidate, 'detect_objects', {'image_id': candidate})
    return {'quality_original': before['quality'], 'quality_candidate': after['quality'],
            'parameters': correction['parameters'], 'detections_original': raw['detections'],
            'detections_candidate': fixed['detections'], 'counts': state.counts,
            'model_sha256': getattr(detector, 'model_sha256', None),
            'configuration_sha256': state.config_hash}


async def run_demo(path, output, method='gamma_only'):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    base = Path(__file__).resolve().parent
    records = []
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    server = StdioServerParameters(command=sys.executable,
        args=['-u', '-X', 'utf8', '-B', str(base / 'serve_mcp.py')], cwd=str(base),
        env={'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
             'LANGSMITH_TRACING': 'false', 'LANGCHAIN_TRACING_V2': 'false'})
    # Force the standard initialize handshake, and send every call to the server.
    async with Client(server, mode='legacy', cache=None, read_timeout_seconds=180) as client:
        listing = await client.list_tools()
        names = {tool.name for tool in listing.tools}
        if names != TOOLS or len(listing.tools) != 3:
            raise AssertionError('协议工具列表与预期不符')
        print('已初始化 MCP 子进程，发现：' + '、'.join(sorted(names)), flush=True)

        async def call(name, args):
            result = await client.call_tool(name, args)
            value = result.structured_content
            if not isinstance(value, dict):
                raise AssertionError('缺少 MCP 结构化结果')
            records.append({'tool': name, 'arguments': args, 'is_error': result.is_error, 'result': value})
            print(value['summary'], flush=True)
            if result.is_error or not value.get('ok'):
                raise RuntimeError(value['summary'])
            return value

        quality = await call('assess_image_quality', {'image_path': str(path)})
        original_id = quality['data']['image_id']
        correction = await call('generate_exposure_candidate', {'image_id': original_id, 'method': method})
        candidate_id = correction['data']['image_id']
        after = await call('assess_image_quality', {'image_id': candidate_id})
        raw = await call('detect_objects', {'image_id': original_id})
        fixed = await call('detect_objects', {'image_id': candidate_id})
        repeated = [await call('assess_image_quality', {'image_id': original_id}),
                    await call('generate_exposure_candidate', {'image_id': original_id, 'method': method}),
                    await call('detect_objects', {'image_id': original_id})]
        if not all(r['cached'] for r in repeated) or repeated[1]['data']['image_id'] != candidate_id:
            raise AssertionError('重复调用没有复用缓存')
    direct = direct_results(path, method)
    checks = {
        'original_quality_equal': quality['data']['quality'] == direct['quality_original'],
        'candidate_quality_equal': after['data']['quality'] == direct['quality_candidate'],
        'correction_parameters_equal': correction['data']['parameters'] == direct['parameters'],
        'original_boxes_equal': raw['data']['detections'] == direct['detections_original'],
        'candidate_boxes_equal': fixed['data']['detections'] == direct['detections_candidate'],
        'configuration_equal': quality['data']['configuration_sha256'] == direct['configuration_sha256'],
        'model_equal': raw['data']['model']['sha256'] == direct['model_sha256'],
        'cache_reused': all(r['cached'] for r in repeated),
        'original_file_preserved': before_hash == hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {'app_version': __version__, 'mcp_version': version('mcp'), 'transport': 'stdio',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'initialization': 'legacy_initialize',
              'source': str(path), 'source_sha256': before_hash, 'tools': sorted(names),
              'code_sha256': {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in
                  ['serve_mcp.py', 'demo_mcp.py', 'optical_agent/mcp_adapter.py', 'optical_agent/mcp_server.py',
                   'optical_agent/adaptive_tools.py', 'optical_agent/adaptive_goals.py', 'quality.py', 'detector.py']},
              'checks': checks, 'all_checks_passed': all(checks.values()),
              'paid_api_requests': 0, 'tool_calls': len(records), 'calls': records, 'direct': direct,
              'scope': '真实 MCP 协议及现有视觉工具一致性验证；不代表 LLM 调度或视觉精度提高。'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(redact(record), ensure_ascii=False, indent=2), encoding='utf-8')
    print('检查结果：' + json.dumps(checks, ensure_ascii=False), flush=True)
    print('记录保存到：' + str(output), flush=True)
    print('候选图片：' + correction['data']['image_path'], flush=True)
    if not record['all_checks_passed']:
        raise SystemExit(1)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--method', choices=['gamma_only', 'local_bounded'], default='gamma_only')
    parser.add_argument('--output', type=Path, default=MCP_ROOT / 'runs' / 'demo_v053.json')
    args = parser.parse_args()
    if args.image is None:
        from data import SODDDataset
        dataset = SODDDataset(DATA_ROOT / 'sodd' / 'SODD' / 'data', 'test')
        args.image = dataset.items[6][0]
    if not args.output.resolve().is_relative_to(MCP_ROOT.resolve()):
        parser.error('演示记录必须保存在 D 盘 MCP 目录内。')
    asyncio.run(run_demo(args.image.resolve(), args.output.resolve(), args.method))


if __name__ == '__main__':
    main()
