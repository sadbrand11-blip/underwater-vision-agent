"""Publish summaries and adoption decision from the complete frozen experiment."""
import csv
import json
from pathlib import Path
import shutil

from optical_agent.rag_engine import DATA_ROOT, MODEL_REVISION, file_hash
from optical_agent.rag_eval import adoption_gate, paired_interval, summarize, verify_freeze, KS

REPO=Path(__file__).parent
RUN=DATA_ROOT/'runs/rag_v050_20261003'


def main():
    archived=(RUN/'frozen_code').exists()
    verify_freeze(REPO,RUN,archived=archived)
    frozen=json.loads((RUN/'freeze.json').read_text(encoding='utf8'))
    before=dict(frozen);before.pop('selection_sha256',None)
    preselection_bytes=(json.dumps(before,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    import hashlib
    dev=json.loads((RUN/'dev.json').read_text(encoding='utf8'))
    if hashlib.sha256(preselection_bytes).hexdigest()!=dev['freeze_sha256']:
        preselection_bytes=preselection_bytes.replace(b'\n',b'\r\n')
    if hashlib.sha256(preselection_bytes).hexdigest()!=dev['freeze_sha256']:
        raise ValueError('Pre-selection freeze cannot be reconstructed exactly')
    test=json.loads((RUN/'test.json').read_text(encoding='utf8'))
    selection=json.loads((RUN/'selection.json').read_text(encoding='utf8'))
    results=test['results']; candidate=selection['candidate']
    for key, value in results.items():
        expected=50 if key.startswith('C1/') else 40
        if len(value['records'])!=expected or len({r['id'] for r in value['records']})!=expected:
            raise ValueError('Incomplete or duplicate result records')
    chosen=results['C1/'+candidate]; reference=results['C1/tfidf']; old=results['C0/tfidf']
    gates=adoption_gate(chosen['delivered'],reference['delivered'],chosen['latency']['cpu_warm_p95_ms'])
    a,b=old['old_questions'],chosen['old_questions']
    knowledge_gate=all(b['recall'][str(k)]>=a['recall'][str(k)]-1e-12 for k in KS) and b['mrr10']>=a['mrr10']-1e-12
    release_mode=candidate if all(gates.values()) and knowledge_gate else 'legacy'
    config={'version':'0.5.0','mode':release_mode,'corpus_hash':chosen['metadata']['corpus_hash'],
        'model_revision':MODEL_REVISION,'thresholds':chosen['metadata']['thresholds'],
        'profiles':{m:selection['profiles']['C1/'+m]['thresholds'] for m in ('tfidf','embedding','hybrid')},
        'run_id':RUN.name,'adoption_gates':gates,'knowledge_replacement_gate':knowledge_gate,
        'legacy_corpus':'frozen_C0','holdout_tuning':False}
    intervals={}
    for mode in ('embedding','hybrid'):
        intervals['C1_'+mode+'_vs_tfidf']={metric:paired_interval(reference['records'],results['C1/'+mode]['records'],metric)
            for metric in ('recall3','mrr10')}
    for mode in ('tfidf','embedding','hybrid'):
        left=[r for r in results['C0_struct/'+mode]['records'] if r['kind']=='existing']
        right=[r for r in results['C1/'+mode]['records'] if r['kind']=='existing']
        intervals['knowledge_only_'+mode]=paired_interval(left,right)
        left=[r for r in results['C0/'+mode]['records'] if r['kind']=='existing']
        intervals['combined_'+mode]=paired_interval(left,right)
    summary={'version':'0.5.0','run_id':RUN.name,'evaluation_status':'complete',
        'release':config,'metrics':{k:{f:v[f] for f in ('delivered','raw','old_questions','literature','latency',
           'index_build_ms','load_ms','process_rss_mb','matrix_bytes','vector_bytes','metadata')} for k,v in results.items()},
        'development':{k:{'delivered':v['delivered'],'thresholds':v['metadata']['thresholds']} for k,v in dev['results'].items()},
        'paired_intervals':intervals,'http_requests_during_evaluation':0,'paid_api_requests':0,
        'labels':'automatic_source_anchored_independently_reviewed_no_external_human_acceptance',
        'independent_review':'same-family/provisional','freeze_sha256':file_hash(RUN/'freeze.json'),
        'delivery_revision':'v050_rag_delivery_r3','evaluated_source':'D-drive frozen_code archive',
        'original_freeze_limitations':['indexes not anchored before evaluation','publisher not frozen before evaluation'],
        'results_sha256':{'dev':file_hash(RUN/'dev.json'),'test':file_hash(RUN/'test.json'),'selection':file_hash(RUN/'selection.json')},
        'limitations':['small automatic question set','not LLM answer quality','not visual precision',
            'negative return is strict abstention proxy, including valid limitation passages',
            'process RSS includes shared encoder even for TF-IDF','latency repeated five times, not independent tasks']}
    (REPO/'RAG_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (REPO/'RAG_CASES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['split','corpus','mode','question_id','fact_group','kind','query','recall_at3','mrr_at10',
                'full_at3','false_return','missing_at10','failure_class','delivered_citations','raw_citations']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for split,payload in (('dev',dev),('test',test)):
            for key,v in payload['results'].items():
                corpus,mode=key.split('/')
                for r in v['records']:
                    writer.writerow({'split':split,'corpus':corpus,'mode':mode,'question_id':r['id'],'fact_group':r['fact_group'],
                        'kind':r['kind'],'query':r['query'],'recall_at3':r['delivered']['recall']['3'],
                        'mrr_at10':r['delivered']['mrr10'],'full_at3':r['delivered']['full']['3'],
                        'false_return':r['delivered']['false_return'],'missing_at10':'|'.join(r['delivered']['missing_at10']),
                        'failure_class':r['failure_class'],'delivered_citations':'|'.join(c['citation_id'] for c in r['delivered_ranking'][:10]),
                        'raw_citations':'|'.join(c['citation_id'] for c in r['raw_ranking'][:10])})
    lines=['# 0.5.0 本地 RAG 对照实验','', '日期：2026-10-03。本轮没有付费 API 调用，没有重新训练视觉模型。',
        '',f'## 采用结论\n\n开发集先选出 **{candidate}**，保留集按预设门槛确认；发布默认为 **{release_mode}**。',
        f'门槛：{gates}；旧问题知识库替换门槛：{knowledge_gate}。没有用保留集继续调参。',
        '', '## 实际交付排名（保留集）','',
        '| 语料 / 方法 | 可答 / 无答 | Recall@1 | @3 | @5 | @10 | MRR@10 | 无答错误返回 | CPU热P95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,v in results.items():
        s=v['delivered']
        lines.append(f"|{key}|{s['answerable_n']}/{s['negative_n']}|"+'|'.join(f"{s['recall'][str(k)]:.1%}" for k in KS)+
                     f"|{s['mrr10']:.4f}|{s['no_answer_false_return']:.1%}|{v['latency']['cpu_warm_p95_ms']:.1f}|")
    lines+=['','C0与C0_struct只有旧问题：每集30可答＋10无答。C1每集40可答＋10无答，包含10道新文献问题，不能混用分母比较总成绩。',
        '', '## 未过滤排名（保留集）','','| 语料 / 方法 | Recall@1 | @3 | @5 | @10 | MRR@10 |','|---|---:|---:|---:|---:|---:|']
    for key,v in results.items():
        s=v['raw'];lines.append('|'+key+'|'+'|'.join(f"{s['recall'][str(k)]:.1%}" for k in KS)+f"|{s['mrr10']:.4f}|")
    lines+=['','本轮主要收益在**带拒绝阈值的实际交付**，未过滤的C1 Hybrid和TF-IDF Recall@3相同，不能概括为语义排序全面优于词语匹配。拒绝保护会漏掉许多可答问题。',
        '', '## 旧问题：算法与知识补充分开','', '| 方法 | C0 R@3 / MRR | C0_struct R@3 / MRR | C1旧题 R@3 / MRR | 知识单独增益 |',
        '|---|---:|---:|---:|---:|']
    for mode in ('tfidf','embedding','hybrid'):
        cells=[]
        for corpus in ('C0','C0_struct','C1'):
            s=results[corpus+'/'+mode]['old_questions'];cells.append(f"{s['recall']['3']:.1%} / {s['mrr10']:.4f}")
        delta=intervals['knowledge_only_'+mode]['delta']
        lines.append('|'+mode+'|'+'|'.join(cells)+f'|{delta:+.1%}|')
    lines+=['','C0保留原8份资料、55个400/80字符片段；C0_struct是同8份资料、96个结构化片段＋版本过滤；C1加6张经核实知识卡，共126片段。',
        'C0→C1同时改变分段和资料过滤，不能全部归因于补充论文。C0_struct→C1的旧题Recall@3增益为0；新文献问题单列。阈值在各语料开发集校准，属于校准后系统对照。',
        '', '## 配对不确定性','', '| 对照 | Δ Recall@3 | 95%事实组bootstrap区间 |','|---|---:|---:|']
    for mode in ('embedding','hybrid'):
        ci=intervals['C1_'+mode+'_vs_tfidf']['recall3'];lines.append(f"|C1 {mode} − TF-IDF|{ci['delta']:+.1%}|{ci['ci95'][0]:+.1%} ～ {ci['ci95'][1]:+.1%}|")
    lines+=['','事实组配对重采样2000次、种子20261003；同事实改写一起采样。样本较小，门槛通过不等于显著性或泛化保证。',
        '', '## 指标、时间与资源','',
        '- Recall按必要证据组的命中比例逐题平均，重叠片段/等价来源不重复计分；MRR为首个相关结果倒数。跨来源问题另外报告完整命中率。',
        '- 无答题不进入Recall/MRR分母。“任何返回算错误”是严格拒绝代理指标；返回“缺少深度”限制片段也算返回，不能据此说系统幻觉或回答错误。未评测LLM引用事实性。',
        '- 每题热查询重复5次，清空查询缓存，包含CPU编码和排序；模型加载、建索引另列。另报缓存命中时间。重复耗时不增加独立题数。',
        '- 建索引使用RTX3060 Laptop，查询固定CPU/2线程。进程RSS包含共享Embedding模型，TF-IDF行不是其单独进程内存；另报稀疏矩阵和向量实际字节。',
        '', '| 方法(C1) | 完整证据@3 | 平均 / P95 ms | 缓存P95 ms | GPU建索引 ms | 向量 KiB | 进程RSS MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('tfidf','embedding','hybrid'):
        v=results['C1/'+mode];s=v['delivered'];l=v['latency'];lines.append(f"|{mode}|{s['full_hit']['3']:.1%}|{l['cpu_warm_mean_ms']:.1f} / {l['cpu_warm_p95_ms']:.1f}|{l['cached_p95_ms']:.1f}|{v['index_build_ms']:.1f}|{v['vector_bytes']/1024:.1f}|{v['process_rss_mb']:.1f}|")
    lines+=['','TF-IDF建索引在CPU；Hybrid复用同一Dense向量索引，其建索引时间不是独立重复。模型权重约95.8MB十进制，未升级Torch。',
        '', '## 来源、审核与冻结','',
        '100道自动构造问题：开发/保留各30旧资料可答＋10新文献可答＋10无答。41证据组、60个来源原句，原句/位置先独立审查后冻结。无外部人工验收；同模型家族审核为暂定意见。',
        '原标签审核FAIL记录保留，补齐等价出处、收窄问法、修正每次一种增强器的论文描述后通过标签门槛。完整脚本、JSON、语料、模型资产和论文SHA256已冻结；construction_sha256单独字段不足以覆盖生成逻辑。',
        f'模型：BAAI/bge-small-zh-v1.5，固定revision `{MODEL_REVISION}`，CLS池化、归一化、官方查询指令，最长512 tokens、不静默截断。',
        '下载只使用作者/官方入口。assets_all保留UIEB最初失败记录，assets_papers是后续成功清单；不能把初始失败删除为从未发生。独立审核未能重下载全部远端字节，本地SHA与清单一致。',
        '模型卡和实验报告按文件版本过滤；混合README仍含历史默认能力说明，尚未完成逐片段作用域隔离。交付层额外标注历史版本警告，不改变已评测排序。论文不是Agent已实现功能。模型/索引缺失显示不可用；只有默认入口允许显式提示后降级TF-IDF。',
        '审计发现干净Git字节与原快照存在混合换行差异，交付新增逐行换行策略并在D盘隔离目录真实重建，C0全部55片段及C1发布指纹一致。会话选择器现在锁定，发布/语料/模型身份进入服务缓存键；已有会话保持固定。',
        '原冻结未覆盖索引和发布程序：当前真实索引经独立编码核对，后补哈希不称为预冻结。交付已加强后续freeze覆盖与发布前验证。原评测代码保存在frozen_code；交付修复未重跑保留集或改变成绩。开发阶段初始freeze摘要按原字节重建并匹配dev记录，明确命名为reconstructed，不冒称事前保存副本。',
        '', '## 查看与复现','',
        '- [操作与优化思路](docs/11_RAG_OPTIMIZATION.md)；网页 `/knowledge` 无需图片对照三种检索，`/agent` 创建会话前选择检索方式，创建后固定。',
        '- [机器摘要](RAG_RESULTS.json)、[逐题缺项和返回片段](RAG_CASES.csv)、[实验审计](RAG_EXPERIMENT_AUDIT.md)。CSV是检索证据命中诊断，不是虚构LLM引用统计。',
        f'- 本地原始记录：`{RUN}`，包括dev/test逐题排名、阈值网格、freeze及选择文件；公开仓库不包含模型下载物、索引或原始运行日志。',
        '- 首次准备：`python prepare_rag.py --assets all`；`python evaluate_rag.py snapshot` 从固定旧提交重建语料；`python evaluate_rag.py prepare` 建GPU索引。已有快照不会被覆盖。',
        '- 复测用新run-id，依次执行 `freeze`、`dev`、`test`。旧结果不能覆盖；代码/标签/语料/模型变化会停止。',
        '', '## 剩余不足','',
        '实际交付仍漏掉一半左右可答证据。领域内未知题与领域外题各占一半，题集仅40个主事实家庭，不是通用中文RAG基准。下一步应在新的开发题上研究证据充分性与重排，再使用新保留题验收。',
        '旧视觉成绩、0.2.4正式Agent Eval保持原样。本轮仅说明检索效果，不等于LLM回答质量或识别精度提高。',
        '', '[BGE官方模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)；[RRF原文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)。六份知识来源见 knowledge/sources_v2.json。']
    (REPO/'RAG_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # Preserve executable inputs and resolve dev's pre-selection hash explicitly.
    p=RUN/'freeze_before_selection_reconstructed.json'
    p.write_bytes(preselection_bytes)
    if file_hash(p)!=dev['freeze_sha256']:raise ValueError('Pre-selection freeze cannot be reconstructed exactly')
    if not archived:
        for path,digest in frozen['files'].items():
            path=Path(path)
            if path.is_relative_to(REPO):
                target=RUN/'frozen_code'/path.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
                if file_hash(target)!=digest:raise ValueError('Archived source mismatch')
    # Release adoption is the last write, after all evidence and archival checks.
    (REPO/'knowledge/rag_release.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Published',release_mode,'gates',gates,'old-question gate',knowledge_gate)


if __name__=='__main__':main()
