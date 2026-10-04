import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from optical_agent.rag_engine import LocalEncoder, RagRetriever, RagUnavailable, build_chunks, split_block
from optical_agent.rag_eval import score_question, summarize, paired_interval, adoption_gate, validate_labels


class FakeEncoder:
    model_hash='fake-weights'
    asset_hash='fake-assets'
    device='cpu'
    calls=0
    def encode(self,texts,query=False):
        self.calls+=1
        x=np.zeros((len(texts),512),np.float32)
        for i,t in enumerate(texts):
            x[i,0 if '曝光' in t else 1]=1
        return x


def chunks():
    return [{'citation_id':'a:1','source_id':'a','title':'曝光','text':'曝光不能恢复丢失纹理。','vision_modes':['legacy']},
            {'citation_id':'b:1','source_id':'b','title':'机器人','text':'机器人模型是实验能力。','vision_modes':['experiment_candidate']}]


class RagTests(unittest.TestCase):
    def test_group_recall_deduplicates_overlap_and_mrr_hand_calculation(self):
        gold={'x':{'alternatives':[{'source_id':'a','quote':'曝光'}]},
              'y':{'alternatives':[{'source_id':'b','quote':'机器人'}]}}
        q={'required_groups':['x','y']}
        ranking=[chunks()[1],chunks()[1],chunks()[0]]
        s=score_question(q,ranking,gold)
        self.assertEqual(s['recall']['1'],.5)
        self.assertEqual(s['recall']['3'],1)
        self.assertEqual(s['mrr10'],1)
        self.assertTrue(s['full']['3'])
        self.assertEqual(score_question(q,[],gold)['recall']['10'],0)

    def test_no_answer_excluded_from_recall_denominator(self):
        gold={'x':{'alternatives':[{'source_id':'a','quote':'曝光'}]}}
        rows=[{'kind':'existing','delivered':score_question({'required_groups':['x']},chunks()[:1],gold)},
              {'kind':'unanswerable','delivered':score_question({'required_groups':[]},chunks(),gold)}]
        s=summarize(rows)
        self.assertEqual(s['answerable_n'],1)
        self.assertEqual(s['recall']['3'],1)
        self.assertEqual(s['no_answer_false_return'],1)

    def test_tables_and_source_offsets_preserved_no_labels_indexed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'knowledge').mkdir()
            text='# Heading\n\n'+('曝光校正应保留原图。'*60)+'\n\n| a | b |\n|---|---|\n| 1 | 2 |'
            (root/'doc.md').write_text(text,encoding='utf8')
            source={'id':'doc','path':'doc.md','title':'标题','url':'https://example.org',
                    'verified_at':'2026-10-03','source_type':'project','vision_modes':['legacy']}
            (root/'knowledge/sources_v2.json').write_text(json.dumps([source]),encoding='utf8')
            (root/'labels.json').write_text('SECRET LABEL',encoding='utf8')
            result=build_chunks(root)
            for c in result:
                self.assertEqual(text[c['source_start']:c['source_end']],c['text'])
                self.assertLessEqual(len(c['text']),400)
                self.assertNotIn('SECRET',c['text'])
            self.assertTrue(any('| 1 | 2 |' in c['text'] for c in result))
        with self.assertRaises(ValueError): list(split_block('|'+'x'*410))

    def test_scope_threshold_cache_isolation_and_scores_not_probabilities(self):
        a=RagRetriever(chunks(),vision_mode='legacy',thresholds={'tfidf':.01,'dense':.75})
        b=RagRetriever(chunks(),vision_mode='experiment_candidate',thresholds={'tfidf':.01,'dense':.75})
        self.assertTrue(all(c['source_id']=='a' for c in a.search('曝光')))
        self.assertTrue(all(c['source_id']=='b' for c in b.rank('曝光机器人',filtered=False)))
        self.assertIsNot(a.query_cache,b.query_cache)
        self.assertFalse(a.search('曝光')[0]['score_is_probability'])
        self.assertEqual(a.search('ΩΩΩ'),[])

    def test_model_missing_and_corrupt_index_are_not_empty_results(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RagUnavailable): LocalEncoder(Path(d))
            enc=FakeEncoder()
            r=RagRetriever(chunks(),'embedding',encoder=enc,cache_root=d,build=True)
            self.assertTrue(r.search('曝光'))
            metadata=next(Path(d).rglob('index.json'))
            metadata.write_text('{}')
            with self.assertRaises(RagUnavailable):
                RagRetriever(chunks(),'embedding',encoder=enc,cache_root=d)

    def test_dense_rrf_and_query_cache(self):
        with tempfile.TemporaryDirectory() as d:
            enc=FakeEncoder()
            r=RagRetriever(chunks(),'hybrid',encoder=enc,cache_root=d,build=True,
                           thresholds={'tfidf':.01,'dense':.75})
            ranked=r.search('曝光')
            self.assertEqual(ranked[0]['citation_id'],'a:1')
            self.assertAlmostEqual(ranked[0]['score'],2/61,places=6)
            calls=enc.calls; r.search('曝光'); self.assertEqual(enc.calls,calls)
            r.thresholds={'tfidf':1.1,'dense':1.1}; self.assertEqual(r.search('曝光'),[])

    def test_freeze_dataset_and_selection_gates(self):
        from optical_agent.rag_engine import DATA_ROOT
        repo=Path(__file__).resolve().parents[1]
        dataset=json.loads((repo/'eval/rag_questions.json').read_text(encoding='utf8'))
        if (DATA_ROOT/'c1_source').exists(): self.assertTrue(validate_labels(dataset,DATA_ROOT/'c1_source'))
        duplicate=copy.deepcopy(dataset);duplicate['questions'][0]['id']=duplicate['questions'][1]['id']
        with self.assertRaises(ValueError): validate_labels(duplicate,repo)
        s={'recall':{'3':.8},'mrr10':.9,'no_answer_false_return':.1}
        self.assertFalse(adoption_gate(s,s,50)['recall3_gain'])
        rows=[{'id':'q','kind':'existing','fact_group':'f','delivered':{'recall':{'3':.8},'mrr10':.9}}]
        self.assertEqual(paired_interval(rows,rows)['ci95'],[0,0])

    def test_checkout_newlines_reconstruct_release_bytes(self):
        import subprocess
        from prepare_rag_corpus import render
        repo=Path(__file__).resolve().parents[1]
        layout=json.loads((repo/'knowledge/corpus_layout.json').read_text(encoding='utf8'))
        baseline=subprocess.run(['git','cat-file','-e',layout['base_commit']],cwd=repo,capture_output=True)
        if baseline.returncode:
            self.skipTest('Private historical baseline is deliberately excluded from the sanitized Git history.')
        for path,spec in layout['c0'].items():
            raw=subprocess.run(['git','show',layout['base_commit']+':'+path],cwd=repo,capture_output=True,check=True).stdout
            self.assertEqual(render(raw,spec),render(raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'),spec))

    def test_delivery_warning_does_not_change_rank_or_measurement(self):
        from optical_agent.rag_delivery import KnowledgeView
        r=RagRetriever([dict(chunks()[0],source_id='readme',citation_id='readme:1')],thresholds={'tfidf':.01,'dense':.6})
        delivered=KnowledgeView(r,'experiment_candidate').search('曝光')
        self.assertIn('冻结README',delivered[0]['version_warning'])
        for key,value in r.search('曝光')[0].items():self.assertEqual(delivered[0][key],value)


if __name__=='__main__': unittest.main()
