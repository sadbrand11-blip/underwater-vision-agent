import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app as web
from optical_agent.rag_engine import RagUnavailable
from optical_agent.state import SessionStore


class StubRetriever:
    def __init__(self,mode,vision):self.mode=mode;self.vision=vision
    def metadata(self):return {'mode':self.mode,'corpus':'C1','corpus_hash':'fixed','vision_mode':self.vision}
    def search(self,query,top_k=3):return [{'citation_id':'real:1','title':'依据','text':'项目限制','source':'https://example.org','score':.5}]


class RagWebTests(unittest.TestCase):
    def setUp(self):self.client=web.app.test_client()

    def test_comparison_without_image_and_unavailable_is_distinct(self):
        def factory(mode,vision):
            if mode=='embedding':raise RagUnavailable('模型缺失')
            return StubRetriever(mode,vision)
        with patch.object(web,'get_retriever',side_effect=factory):
            result=self.client.post('/api/knowledge/search',json={'query':'为什么框变多不能说更准确？'}).get_json()
        self.assertEqual(result['paid_api_requests'],0)
        self.assertTrue(result['comparisons']['tfidf']['available'])
        self.assertFalse(result['comparisons']['embedding']['available'])
        self.assertIsNone(result['comparisons']['embedding']['results'])
        self.assertEqual(self.client.get('/knowledge').status_code,200)

    def test_search_validation_and_local_origin(self):
        for body in ({'query':''},{'query':'x','top_k':True},{'query':'x','top_k':11},
                     {'query':'x','modes':['unknown']},{'query':'x','vision_mode':'unknown'}):
            self.assertEqual(self.client.post('/api/knowledge/search',json=body).status_code,400)
        self.assertEqual(self.client.post('/api/knowledge/search',json={'query':'曝光'},
                        headers={'Origin':'https://other.example'}).status_code,403)

    def test_release_update_refreshes_new_retrievers_but_old_stays_pinned(self):
        config={'mode':'tfidf','thresholds':{'tfidf':.1,'dense':.6}}
        with patch.object(web,'_retriever',None),patch.object(web,'_rag_retrievers',{}),\
             patch.object(web,'release_config',side_effect=lambda root:dict(config)),\
             patch.object(web,'build_chunks',return_value=[{'citation_id':'x','text':'a'}]),\
             patch.object(web,'create_retriever',side_effect=lambda root,mode,vision,**kw:StubRetriever(mode or 'tfidf',vision)):
            old=web.get_retriever()
            self.assertIs(old,web.get_retriever())
            config['thresholds']={'tfidf':.2,'dense':.6}
            new=web.get_retriever()
            self.assertIsNot(old,new)
            self.assertEqual(old.metadata()['corpus_hash'],'fixed')

    def test_session_pins_mode_version_and_visual_scope(self):
        ok,encoded=cv2.imencode('.png',np.full((32,32,3),100,np.uint8));self.assertTrue(ok)
        store=SessionStore()
        with patch.object(web,'_sessions',store),patch.object(web,'get_vision_agent',return_value=SimpleNamespace(detector=object())),\
             patch.object(web,'get_retriever',side_effect=lambda mode,vision:StubRetriever(mode,vision)):
            r=self.client.post('/api/sessions',data={'image':(io.BytesIO(encoded.tobytes()),'image.png'),
                       'rag_mode':'hybrid','vision_mode':'experiment_candidate'})
            self.assertEqual(r.status_code,201)
            session=store.get(r.get_json()['session_id'])
            self.assertEqual(session.rag_metadata['mode'],'hybrid')
            self.assertEqual(session.context.retriever.vision,'experiment_candidate')
            self.assertEqual(session.rag_metadata['corpus_hash'],'fixed')


if __name__=='__main__':unittest.main()
