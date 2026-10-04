"""Presentation context only; leaves evaluated ranking, chunks and scores unchanged."""


class KnowledgeView:
    def __init__(self,retriever,vision_mode):
        self.retriever=retriever
        self.vision_mode=vision_mode

    def __getattr__(self,name):return getattr(self.retriever,name)

    def metadata(self):
        public=self.retriever.metadata().get('corpus','').startswith('public')
        return dict(self.retriever.metadata(),delivery_revision='v050_rag_delivery_r3',
                    historical_readme_warning=not public, public_corpus=public)

    def search(self,query,top_k=3):
        result=[]
        for original in self.retriever.search(query,top_k):
            item=dict(original)
            if item.get('source_id',item['citation_id'].split(':')[0])=='readme' and not self.metadata()['public_corpus']:
                item['version_warning']='冻结README属于历史实验语料；论文或实验模型不能表述为当前默认能力。'
            item['active_vision_mode']=self.vision_mode
            result.append(item)
        return result
