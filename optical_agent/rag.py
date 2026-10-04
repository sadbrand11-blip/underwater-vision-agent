"""Local, source-labelled TF-IDF retrieval; no embedding download."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeRetriever:
    def __init__(self, root=None, chunk_size=400, overlap=80, min_score=0.10):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.min_score = min_score
        manifest = json.loads((self.root / 'knowledge' / 'sources.json').read_text(encoding='utf-8'))
        self.chunks = []
        for source in manifest:
            text = (self.root / source['path']).read_text(encoding='utf-8')
            for index, start in enumerate(range(0, len(text), chunk_size - overlap)):
                chunk = text[start:start + chunk_size].strip()
                if not chunk:
                    continue
                self.chunks.append({'citation_id': f'{source["id"]}:{index}', 'title': source['title'],
                    'source': source['url'], 'path': source['path'], 'verified_at': source['verified_at'],
                    'text': chunk})
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c['title'] + '\n' + c['text'] for c in self.chunks])

    def search(self, query, top_k=3):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [dict(self.chunks[i], score=round(float(scores[i]), 4))
                for i in ranked[:top_k] if scores[i] >= self.min_score]
