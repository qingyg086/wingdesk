"""Versioned policy index, native Chinese BM25 + dense retrieval and optional reranking."""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / 'agent' / '.env')
MODEL = 'text-embedding-v2'
DIM = 1536
DENSE_INDEX = {'index_type': 'HNSW', 'metric_type': 'COSINE', 'params': {'M': 16, 'efConstruction': 200}}
DENSE_SEARCH = {'metric_type': 'COSINE', 'params': {'ef': 64}}
ANALYZER = {'tokenizer': 'jieba', 'filter': ['lowercase']}
CANDIDATES = 10
RRF_K = 60
OUTPUT_FIELDS = ['text', 'source', 'section', 'title', 'category']
MODES = ('vector', 'bm25', 'hybrid', 'hybrid_rerank')


def corpus():
    # One splitter shared with the existing importer; no policy text is rewritten.
    from database.import_airline_policy import load_and_split
    chunks = load_and_split()
    rows = []
    for doc in chunks:
        identity = json.dumps([doc.metadata['source'], doc.metadata['section'], doc.page_content], ensure_ascii=False)
        rows.append({'id': hashlib.sha256(identity.encode()).hexdigest(), 'text': doc.page_content, **doc.metadata})
    fingerprint = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return rows, fingerprint


def collection_name(fingerprint):
    return f'airline_policy_hybrid_v1_{fingerprint[:12]}'


def client():
    return MilvusClient(uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}",
                        token=os.getenv('MILVUS_API_KEY', ''), timeout=10)


def embeddings():
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(model=MODEL, dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))


def build_index(db, rows, fingerprint, name=None):
    """Create a new versioned collection. Never drop the existing policy or cache collections."""
    name = name or collection_name(fingerprint)
    description = f'wingdesk-policy-v1:{fingerprint}'
    if db.has_collection(name):
        if db.describe_collection(name)['description'] != description:
            raise ValueError('policy collection belongs to another corpus or schema')
        count = db.query(name, filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
        if count != len(rows):
            raise ValueError(f'incomplete index: {count}/{len(rows)}; remove this incomplete collection before retrying')
        db.load_collection(name)
        return name
    # Compute vectors before creating the collection, so provider failure leaves no half-built index.
    vectors = embeddings().embed_documents([row['text'] for row in rows])
    if len(vectors) != len(rows) or any(len(vec) != DIM for vec in vectors):
        raise ValueError('embedding count or dimensions do not match the policy corpus')
    schema = db.create_schema(auto_id=False, enable_dynamic_field=False, description=description)
    schema.add_field('id', DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field('text', DataType.VARCHAR, max_length=8192, enable_analyzer=True, analyzer_params=ANALYZER)
    for field in OUTPUT_FIELDS[1:]:
        schema.add_field(field, DataType.VARCHAR, max_length=512)
    schema.add_field('dense', DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field('sparse', DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name='policy_bm25', input_field_names=['text'], output_field_names=['sparse'],
                                 function_type=FunctionType.BM25))
    index = db.prepare_index_params()
    index.add_index(field_name='dense', **DENSE_INDEX)
    index.add_index(field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25',
                    params={'inverted_index_algo': 'DAAT_MAXSCORE', 'bm25_k1': 1.2, 'bm25_b': 0.75})
    db.create_collection(name, schema=schema, index_params=index, description=description,
                         consistency_level='Strong')
    db.insert(name, data=[dict(row, dense=vec) for row, vec in zip(rows, vectors)])
    db.flush(name)
    count = db.query(name, filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
    if count != len(rows):
        raise ValueError(f'incomplete index after insert: {count}/{len(rows)}')
    db.load_collection(name)
    if db.describe_collection(name)['description'] != description:
        raise ValueError('policy index version was not persisted in the schema')
    return name


class PolicyRetriever:
    def __init__(self, db, collection, embedder=None, reranker=None):
        self.db, self.collection = db, collection
        self.embedder = embedder or embeddings()
        self.reranker = reranker or rerank

    def search(self, query, mode='hybrid', limit=3):
        if mode not in MODES:
            raise ValueError(f'unknown policy retrieval mode: {mode}')
        if not query.strip():
            return []
        if mode == 'bm25':
            hits = self.db.search(self.collection, data=[query], anns_field='sparse',
                                 search_params={'metric_type': 'BM25'}, limit=limit, output_fields=OUTPUT_FIELDS)[0]
        else:
            vec = self.embedder.embed_query(query)
            if mode == 'vector':
                hits = self.db.search(self.collection, data=[vec], anns_field='dense', search_params=DENSE_SEARCH,
                                      limit=limit, output_fields=OUTPUT_FIELDS)[0]
            else:
                requests = [AnnSearchRequest(data=[vec], anns_field='dense', param=DENSE_SEARCH, limit=CANDIDATES),
                            AnnSearchRequest(data=[query], anns_field='sparse', param={'metric_type': 'BM25'}, limit=CANDIDATES)]
                hits = self.db.hybrid_search(self.collection, reqs=requests, ranker=RRFRanker(k=RRF_K),
                                            limit=CANDIDATES if mode == 'hybrid_rerank' else limit,
                                            output_fields=OUTPUT_FIELDS)[0]
        results = [dict(hit['entity'], id=str(hit['id']), score=float(hit['distance'])) for hit in hits]
        return self.reranker(query, results, limit) if mode == 'hybrid_rerank' and results else results


def rerank(query, candidates, limit):
    import dashscope
    response = dashscope.TextReRank.call(model=os.getenv('POLICY_RERANK_MODEL', 'qwen3-rerank'), query=query,
        documents=[row['text'] for row in candidates], top_n=limit, return_documents=False,
        api_key=os.getenv('DASHSCOPE_API_KEY'), timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f'rerank unavailable: {response.status_code}')
    ranked = response.output['results']
    indices = [row['index'] for row in ranked]
    if len(indices) != min(limit, len(candidates)) or len(set(indices)) != len(indices) or any(
            not isinstance(i, int) or i < 0 or i >= len(candidates) for i in indices):
        raise ValueError('invalid reranker result indices')
    if any(not math.isfinite(float(row['relevance_score'])) for row in ranked):
        raise ValueError('non-finite reranker score')
    return [dict(candidates[row['index']], rerank_score=float(row['relevance_score'])) for row in ranked]


@lru_cache(maxsize=1)
def get_retriever():
    rows, fingerprint = corpus()
    name = collection_name(fingerprint)
    db = client()
    if not db.has_collection(name):
        db.close()
        raise RuntimeError('policy search index not imported')
    if db.describe_collection(name)['description'] != f'wingdesk-policy-v1:{fingerprint}':
        db.close()
        raise RuntimeError('policy search index version mismatch')
    count = db.query(name, filter='', output_fields=['count(*)'], consistency_level='Strong')[0]['count(*)']
    if count != len(rows):
        db.close()
        raise RuntimeError('policy search index incomplete')
    return PolicyRetriever(db, name)
