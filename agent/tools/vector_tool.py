import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections
from langchain_core.tools import tool

# ==============================================================================
# 修复 pymilvus 2.6.x 与 langchain-milvus 0.3.x 之间的兼容性问题
# (MilvusClient 连接不注册到 connections 导致的 ConnectionNotExistException)
# 入库脚本 agent/database/import_airline_policy.py 也通过 import 本模块复用这个补丁。
# ==============================================================================
original_fetch = connections._fetch_handler
def patched_fetch(alias):
    try:
        return original_fetch(alias)
    except Exception:
        from pymilvus.client.connection_manager import ConnectionManager
        mgr = ConnectionManager.get_instance()
        for mc in mgr._registry.values():
            if f"cm-{id(mc.handler)}" == alias:
                return mc.handler
        for mc in mgr._dedicated.values():
            if f"cm-{id(mc.handler)}" == alias:
                return mc.handler
        raise
connections._fetch_handler = patched_fetch
# ==============================================================================

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# ------------------------------------------------------------------------------
# 航空政策知识库的集合配置。入库脚本和检索共用这一份，保证两边一致。
# COSINE：Milvus 返回的分数是余弦相似度，越大越相关（不是距离）。
# ------------------------------------------------------------------------------
COLLECTION_NAME = "airline_policy_docs"
EMBEDDING_MODEL = "text-embedding-v2"          # 1536 维
INDEX_PARAMS = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}
TOP_K = 3

_milvus_instance = None


def get_embeddings():
    return DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model=EMBEDDING_MODEL)


def get_milvus_uri():
    return f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"


def _get_milvus_store():
    global _milvus_instance
    if _milvus_instance is not None:
        return _milvus_instance

    milvus_uri = get_milvus_uri()
    print(f"[Init] 正在连接 Milvus 向量数据库: {milvus_uri}")
    _milvus_instance = Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": milvus_uri},
        collection_name=COLLECTION_NAME,
        index_params=INDEX_PARAMS,
        search_params=SEARCH_PARAMS,
        auto_id=True,
        drop_old=False,
    )
    return _milvus_instance


@tool
def query_vector_db(query: str) -> str:
    """
    检索航空公司的政策文档（RAG）。
    当用户询问规则类问题时使用，例如：托运和随身行李规定、旅行保险、改签和退票条件、
    航班取消或延误的补偿、婴儿儿童乘机、宠物、特殊协助、值机登机、会员权益、支付方式等。
    不用于查询具体的航班、订单或用户信息（这些用对应的业务工具查询）。
    """
    try:
        store = _get_milvus_store()
        results = store.similarity_search_with_score(query, k=TOP_K)

        if not results:
            return "未在政策文档中检索到相关信息。"

        formatted_results = []
        for doc, score in results:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            tag = "核心政策" if doc.metadata.get("category") == "core" else "仿真扩充政策"
            formatted_results.append(f"【来源: {source}（{tag}）】\n{doc.page_content.strip()}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"查询向量数据库时发生错误: {str(e)}"
