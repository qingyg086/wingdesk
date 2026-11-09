"""Policy evidence lookup used by the policy specialist."""
import json
import logging
import os

from langchain_core.tools import tool
from core.policy_retrieval import get_retriever

log = logging.getLogger(__name__)


def evidence_payload(hits, *, degraded=False):
    if not hits:
        return {'status': 'no_evidence', 'sources': [],
                'guidance': '没有找到可用政策资料。若问题对象不明确，请只追问必要信息；否则说明当前资料无法确认。不得编造政策。'}
    return {'status': 'candidates', 'degraded': degraded,
            'guidance': '以下是候选资料，不保证包含答案。逐项核对问题与条款；相似主题不等于适用依据。缺少条件先追问，未写明的价格、许可或例外请明确说无法确认。',
            'sources': [{'citation': f"{h['source']}#{h['section']}", 'title': h['title'],
                         'section': h['section'], 'category': '核心政策' if h['category']=='core' else '仿真扩充政策',
                         'text': h['text']} for h in hits]}


@tool
def query_vector_db(query: str) -> str:
    """检索航空政策证据（行李、退改、保险、补偿、宠物、值机等）。
    返回带来源和小节的候选资料；必须确认条款直接回答了问题，不能把相关性当作事实依据。
    不能查询个人订单、退款进度或实时航班。资料不足时应追问必要条件或说明无法确认。
    """
    mode = os.getenv('POLICY_RETRIEVAL_MODE', 'hybrid_rerank').strip().lower()
    try:
        retriever = get_retriever()
        degraded = False
        try:
            hits = retriever.search(query, mode, limit=3)
        except Exception:
            if mode != 'hybrid_rerank':
                raise
            # Dev evaluation favored dense over unreranked fusion. Preserve that fallback.
            log.warning('Policy reranking path failed; falling back to dense retrieval')
            hits = retriever.search(query, 'vector', limit=3)
            degraded = True
        return json.dumps(evidence_payload(hits, degraded=degraded), ensure_ascii=False)
    except Exception as exc:
        log.warning('Policy retrieval unavailable (%s)', type(exc).__name__)
        return json.dumps({'status': 'unavailable', 'sources': [],
                           'guidance': '政策查询暂时不可用，暂时无法核实。请简短说明并建议稍后重试；不要将系统故障说成不存在相关政策，也不要凭记忆补答。'}, ensure_ascii=False)
