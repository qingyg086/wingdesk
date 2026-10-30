// 服务线路定义：key 与后端 LangGraph 节点名保持一致（agent/core/workflow/graph_manager.py）
export interface ServiceLine {
  key: string
  name: string
  role: string
  color: string
}

export const SERVICE_LINES: ServiceLine[] = [
  { key: 'semantic_cache', name: '语义缓存', role: '高频政策问答直接命中，不调用模型', color: '#66727f' },
  { key: 'orchestrator', name: '总控路由', role: '识别意图，分派给对应专员', color: '#1b2530' },
  { key: 'flight_search_agent', name: '航班查询', role: '直飞 / 中转航班、航线图谱', color: '#2f68d8' },
  { key: 'booking_agent', name: '订票', role: '核对乘客与支付，预订新机票', color: '#23895b' },
  { key: 'reservation_agent', name: '订单管理', role: '查订单，改航班、舱位、行李、乘客', color: '#c9771f' },
  { key: 'cancel_refund_agent', name: '退票补偿', role: '按政策判断能否取消，延误补偿', color: '#b8396f' },
  { key: 'policy_agent', name: '政策咨询', role: '行李、保险、改签等规则（RAG）', color: '#6f55c2' },
  { key: 'human_handoff', name: '转人工', role: '超出线上处理范围时转接', color: '#8a5a3c' },
]

// 不在图例里展示、但会出现在路线条上的节点
const EXTRA: ServiceLine[] = [
  { key: 'confirm', name: '人工确认', role: '用户确认 / 取消待执行的操作', color: '#1b2530' },
]

const byKey = new Map([...SERVICE_LINES, ...EXTRA].map((l) => [l.key, l]))

export function lineOf(key: string): ServiceLine {
  return byKey.get(key) ?? { key, name: key, role: '', color: '#66727f' }
}
