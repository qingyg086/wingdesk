// 中文展示用的词表：数据库和工具参数保持 τ-bench 英文原值，只在界面上翻译。
// AIRPORT_CITY / ENUMS 由 data/airline/airports_zh.json、enums_zh.json 生成，改动请两边同步。

export const AIRPORT_CITY: Record<string, string> = {
  "SFO": "旧金山",
  "JFK": "纽约",
  "LAX": "洛杉矶",
  "ORD": "芝加哥",
  "DFW": "达拉斯",
  "DEN": "丹佛",
  "SEA": "西雅图",
  "ATL": "亚特兰大",
  "MIA": "迈阿密",
  "BOS": "波士顿",
  "PHX": "菲尼克斯",
  "IAH": "休斯敦",
  "LAS": "拉斯维加斯",
  "MCO": "奥兰多",
  "EWR": "纽瓦克",
  "CLT": "夏洛特",
  "MSP": "明尼阿波利斯",
  "DTW": "底特律",
  "PHL": "费城",
  "LGA": "纽约"
}

export const ENUMS: Record<string, Record<string, string>> = {
  "cabin": {
    "basic_economy": "基础经济舱",
    "economy": "经济舱",
    "business": "商务舱"
  },
  "flight_type": {
    "one_way": "单程",
    "round_trip": "往返"
  },
  "membership": {
    "regular": "普通会员",
    "silver": "银卡会员",
    "gold": "金卡会员"
  },
  "flight_status": {
    "available": "可预订",
    "on time": "准点",
    "delayed": "延误",
    "flying": "飞行中",
    "landed": "已降落",
    "cancelled": "已取消"
  },
  "payment_source": {
    "credit_card": "信用卡",
    "gift_card": "礼品卡",
    "certificate": "旅行代金券"
  },
  "card_brand": {
    "visa": "Visa",
    "mastercard": "万事达卡"
  },
  "insurance": {
    "yes": "已购买",
    "no": "未购买"
  },
  "reservation_status": {
    "active": "有效",
    "cancelled": "已取消"
  }
}

// 工具名 → 中文动作
export const TOOL_LABELS: Record<string, string> = {
  get_user_details: '查询会员资料',
  get_reservation_details: '查询订单',
  list_all_airports: '查询机场列表',
  search_direct_flight: '搜索直飞航班',
  search_onestop_flight: '搜索中转航班',
  book_reservation: '预订机票',
  update_reservation_flights: '修改航班 / 舱位',
  update_reservation_passengers: '修改乘客信息',
  update_reservation_baggages: '修改托运行李',
  cancel_reservation: '取消订单',
  send_certificate: '发放代金券',
  calculate: '计算金额',
  think: '梳理思路',
  transfer_to_human_agents: '转接人工客服',
  query_flight_routes: '查询航线图谱',
  query_vector_db: '检索政策文档',
}

export function airport(code: unknown): string {
  const c = String(code ?? '')
  const city = AIRPORT_CITY[c]
  return city ? `${city} ${c}` : c
}

export function enumZh(category: string, value: unknown): string {
  const v = String(value ?? '')
  return ENUMS[category]?.[v] ?? v
}

// 把一次工具调用翻译成一行中文，例如"搜索直飞航班 · 纽约 JFK → 西雅图 SEA · 2024-05-20"
export function describeTool(name: string, args: Record<string, unknown> = {}): string {
  const parts = [TOOL_LABELS[name] ?? name]
  if (args.origin && args.destination) parts.push(`${airport(args.origin)} → ${airport(args.destination)}`)
  if (args.date) parts.push(String(args.date))
  if (args.reservation_id) parts.push(`订单 ${args.reservation_id}`)
  if (args.cabin) parts.push(enumZh('cabin', args.cabin))
  if (name === 'send_certificate' && args.amount != null) parts.push(`$${args.amount}`)
  if (name === 'calculate' && args.expression) parts.push(String(args.expression))
  if (name === 'query_vector_db' && args.query) parts.push(`“${args.query}”`)
  return parts.join(' · ')
}
