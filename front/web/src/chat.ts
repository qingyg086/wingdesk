export interface Step {
  kind: 'node' | 'tool'
  key?: string
  name?: string
  args?: Record<string, unknown>
}

export interface ConfirmPayload {
  type: string
  action: string
  args: Record<string, unknown>
  summary_zh: string
}

export type ConfirmState = 'waiting' | 'submitting' | 'approved' | 'rejected' | 'replied' | 'uncertain'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  steps?: Step[]
  notes?: string[]
  meta?: {
    cache: string | null
    route: string[]
    similarity?: number
    busy?: boolean
    timing?: { first_token_ms: number | null; total_ms: number }
  }
  confirm?: ConfirmPayload
  confirmState?: ConfirmState
  pending?: boolean
  error?: string
}

export interface Session {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

// Keep partial events between network chunks, including CRLF split across chunks.
export function createEventParser(onEvent: (event: Record<string, unknown>) => void) {
  let buffer = ''
  function drain() {
    let match: RegExpExecArray | null
    while ((match = /\r?\n\r?\n/.exec(buffer))) {
      const block = buffer.slice(0, match.index)
      buffer = buffer.slice(match.index + match[0].length)
      const data = block.split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
        .join('\n')
      if (data && data !== '[DONE]') onEvent(JSON.parse(data))
    }
  }
  return {
    push(chunk: string) { buffer += chunk; drain() },
    finish() { if (buffer.trim()) { buffer += '\n\n'; drain() } },
  }
}

export function applyEvent(reply: Message, data: Record<string, unknown>) {
  if (data.step) {
    const step = data.step as Step
    reply.steps?.push(step)
    if (step.kind === 'tool' && reply.content.trim()) {
      reply.notes?.push(reply.content.trim())
      reply.content = ''
    }
  }
  if (data.confirm) {
    reply.confirm = data.confirm as ConfirmPayload
    reply.confirmState = 'waiting'
  }
  if (typeof data.content === 'string' && !reply.confirm) reply.content += data.content
  if (typeof data.error === 'string') reply.error = data.error
  if (data.meta) reply.meta = data.meta as Message['meta']
}

export function settleConfirmation(reply: Message, complete: boolean, choice?: boolean): ConfirmState {
  if (reply.meta?.busy) return 'waiting'
  if (!complete || reply.error) return 'uncertain'
  return choice === undefined ? 'replied' : choice ? 'approved' : 'rejected'
}

export function restoreSessions(raw: string | null): Session[] {
  try {
    const parsed: unknown = JSON.parse(raw ?? 'null')
    if (!Array.isArray(parsed)) return []
    return parsed.filter((s): s is Session => Boolean(s && typeof s.id === 'string'
      && typeof s.title === 'string' && typeof s.updatedAt === 'number'
      && Array.isArray(s.messages) && s.messages.every((m: Message) => m
        && ['user', 'assistant'].includes(m.role) && typeof m.content === 'string')))
      .map((s) => ({ ...s, messages: s.messages.map((m) => ({
        ...m,
        pending: false,
        ...(m.pending ? { error: '上次回复未完整接收。你可以继续对话；涉及订单的操作，请先查询订单状态。' } : {}),
        ...(m.confirmState === 'submitting' ? { confirmState: 'uncertain' as const } : {}),
      })) }))
  } catch { return [] }
}
