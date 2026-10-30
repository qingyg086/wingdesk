<template>
  <div class="shell" :class="{ 'nav-open': navOpen }">
    <!-- 侧栏：品牌 / 会话 / 服务线路图例 / 账号 -->
    <aside class="side" aria-label="会话与服务线路">
      <div class="brand">
        <span class="brand-mark">翼枢</span>
        <span class="brand-sub">WingDesk · 航空智能客服</span>
      </div>

      <button class="btn-outline" type="button" :disabled="busy" @click="createSession">新会话</button>

      <nav class="sessions" aria-label="会话列表">
        <button
          v-for="s in sortedSessions"
          :key="s.id"
          type="button"
          :class="['session', { active: s.id === currentId }]"
          :disabled="busy && s.id !== currentId"
          @click="switchSession(s.id)"
        >
          {{ s.title }}
        </button>
      </nav>

      <section class="legend" aria-labelledby="legend-title">
        <h2 id="legend-title" class="legend-title">服务线路</h2>
        <ul>
          <li v-for="line in SERVICE_LINES" :key="line.key">
            <span class="legend-line" :style="{ background: line.color }"></span>
            <span class="legend-name">{{ line.name }}</span>
            <span class="legend-role">{{ line.role }}</span>
          </li>
        </ul>
      </section>

      <div class="account">
        <label for="account">演示账号</label>
        <select id="account" v-model="userId" :disabled="busy">
          <option v-for="a in ACCOUNTS" :key="a.id" :value="a.id">{{ a.name }} · {{ a.tier }}</option>
        </select>
        <span class="account-id">{{ userId }}</span>
      </div>
    </aside>

    <div class="scrim" @click="navOpen = false"></div>

    <!-- 主区域 -->
    <main class="main">
      <header class="topbar">
        <button class="nav-toggle" type="button" aria-label="打开会话列表" @click="navOpen = true">
          <svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true">
            <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
          </svg>
        </button>
        <h1 class="topbar-title">{{ current?.title ?? '新会话' }}</h1>
      </header>

      <div ref="scrollRef" class="scroll">
        <div class="column">
          <!-- 空状态 -->
          <section v-if="messages.length === 0" class="welcome">
            <h2 class="welcome-title">今天要办理哪项出行业务？</h2>
            <p class="welcome-desc">
              查航班、订票、改签、退票、问政策都可以。订票、改签、退票等操作会先弹出确认卡片，点确认后才会执行。演示数据来自
              τ-bench airline，系统时间设定为 2024 年 5 月 15 日。
            </p>

            <div class="starters">
              <div v-for="group in STARTERS" :key="group.line" class="starter-group">
                <h3 class="starter-head">
                  <span class="starter-line" :style="{ background: lineOf(group.line).color }"></span>
                  {{ group.title }}
                </h3>
                <button
                  v-for="q in group.prompts"
                  :key="q"
                  type="button"
                  class="starter"
                  :disabled="busy"
                  @click="send(q)"
                >
                  {{ q }}
                </button>
              </div>
            </div>
          </section>

          <!-- 消息流 -->
          <template v-for="(m, i) in messages" :key="i">
            <div v-if="m.role === 'user'" class="msg-user">{{ m.content }}</div>

            <article v-else class="msg-ai" :style="{ '--line': finalColor(m) }">
              <!-- 路线条：随 SSE 的 step 事件实时生长 -->
              <ol class="route" :aria-label="routeLabel(m)">
                <li v-if="nodesOf(m).length === 0 && m.pending" class="stop pass">
                  <span class="dot pulse" :style="{ '--c': lineOf('orchestrator').color }"></span>
                  正在连接
                </li>
                <li v-if="showCacheMiss(m)" class="stop pass">
                  <span class="dot hollow"></span>
                  缓存未命中
                </li>
                <li
                  v-for="(key, k) in nodesOf(m)"
                  :key="key"
                  class="stop"
                  :class="{ live: m.pending && k === nodesOf(m).length - 1 }"
                  :style="{ '--c': lineOf(key).color }"
                >
                  <span v-if="k > 0 || showCacheMiss(m)" class="seg"></span>
                  <span
                    class="dot"
                    :class="{ pulse: m.pending && k === nodesOf(m).length - 1 }"
                    :style="{ background: lineOf(key).color }"
                  ></span>
                  {{ key === 'semantic_cache' ? '语义缓存命中' : lineOf(key).name }}
                </li>
              </ol>

              <!-- 工具调用：随 step(kind=tool) 事件出现 -->
              <ul v-if="toolsOf(m).length" class="tools">
                <li v-for="(t, k) in toolsOf(m)" :key="k">{{ describeTool(t.name ?? '', t.args) }}</li>
              </ul>
              <p v-for="(n, k) in m.notes ?? []" :key="'n' + k" class="note">{{ n }}</p>

              <div v-if="m.content" class="md" v-html="render(m.content)"></div>
              <div v-if="m.error" class="msg-error">{{ m.error }}</div>

              <!-- 人工确认卡片 -->
              <section v-if="m.confirm" class="confirm" :class="m.confirmState">
                <h3 class="confirm-head">请确认：{{ TOOL_LABELS[m.confirm.action] ?? m.confirm.action }}</h3>
                <p class="confirm-body">{{ m.confirm.summary_zh }}</p>
                <div v-if="m.confirmState === 'waiting'" class="confirm-actions">
                  <button class="btn-send" type="button" :disabled="busy" @click="answerConfirm(m, true)">确认</button>
                  <button class="btn-outline" type="button" :disabled="busy" @click="answerConfirm(m, false)">
                    取消
                  </button>
                  <span class="confirm-hint">需要修改的话，直接在下方输入要改什么</span>
                </div>
                <p v-else class="confirm-done">{{ CONFIRM_STATE_TEXT[m.confirmState ?? 'replied'] }}</p>
              </section>

              <p v-if="m.meta?.timing" class="timing">{{ timingText(m) }}</p>
            </article>
          </template>
        </div>
      </div>

      <!-- 输入区 -->
      <form class="composer" @submit.prevent="send(input)">
        <div class="column">
          <div class="composer-box">
            <label for="ask" class="sr-only">输入问题</label>
            <textarea
              id="ask"
              ref="inputRef"
              v-model="input"
              rows="1"
              placeholder="例如：5 月 20 日从纽约飞西雅图有哪些直飞航班？"
              :disabled="busy"
              @input="autosize"
              @keydown.enter.exact.prevent="send(input)"
            ></textarea>
            <button class="btn-send" type="submit" :disabled="busy || !input.trim()">
              {{ busy ? '处理中' : '发送' }}
            </button>
          </div>
          <p class="composer-hint">Enter 发送，Shift + Enter 换行</p>
        </div>
      </form>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { marked } from 'marked'
import { SERVICE_LINES, lineOf } from './agents'
import { TOOL_LABELS, describeTool } from './i18n/zh'

// ---------- 配置 ----------
const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:5000/api/chat'

// τ-bench airline 里的两个演示账号
const ACCOUNTS = [
  { id: 'mia_li_3668', name: 'Mia Li', tier: '金卡会员' },
  { id: 'lucas_wilson_8118', name: 'Lucas Wilson', tier: '银卡会员' },
]
const ACCOUNT_KEY = 'wingdesk.account'
const sessionsKey = (uid: string) => `wingdesk.sessions.v1.${uid}`

const STARTERS = [
  {
    line: 'flight_search_agent',
    title: '航班查询与订票',
    prompts: ['5 月 20 日从纽约 JFK 飞西雅图 SEA 有哪些直飞航班？', '帮我订一张 5 月 20 日 JFK 到 SEA 的经济舱单程票'],
  },
  {
    line: 'reservation_agent',
    title: '订单管理',
    prompts: ['帮我查一下我名下都有哪些订单', '我想给我的一个订单加一件托运行李'],
  },
  {
    line: 'cancel_refund_agent',
    title: '退票与补偿',
    prompts: ['我想取消一个订单，帮我看看能不能退', '我的航班延误了，可以申请补偿吗？'],
  },
  {
    line: 'policy_agent',
    title: '政策咨询',
    prompts: ['金卡会员坐经济舱能免费托运几件行李？', '基础经济舱的票可以退吗？'],
  },
]

const CONFIRM_STATE_TEXT: Record<string, string> = {
  approved: '已确认',
  rejected: '已取消，操作没有执行',
  replied: '已在对话中回复',
}

// ---------- 类型（与 app/service/chat_service.py 的 SSE 事件对应） ----------
interface Step {
  kind: 'node' | 'tool'
  key?: string
  name?: string
  args?: Record<string, unknown>
}

interface Timing {
  first_token_ms: number | null
  total_ms: number
}

interface RouteMeta {
  cache: string | null
  route: string[]
  similarity?: number
  timing?: Timing
}

interface ConfirmPayload {
  type: string
  action: string
  args: Record<string, unknown>
  summary_zh: string
}

type ConfirmState = 'waiting' | 'approved' | 'rejected' | 'replied'

interface Message {
  role: 'user' | 'assistant'
  content: string
  steps?: Step[]
  notes?: string[]
  meta?: RouteMeta
  confirm?: ConfirmPayload
  confirmState?: ConfirmState
  pending?: boolean
  error?: string
}

interface Session {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

// ---------- 账号与会话（浏览器本地保存，按账号分开；读写失败时退化为内存） ----------
function loadAccount(): string {
  try {
    const v = localStorage.getItem(ACCOUNT_KEY)
    if (v && ACCOUNTS.some((a) => a.id === v)) return v
  } catch {
    /* ignore */
  }
  return ACCOUNTS[0]?.id ?? 'mia_li_3668'
}

function newSession(): Session {
  return { id: `session_${Date.now()}`, title: '新会话', messages: [], updatedAt: Date.now() }
}

function loadSessions(uid: string): Session[] {
  try {
    const raw = localStorage.getItem(sessionsKey(uid))
    const parsed = raw ? (JSON.parse(raw) as Session[]) : []
    if (Array.isArray(parsed) && parsed.length) {
      // 丢弃上次未完成的请求状态
      parsed.forEach((s) => s.messages.forEach((m) => (m.pending = false)))
      return parsed
    }
  } catch {
    /* ignore */
  }
  return [newSession()]
}

const userId = ref<string>(loadAccount())
const sessions = ref<Session[]>(loadSessions(userId.value))
const currentId = ref<string>(sessions.value[0]?.id ?? '')
const current = computed(() => sessions.value.find((s) => s.id === currentId.value))
const messages = computed<Message[]>(() => current.value?.messages ?? [])
const sortedSessions = computed(() => [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt))

watch(
  sessions,
  (v) => {
    try {
      localStorage.setItem(sessionsKey(userId.value), JSON.stringify(v))
    } catch {
      /* ignore */
    }
  },
  { deep: true },
)

watch(userId, (uid) => {
  try {
    localStorage.setItem(ACCOUNT_KEY, uid)
  } catch {
    /* ignore */
  }
  sessions.value = loadSessions(uid)
  currentId.value = sessions.value[0]?.id ?? ''
  scrollToBottom()
})

const navOpen = ref(false)

function createSession() {
  const empty = sessions.value.find((s) => s.messages.length === 0)
  const s = empty ?? newSession()
  if (!empty) sessions.value.push(s)
  s.updatedAt = Date.now()
  currentId.value = s.id
  navOpen.value = false
}

function switchSession(id: string) {
  currentId.value = id
  navOpen.value = false
  scrollToBottom()
}

// ---------- 渲染 ----------
marked.setOptions({ breaks: true, gfm: true })

function render(text: string): string {
  return marked.parse(text, { async: false }) as string
}

function nodesOf(m: Message): string[] {
  return (m.steps ?? []).filter((s) => s.kind === 'node' && s.key).map((s) => s.key as string)
}

function toolsOf(m: Message): Step[] {
  return (m.steps ?? []).filter((s) => s.kind === 'tool' && s.name !== 'think')
}

function showCacheMiss(m: Message): boolean {
  const first = nodesOf(m)[0]
  return first === 'orchestrator'
}

function finalColor(m: Message): string {
  if (m.error && !m.content) return 'var(--danger)'
  const nodes = nodesOf(m)
  const last = nodes[nodes.length - 1]
  return last ? lineOf(last).color : 'var(--rule)'
}

function routeLabel(m: Message): string {
  const nodes = nodesOf(m)
  if (!nodes.length) return '处理路线'
  return '处理路线：' + nodes.map((k) => (k === 'semantic_cache' ? '语义缓存命中' : lineOf(k).name)).join('，')
}

function seconds(ms: number | null | undefined): string {
  return ms == null ? '—' : `${(ms / 1000).toFixed(1)} s`
}

function timingText(m: Message): string {
  const meta = m.meta
  const t = meta?.timing
  if (!meta || !t) return ''
  if (m.confirm && !m.content) return `总耗时 ${seconds(t.total_ms)}`
  if (meta.cache) {
    const sim = meta.similarity != null ? `（相似度 ${meta.similarity.toFixed(2)}）` : ''
    return `语义缓存命中${sim} · 总耗时 ${seconds(t.total_ms)}`
  }
  return `首字 ${seconds(t.first_token_ms)} · 总耗时 ${seconds(t.total_ms)}`
}

// ---------- 交互 ----------
const input = ref('')
const busy = ref(false)
const scrollRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLTextAreaElement | null>(null)

async function scrollToBottom() {
  await nextTick()
  const el = scrollRef.value
  if (el) el.scrollTop = el.scrollHeight
}

function autosize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 168) + 'px'
}

function answerConfirm(m: Message, approved: boolean) {
  if (busy.value) return
  m.confirmState = approved ? 'approved' : 'rejected'
  send(approved ? '确认' : '取消', approved)
}

// 处理一条 SSE 事件，更新正在生成的那条回复
function applyEvent(reply: Message, data: Record<string, unknown>, state: { confirmed: boolean }) {
  if (data.step) {
    const step = data.step as Step
    reply.steps?.push(step)
    // 模型在调用工具前顺带说的话属于"过程说明"，挪到回复正文之外
    if (step.kind === 'tool' && reply.content.trim()) {
      reply.notes?.push(reply.content.trim())
      reply.content = ''
    }
  }
  if (data.confirm) {
    reply.confirm = data.confirm as ConfirmPayload
    reply.confirmState = 'waiting'
    state.confirmed = true
  }
  // 确认事件之后的文字是"请确认以下操作……"提示，卡片里已经展示了，不再重复
  if (typeof data.content === 'string' && !state.confirmed) {
    reply.content += data.content
  }
  if (typeof data.error === 'string') reply.error = data.error
  if (data.meta) reply.meta = data.meta as RouteMeta
}

async function send(raw: string, confirm?: boolean) {
  const text = raw.trim()
  const session = current.value
  if (!text || busy.value || !session) return

  input.value = ''
  await nextTick()
  autosize()

  // 有待确认的卡片、用户却直接打字回复：卡片失效，由后端根据文字判断是确认还是修改
  if (confirm === undefined) {
    session.messages.forEach((m) => {
      if (m.confirmState === 'waiting') m.confirmState = 'replied'
    })
  }

  if (session.messages.length === 0) {
    session.title = text.length > 18 ? text.slice(0, 18) + '…' : text
  }
  session.updatedAt = Date.now()
  session.messages.push({ role: 'user', content: text })
  session.messages.push({ role: 'assistant', content: '', steps: [], notes: [], pending: true })
  // 通过响应式代理修改，保证界面更新
  const reply = session.messages[session.messages.length - 1] as Message
  busy.value = true
  scrollToBottom()

  try {
    const body: Record<string, unknown> = { query: text, user_id: userId.value, session_id: session.id }
    if (confirm !== undefined) body.confirm = confirm
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    const state = { confirmed: false }
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // SSE 以空行分隔事件；一次 read 可能包含半个事件，剩下的留到下一次拼接
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6).trim()
        if (!payload) continue
        try {
          applyEvent(reply, JSON.parse(payload), state)
        } catch (e) {
          console.error('SSE 解析失败', e, payload)
        }
      }
      scrollToBottom()
    }
    if (!reply.content && !reply.confirm && !reply.error) {
      reply.error = '后端没有返回内容。请查看 FastAPI 终端日志后重新发送。'
    }
  } catch (err) {
    console.error(err)
    reply.error = '没有连上后端服务。确认 FastAPI 已在 5000 端口启动，然后重新发送。'
  } finally {
    reply.pending = false
    busy.value = false
    session.updatedAt = Date.now()
    scrollToBottom()
    nextTick(() => inputRef.value?.focus())
  }
}
</script>

<style scoped>
/* ---------- 布局 ---------- */
.shell {
  display: grid;
  grid-template-columns: 284px 1fr;
  height: 100vh;
  height: 100dvh;
}

.side {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 28px 20px 20px;
  background: var(--surface);
  border-right: 1px solid var(--rule);
  min-height: 0;
}

.main {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

.column {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 0 28px;
}

/* ---------- 品牌 ---------- */
.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding-left: 2px;
}
.brand {
  flex-wrap: wrap;
}
.brand-mark {
  font-size: 30px;
  font-weight: 600;
  letter-spacing: 0.06em;
  line-height: 1;
}
.brand-sub {
  font-size: 13px;
  color: var(--muted);
  letter-spacing: 0.02em;
}

/* ---------- 按钮 ---------- */
.btn-outline {
  height: 38px;
  border: 1px solid var(--ink);
  border-radius: 10px;
  background: transparent;
  cursor: pointer;
  font-weight: 500;
}
.btn-outline:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-outline:hover:not(:disabled) {
  background: var(--ink);
  color: var(--surface);
}

/* ---------- 会话列表 ---------- */
.sessions {
  flex: 1;
  min-height: 80px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin: 0 -8px;
}
.session {
  text-align: left;
  border: 0;
  background: transparent;
  padding: 8px 10px 8px 14px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  position: relative;
}
.session:hover {
  background: var(--sunk);
  color: var(--ink);
}
.session:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.session.active {
  color: var(--ink);
  font-weight: 500;
  background: var(--sunk);
}
.session.active::before {
  content: '';
  position: absolute;
  left: 4px;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 2px;
  background: var(--ink);
}

/* ---------- 服务线路图例 ---------- */
.legend {
  border-top: 1px solid var(--rule);
  padding-top: 16px;
}
.legend-title {
  margin: 0 0 10px;
  font-size: 13px;
  font-weight: 600;
}
.legend ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 8px;
}
.legend li {
  display: grid;
  grid-template-columns: 14px auto;
  column-gap: 8px;
  align-items: center;
  font-size: 13px;
  line-height: 1.45;
}
.legend-line {
  grid-row: span 2;
  align-self: stretch;
  width: 4px;
  border-radius: 2px;
  justify-self: center;
}
.legend-name {
  font-weight: 500;
}
.legend-role {
  color: var(--muted);
  font-size: 12px;
}

.account {
  font-size: 13px;
  color: var(--muted);
  border-top: 1px solid var(--rule);
  padding-top: 14px;
}
.account {
  display: grid;
  gap: 6px;
}
.account select {
  height: 34px;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--surface);
  padding: 0 8px;
  font: inherit;
  color: var(--ink);
}
.account-id {
  font-family: var(--mono);
  font-size: 12px;
}

/* ---------- 顶栏 ---------- */
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 60px;
  padding: 0 28px;
  border-bottom: 1px solid var(--rule);
  flex-shrink: 0;
}
.topbar-title {
  margin: 0;
  font-size: 15px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.nav-toggle {
  display: none;
  border: 0;
  background: transparent;
  padding: 6px;
  margin-left: -6px;
  cursor: pointer;
  border-radius: 8px;
}

.scroll {
  flex: 1;
  overflow-y: auto;
  padding: 36px 0 24px;
  scroll-behavior: smooth;
}

/* ---------- 空状态 ---------- */
.welcome-title {
  margin: 24px 0 8px;
  font-size: 34px;
  line-height: 1.25;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.welcome-desc {
  margin: 0 0 36px;
  color: var(--muted);
  max-width: 34em;
}
.starters {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 28px 32px;
}
.starter-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 6px;
  font-size: 15px;
  font-weight: 600;
}
.starter-line {
  width: 22px;
  height: 4px;
  border-radius: 2px;
}
.starter {
  display: block;
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid var(--rule);
  background: transparent;
  padding: 10px 0;
  color: var(--ink);
  cursor: pointer;
  line-height: 1.5;
}
.starter:hover:not(:disabled) {
  color: var(--focus);
}
.starter:disabled {
  color: var(--muted);
  cursor: not-allowed;
}

/* ---------- 消息 ---------- */
.msg-user {
  margin: 0 0 28px auto;
  width: fit-content;
  max-width: 78%;
  background: var(--sunk);
  padding: 10px 16px;
  border-radius: 14px 14px 4px 14px;
  white-space: pre-wrap;
  word-break: break-word;
}

.msg-ai {
  margin: 0 0 36px;
}

/* 路线条：本页的标志元素 */
.route {
  list-style: none;
  margin: 0 0 10px;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  row-gap: 6px;
  font-size: 12.5px;
  color: var(--muted);
  min-height: 20px;
}
.stop {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.stop.hit,
.stop:not(.pass) {
  color: var(--ink);
}
.seg {
  width: 22px;
  height: 2px;
  margin: 0 6px 0 8px;
  background: var(--c, var(--rule));
  border-radius: 1px;
}
.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot.hollow {
  border: 1.5px solid var(--muted);
  background: transparent;
}
.dot.pulse {
  background: var(--c);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(27, 37, 48, 0.35); }
  50% { box-shadow: 0 0 0 5px rgba(27, 37, 48, 0); }
}

.md,
.msg-error {
  border-left: 2px solid var(--line);
  padding: 2px 0 2px 18px;
  word-break: break-word;
}
.msg-error {
  color: var(--danger);
}

/* markdown 内容 */
.md :deep(p) { margin: 0 0 12px; }
.md :deep(> :last-child) { margin-bottom: 0; }
.md :deep(h1),
.md :deep(h2),
.md :deep(h3),
.md :deep(h4) {
  margin: 18px 0 8px;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.5;
}
.md :deep(> :first-child) { margin-top: 0; }
.md :deep(ul),
.md :deep(ol) { margin: 0 0 12px; padding-left: 1.4em; }
.md :deep(li) { margin: 2px 0; }
.md :deep(strong) { font-weight: 600; }
.md :deep(a) { color: var(--focus); text-underline-offset: 2px; }
.md :deep(img) {
  display: block;
  max-width: min(100%, 360px);
  border-radius: 10px;
  margin: 10px 0;
  border: 1px solid var(--rule);
}
.md :deep(code) {
  font-family: var(--mono);
  font-size: 0.88em;
  background: var(--sunk);
  padding: 1px 5px;
  border-radius: 4px;
}
.md :deep(pre) {
  background: var(--sunk);
  padding: 12px 14px;
  border-radius: 8px;
  overflow-x: auto;
}
.md :deep(pre code) { background: none; padding: 0; }
.md :deep(blockquote > :last-child) { margin-bottom: 0; }
.md :deep(blockquote) {
  margin: 0 0 12px;
  padding: 8px 14px;
  background: var(--surface);
  border-radius: 8px;
  color: var(--muted);
}
.md :deep(table) {
  display: block;
  overflow-x: auto;
  border-collapse: collapse;
  margin: 0 0 12px;
  font-size: 14px;
}
.md :deep(th),
.md :deep(td) {
  border-bottom: 1px solid var(--rule);
  padding: 7px 14px 7px 0;
  text-align: left;
  white-space: nowrap;
}
.md :deep(th) { font-weight: 600; }

/* ---------- 工具调用 / 过程说明 / 计时 ---------- */
.tools {
  list-style: none;
  margin: 0 0 10px;
  padding: 0 0 0 20px;
  font-size: 12.5px;
  color: var(--muted);
  display: grid;
  gap: 2px;
}
.tools li::before {
  content: '↳ ';
  color: var(--line);
}
.note {
  margin: 0 0 8px 20px;
  font-size: 13px;
  color: var(--muted);
  font-style: italic;
}
.timing {
  margin: 8px 0 0 20px;
  font-size: 12px;
  color: var(--muted);
}

/* ---------- 人工确认卡片 ---------- */
.confirm {
  margin: 12px 0 0 20px;
  padding: 14px 16px;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--line);
  border-radius: 10px;
}
.confirm-head {
  margin: 0 0 6px;
  font-size: 14px;
  font-weight: 600;
}
.confirm-body {
  margin: 0 0 12px;
  white-space: pre-wrap;
  font-size: 14px;
  line-height: 1.65;
}
.confirm-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.confirm-actions .btn-outline {
  padding: 0 18px;
}
.confirm-hint {
  font-size: 12px;
  color: var(--muted);
}
.confirm-done {
  margin: 0;
  font-size: 13px;
  color: var(--muted);
}
.confirm.rejected .confirm-body,
.confirm.replied .confirm-body {
  color: var(--muted);
}

/* ---------- 输入区 ---------- */
.composer {
  flex-shrink: 0;
  padding: 12px 0 14px;
  background: linear-gradient(to bottom, transparent, var(--paper) 30%);
}
.composer-box {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 14px;
  padding: 8px 8px 8px 16px;
}
.composer-box:focus-within {
  border-color: var(--ink);
}
.composer-box textarea {
  flex: 1;
  border: 0;
  outline: none;
  resize: none;
  background: transparent;
  padding: 7px 0;
  line-height: 1.6;
  max-height: 168px;
}
.composer-box textarea::placeholder { color: #8a949a; }
.btn-send {
  height: 38px;
  min-width: 76px;
  padding: 0 16px;
  border: 0;
  border-radius: 10px;
  background: var(--ink);
  color: var(--surface);
  font-weight: 500;
  cursor: pointer;
}
.btn-send:disabled {
  background: var(--sunk);
  color: var(--muted);
  cursor: not-allowed;
}
.composer-hint {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--muted);
  text-align: right;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

.scrim { display: none; }

/* ---------- 窄屏 ---------- */
@media (max-width: 860px) {
  .shell { grid-template-columns: 1fr; }
  .side {
    position: fixed;
    inset: 0 auto 0 0;
    width: min(300px, 86vw);
    z-index: 20;
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    overflow-y: auto;
  }
  .nav-open .side { transform: none; }
  .nav-open .scrim {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 10;
    background: rgba(27, 37, 48, 0.3);
  }
  .nav-toggle { display: inline-flex; }
  .topbar { padding: 0 16px; }
  .column { padding: 0 16px; }
  .welcome-title { font-size: 26px; margin-top: 4px; }
  .starters { grid-template-columns: 1fr; gap: 24px; }
  .msg-user { max-width: 90%; }
  .composer-hint { display: none; }
}
</style>
