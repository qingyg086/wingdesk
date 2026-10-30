<template>
  <div class="app-shell" :class="{ 'nav-open': navOpen }" @keydown.esc="closeNav">
    <a class="skip-link" href="#ask">跳转到输入框</a>
    <button v-if="navOpen" class="sidebar-scrim" aria-label="关闭会话菜单" @click="closeNav"></button>
    <aside id="sidebar" ref="sidebarRef" class="sidebar" :inert="isMobile && !navOpen" aria-label="会话导航" @keydown="trapNavFocus">
      <div class="brand-row">
        <div class="brand-symbol"><AppIcon name="plane" /></div>
        <div class="brand-wordmark"><strong>翼枢<span>WingDesk</span></strong><small>航空智能客服</small></div>
        <button class="icon-button mobile-close" aria-label="关闭会话菜单" @click="closeNav"><AppIcon name="close" /></button>
      </div>
      <button class="new-chat-button" :disabled="busy" @click="createSession"><AppIcon name="plus" />开启新会话<AppIcon class="new-chat-arrow" name="arrow" /></button>
      <div class="history-heading"><span>最近会话</span><span>{{ sessions.length.toString().padStart(2, '0') }}</span></div>
      <label class="history-search"><AppIcon name="search" /><input v-model="sessionSearch" placeholder="搜索会话" aria-label="搜索会话" /></label>
      <nav class="sessions" aria-label="会话列表">
        <button v-for="s in filteredSessions" :key="s.id" class="session-button" :class="{ active: s.id === currentId }" :disabled="busy && s.id !== currentId" :aria-current="s.id === currentId ? 'page' : undefined" :title="s.title" @click="switchSession(s.id)">
          <AppIcon name="chat" /><span>{{ s.title }}</span><span v-if="s.id === currentId" class="session-dot"></span>
        </button>
        <p v-if="!filteredSessions.length" class="no-sessions">没有找到相关会话</p>
      </nav>
      <details class="service-guide">
        <summary><AppIcon name="layers" /><span>了解服务分工</span><AppIcon class="details-chevron" name="chevron" /></summary>
        <ul><li v-for="line in SERVICE_LINES" :key="line.key"><span :style="{ background: line.color }"></span><div><strong>{{ line.name }}</strong><p>{{ line.role }}</p></div></li></ul>
      </details>
      <div class="sidebar-note"><AppIcon name="shield" /><p>每个重要决定<br /><strong>都由你来确认。</strong></p></div>
      <div class="account-panel">
        <div class="account-avatar">{{ account.initials }}</div>
        <div class="account-select"><label for="account">演示账号<span>{{ account.tier }}</span></label><select id="account" v-model="userId" :disabled="busy"><option v-for="a in ACCOUNTS" :key="a.id" :value="a.id">{{ a.name }}</option></select></div>
        <AppIcon name="chevron" class="account-chevron" />
      </div>
      <div class="sidebar-footer"><span>WINGDESK</span><span>让旅程更从容</span></div>
    </aside>

    <main class="workspace" :inert="isMobile && navOpen">
      <header class="topbar">
        <div class="topbar-left"><button ref="menuRef" class="icon-button menu-button" aria-label="打开会话菜单" :aria-expanded="navOpen" aria-controls="sidebar" @click="openNav"><AppIcon name="menu" /></button><span class="workspace-label">服务工作台</span><span class="topbar-divider">/</span><h1>{{ current?.title === '新会话' ? '新的旅程' : current?.title }}</h1></div>
        <div class="topbar-right"><span class="demo-badge">演示空间</span><button class="service-health" :class="health" :aria-label="`${healthLabel}，点击重新检查`" :disabled="health === 'checking'" @click="checkHealth"><span></span>{{ healthLabel }}</button></div>
      </header>

      <div ref="scrollRef" class="conversation-scroll" @scroll.passive="onScroll">
        <div class="content-column">
          <section v-if="messages.length === 0" class="welcome">
            <div class="welcome-hero">
              <div class="hero-copy"><div class="eyebrow hero-eyebrow"><span></span>YOUR JOURNEY, OUR CARE</div><p class="welcome-greeting">你好，{{ account.name.split(' ')[0] }}</p><h2>下一程，<br />交给翼枢<span>为你打点。</span></h2><p class="welcome-description">从挑选航班到调整行程，<br class="desktop-break" />你的专属出行助手，随时为你解答。</p><span class="hero-caption"><AppIcon name="shield" />订票与退改签，确认后才办理</span></div>
              <div class="journey-art" aria-hidden="true">
                <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
                <div class="boarding-pass">
                  <div class="pass-top"><span><AppIcon name="plane" />WINGDESK</span><span>旅程服务卡</span></div>
                  <div class="pass-route"><div><small>从这里</small><strong>此刻</strong><span>YOUR PLAN</span></div><div class="pass-flight"><span></span><AppIcon name="plane" /></div><div><small>去往</small><strong>下一程</strong><span>WHAT'S NEXT</span></div></div>
                  <div class="pass-bottom"><span>每一程，都有回应<small>READY WHEN YOU ARE</small></span><div class="barcode"></div></div>
                </div>
                <div class="art-caption"><span></span>把繁琐留给我们，把期待留给旅程</div>
              </div>
            </div>

            <div class="section-heading"><h3>今天，有什么可以帮你？</h3><span>选择服务，开始对话</span></div>
            <div class="service-grid">
              <button v-for="service in SERVICES" :key="service.title" class="service-card" :class="service.tone" :disabled="busy" @click="send(service.prompt)"><div class="service-card-top"><span class="service-icon"><AppIcon :name="service.icon" /></span><AppIcon class="service-arrow" name="arrow" /></div><h4>{{ service.title }}</h4><p>{{ service.description }}</p><span class="service-card-action">{{ service.action }}<AppIcon name="chevron" /></span></button>
            </div>
            <div class="suggestion-row"><span><AppIcon name="chat" />也可以问</span><button v-for="q in QUICK_QUESTIONS" :key="q.label" :disabled="busy" @click="send(q.prompt)">{{ q.label }}<AppIcon name="plus" /></button></div>
            <div class="welcome-footnote"><AppIcon name="globe" /><span>航班与订单均为仿真数据 · 出行日期以美国东部时间为准</span></div>
          </section>

          <section v-else class="message-list" aria-label="对话记录" :aria-busy="busy">
            <div class="conversation-start"><span></span>每一程，都有回应<span></span></div>
            <template v-for="(m, i) in messages" :key="i">
              <article v-if="m.role === 'user'" class="user-message"><div class="message-author">你<span class="small-avatar">{{ account.initials }}</span></div><div class="user-bubble">{{ m.content }}</div></article>
              <article v-else class="assistant-message">
                <div class="assistant-avatar"><AppIcon name="plane" /></div>
                <div class="assistant-body">
                  <div class="assistant-heading"><strong>翼枢</strong><span>出行助手</span><span v-if="m.pending" class="live-label"><i></i>正在处理</span></div>
                  <div v-if="m.pending" class="processing-status" role="status"><span class="spinner"></span>{{ progressLabel(m) }}</div>
                  <div v-if="m.content" class="markdown" v-html="render(m.content)"></div>
                  <div v-if="m.error" class="message-error" role="alert"><AppIcon name="alert" /><div><strong>本次回复未能完整完成</strong><p>{{ m.error }}</p><button class="text-button" :disabled="health === 'checking'" @click="checkHealth">检查服务连接<AppIcon name="refresh" /></button></div></div>
                  <ConfirmationCard v-if="m.confirm" :payload="m.confirm" :state="m.confirmState ?? 'waiting'" :busy="busy" @answer="send($event ? '确认' : '取消', $event, m)" />
                  <div v-if="m.steps?.length || m.meta || m.notes?.length" class="message-details-row">
                    <details class="processing-details"><summary><AppIcon name="layers" /><span>处理详情</span><span v-if="m.meta?.timing" class="detail-duration">{{ seconds(m.meta.timing.total_ms) }}</span><AppIcon class="details-chevron" name="chevron" /></summary><div class="processing-content"><ol class="processing-route" aria-label="服务处理路线"><li v-for="(key, k) in nodesOf(m)" :key="`${key}-${k}`"><span :style="{ background: lineOf(key).color }"></span>{{ lineOf(key).name }}</li></ol><ul v-if="toolsOf(m).length" class="tool-list"><li v-for="(t, k) in toolsOf(m)" :key="k"><AppIcon name="chevron" />{{ describeTool(t.name ?? '', t.args) }}</li></ul><p v-for="(note, k) in m.notes" :key="k" class="process-note">{{ note }}</p><p v-if="m.meta?.timing" class="timing">{{ timingText(m) }}</p></div></details>
                    <button v-if="m.content && !m.pending" class="icon-button copy-button" :aria-label="copiedIndex === i ? '已复制回复' : '复制回复'" :title="copiedIndex === i ? '已复制' : '复制回复'" @click="copyMessage(m.content, i)"><AppIcon :name="copiedIndex === i ? 'check' : 'copy'" /></button>
                  </div>
                </div>
              </article>
            </template>
          </section>
        </div>
      </div>

      <div class="composer-area">
        <button v-if="!nearBottom && messages.length" class="jump-latest" @click="scrollToBottom(true)"><AppIcon name="arrowDown" />回到最新消息</button>
        <div class="composer-column">
          <p v-if="waitingConfirmation" class="composer-context"><AppIcon name="shield" />有一项操作等待你确认，也可以输入修改意见。</p>
          <form class="composer" :class="{ 'is-busy': busy }" @submit.prevent="send(input)">
            <label for="ask" class="sr-only">输入问题</label>
            <textarea id="ask" ref="inputRef" v-model="input" rows="1" :placeholder="busy ? '可以先写下一个问题，当前回复完成后发送…' : '告诉我你的出行计划，或问一个旅行问题…'" @input="autosize" @keydown="onComposerKeydown"></textarea>
            <div class="composer-bottom"><span class="composer-capability"><AppIcon name="chat" />{{ busy ? '正在为你处理，请稍候' : '查航班 · 办业务 · 问政策' }}</span><button class="send-button" type="submit" :disabled="busy || !input.trim()" :aria-label="busy ? '正在处理，暂不能发送' : '发送消息'"><span v-if="busy" class="spinner"></span><template v-else><span>发送</span><AppIcon name="arrowUp" /></template></button></div>
          </form>
          <div class="composer-footer"><span><AppIcon name="shield" />重要操作由你确认</span><span class="keyboard-hint">Enter 发送<span> · </span>Shift + Enter 换行</span><span class="mobile-disclaimer">仿真数据 · 美国东部时间</span></div>
        </div>
      </div>
      <div class="sr-only" role="status">{{ announcement }}</div>
      <div v-if="toast" class="toast" role="status">{{ toast }}</div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import AppIcon from './components/AppIcon.vue'
import ConfirmationCard from './components/ConfirmationCard.vue'
import { SERVICE_LINES, lineOf } from './agents'
import { TOOL_LABELS, describeTool } from './i18n/zh'
import { applyEvent, createEventParser, restoreSessions, settleConfirmation } from './chat'
import type { Message, Session } from './chat'

const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:5000/api/chat'
const HEALTH_URL = API_URL.replace(/\/chat\/?$/, '/health')
const ACCOUNTS = [
  { id: 'mia_li_3668', name: 'Mia Li', initials: 'ML', tier: '金卡会员' },
  { id: 'lucas_wilson_8118', name: 'Lucas Wilson', initials: 'LW', tier: '银卡会员' },
]
const SERVICES = [
  { title: '航班与订票', description: '寻找合适的航班，安排下一程', action: '查询航班', icon: 'plane', tone: 'blue', prompt: '后天从纽约 JFK 飞西雅图 SEA 有哪些直飞航班？' },
  { title: '我的订单', description: '查看行程，管理乘客与行李', action: '查看我的订单', icon: 'ticket', tone: 'teal', prompt: '帮我查一下我名下都有哪些订单' },
  { title: '退改与补偿', description: '计划有变？一起找到解决办法', action: '了解退改规则', icon: 'exchange', tone: 'amber', prompt: '我想了解机票退改签和航班延误补偿的规则' },
  { title: '出行政策', description: '行李、保险与会员权益，一问即知', action: '咨询出行政策', icon: 'book', tone: 'violet', prompt: '金卡会员坐经济舱能免费托运几件行李？' },
]
const QUICK_QUESTIONS = [
  { label: '托运行李额度', prompt: '金卡会员坐经济舱能免费托运几件行李？' },
  { label: '基础经济舱退票', prompt: '基础经济舱的票可以退吗？' },
  { label: '航班延误怎么办', prompt: '我的航班延误了，可以申请补偿吗？' },
]
const sessionsKey = (uid: string) => `wingdesk.sessions.v1.${uid}`
function loadAccount() {
  try { return ACCOUNTS.find((a) => a.id === localStorage.getItem('wingdesk.account'))?.id ?? ACCOUNTS[0]!.id }
  catch { return ACCOUNTS[0]!.id }
}
function newSession(): Session { return { id: `session_${crypto.randomUUID()}`, title: '新会话', messages: [], updatedAt: Date.now() } }
function loadSessions(uid: string): Session[] {
  try { const saved = restoreSessions(localStorage.getItem(sessionsKey(uid))); if (saved.length) return saved }
  catch { /* Storage may be unavailable in private browsing. */ }
  return [newSession()]
}
const userId = ref(loadAccount())
const account = computed(() => ACCOUNTS.find((a) => a.id === userId.value) ?? ACCOUNTS[0]!)
const sessions = ref(loadSessions(userId.value))
const currentId = ref([...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt)[0]!.id)
const current = computed(() => sessions.value.find((s) => s.id === currentId.value))
const messages = computed(() => current.value?.messages ?? [])
const sessionSearch = ref('')
const filteredSessions = computed(() => [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt).filter((s) => s.title.toLocaleLowerCase().includes(sessionSearch.value.trim().toLocaleLowerCase())))
const waitingConfirmation = computed(() => messages.value.some((m) => m.confirmState === 'waiting'))
const input = ref('')
const busy = ref(false)
const scrollRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLTextAreaElement | null>(null)
const sidebarRef = ref<HTMLElement | null>(null)
const menuRef = ref<HTMLButtonElement | null>(null)
const nearBottom = ref(true)
const navOpen = ref(false)
const isMobile = ref(window.matchMedia('(max-width: 760px)').matches)
const announcement = ref('')
const toast = ref('')
const copiedIndex = ref<number | null>(null)
let saveTimer: ReturnType<typeof setTimeout> | undefined
let toastTimer: ReturnType<typeof setTimeout> | undefined
let copyTimer: ReturnType<typeof setTimeout> | undefined
let requestController: AbortController | undefined

function persistSessions(uid = userId.value, list = sessions.value) {
  try { localStorage.setItem(sessionsKey(uid), JSON.stringify(list)) }
  catch { showToast('浏览器存储空间不足，部分对话可能无法在刷新后保留。') }
}
watch(sessions, () => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => persistSessions(), 400)
}, { deep: true })
watch(userId, (uid, previous) => {
  clearTimeout(saveTimer)
  persistSessions(previous)
  try { localStorage.setItem('wingdesk.account', uid) } catch { /* Keep the selected account in memory. */ }
  sessions.value = loadSessions(uid)
  currentId.value = [...sessions.value].sort((a, b) => b.updatedAt - a.updatedAt)[0]!.id
  input.value = ''
  sessionSearch.value = ''
  closeNav()
  nextTick(autosize)
  scrollToBottom(true)
})
function createSession() {
  if (busy.value) return
  const s = sessions.value.find((s) => !s.messages.length) ?? newSession()
  if (!sessions.value.includes(s)) sessions.value.push(s)
  s.updatedAt = Date.now()
  switchSession(s.id)
  sessionSearch.value = ''
  if (!isMobile.value) nextTick(() => inputRef.value?.focus())
}
function switchSession(id: string) {
  if (busy.value && id !== currentId.value) return
  currentId.value = id
  input.value = ''
  closeNav()
  nextTick(autosize)
  scrollToBottom(true)
}
function openNav() { navOpen.value = true; nextTick(() => sidebarRef.value?.querySelector<HTMLButtonElement>('.mobile-close')?.focus()) }
function closeNav() { if (navOpen.value) { navOpen.value = false; nextTick(() => menuRef.value?.focus()) } }
function trapNavFocus(event: KeyboardEvent) {
  if (!isMobile.value || !navOpen.value || event.key !== 'Tab') return
  const elements = [...(sidebarRef.value?.querySelectorAll<HTMLElement>('button:not(:disabled), input, select:not(:disabled), summary') ?? [])].filter((el) => el.getClientRects().length)
  const first = elements[0], last = elements[elements.length - 1]
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
}
function onScroll() { const el = scrollRef.value; if (el) nearBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 100 }
async function scrollToBottom(force = false) {
  if (!force && !nearBottom.value) return
  await nextTick()
  if (!force && !nearBottom.value) return
  const el = scrollRef.value
  if (el) { el.scrollTop = messages.value.length ? el.scrollHeight : 0; nearBottom.value = true }
}
function autosize() { const el = inputRef.value; if (el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 150) + 'px' } }
function onComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); send(input.value) }
}
function showToast(message: string) { toast.value = message; clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.value = '' }, 4000) }
async function copyMessage(text: string, index: number) {
  try { await navigator.clipboard.writeText(text); copiedIndex.value = index; announcement.value = '回复已复制'; clearTimeout(copyTimer); copyTimer = setTimeout(() => { copiedIndex.value = null }, 2000) }
  catch { showToast('暂时无法复制，请选中回复内容后复制。') }
}

marked.setOptions({ breaks: true, gfm: true })
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A' && node.getAttribute('href')) { node.setAttribute('target', '_blank'); node.setAttribute('rel', 'noopener noreferrer') }
})
function render(text: string) {
  const safeHtml = DOMPurify.sanitize(marked.parse(text, { async: false }) as string, {
    FORBID_TAGS: ['style', 'iframe', 'form', 'input', 'button', 'textarea', 'select', 'object', 'embed'], FORBID_ATTR: ['style'],
  })
  const fragment = document.createElement('template')
  fragment.innerHTML = safeHtml
  fragment.content.querySelectorAll('table').forEach((table) => {
    const wrapper = document.createElement('div')
    wrapper.className = 'table-scroll'
    wrapper.tabIndex = 0
    wrapper.setAttribute('role', 'region')
    wrapper.setAttribute('aria-label', '数据表格，可横向滚动')
    table.replaceWith(wrapper)
    wrapper.append(table)
  })
  return fragment.innerHTML
}
const nodesOf = (m: Message) => (m.steps ?? []).filter((s) => s.kind === 'node' && s.key).map((s) => s.key!)
const toolsOf = (m: Message) => (m.steps ?? []).filter((s) => s.kind === 'tool' && s.name !== 'think')
const seconds = (ms: number | null | undefined) => ms == null ? '—' : `${(ms / 1000).toFixed(1)} 秒`
function timingText(m: Message) {
  const meta = m.meta, t = meta?.timing
  if (!t) return ''
  if (meta?.cache) return `语义缓存命中${meta.similarity != null ? ` · 相似度 ${meta.similarity.toFixed(2)}` : ''} · 总耗时 ${seconds(t.total_ms)}`
  return `首字 ${seconds(t.first_token_ms)} · 总耗时 ${seconds(t.total_ms)}`
}
function progressLabel(m: Message) {
  if (m.confirm) return '正在整理操作明细'
  if (m.content) return '正在为你整理回复'
  const tools = toolsOf(m), lastTool = tools[tools.length - 1]
  if (lastTool) return `${TOOL_LABELS[lastTool.name ?? ''] ?? '处理你的请求'}…`
  const nodes = nodesOf(m), last = nodes[nodes.length - 1]
  if (last === 'semantic_cache') return '已找到相关政策，正在整理答案'
  if (last && last !== 'orchestrator') return `${lineOf(last).name}正在为你服务`
  return '正在了解你的需求…'
}

const health = ref<'checking' | 'online' | 'degraded' | 'offline'>('checking')
const healthLabel = computed(() => ({ checking: '检查连接中', online: '服务已连接', degraded: '服务暂不可用', offline: '服务未连接' })[health.value])
let healthController: AbortController | undefined
async function checkHealth() {
  healthController?.abort()
  healthController = new AbortController()
  const controller = healthController
  health.value = 'checking'
  const timer = setTimeout(() => controller.abort(), 10000)
  try {
    const res = await fetch(HEALTH_URL, { signal: controller.signal })
    const body = await res.json()
    if (controller !== healthController) return
    health.value = res.ok && body.status === 'ok' ? 'online' : 'degraded'
  } catch { if (controller === healthController) health.value = 'offline' }
  finally { clearTimeout(timer) }
}

async function send(raw: string, confirm?: boolean, confirmation?: Message) {
  const text = raw.trim(), session = current.value
  if (!text || busy.value || !session) return
  const affected = confirmation ? [confirmation] : session.messages.filter((m) => m.confirmState === 'waiting')
  affected.forEach((m) => { m.confirmState = 'submitting' })
  if (input.value.trim() === text) { input.value = ''; nextTick(autosize) }
  if (!session.messages.length) session.title = text.length > 20 ? text.slice(0, 20) + '…' : text
  session.updatedAt = Date.now()
  session.messages.push({ role: 'user', content: text }, { role: 'assistant', content: '', steps: [], notes: [], pending: true })
  const reply = session.messages[session.messages.length - 1]!
  busy.value = true
  announcement.value = '正在处理你的请求'
  scrollToBottom(true)
  requestController = new AbortController()
  let complete = false
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
  let idleTimer: ReturnType<typeof setTimeout> | undefined
  const resetIdleTimeout = () => { clearTimeout(idleTimer); idleTimer = setTimeout(() => requestController?.abort(), 120000) }
  try {
    resetIdleTimeout()
    const body: Record<string, unknown> = { query: text, user_id: userId.value, session_id: session.id }
    if (confirm !== undefined) body.confirm = confirm
    const res = await fetch(API_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: requestController.signal })
    if (!res.ok || !res.body) throw new Error(`服务暂时无法处理请求（${res.status}），请稍后再试。`)
    reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    const parser = createEventParser((data) => { applyEvent(reply, data); if (data.done === true) complete = true })
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      resetIdleTimeout()
      parser.push(decoder.decode(value, { stream: true }))
      scrollToBottom()
    }
    parser.push(decoder.decode())
    parser.finish()
    if (!complete && !reply.error) reply.error = '连接中断，回复可能不完整。涉及订单的操作，请先查询订单状态。'
    if (!reply.content && !reply.confirm && !reply.error) reply.error = '暂时没有收到有效回复，请稍后继续对话。'
  } catch (err) {
    reply.error = err instanceof Error && err.message.startsWith('服务暂时') ? err.message
      : affected.length ? '连接中断，暂时无法确认操作结果。请先查询订单状态，避免重复办理。'
      : '服务连接中断，请检查网络或服务状态后继续对话。涉及订单的操作，请先查询订单状态。'
  } finally {
    clearTimeout(idleTimer)
    if (reader) { try { await reader.cancel() } catch { /* The network may already be closed. */ } reader.releaseLock() }
    reply.pending = false
    affected.forEach((m) => { m.confirmState = settleConfirmation(reply, complete, confirm) })
    busy.value = false
    session.updatedAt = Date.now()
    announcement.value = reply.error ? '回复未完整完成，请查看提示' : reply.confirm ? '有一项操作等待你确认' : '回复已完成'
    persistSessions()
    scrollToBottom()
    if (!isMobile.value && document.activeElement === document.body) nextTick(() => inputRef.value?.focus())
  }
}
const media = window.matchMedia('(max-width: 760px)')
function onMediaChange() { isMobile.value = media.matches; if (!media.matches) navOpen.value = false }
function onPageHide() { clearTimeout(saveTimer); persistSessions() }
onMounted(() => { checkHealth(); media.addEventListener('change', onMediaChange); window.addEventListener('pagehide', onPageHide); scrollToBottom(true) })
onBeforeUnmount(() => { onPageHide(); clearTimeout(toastTimer); clearTimeout(copyTimer); requestController?.abort(); healthController?.abort(); media.removeEventListener('change', onMediaChange); window.removeEventListener('pagehide', onPageHide) })
</script>
