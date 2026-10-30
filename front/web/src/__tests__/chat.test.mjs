import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(new URL('../chat.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } })
const { applyEvent, createEventParser, restoreSessions, settleConfirmation } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
)

test('stream events survive fragmented CRLF and Chinese text chunks', () => {
  const events = []
  const parser = createEventParser((event) => events.push(event))
  const stream = 'data: {"content":"你好，旅客"}\r\n\r\ndata: {"done":true}\n\n'
  for (const char of stream) parser.push(char)
  parser.finish()
  assert.deepEqual(events, [{ content: '你好，旅客' }, { done: true }])
})

test('stream supports comments, multiple data lines and a final unterminated event', () => {
  const events = []
  const parser = createEventParser((event) => events.push(event))
  parser.push(': heartbeat\n\ndata: {\ndata: "content": "答复"\ndata: }\n\ndata: {"done":true}')
  parser.finish()
  assert.deepEqual(events, [{ content: '答复' }, { done: true }])
})

test('confirmation prompt is rendered only in its card and tool narration becomes a detail', () => {
  const reply = { role: 'assistant', content: '正在查找', steps: [], notes: [] }
  applyEvent(reply, { step: { kind: 'tool', name: 'search_direct_flight' } })
  assert.deepEqual(reply.notes, ['正在查找'])
  assert.equal(reply.content, '')
  applyEvent(reply, { confirm: { type: 'confirm_action', action: 'book_reservation', args: {}, summary_zh: '真实操作摘要' } })
  applyEvent(reply, { content: '请确认以下操作：重复提示' })
  assert.equal(reply.content, '')
  assert.equal(reply.confirmState, 'waiting')
})

test('confirmation never claims success on incomplete streams or server errors', () => {
  const reply = { role: 'assistant', content: '部分回复' }
  assert.equal(settleConfirmation(reply, false, true), 'uncertain')
  assert.equal(settleConfirmation({ ...reply, error: '工具调用失败' }, true, false), 'uncertain')
  assert.equal(settleConfirmation(reply, true, true), 'approved')
  assert.equal(settleConfirmation(reply, true, false), 'rejected')
  assert.equal(settleConfirmation(reply, true), 'replied')
})

test('a busy rejection restores the existing confirmation instead of losing it', () => {
  const reply = { role: 'assistant', content: '', error: '会话忙碌', meta: { cache: null, route: [], busy: true } }
  assert.equal(settleConfirmation(reply, true, true), 'waiting')
})

test('restoring a session preserves history but marks interrupted operations uncertain', () => {
  const data = [{ id: 'existing-session', title: '订票', updatedAt: 1, messages: [
    { role: 'user', content: '确认' },
    { role: 'assistant', content: '', pending: true },
    { role: 'assistant', content: '', confirmState: 'submitting' },
    { role: 'assistant', content: '之前的回复' },
  ] }]
  const restored = restoreSessions(JSON.stringify(data))
  assert.equal(restored[0]?.id, 'existing-session')
  assert.equal(restored[0]?.messages[1]?.pending, false)
  assert.match(restored[0]?.messages[1]?.error ?? '', /未完整接收/)
  assert.equal(restored[0]?.messages[2]?.confirmState, 'uncertain')
  assert.equal(restored[0]?.messages[3]?.content, '之前的回复')
})

test('invalid stored sessions do not crash startup', () => {
  assert.deepEqual(restoreSessions('{'), [])
  assert.deepEqual(restoreSessions(null), [])
  assert.deepEqual(restoreSessions('{"messages":[]}'), [])
  assert.deepEqual(restoreSessions('[null, {"id":"x","messages":[null]}]'), [])
})
