<template>
  <section class="confirmation-card" :class="state" :aria-label="title">
    <header class="confirmation-header">
      <span class="confirmation-icon"><AppIcon name="shield" /></span>
      <div><span class="eyebrow">操作确认</span><h3>{{ title }}</h3></div>
      <span class="confirmation-badge">{{ statusLabel }}</span>
    </header>
    <div v-if="payload.args.origin && payload.args.destination" class="confirmation-flight">
      <div><strong>{{ payload.args.origin }}</strong><span>{{ AIRPORT_CITY[String(payload.args.origin)] ?? '出发机场' }}</span></div>
      <div class="flight-connector"><span></span><AppIcon name="plane" /><span></span></div>
      <div><strong>{{ payload.args.destination }}</strong><span>{{ AIRPORT_CITY[String(payload.args.destination)] ?? '到达机场' }}</span></div>
    </div>
    <dl v-if="facts.length" class="confirmation-facts">
      <div v-for="fact in facts" :key="fact.label"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div>
    </dl>
    <div class="confirmation-summary"><span class="field-label">完整操作明细</span><p>{{ payload.summary_zh }}</p></div>
    <div v-if="state === 'waiting'" class="confirmation-footer">
      <p><AppIcon name="shield" />请核对以上信息，确认后将提交办理。</p>
      <div class="confirmation-buttons">
        <button class="primary-button" :disabled="busy" @click="$emit('answer', true)"><AppIcon name="check" />确认{{ title }}</button>
        <button class="secondary-button" :disabled="busy" @click="$emit('answer', false)">暂不办理</button>
      </div>
      <span class="confirmation-help">需要调整？直接在下方输入修改意见。</span>
    </div>
    <p v-else class="confirmation-result" role="status"><AppIcon :name="state === 'uncertain' ? 'alert' : state === 'submitting' ? 'clock' : 'check'" />{{ stateText }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import AppIcon from './AppIcon.vue'
import { AIRPORT_CITY, TOOL_LABELS, enumZh } from '../i18n/zh'
import type { ConfirmPayload, ConfirmState } from '../chat'

const props = defineProps<{ payload: ConfirmPayload; state: ConfirmState; busy: boolean }>()
defineEmits<{ answer: [approved: boolean] }>()
const title = computed(() => TOOL_LABELS[props.payload.action] ?? '提交操作')
const statusLabel = computed(() => ({ waiting: '待你确认', submitting: '正在提交', approved: '已提交确认', rejected: '已回复取消', replied: '已回复', uncertain: '待核实' })[props.state])
const stateText = computed(() => ({
  waiting: '', submitting: '正在提交你的决定，请稍候…',
  approved: '确认已提交，办理结果请查看下方回复。',
  rejected: '取消决定已提交，处理结果请查看下方回复。',
  replied: '已收到你的回复，请查看后续方案。',
  uncertain: '未能确认本次处理结果。请先查询订单状态，避免重复办理。',
})[props.state])
const facts = computed(() => {
  const a = props.payload.args
  const rows: { label: string; value: string }[] = []
  if (a.reservation_id) rows.push({ label: '订单编号', value: String(a.reservation_id) })
  if (a.cabin) rows.push({ label: '舱位', value: enumZh('cabin', a.cabin) })
  if (a.flight_type) rows.push({ label: '行程类型', value: enumZh('flight_type', a.flight_type) })
  if (a.total_baggages != null) rows.push({ label: '托运行李', value: `${a.total_baggages} 件` })
  if (typeof a.amount === 'number') rows.push({ label: '金额（美元）', value: `$${a.amount.toFixed(2)}` })
  if (Array.isArray(a.payment_methods) && a.payment_methods.length && a.payment_methods.every((p) => typeof p.amount === 'number')) {
    rows.push({ label: '合计（美元）', value: `$${a.payment_methods.reduce((sum, p) => sum + p.amount, 0).toFixed(2)}` })
  }
  return rows
})
</script>
