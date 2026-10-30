# 翼枢 WingDesk · Web 工作台

航空客服多智能体平台的前端，基于 Vue 3 + TypeScript + Vite。

## 启动

```sh
npm install
npm run dev        # 默认 http://localhost:5173
```

后端默认地址为 `http://127.0.0.1:5000/api/chat`，可通过环境变量 `VITE_API_URL` 覆盖。

## 功能

- 真流式：路由到哪个专员、调用了哪个工具、回复的每个字都实时显示（SSE）
- 人工确认：订票、改签、退票等写操作会弹出确认卡片，点【确认】才执行
- 演示账号切换：mia_li_3668（金卡）/ lucas_wilson_8118（银卡），会话按账号分开保存在浏览器里
- 每条回复下方显示首字时间和总耗时

## 构建

```sh
npm run build
```
