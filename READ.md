# SleepGPT — AI 智能睡眠助理 

SleepGPT 是一個基於 **生成式 AI（Generative AI）** 構建的睡眠助理，
結合對話式互動與心理學知識，協助使用者改善睡眠品質、培養放鬆習慣。

---

## 概述
- **目標**：利用 AI 提供個人化睡眠建議
- **概念**：融合心理學的 **睡眠衛生（Sleep Hygiene）** 與生成式 AI 技術
- **應用場景**：整合於 LINE Bot，幫助使用者透過簡單對話來了解或改善睡眠

---

##  技術架構
- **語言模型**：OpenAI GPT（搭配 prompt 控制回應）
- **框架**：LangChain

---
## LangChain日前大幅更新
- 將在 LangChain v1.0 穩定後，更新至最新 LCEL 結構。
- 計畫使用 **RunnableSequence** 與 **RunnableParallel** 重新設計 RAG 模組，以簡化 pipeline 並提升擴展性。
