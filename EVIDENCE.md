# Evidence

> Дополнено 2026-11-26 (Prompt Injection / trust-boundary). Мастер-разбор:
> `~/Desktop/Services/docs/ANALYSIS_2026-11-26_PromptInjection_Retail_RAG.md`.

## 🔥 NEW 2026-11-26 — Indirect Prompt Injection / capability-perimeter (веб- и MCP-извлечение)

| Тезис | Источник | ID | Сила | Что делать в argus-pre |
|---|---|---|---|---|
| **Indirect prompt injection:** инструкции, встроенные в извлечённый контент (http.get / browser / MCP-инструменты), могут компрометировать агента во время легитимной задачи | Greshake 2023 (фундам.); residual privacy leakage (tool-using agents) | **DOI 10.1145/3605764.3623985**; DOI 10.21203/rs.3.rs-11428254/v1 | **Сильная** (фундам. рецензир.) | Подтверждает существующий SSRF-guard (`http.get`/`browser` private/loopback блок). Принцип: **извлечённое содержимое — недоверенные данные, не инструкции**. Поля результата http.get/browser считать данными, никогда не подставлять как системную директиву |
| **Capability-периметр:** права инструментов не должны эскалироваться подачей из внешнего контента; атака на tool selection агента реальна | Das 2026 (capability-mediated perimeters); **NDSS 2026** tool-selection | DOI 10.21203/rs.3.rs-10889863/v1; **10.14722/ndss.2026.230675** | **Сильная** (NDSS — рецензир.) | Подтверждает существующий scope-гейт (`AgentLoop.mandate` + `allows_scope`). Добавить unit-тест: извлечённый контент пытается вызвать `shell` в обход mandate_scope_gate — должен блокироваться `allows_scope` |
| **Trust-boundary:** различие доверенных/недоверенных источников — защита от индирект-инъекции | Beyond Prompt Injection: Trust-Boundary Security | DOI 10.21203/rs.3.rs-10798245/v1 | Умеренная (препринт) | Источники, из которых агент читает (файлы/URL/MCP), помечать доверенными/недоверенными; поведенческие директивы — только из доверенных |
| **Red teaming в клиническом ИИ** (перекрёстный, для ко-пилотов PP): ~20% выходов LLM с рисками | *BDJ Open* 2026 | **PMID 42420260** / DOI 10.1038/s41405-026-00462-9 | Сильная | Полезно как бенчмарк-аргумент в грант-подачах (OSC $17K, Foresight $50K, STF): безопасность агента — измеримый критерий |

**Мета-анализ:** механика indirect prompt injection подтверждена независимо (security-базы: Greshake 2023, NDSS 2026; свежие препринты 2026). Существующий стек argus-pre (SSRF-guard, mandate_scope_gate, plan-mode, approval-gated shell) уже закрывает 2 из 3 «границ» атаки; недостающее звено — формальная метка доверия извлечённого контента и тест на tool-selection инъекцию.

**Оговорки:** часть подтверждений — Research Square 2026 препринты `[не проверено по рецензии]`; NDSS DOI (10.14722/ndss.2026.230675) проверен в Crossref, полный текст не читал.
