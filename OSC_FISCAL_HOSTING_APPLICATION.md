# ARGUS-OS1 — Apply to Open Source Collective (Fiscal Hosting)

> **Что это:** заявка на РАЗМЕЩЕНИЕ проекта (fiscal hosting) под зонтиком
> Open Source Collective (OSC) на платформе Open Collective.
> **Это НЕ грант.** OSC — фискальный хост (10% host fee на входящие средства
> + комиссии платёжных систем). Цель — дать ARGUS-OS1 юридическую/финансовую
> инфраструктуру для приёма GitHub Sponsors, донатов и спонсорства.
>
> Подача: https://opencollective.com/opensource → кнопка **Apply**.
> Основа: https://docs.oscollective.org (terms of fiscal sponsorship, fees).

---

## Данные для формы Apply

| Поле | Значение |
|---|---|
| **Название коллектива** | ARGUS-OS1 |
| **Категория** | Open source software |
| **Краткое описание** | Local-first AI agent core in Rust (MIT): single binary, file-based memory, real tools, skills hub, MCP client, reproducible evals. |
| **Репозиторий** | https://github.com/djabbat/argus-pre |
| **Лицензия** | MIT |
| **Сайт/доки** | https://github.com/djabbat/argus-pre (README, DOCS.md) |
| **Автор / мейнтейнер** | Jaba Tqemaladze, MD — Georgia Longevity Alliance (jaba@longevity.ge) |
| **Статус** | Активная разработка, V3.6, 46 unit-тестов, CI, инсталлятор |

---

## Текст «About» для коллектива (EN, копировать в форму)

ARGUS-OS1 is a local-first AI agent core written in Rust. A single binary runs the
full agent loop — decide, run, compress, repeat — and keeps all state in plain
files on the user's machine: sessions, memory, tasks, skills. It works with any
OpenAI-compatible model (Ollama, LM Studio, llama.cpp) and costs nothing per
inference. The core drives real tools (files, shell, web, a real Chrome browser
over CDP), remembers across sessions, schedules durable tasks, and ships a
reproducible eval suite.

Why it exists: agent platforms increasingly hold users' prompts, files and
secrets in the cloud. ARGUS-OS1 is the working counter-path — privacy by default,
no API fees, no lock-in, fully inspectable state. It was built for a concrete
need: research data that cannot leave the machine, and it applies equally to
clinics, labs and public institutions handling sensitive data.

Status: version 3.6, MIT license, 46 passing unit tests, clean clippy, CI,
installer, reproducible GAIA-style benchmarks. Repo: https://github.com/djabbat/argus-pre

---

## Почему размещение у OSC (для ответа в форме, если спросят)

- Нужен юридический адрес для приёма GitHub Sponsors и спонсорства без создания
  собственного юрлица.
- Прозрачные финансы (Open Collective) — совпадает с принципами проекта.
- Принятые донаты пойдут на: мейнтейнерскую поддержку, CI/инфраструктуру,
  бенчмарки, документацию и сообщество.

---

## Ожидания и комиссия

- Host fee OSC: **10%** от входящих средств (фиксированная).
- Платёжные системы (Stripe/PayPal/Wise) — свои комиссии.
- OSC не берёт комиссию за перевод между hosted коллективами/фондами.
- Скидок нет.

---

## Чек-лист перед Apply

- [ ] Убедиться, что репо `github.com/djabbat/argus-pre` публичный и MIT.
- [ ] Подтвердить, что проект соответствует eligibility OSC (open source, OSI-лицензия).
- [ ] Прочитать Terms of Fiscal Sponsorship (docs.oscollective.org).
- [ ] Заполнить форму Apply на opencollective.com/opensource.
- [ ] Сохранить подтверждение/ID коллектива после одобрения (обычно 1–2 недели).
