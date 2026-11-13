//! Канал диалога — «консилиум» бесплатных моделей (`--probe dialogue`).
//!
//! Задача (Джаба, 2026-11-13): до принятия решения по вопросу отправить его с описанием
//! проблемы бесплатным чатам, разобрать их ответы и применить результат разбора к решению.
//!
//! **Что здесь бесплатно:** локальные модели Ollama (разные семейства) — платить не за что.
//! Если появятся ключи бесплатных облачных тиров (Groq / OpenRouter :free / Cerebras /
//! Mistral / HuggingFace), они подключаются через тот же OpenAI-совместимый протокол:
//! `ARGUS_COUNCIL_MODELS="qwen2.5:7b-16k@http://127.0.0.1:11434/api/chat|llama-3.1-8b@groq"`
//! — без правки кода.
//!
//! **Эпистемическая рамка (важно).** Ответ чата — не свидетельство, а мнение. Поэтому:
//! 1) консилиум даёт ТОЛЬКО слабый сдвиг приора (`LR = 1.5 / 0.67`), а не `LR = 3`;
//! 2) действует лишь при согласии ≥ `MIN_AGREE` моделей и уверенности ≥ `ACT_CONF`;
//! 3) расхождение моделей → воздержание;
//! 4) никогда не закрывает вопрос (не достигает порогов 0.90 / 0.10);
//! 5) не более одного сдвига на вопрос в сутки (`COUNCIL_MAX_SHIFTS_PER_DAY`);
//! 6) всё пишется в журнал цикла с источником `council:<модели>` — аудируемо.
//!
//! Единственное место, где консилиум обязателен по решению Джабы, — лабораторный канал:
//! прежде чем применить лабораторные данные (`--probe lab --require-council`), вопрос
//! уходит в консилиум, и расхождение блокирует применение.

use crate::stance::Stance;
use serde::{Deserialize, Serialize};

/// Минимум согласных моделей, чтобы мнение считалось согласием.
pub const MIN_AGREE: usize = 2;
/// Порог уверенности отдельной модели, чтобы её голос учитывался.
pub const MIN_MODEL_CONF: f32 = 0.60;
/// Порог итоговой уверенности консилиума для действия.
pub const ACT_CONF: f32 = 0.70;
/// Сколько символов контекста задачи отправляем (экономия времени CPU-моделей).
pub const MAX_CONTEXT_CHARS: usize = 2000;

pub const SYSTEM_PROMPT: &str = "Ты участник научного консилиума. Тебе дают вопрос проекта и \
описание проблемы. Оцени, насколько известные данные позволяют ответить «да» или «нет». \
Ответ — строго JSON: {\"stance\":\"support|contradict|neutral\",\"confidence\":0.0,\"key_point\":\"одна фраза по-русски\",\"missing\":\"чего не хватает для ответа\"}. \
support = данные говорят «да», contradict = «нет», neutral = данных недостаточно. \
Не выдумывай числа и ссылки. Если не хватает данных — neutral с низкой уверенностью.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Api {
    Ollama,
    OpenAi,
}

#[derive(Debug, Clone)]
pub struct Member {
    /// имя модели для API-запроса (ровно то, что ждёт провайдер)
    pub model: String,
    /// подпись для журнала и отчёта: `groq/openai/gpt-oss-20b`
    pub label: String,
    pub url: String,
    pub api: Api,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CouncilConfig {
    pub members: Vec<Member>,
    pub timeout_secs: u64,
    pub min_agree: usize,
}

impl CouncilConfig {
    /// Состав по умолчанию — три локальные модели разных семейств (платить не за что).
    pub fn from_env() -> Self {
        let raw = std::env::var("ARGUS_COUNCIL_MODELS").unwrap_or_default();
        let mut members: Vec<Member> = Vec::new();
        if raw.trim().is_empty() {
            // разные семейства, все влезают в CPU без долгих раздумий;
            // reason-модели (deepseek-r1) подключаются через ARGUS_COUNCIL_MODELS
            for m in ["qwen2.5:7b-16k", "llama3.2:latest", "hermes3:latest"] {
                members.push(Member {
                    model: m.to_string(),
                    label: m.to_string(),
                    url: std::env::var("ARGUS_COUNCIL_OLLAMA")
                        .unwrap_or_else(|_| "http://127.0.0.1:11434/api/chat".to_string()),
                    api: Api::Ollama,
                    api_key: None,
                });
            }
        } else {
            for spec in raw.split(',') {
                let spec = spec.trim();
                if spec.is_empty() {
                    continue;
                }
                // формат: model@url  либо  model  (по умолчанию Ollama)
                let (model, url) = match spec.split_once('@') {
                    Some((m, u)) => (m.trim().to_string(), u.trim().to_string()),
                    None => (
                        spec.to_string(),
                        std::env::var("ARGUS_COUNCIL_OLLAMA")
                            .unwrap_or_else(|_| "http://127.0.0.1:11434/api/chat".to_string()),
                    ),
                };
                let is_ollama = url.contains("11434") || url.ends_with("/api/chat");
                let key_var = format!(
                    "ARGUS_COUNCIL_KEY_{}",
                    model
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
                        .collect::<String>()
                );
                members.push(Member {
                    api_key: std::env::var(&key_var).ok().or_else(|| std::env::var("ARGUS_COUNCIL_KEY").ok()),
                    label: model.clone(),
                    model,
                    url,
                    api: if is_ollama { Api::Ollama } else { Api::OpenAi },
                });
            }
        }
        // облачные бесплатные тиры (ключи берём из конфига pi, не из окружения)
        if std::env::var("ARGUS_COUNCIL_CLOUD").map(|v| v != "0").unwrap_or(false) {
            // cerebras → 402 payment required, together/fireworks → 403/биллинг,
            // huggingface без ключа-router → не подключаем по умолчанию
            let list = std::env::var("ARGUS_COUNCIL_PROVIDERS")
                .unwrap_or_else(|_| "groq,openrouter".to_string());
            for prov in list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                let Some((base, key)) = CouncilConfig::provider_creds(prov) else {
                    continue;
                };
                let model = CouncilConfig::discover_model(prov, 20)
                    .or_else(|| CouncilConfig::free_model(prov).map(|m| m.to_string()));
                let Some(model) = model else {
                    continue;
                };
                members.push(Member {
                    model: model.clone(),
                    label: format!("{prov}/{model}"),
                    url: format!("{base}/chat/completions"),
                    api: Api::OpenAi,
                    api_key: Some(key),
                });
            }
        }

        CouncilConfig {
            members,
            timeout_secs: std::env::var("ARGUS_COUNCIL_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120),
            min_agree: std::env::var("ARGUS_COUNCIL_MIN_AGREE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(MIN_AGREE),
        }
    }

    /// Ключи и адреса бесплатных тиров из `~/.pi/agent/models.json`.
    ///
    /// Ключи уже лежат в конфиге pi — их не надо спрашивать у человека (решение Джабы
    /// 14.11: «найди сам»). Читаем `providers.<name>.apiKey/baseUrl`, ничего не печатаем.
    pub fn provider_creds(provider: &str) -> Option<(String, String)> {
        let path = std::env::var("ARGUS_PROVIDERS_JSON")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                dirs::home_dir().map(|h| h.join(".pi/agent/models.json"))
            })?;
        let raw = std::fs::read_to_string(path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let p = v.get("providers")?.get(provider)?;
        let key = p.get("apiKey")?.as_str()?.trim().to_string();
        // `$DEEPINFRA_API_KEY` и подобные подстановки не раскрываем — только литеральные ключи
        if key.is_empty() || key.starts_with('$') {
            return None;
        }
        let mut base = p.get("baseUrl")?.as_str()?.trim_end_matches('/').to_string();
        // домен api-inference.huggingface.co больше не резолвится — рабочий роутер
        if provider == "huggingface" {
            base = "https://router.huggingface.co/v1".to_string();
        }
        Some((base, key))
    }

    /// Модель по умолчанию для бесплатного тира провайдера (фолбэк, если discovery не сработал).
    pub fn free_model(provider: &str) -> Option<&'static str> {
        match provider {
            "groq" => Some("openai/gpt-oss-20b"),
            "cerebras" => Some("qwen-3.8-27b"),
            "openrouter" => Some("meta-llama/llama-3.1-8b-instruct:free"),
            "mistral" => Some("mistral-small-latest"),
            "huggingface" => Some("meta-llama/Llama-3.1-8B-Instruct"),
            _ => None,
        }
    }

    /// Спросить у провайдера список моделей и выбрать разумную бесплатную.
    /// Список моделей у бесплатных тиров меняется чаще, чем выходит наш код
    /// (14.11: `llama-3.1-8b-instant` у Groq уже нет — вернулся 404).
    pub fn discover_model(provider: &str, timeout_secs: u64) -> Option<String> {
        let (base, key) = CouncilConfig::provider_creds(provider)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs.max(5)))
            .user_agent("argus-agent/council")
            .build()
            .ok()?;
        let resp = client
            .get(format!("{base}/models"))
            .bearer_auth(&key)
            .send()
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: serde_json::Value = resp.json().ok()?;
        let ids: Vec<String> = v
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|i| i.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        const BAD: [&str; 11] = [
            "whisper", "guard", "tts", "embed", "rerank", "moderation", "image", "audio",
            "vision-only", "code", "ocr",
        ];
        let mut cands: Vec<String> = ids
            .into_iter()
            .filter(|id| {
                let low = id.to_lowercase();
                !BAD.iter().any(|b| low.contains(b))
            })
            .collect();
        // предпочтение: gpt-oss / qwen / llama / mistral — то, что умеет держать JSON-инструкцию
        cands.sort_by_key(|id| {
            let low = id.to_lowercase();
            // у OpenRouter бесплатны только модели с суффиксом :free
            if low.ends_with(":free") {
                0
            } else if low.contains("gpt-oss") {
                1
            } else if low.contains("qwen") {
                2
            } else if low.contains("llama") {
                3
            } else if low.contains("mistral") {
                4
            } else {
                5
            }
        });
        cands.into_iter().next()
    }

    pub fn enabled() -> bool {
        match std::env::var("ARGUS_COUNCIL") {
            Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")),
            Err(_) => true,
        }
    }

    pub fn label(&self) -> String {
        self.members
            .iter()
            .map(|m| m.label.as_str())
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// Мнение одной модели.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opinion {
    pub model: String,
    pub stance: Stance,
    pub confidence: f32,
    pub key_point: String,
    pub missing: String,
}

impl Opinion {
    pub fn stance_str(&self) -> &'static str {
        self.stance.as_str()
    }
    pub fn counts(&self) -> bool {
        self.stance != Stance::Neutral && self.confidence >= MIN_MODEL_CONF
    }
}

/// Итог разбора: согласие, уверенность, объяснение.
#[derive(Debug, Clone)]
pub struct Consensus {
    pub stance: Stance,
    pub confidence: f32,
    pub agree: usize,
    pub total: usize,
    pub reason: String,
    /// подробности по каждой модели — идут в журнал цикла
    pub opinions: Vec<Opinion>,
}

impl Consensus {
    pub fn acts(&self) -> bool {
        self.stance != Stance::Neutral && self.confidence >= ACT_CONF
    }
    pub fn label(&self) -> String {
        format!(
            "council:{}/{}:{}",
            self.agree,
            self.total,
            self.stance.as_str()
        )
    }
}

/// Разбор ответов: детерминированная агрегация, без «усреднения мнений в кашу».
pub fn analyze(opinions: Vec<Opinion>, min_agree: usize) -> Consensus {
    let total = opinions.len();
    let base = Consensus {
        stance: Stance::Neutral,
        confidence: 0.0,
        agree: 0,
        total,
        reason: String::new(),
        opinions: opinions.clone(),
    };
    if total == 0 {
        return Consensus {
            reason: "консилиум недоступен: ни одна модель не ответила".into(),
            ..base
        };
    }
    let n_sup = opinions
        .iter()
        .filter(|o| o.counts() && o.stance == Stance::Support)
        .count();
    let n_con = opinions
        .iter()
        .filter(|o| o.counts() && o.stance == Stance::Contradict)
        .count();

    // оба знака набрали минимум голосов → консилиум спорит
    if n_sup >= min_agree && n_con >= min_agree {
        return Consensus {
            agree: n_sup.max(n_con),
            reason: format!(
                "модели расходятся: за «да» {n_sup}, за «нет» {n_con} — воздержание"
            ),
            ..base
        };
    }
    let stance = if n_sup >= min_agree {
        Stance::Support
    } else if n_con >= min_agree {
        Stance::Contradict
    } else {
        return Consensus {
            agree: n_sup.max(n_con),
            reason: format!(
                "согласия нет: менее {min_agree} моделей одного знака (да {n_sup}, нет {n_con}, всего {total})"
            ),
            ..base
        };
    };
    let (votes, opposite) = if stance == Stance::Support {
        (
            opinions
                .iter()
                .filter(|o| o.counts() && o.stance == Stance::Support)
                .collect::<Vec<_>>(),
            n_con,
        )
    } else {
        (
            opinions
                .iter()
                .filter(|o| o.counts() && o.stance == Stance::Contradict)
                .collect::<Vec<_>>(),
            n_sup,
        )
    };
    if votes.len() * 2 < total {
        return Consensus {
            agree: votes.len(),
            reason: format!(
                "меньшинство: {} из {total} моделей — мнение не принимается",
                votes.len()
            ),
            ..base
        };
    }
    // Уверенность = средняя уверенность согласных × ясность согласия.
    // Ясность считается только по ПРОТИВНИКАМ: «воздержался» — это «не знаю», а не «против»
    // (иначе 2 согласных + 1 воздержавшийся никогда не пробивают порог).
    let mean = votes.iter().map(|o| o.confidence).sum::<f32>() / votes.len() as f32;
    let clarity = votes.len() as f32 / (votes.len() + opposite) as f32;
    let confidence = (mean * clarity).clamp(0.0, 0.95);
    Consensus {
        stance,
        confidence,
        agree: votes.len(),
        total,
        reason: format!(
            "{} из {total} моделей: {} (уверенность {:.2}{})",
            votes.len(),
            stance.as_str(),
            confidence,
            if opposite == 0 {
                ""
            } else {
                ", есть возражающие"
            }
        ),
        opinions,
    }
}

/// Промпт консилиума: вопрос + описание проблемы + известные данные.
pub fn build_prompt(question: &str, problem: &str, context: &str) -> String {
    let ctx: String = context.chars().take(MAX_CONTEXT_CHARS).collect();
    let mut s = format!("ВОПРОС: {question}\n");
    if !problem.trim().is_empty() {
        s.push_str(&format!("ОПИСАНИЕ ПРОБЛЕМЫ: {problem}\n"));
    }
    if !ctx.trim().is_empty() {
        s.push_str(&format!("ИЗВЕСТНЫЕ ДАННЫЕ: {ctx}\n"));
    }
    s
}

/// Опрос консилиума. Каждая модель отвечает независимо (не видит других ответов).
pub fn consult(cfg: &CouncilConfig, question: &str, problem: &str, context: &str) -> Consensus {
    let prompt = build_prompt(question, problem, context);
    let mut opinions: Vec<Opinion> = Vec::new();
    for m in &cfg.members {
        match ask_member(cfg, m, &prompt) {
            Ok(o) => opinions.push(o),
            Err(e) => opinions.push(Opinion {
                model: format!("{}!", m.label),
                stance: Stance::Neutral,
                confidence: 0.0,
                key_point: String::new(),
                missing: format!("не ответила: {e}"),
            }),
        }
    }
    analyze(opinions, cfg.min_agree)
}

fn ask_member(cfg: &CouncilConfig, m: &Member, prompt: &str) -> Result<Opinion, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(cfg.timeout_secs.max(10)))
        .user_agent("argus-agent/council")
        .build()
        .map_err(|e| e.to_string())?;
    let payload = match m.api {
        Api::Ollama => serde_json::json!({
            "model": m.model,
            "stream": false,
            "format": "json",
            "keep_alive": "15m",
            "options": {"temperature": 0.2, "num_predict": 220},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        }),
        Api::OpenAi => serde_json::json!({
            "model": m.model,
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        }),
    };
    let mut req = client.post(&m.url).json(&payload);
    if let Some(k) = &m.api_key {
        req = req.bearer_auth(k);
    }
    let resp = req.send().map_err(|e| format!("сеть: {e}"))?;
    let status = resp.status();
    let body = resp.text().map_err(|e| format!("тело: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "HTTP {}: {}",
            status.as_u16(),
            body.chars().take(120).collect::<String>()
        ));
    }
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("JSON: {e}"))?;
    let content = match m.api {
        Api::Ollama => v
            .get("message")
            .map(|x| {
                let c = x.get("content").and_then(|c| c.as_str()).unwrap_or("");
                if c.trim().is_empty() {
                    // reasoning-модели (deepseek-r1) кладут текст в `thinking`
                    x.get("thinking").and_then(|c| c.as_str()).unwrap_or("")
                } else {
                    c
                }
            }),
        Api::OpenAi => v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .map(|x| {
                let c = x.get("content").and_then(|c| c.as_str()).unwrap_or("");
                if c.trim().is_empty() {
                    // reasoning-модели (gpt-oss, ernie …) кладут текст в reasoning
                    x.get("reasoning")
                        .and_then(|r| r.as_str())
                        .unwrap_or("")
                } else {
                    c
                }
            }),
    }
    .ok_or_else(|| "нет поля с ответом".to_string())?;
    parse_opinion(&m.label, content)
}

/// Разбор ответа одной модели.
pub fn parse_opinion(model: &str, text: &str) -> Result<Opinion, String> {
    let body = match (text.find('{'), text.rfind('}')) {
        (Some(a), Some(b)) if b > a => &text[a..=b],
        _ => text,
    };
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("ответ не JSON: {e}"))?;
    let stance = match v
        .get("stance")
        .and_then(|s| s.as_str())
        .unwrap_or("neutral")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "support" | "supports" | "да" | "yes" => Stance::Support,
        "contradict" | "contradicts" | "нет" | "no" => Stance::Contradict,
        _ => Stance::Neutral,
    };
    Ok(Opinion {
        model: model.to_string(),
        stance,
        confidence: v
            .get("confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0) as f32,
        key_point: clean(v.get("key_point").and_then(|s| s.as_str()).unwrap_or("")),
        missing: clean(v.get("missing").and_then(|s| s.as_str()).unwrap_or("")),
    })
}

fn clean(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(*c as u32, 0x3000..=0x9FFF | 0xAC00..=0xD7AF))
        .take(200)
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(model: &str, stance: Stance, conf: f32) -> Opinion {
        Opinion {
            model: model.into(),
            stance,
            confidence: conf,
            key_point: "k".into(),
            missing: String::new(),
        }
    }

    #[test]
    fn two_agreeing_models_produce_actionable_consensus() {
        let c = analyze(
            vec![
                op("a", Stance::Support, 0.85),
                op("b", Stance::Support, 0.9),
                op("c", Stance::Neutral, 0.2),
            ],
            MIN_AGREE,
        );
        assert_eq!(c.stance, Stance::Support);
        assert!(c.acts(), "conf={} reason={}", c.confidence, c.reason);
        assert_eq!((c.agree, c.total), (2, 3));
    }

    #[test]
    fn single_model_is_not_a_consensus() {
        let c = analyze(
            vec![op("a", Stance::Support, 0.95), op("b", Stance::Neutral, 0.0)],
            MIN_AGREE,
        );
        assert_eq!(c.stance, Stance::Neutral);
        assert!(!c.acts());
        assert!(c.reason.contains("согласия нет"), "{}", c.reason);
    }

    #[test]
    fn split_council_abstains() {
        let c = analyze(
            vec![
                op("a", Stance::Support, 0.9),
                op("b", Stance::Contradict, 0.9),
                op("c", Stance::Neutral, 0.0),
            ],
            1,
        );
        assert_eq!(c.stance, Stance::Neutral);
        assert!(c.reason.contains("расходятся"), "{}", c.reason);
    }

    #[test]
    fn low_confidence_voices_are_ignored() {
        let c = analyze(
            vec![
                op("a", Stance::Support, 0.4),
                op("b", Stance::Support, 0.5),
                op("c", Stance::Support, 0.3),
            ],
            MIN_AGREE,
        );
        assert_eq!(c.stance, Stance::Neutral, "низкая уверенность не считается голосом");
    }

    #[test]
    fn empty_council_is_neutral_with_reason() {
        let c = analyze(vec![], MIN_AGREE);
        assert_eq!(c.stance, Stance::Neutral);
        assert!(c.reason.contains("недоступен"));
    }

    #[test]
    fn parses_opinion_and_cleans_cjk() {
        let o = parse_opinion(
            "m",
            r#"{"stance":"support","confidence":0.8,"key_point":"данные дают 0.6 px","missing":""}"#,
        )
        .unwrap();
        assert_eq!(o.stance, Stance::Support);
        assert!(o.key_point.contains("0.6 px"));
        let o2 = parse_opinion("m", r#"{"stance":"neutral","confidence":0.1,"key_point":"文本"}"#).unwrap();
        assert!(!o2.key_point.contains('文'));
    }

    #[test]
    fn thinking_only_answer_is_parsed() {
        // deepseek-r1 возвращает пустой content и текст в thinking
        let o = parse_opinion(
            "deepseek-r1",
            "думаю... готово: {\"stance\":\"contradict\",\"confidence\":0.7,\"key_point\":\"нет данных\"}",
        )
        .unwrap();
        assert_eq!(o.stance, Stance::Contradict);
    }

    #[test]
    fn prompt_includes_question_problem_and_context() {
        let p = build_prompt("Держит ли фокус?", "72 ч съёмка", "дрейф 0.6 px/24 ч");
        assert!(p.contains("ВОПРОС: Держит ли фокус?"));
        assert!(p.contains("ОПИСАНИЕ ПРОБЛЕМЫ: 72 ч съёмка"));
        assert!(p.contains("ИЗВЕСТНЫЕ ДАННЫЕ: дрейф 0.6 px/24 ч"));
    }

    #[test]
    fn provider_creds_are_read_without_exposing_keys() {
        if let Some((base, key)) = CouncilConfig::provider_creds("groq") {
            assert!(base.starts_with("http"), "{base}");
            assert!(!key.is_empty());
            assert!(!key.starts_with('$'), "подстановки не раскрываем");
        }
        // несуществующий провайдер — просто None, без паники
        assert!(CouncilConfig::provider_creds("нет-такого").is_none());
    }

    #[test]
    fn free_model_table_has_no_paid_defaults() {
        assert!(CouncilConfig::free_model("groq").is_some());
        assert_eq!(CouncilConfig::free_model("groq"), Some("openai/gpt-oss-20b"));
        assert!(CouncilConfig::free_model("deepseek").is_none(), "платные не по умолчанию");
        assert!(CouncilConfig::free_model("deepinfra").is_none());
    }

    #[test]
    fn consensus_label_is_auditable() {
        let c = analyze(
            vec![op("a", Stance::Contradict, 0.9), op("b", Stance::Contradict, 0.8)],
            2,
        );
        assert_eq!(c.label(), "council:2/2:contradict");
    }
}
