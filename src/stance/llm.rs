//! LLM-судья позиции свидетельства (`--judge llm|hybrid`).
//!
//! Лексический классификатор (`crate::stance`) работает без сети, но у него есть слепое
//! пятно: русский вопрос + английский абстракт почти всегда даёт «позиции в тексте нет»,
//! даже когда абстракт прямо утверждает претензию. LLM-судья закрывает именно этот случай.
//!
//! Правила (те же, что у лексики — «молчать дешевле, чем соврать»):
//! - формат ответа строгий JSON, вердикт на каждое свидетельство;
//! - `neutral` обязателен, если текст не утверждает ничего о претензии (тематическое
//!   совпадение — НЕ подтверждение);
//! - действует только `confidence >= crate::stance::ACT_CONF`;
//! - строка «чего не хватает» не запрашивается: одна позиция, одна уверенность.
//!
//! Экономика: 1 HTTP-вызов на цикл (все кандидаты в одном запросе), кэш вердиктов по
//! `(вопрос, идентификатор, модель)`, вызов только когда есть новые идентификаторы.
//! На CPU-модели 7B это ~2 с тёплого вызова; таймаут → возврат к лексике (Evidence),
//! цикл не блокируется и не тратит деньги.
//!
//! Конфигурация (env):
//! - `ARGUS_JUDGE=0` — выключить полностью;
//! - `ARGUS_JUDGE_URL` — endpoint; по умолчанию `http://127.0.0.1:11434/api/chat` (Ollama);
//! - `ARGUS_JUDGE_MODEL` — модель; по умолчанию `qwen2.5:7b-16k`;
//! - `ARGUS_JUDGE_KEY` — если задан, используется OpenAI-совместимый протокол
//!   (`/v1/chat/completions`, Bearer), иначе — нативный Ollama;
//! - `ARGUS_JUDGE_TIMEOUT` — секунды на вызов (по умолчанию 45).

use crate::stance::{Decision, Stance};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Сколько символов абстракта отправляем судье (экономия и защита контекста).
pub const MAX_ABSTRACT_CHARS: usize = 1400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Api {
    /// POST {url} {"model","messages","stream":false,"format":"json","keep_alive"}
    Ollama,
    /// POST {url} {"model","messages","max_tokens"} + Authorization: Bearer
    OpenAi,
}

#[derive(Debug, Clone)]
pub struct JudgeConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
    pub api: Api,
}

impl JudgeConfig {
    pub fn from_env() -> Self {
        let model =
            std::env::var("ARGUS_JUDGE_MODEL").unwrap_or_else(|_| "qwen2.5:7b-16k".to_string());
        let api_key = std::env::var("ARGUS_JUDGE_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty());
        let ai = if api_key.is_some() {
            Api::OpenAi
        } else {
            Api::Ollama
        };
        let default_url = match ai {
            Api::Ollama => "http://127.0.0.1:11434/api/chat".to_string(),
            Api::OpenAi => "https://api.deepinfra.com/v1/openai/chat/completions".to_string(),
        };
        JudgeConfig {
            url: std::env::var("ARGUS_JUDGE_URL").unwrap_or(default_url),
            model,
            api_key,
            timeout_secs: std::env::var("ARGUS_JUDGE_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(45),
            api: ai,
        }
    }

    pub fn enabled() -> bool {
        match std::env::var("ARGUS_JUDGE") {
            Ok(v) => {
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            }
            Err(_) => true,
        }
    }

    /// Метка источника для журнала: `llm:qwen2.5:7b-16k`.
    pub fn source_label(&self) -> String {
        format!("llm:{}", self.model)
    }
}

// ── кэш вердиктов ───────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CachedVerdict {
    pub stance: String,
    pub confidence: f32,
    pub why: String,
    pub model: String,
    pub at: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct JudgeCache {
    #[serde(flatten)]
    pub entries: BTreeMap<String, CachedVerdict>,
}

impl JudgeCache {
    pub fn path(state_dir: &Path) -> PathBuf {
        state_dir.join("ledger").join("stance_cache.json")
    }

    pub fn load(state_dir: &Path) -> Self {
        std::fs::read_to_string(Self::path(state_dir))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, state_dir: &Path) -> std::io::Result<()> {
        let p = Self::path(state_dir);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let s = serde_json::to_string_pretty(&self).map_err(std::io::Error::other)?;
        std::fs::write(p, s)
    }

    pub fn get(&self, key: &str) -> Option<&CachedVerdict> {
        self.entries.get(key)
    }

    pub fn put(&mut self, key: String, v: CachedVerdict) {
        self.entries.insert(key, v);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Ключ кэша: вердикт привязан к паре (вопрос, идентификатор) и к модели.
pub fn cache_key(qid: &str, identifier: &str, model: &str) -> String {
    format!("{qid}|{identifier}|{model}")
}

// ── промпт ──────────────────────────────────────────────────────────────────

pub const SYSTEM_PROMPT: &str = "Ты оцениваешь позицию ОДНОГО текста относительно утверждения. \
Ответ — строго JSON: {\"stance\":\"support|contradict|neutral\",\"confidence\":0.0,\"quote\":\"точная цитата из текста или пустая строка\",\"why\":\"до 12 слов по-русски\"}. \
Правила: support — текст явно утверждает, что утверждение верно (нужна цитата); \
contradict — текст явно утверждает противоположное (нужна цитата); \
neutral — текст не упоминает ключевую деталь утверждения, говорит о другом, либо неопределён. \
Отсутствие упоминания — это neutral, а НЕ contradict. Без цитаты support/contradict запрещены.";

pub struct Item<'a> {
    pub id: &'a str,
    pub title: &'a str,
    pub abstract_text: &'a str,
}

pub fn build_user_prompt(question: &str, item: &Item<'_>) -> String {
    let abs: String = item
        .abstract_text
        .chars()
        .take(MAX_ABSTRACT_CHARS)
        .collect();
    format!(
        "УТВЕРЖДЕНИЕ: {question}\n\nЗАГОЛОВОК: {}\nТЕКСТ: {}\n",
        item.title,
        if abs.trim().is_empty() {
            "(нет абстракта — оценивать только по заголовку и, если нельзя, ставь neutral)"
        } else {
            abs.as_str()
        }
    )
}

/// Детерминированные ворота для вердикта LLM. Возвращает причину отказа или Ok(()).
///
/// Зачем: 7B-модель уверенно говорит `support` на текст, где нет ключевой детали
/// (пример 13 ноя: paper про «rigid cage removes thermal drift, drift < 1 px over 24 h»
/// → `support` для утверждения про *fiducial*-автофокус). Поэтому вердикт принимается
/// только если модель привела цитату, цитата реально есть в тексте и содержит
/// якоря утверждения (латинский термин и/или число).
pub fn verdict_gate(question: &str, evidence_text: &str, quote: &str) -> Result<(), String> {
    let q = quote.trim();
    if q.is_empty() {
        return Err("нет цитаты".to_string());
    }
    let ev = norm(evidence_text);
    let qn = norm(q);
    if qn.chars().count() < 8 {
        return Err("цитата слишком короткая".to_string());
    }
    if !ev.contains(&qn) {
        return Err("цитаты нет в тексте".to_string());
    }
    // якорь-термин: латинские слова вопроса (≥4 символов)
    let anchors: Vec<String> = question
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .filter(|w| {
            ![
                "does",
                "with",
                "from",
                "this",
                "that",
                "over",
                "than",
                "published",
                "shown",
            ]
            .contains(w)
        })
        .map(|w| w.to_string())
        .collect();
    if !anchors.is_empty() && !anchors.iter().any(|a| qn.contains(a.as_str())) {
        return Err(format!(
            "в цитате нет якоря утверждения ({})",
            anchors.join("/")
        ));
    }
    // числовой якорь: если в вопросе есть числа, хотя бы одно должно быть в цитате
    let nums = numbers(question);
    if !nums.is_empty() && !nums.iter().any(|n| has_number(&qn, n)) {
        return Err(format!(
            "в цитате нет чисел утверждения ({})",
            nums.join("/")
        ));
    }
    Ok(())
}

fn norm(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn numbers(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || (ch == '.' && !cur.is_empty()) {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|n| !n.trim_matches('.').is_empty() && n != ".");
    out.dedup();
    out
}

/// Число как отдельный токен (чтобы «1» не совпадало внутри «0.1» или «12»).
fn has_number(hay: &str, num: &str) -> bool {
    let mut from = 0;
    while let Some(pos) = hay[from..].find(num) {
        let start = from + pos;
        let end = start + num.len();
        let before_ok = hay[..start]
            .chars()
            .next_back()
            .map(|c| !c.is_ascii_digit() && c != '.')
            .unwrap_or(true);
        let after_ok = hay[end..]
            .chars()
            .next()
            .map(|c| !c.is_ascii_digit() && c != '.')
            .unwrap_or(true);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

// ── парсинг ответа ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Verdict {
    pub stance: Stance,
    pub confidence: f32,
    pub quote: String,
    pub why: String,
}

/// Разбор ответа судьи об ОДНОМ тексте. `None` — ответ не разобран.
pub fn parse_verdict(text: &str) -> Option<Verdict> {
    let body = match (text.find('{'), text.rfind('}')) {
        (Some(a), Some(b)) if b > a => &text[a..=b],
        _ => text,
    };
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let stance = match v
        .get("stance")
        .and_then(|s| s.as_str())
        .unwrap_or("neutral")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "support" | "supports" => Stance::Support,
        "contradict" | "contradicts" | "refute" => Stance::Contradict,
        _ => Stance::Neutral,
    };
    Some(Verdict {
        stance,
        confidence: v
            .get("confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0) as f32,
        quote: v
            .get("quote")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .trim()
            .to_string(),
        why: sanitize(v.get("why").and_then(|w| w.as_str()).unwrap_or("")),
    })
}

/// Убрать CJK: qwen иногда съезжает в китайский, а объяснение читает человек.
fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(*c as u32, 0x3000..=0x9FFF | 0xAC00..=0xD7AF))
        .take(160)
        .collect::<String>()
        .trim()
        .to_string()
}

// ── вызов ───────────────────────────────────────────────────────────────────

pub struct JudgeOutcome {
    pub decisions: Vec<Decision>,
    /// токены, потраченные на вызовы (0 при кэше/ошибке) — идут в дневной бюджет
    pub tokens: u64,
    pub note: String,
}

/// Спросить судью о свидетельствах. Один вызов на свидетельство (батч из 3 текстов
/// 7B-модель выполняла нестабильно: возвращала вердикт только на первый).
/// Кэш по `(вопрос, идентификатор, модель)`; таймаут/ошибка → `Neutral` (цикл не падает).
pub fn judge_batch(
    cfg: &JudgeConfig,
    qid: &str,
    question: &str,
    items: &[Item<'_>],
    cache: &mut JudgeCache,
) -> JudgeOutcome {
    let source: &'static str = source_label(cfg);
    let mut decisions: Vec<Decision> = Vec::with_capacity(items.len());
    let mut tokens: u64 = 0;
    let mut fresh = 0usize;
    let mut gated = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for it in items {
        let key = cache_key(qid, it.id, &cfg.model);
        if let Some(v) = cache.get(&key) {
            let stance = match v.stance.as_str() {
                "support" => Stance::Support,
                "contradict" => Stance::Contradict,
                _ => Stance::Neutral,
            };
            decisions.push(Decision {
                stance,
                confidence: v.confidence,
                source,
                reason: format!("кэш судьи: {}", v.why),
            });
            continue;
        }

        let evidence = format!(
            "{} {}",
            it.title,
            it.abstract_text
                .chars()
                .take(MAX_ABSTRACT_CHARS)
                .collect::<String>()
        );
        let prompt = build_user_prompt(question, it);
        let (body, used) = match call_model(cfg, &prompt) {
            Ok(v) => v,
            Err(e) => {
                errors.push(e);
                decisions.push(Decision::neutral(
                    source,
                    "судья недоступен — остаётся лексика",
                ));
                continue;
            }
        };
        tokens += used;
        fresh += 1;
        let Some(v) = parse_verdict(&body) else {
            decisions.push(Decision::neutral(source, "ответ судьи не разобран"));
            continue;
        };
        // ворота: действует только подтверждённая цитатой позиция
        let mut final_stance = v.stance;
        let mut reason = format!("судья: {}", v.why);
        if final_stance != Stance::Neutral {
            match verdict_gate(question, &evidence, &v.quote) {
                Ok(()) => {
                    if v.confidence < crate::stance::ACT_CONF {
                        reason = format!(
                            "судья: {} с низкой уверенностью {:.2} < {} (ворота пройдены, но рано)",
                            final_stance.as_str(),
                            v.confidence,
                            crate::stance::ACT_CONF
                        );
                        final_stance = Stance::Neutral;
                    }
                }
                Err(why) => {
                    gated += 1;
                    reason = format!(
                        "судья сказал {}, но ворота отклонили: {why}",
                        final_stance.as_str()
                    );
                    final_stance = Stance::Neutral;
                }
            }
        }
        let d = if final_stance == Stance::Neutral {
            Decision::neutral(source, reason)
        } else {
            Decision {
                stance: final_stance,
                confidence: v.confidence,
                source,
                reason,
            }
        };
        cache.put(
            key,
            CachedVerdict {
                stance: d.stance.as_str().to_string(),
                confidence: d.confidence,
                why: d.reason.chars().take(200).collect(),
                model: cfg.model.clone(),
                at: chrono::Local::now().to_rfc3339(),
            },
        );
        decisions.push(d);
    }

    let note = if !errors.is_empty() {
        format!(
            "судья недоступен ({}), вердиктов нет — остаётся лексика",
            errors[0]
        )
    } else if fresh == 0 {
        format!("судья: все {} вердиктов из кэша", items.len())
    } else {
        format!("судья: {fresh} новых вердиктов, отклонено воротами {gated}, {tokens} токенов")
    };
    JudgeOutcome {
        decisions,
        tokens,
        note,
    }
}

/// Метка источника как `&'static str` (Decision хранит статическую строку).
/// Внутри процесса конфигурация одна, поэтому метка кэшируется, а не течёт каждый вызов.
fn source_label(cfg: &JudgeConfig) -> &'static str {
    use std::sync::OnceLock;
    static LABEL: OnceLock<&'static str> = OnceLock::new();
    LABEL.get_or_init(|| Box::leak(cfg.source_label().into_boxed_str()))
}

fn call_model(cfg: &JudgeConfig, user: &str) -> Result<(String, u64), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(cfg.timeout_secs.max(5)))
        .user_agent("argus-agent/stance-judge")
        .build()
        .map_err(|e| e.to_string())?;

    let (payload, url) = match cfg.api {
        Api::Ollama => (
            serde_json::json!({
                "model": cfg.model,
                "stream": false,
                "format": "json",
                "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": 300},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}
                ]
            }),
            cfg.url.clone(),
        ),
        Api::OpenAi => (
            serde_json::json!({
                "model": cfg.model,
                "temperature": 0.0,
                "max_tokens": 400,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}
                ]
            }),
            cfg.url.clone(),
        ),
    };

    let mut req = client.post(&url).json(&payload);
    if let Some(k) = &cfg.api_key {
        req = req.bearer_auth(k);
    }
    let resp = req.send().map_err(|e| format!("сеть: {e}"))?;
    let status = resp.status();
    let body = resp.text().map_err(|e| format!("тело: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "HTTP {}: {}",
            status.as_u16(),
            body.chars().take(160).collect::<String>()
        ));
    }
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("JSON: {e}"))?;
    let content = match cfg.api {
        Api::Ollama => v
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str()),
        Api::OpenAi => v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str()),
    }
    .ok_or_else(|| "нет поля с ответом модели".to_string())?
    .to_string();

    let tokens = match cfg.api {
        Api::Ollama => {
            let a = v
                .get("prompt_eval_count")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            let b = v.get("eval_count").and_then(|x| x.as_u64()).unwrap_or(0);
            a + b
        }
        Api::OpenAi => v
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0),
    };
    Ok((content, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_unreachable() -> JudgeConfig {
        JudgeConfig {
            url: "http://127.0.0.1:1/never".into(),
            model: "test-model".into(),
            api_key: None,
            timeout_secs: 1,
            api: Api::Ollama,
        }
    }

    const Q: &str = "Показано ли, что fiducial-автофокус удерживает дрейф < 1 px за 24 ч?";

    #[test]
    fn parses_single_verdict_with_quote() {
        let v = parse_verdict(
            r#"{"stance":"support","confidence":0.9,"quote":"drift stayed below 1 px over 24 h","why":"дрейф в норме"}"#,
        )
        .unwrap();
        assert_eq!(v.stance, Stance::Support);
        assert!((v.confidence - 0.9).abs() < 1e-6);
        assert!(v.quote.contains("1 px"));
    }

    #[test]
    fn cjk_is_stripped_from_why() {
        let v = parse_verdict(
            r#"{"stance":"neutral","confidence":0.2,"quote":"","why":"文本未提及 fiducial"}"#,
        )
        .unwrap();
        assert!(!v.why.contains('文'), "{}", v.why);
        assert!(v.why.contains("fiducial"), "{}", v.why);
    }

    #[test]
    fn json_in_code_fence_is_tolerated() {
        let v = parse_verdict(
            "```json\n{\"stance\":\"contradict\",\"confidence\":0.77,\"quote\":\"x\"}\n```",
        )
        .unwrap();
        assert_eq!(v.stance, Stance::Contradict);
    }

    #[test]
    fn non_json_answer_is_rejected() {
        assert!(parse_verdict("не JSON вообще").is_none());
    }

    #[test]
    fn gate_accepts_quote_with_anchor_and_number() {
        let ev = "We evaluated fiducial-based autofocus on an open microscope. \
                  Focus drift stayed below 1 px over 24 h in all runs.";
        // цитата, которая действительно подтверждает претензию — с якорем и числом
        assert!(verdict_gate(
            Q,
            ev,
            "fiducial-based autofocus on an open microscope. Focus drift stayed below 1 px over 24 h"
        )
        .is_ok());
        // та же мысль без якоря отклоняется (это и есть защита от ложного support)
        assert!(verdict_gate(Q, ev, "Focus drift stayed below 1 px over 24 h").is_err());
    }

    #[test]
    fn gate_rejects_quote_missing_from_text() {
        let ev = "We evaluated fiducial autofocus.";
        assert!(verdict_gate(Q, ev, "drift stayed below 1 px over 24 h").is_err());
    }

    #[test]
    fn gate_rejects_quote_without_claim_anchor() {
        // реальный провал 13 ноя: rigid cage вместо fiducial-автофокуса
        let ev = "A rigid cage system removes thermal drift; drift stayed below 1 px over 24 h.";
        let err = verdict_gate(Q, ev, "drift stayed below 1 px over 24 h").unwrap_err();
        assert!(err.contains("якоря"), "{err}");
    }

    #[test]
    fn gate_rejects_quote_without_claim_numbers() {
        let ev = "Fiducial autofocus drifted 7 px over 6 h.";
        let err = verdict_gate(Q, ev, "Fiducial autofocus drifted 7 px over 6 h").unwrap_err();
        assert!(err.contains("чисел"), "{err}");
    }

    #[test]
    fn number_token_match_is_exact() {
        assert!(has_number("below 1 px", "1"));
        assert!(
            !has_number("below 0.1 px", "1"),
            "«1» не должна совпадать внутри «0.1»"
        );
        assert!(!has_number("over 12 h", "1"));
    }

    #[test]
    fn cache_roundtrip_and_key_is_model_specific() {
        let dir = std::env::temp_dir().join(format!("argus-judge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut c = JudgeCache::load(&dir);
        assert!(c.is_empty());
        let k = cache_key("q-1", "PMID:1", "m1");
        assert_ne!(k, cache_key("q-1", "PMID:1", "m2"));
        c.put(
            k.clone(),
            CachedVerdict {
                stance: "support".into(),
                confidence: 0.9,
                why: "w".into(),
                model: "m1".into(),
                at: "now".into(),
            },
        );
        c.save(&dir).unwrap();
        let c2 = JudgeCache::load(&dir);
        assert_eq!(c2.len(), 1);
        assert_eq!(c2.get(&k).unwrap().stance, "support");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unreachable_endpoint_returns_neutral_without_panic() {
        let mut cache = JudgeCache::default();
        let items = vec![Item {
            id: "PMID:1",
            title: "t",
            abstract_text: "a",
        }];
        let out = judge_batch(&cfg_unreachable(), "q-1", Q, &items, &mut cache);
        assert_eq!(out.decisions.len(), 1);
        assert_eq!(out.decisions[0].stance, Stance::Neutral);
        assert_eq!(out.tokens, 0);
        assert!(out.note.contains("недоступен"), "{}", out.note);
    }

    #[test]
    fn cache_hit_skips_call() {
        let mut cache = JudgeCache::default();
        let items = vec![Item {
            id: "PMID:7",
            title: "t",
            abstract_text: "a",
        }];
        cache.put(
            cache_key("q-1", "PMID:7", "test-model"),
            CachedVerdict {
                stance: "neutral".into(),
                confidence: 0.0,
                why: "нет позиции".into(),
                model: "test-model".into(),
                at: "now".into(),
            },
        );
        let out = judge_batch(&cfg_unreachable(), "q-1", Q, &items, &mut cache);
        assert!(out.note.contains("из кэша"), "{}", out.note);
        assert_eq!(out.decisions[0].stance, Stance::Neutral);
    }

    #[test]
    fn prompt_contains_claim_title_and_flag_for_missing_abstract() {
        let it = Item {
            id: "PMID:1",
            title: "T1",
            abstract_text: "",
        };
        let p = build_user_prompt(Q, &it);
        assert!(p.contains("УТВЕРЖДЕНИЕ: "));
        assert!(p.contains("ЗАГОЛОВОК: T1"));
        assert!(p.contains("нет абстракта"));
    }
}
