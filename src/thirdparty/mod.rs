//! Чужие эксперименты и данные (решение Джабы 14.11: «изучать чужие эксперименты и данные»).
//!
//! Лабораторный канал отвечает нашими измерениями; здесь — **внешние** данные: наборы
//! данных и репозитории, где другие люди уже измерили то, о чём мы спрашиваем.
//!
//! Источники (все бесплатные, без ключей):
//! - **Zenodo** — датасеты и препринт-материалы: `zenodo.org/api/records?q=…`;
//! - **GitHub** — репозитории открытых микроскопов/трекеров: `api.github.com/search/repositories`;
//! - далее по конвейеру каждый кандидат проходит LLM-судью позиции (`crate::stance::llm`)
//!   с воротами цитаты: подтверждение принимается только с цитатой из описания.
//!
//! Правила честности:
//! 1) найденное чужое измерение **не равно** нашему: провенанс пишется как `third-party`;
//! 2) применяется только при наличии живого идентификатора (`doi:` / `url:`) и только если
//!    ссылка отвечает (проверка доступности);
//! 3) при равенстве голосов или отсутствии цитаты — воздержание;
//! 4) ничего не «додумывается»: цитата обязана быть в тексте источника.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Сколько кандидатов берём с каждого источника.
pub const PER_SOURCE: usize = 4;
/// Максимум символов описания, отправляемых судье.
pub const MAX_SUMMARY_CHARS: usize = 1800;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Zenodo,
    Github,
}

impl SourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::Zenodo => "zenodo",
            SourceKind::Github => "github",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// `doi:10.5281/zenodo.…` либо `url:https://…`
    pub id: String,
    pub title: String,
    pub summary: String,
    pub url: String,
    pub kind: SourceKind,
}

/// Запись о чужом эксперименте, принятая в леджер.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalEvidence {
    pub qid: String,
    /// `support` | `contradict`
    pub verdict: String,
    /// цитата из источника (то, на чём основан вердикт)
    pub value: String,
    /// идентификатор источника: doi:… либо url:…
    pub source: String,
    pub title: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub at: Option<String>,
    /// всегда true: это чужие данные, не наше измерение
    #[serde(default)]
    pub third_party: bool,
}

pub fn path_for(state_dir: &Path, qid: &str) -> PathBuf {
    state_dir
        .join("lab")
        .join("evidence")
        .join(format!("{qid}.external.json"))
}

pub fn load(state_dir: &Path, qid: &str) -> Option<ExternalEvidence> {
    std::fs::read_to_string(path_for(state_dir, qid))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

pub fn record(state_dir: &Path, ev: &ExternalEvidence) -> std::io::Result<PathBuf> {
    let p = path_for(state_dir, &ev.qid);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let s = serde_json::to_string_pretty(ev).map_err(std::io::Error::other)?;
    std::fs::write(&p, s)?;
    Ok(p)
}

fn http_get(url: &str) -> Result<(u16, String), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(25))
        .user_agent("argus-agent/thirdparty (research; contact: lab)")
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url).send().map_err(|e| format!("сеть: {e}"))?;
    let status = resp.status().as_u16();
    let body = resp.text().map_err(|e| format!("тело: {e}"))?;
    Ok((status, body))
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Zenodo: датасеты/материалы, у которых есть DOI.
pub fn search_zenodo(query: &str, limit: usize) -> Result<Vec<Candidate>, String> {
    let url = format!(
        "https://zenodo.org/api/records?q={}&size={}&sort=bestmatch",
        urlencode(query),
        limit.clamp(1, 10)
    );
    let (status, body) = http_get(&url)?;
    if status != 200 {
        return Err(format!("zenodo: HTTP {status}"));
    }
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("zenodo json: {e}"))?;
    let mut out = Vec::new();
    let Some(hits) = v
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|h| h.as_array())
    else {
        return Ok(out);
    };
    for r in hits {
        let meta = r.get("metadata").unwrap_or(&serde_json::Value::Null);
        let title = meta
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let doi = meta
            .get("doi")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let desc = meta
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let html = r
            .get("links")
            .and_then(|l| l.get("self_html"))
            .and_then(|u| u.as_str())
            .unwrap_or("")
            .to_string();
        if title.is_empty() {
            continue;
        }
        let id = if !doi.is_empty() {
            format!("doi:{doi}")
        } else if !html.is_empty() {
            format!("url:{html}")
        } else {
            continue;
        };
        out.push(Candidate {
            id,
            title,
            summary: strip_html(&desc),
            url: html,
            kind: SourceKind::Zenodo,
        });
    }
    Ok(out)
}

/// GitHub: репозитории (описания часто содержат измеренные характеристики).
pub fn search_github(query: &str, limit: usize) -> Result<Vec<Candidate>, String> {
    let url = format!(
        "https://api.github.com/search/repositories?q={}&per_page={}",
        urlencode(query),
        limit.clamp(1, 10)
    );
    let (status, body) = http_get(&url)?;
    if status == 403 || status == 429 {
        return Err("github: rate limit (10 запросов/мин без токена)".to_string());
    }
    if status != 200 {
        return Err(format!("github: HTTP {status}"));
    }
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("github json: {e}"))?;
    let mut out = Vec::new();
    let Some(items) = v.get("items").and_then(|i| i.as_array()) else {
        return Ok(out);
    };
    for r in items {
        let full = r.get("full_name").and_then(|s| s.as_str()).unwrap_or("");
        let html = r.get("html_url").and_then(|s| s.as_str()).unwrap_or("");
        if full.is_empty() || html.is_empty() {
            continue;
        }
        let desc = r.get("description").and_then(|d| d.as_str()).unwrap_or("");
        let topics = r
            .get("topics")
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        out.push(Candidate {
            id: format!("url:{html}"),
            title: full.to_string(),
            summary: format!("{desc}\n{topics}").trim().to_string(),
            url: html.to_string(),
            kind: SourceKind::Github,
        });
    }
    Ok(out)
}

/// Собрать кандидатов из всех источников (ошибки источников не валят цикл).
pub fn gather(query: &str, per_source: usize) -> (Vec<Candidate>, Vec<String>) {
    let mut cands = Vec::new();
    let mut errors = Vec::new();
    match search_zenodo(query, per_source) {
        Ok(mut c) => cands.append(&mut c),
        Err(e) => errors.push(e),
    }
    match search_github(query, per_source) {
        Ok(mut c) => cands.append(&mut c),
        Err(e) => errors.push(e),
    }
    cands.dedup_by(|a, b| a.id == b.id);
    (cands, errors)
}

/// Доступен ли источник прямо сейчас (защита от «применю ссылку, которой нет»).
pub fn source_alive(id: &str) -> Result<(), String> {
    let url = if let Some(doi) = id.strip_prefix("doi:") {
        format!("https://doi.org/{doi}")
    } else if let Some(u) = id.strip_prefix("url:") {
        u.to_string()
    } else {
        return Err(format!("неизвестный идентификатор источника: {id}"));
    };
    let (status, _) = http_get(&url)?;
    if (200..400).contains(&status) {
        Ok(())
    } else {
        Err(format!("источник отвечает HTTP {status}"))
    }
}

fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_html_and_collapses_space() {
        let s = strip_html("<p>Focus <b>drift</b> 0.58 px</p>   over 72 h");
        assert_eq!(s, "Focus drift 0.58 px over 72 h");
    }

    #[test]
    fn record_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("argus-tp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ev = ExternalEvidence {
            qid: "q-8c6d5868".into(),
            verdict: "support".into(),
            value: "drift stayed below 1 px over 24 h".into(),
            source: "doi:10.5281/zenodo.123".into(),
            title: "Fiducial autofocus dataset".into(),
            note: "набор данных сообщества".into(),
            at: Some("2026-11-14".into()),
            third_party: true,
        };
        record(&dir, &ev).unwrap();
        let back = load(&dir, "q-8c6d5868").unwrap();
        assert_eq!(back.verdict, "support");
        assert!(back.third_party);
        assert!(back.value.contains("1 px"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_identifier_rejected_before_network() {
        assert!(source_alive("garbage").is_err());
    }

    #[test]
    fn urlencode_handles_spaces_and_cyrillic() {
        let e = urlencode("fiducial autofocus дрейф");
        assert!(e.contains("fiducial+autofocus"));
        assert!(e.contains('%'));
    }

    #[test]
    fn html_stripping_keeps_numbers() {
        let s = strip_html("<div>0.58&nbsp;px</div>");
        assert!(s.contains("0.58"));
    }
}
