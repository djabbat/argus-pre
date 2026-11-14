//! P9/M2 — verify: резолв идентификаторов (PMID / DOI / arXiv) с файловым кэшем.
//!
//! Идея: `p_true` в леджере двигается только по свидетельству, у которого есть
//! **резолвнутый** идентификатор. Значит нужен слой, который отличает три вещи:
//!   * `Resolved` — идентификатор существует, метаданные получены;
//!   * `NotFound` — идентификатор не существует (это и есть «выдуманный», гейт падает);
//!   * `Unverified` — сеть/лимит/офлайн: **не** улика подделки, но и не свидетельство.
//!
//! Кэш: `<state>/ledger/verify_cache.json`, TTL 30 дней, ключ — нормализованный id.
//! Офлайн: `ARGUS_OFFLINE=1` (или `--offline`) — сеть не трогаем, кэш читаем.

use chrono::{DateTime, Duration, Local, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const CACHE_TTL_DAYS: i64 = 30;
pub const USER_AGENT: &str = "argus-pre/0.3.6 (idle-inference verify; mailto:jaba@longevity.ge)";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdKind {
    Pmid,
    Doi,
    Arxiv,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Id {
    pub kind: IdKind,
    pub value: String,
}

impl Id {
    /// Разбирает `PMID:123`, `doi:10.x/y`, `arXiv:2504.13171` (регистр не важен).
    pub fn parse(raw: &str) -> Option<Id> {
        let s = raw.trim();
        let low = s.to_lowercase();
        if let Some(rest) = low.strip_prefix("pmid:") {
            let v: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            if v.len() >= 4 {
                return Some(Id {
                    kind: IdKind::Pmid,
                    value: v,
                });
            }
            return None;
        }
        if let Some(rest) = low.strip_prefix("doi:") {
            let v = rest.trim().to_string();
            if v.starts_with("10.") && v.contains('/') {
                return Some(Id {
                    kind: IdKind::Doi,
                    value: v,
                });
            }
            return None;
        }
        if let Some(rest) = low.strip_prefix("arxiv:") {
            let v = rest.trim().to_string();
            if v.contains('.') {
                return Some(Id {
                    kind: IdKind::Arxiv,
                    value: v,
                });
            }
            return None;
        }
        None
    }

    pub fn key(&self) -> String {
        match self.kind {
            IdKind::Pmid => format!("PMID:{}", self.value),
            IdKind::Doi => format!("doi:{}", self.value),
            IdKind::Arxiv => format!("arXiv:{}", self.value),
        }
    }

    pub fn api_url(&self) -> String {
        match self.kind {
            IdKind::Pmid => format!(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A{}%20AND%20SRC%3AMED&format=json&pageSize=1",
                self.value
            ),
            IdKind::Doi => format!("https://api.crossref.org/works/{}", self.value),
            IdKind::Arxiv => format!(
                "http://export.arxiv.org/api/query?id_list={}&max_results=1",
                self.value
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Resolved,
    NotFound,
    Unverified,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub container: String,
    #[serde(default)]
    pub year: Option<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub status: Status,
    #[serde(default)]
    pub meta: Option<Meta>,
    pub checked_at: String,
    #[serde(default)]
    pub err: Option<String>,
}

impl Entry {
    fn new(key: &str, status: Status, meta: Option<Meta>, err: Option<String>) -> Entry {
        Entry {
            id: key.to_string(),
            status,
            meta,
            checked_at: Local::now().to_rfc3339(),
            err,
        }
    }
}

/// Файловый кэш результатов резолва.
pub struct Cache {
    pub entries: BTreeMap<String, Entry>,
    path: PathBuf,
}

impl Cache {
    pub fn load(state_dir: &Path) -> Cache {
        let path = state_dir.join("ledger").join("verify_cache.json");
        fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).ok();
        let entries: BTreeMap<String, Entry> = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Cache { entries, path }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(&self.entries).map_err(std::io::Error::other)?;
        fs::write(&self.path, s)
    }

    fn is_fresh(e: &Entry) -> bool {
        DateTime::parse_from_rfc3339(&e.checked_at)
            .ok()
            .map(|t| {
                let now = Utc::now();
                now - t.with_timezone(&Utc) < Duration::days(CACHE_TTL_DAYS)
            })
            .unwrap_or(false)
    }

    pub fn get_fresh(&self, key: &str) -> Option<&Entry> {
        self.entries.get(key).filter(|e| Cache::is_fresh(e))
    }

    pub fn put(&mut self, e: Entry) {
        self.entries.insert(e.id.clone(), e);
    }

    /// Только `Resolved` остаётся пригодным навсегда (существование ID не протухает);
    /// `Unverified` перепроверяем после TTL, `NotFound` — тоже (ID могли доиндексировать).
    pub fn stats(&self) -> (usize, usize, usize) {
        let r = self
            .entries
            .values()
            .filter(|e| e.status == Status::Resolved)
            .count();
        let n = self
            .entries
            .values()
            .filter(|e| e.status == Status::NotFound)
            .count();
        let u = self
            .entries
            .values()
            .filter(|e| e.status == Status::Unverified)
            .count();
        (r, n, u)
    }
}

/// Транспорт: реальный HTTP или подделка для тестов.
pub trait Fetcher {
    /// `Ok(Some(meta))` — найден; `Ok(None)` — не существует; `Err(_)` — сеть/лимит/офлайн.
    fn fetch(&self, id: &Id) -> Result<Option<Meta>, String>;
}

/// Сколько раз повторяем сетевой запрос (ARGUS_HTTP_RETRIES, по умолчанию 2 доп. попытки).
pub fn http_retries() -> u32 {
    std::env::var("ARGUS_HTTP_RETRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

/// Пауза перед повтором: 0.5 с → 1 с → 2 с (потолок 8 с).
pub fn backoff_ms(attempt: u32) -> u64 {
    (500u64 << attempt.min(4)).min(8_000)
}

/// Что имеет смысл повторять: лимит частоты и серверные ошибки.
pub fn should_retry(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

pub struct HttpFetcher {
    pub offline: bool,
    pub retries: u32,
}

impl HttpFetcher {
    pub fn new(offline: bool) -> Self {
        // env-флаг уважаем всегда: AGENTS/CI не должны ходить в сеть случайно
        let env_offline = std::env::var("ARGUS_OFFLINE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        HttpFetcher {
            offline: offline || env_offline,
            retries: http_retries(),
        }
    }

    /// DataCite: DOI из Zenodo / arXiv (10.48550) / репозиториев — Crossref их не знает.
    fn fetch_datacite(&self, doi: &str) -> Result<Option<Meta>, String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;
        let url = format!("https://api.datacite.org/dois/{doi}");
        let resp = match client.get(&url).header("Accept", "application/json").send() {
            Ok(r) => r,
            Err(e) => return Err(e.to_string()),
        };
        let code = resp.status().as_u16();
        if code == 404 || code == 410 {
            return Ok(None);
        }
        if !(200..300).contains(&code) {
            return Err(format!("DataCite HTTP {code}"));
        }
        let body = resp.text().map_err(|e| e.to_string())?;
        Ok(parse_datacite(&body))
    }
}

impl Fetcher for HttpFetcher {
    fn fetch(&self, id: &Id) -> Result<Option<Meta>, String> {
        if self.offline {
            return Err("offline".to_string());
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;
        let url = id.api_url();
        let mut attempt = 0u32;
        let resp = loop {
            match client
                .get(&url)
                .header("Accept", "application/json, application/atom+xml;q=0.9")
                .send()
            {
                Ok(r) if should_retry(r.status().as_u16()) && attempt < self.retries => {
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms(attempt)));
                    attempt += 1;
                }
                Ok(r) => break r,
                Err(e) if attempt < self.retries => {
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms(attempt)));
                    attempt += 1;
                    let _ = e;
                }
                Err(e) => return Err(e.to_string()),
            }
        };
        let code = resp.status().as_u16();
        if code == 429 {
            return Err(format!(
                "rate limited (429) после {} попыток",
                self.retries + 1
            ));
        }
        // DOI, зарегистрированный не в Crossref (Zenodo, arXiv 10.48550/, DataCite-репозитории),
        // Crossref отдаёт 404/400. Проверяем DataCite, иначе честный DOI попадает в «fabricated».
        if matches!(id.kind, IdKind::Doi)
            && (code == 404 || code == 400 || (400..500).contains(&code))
        {
            if let Some(meta) = self.fetch_datacite(&id.value)? {
                return Ok(Some(meta));
            }
            return Ok(None);
        }
        if code == 404 {
            return Ok(None);
        }
        if !(200..300).contains(&code) {
            return Err(format!("HTTP {code}"));
        }
        let body = resp.text().map_err(|e| e.to_string())?;
        match id.kind {
            IdKind::Pmid => Ok(parse_epmc_hit(&body)),
            IdKind::Doi => Ok(parse_crossref(&body)),
            IdKind::Arxiv => Ok(parse_arxiv(&body)),
        }
    }
}

// ── парсеры ответов (чистые функции: тестируются без сети) ────────────────────

pub fn parse_epmc_hit(body: &str) -> Option<Meta> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let hit = v.get("hitCount")?.as_u64().unwrap_or(0);
    if hit == 0 {
        return None;
    }
    let r = v.get("resultList")?.get("result")?.get(0)?;
    Some(Meta {
        title: r
            .get("title")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        container: r
            .get("journalTitle")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        year: r
            .get("pubYear")
            .and_then(|x| x.as_str())
            .and_then(|y| y.parse::<i32>().ok()),
    })
}

pub fn parse_crossref(body: &str) -> Option<Meta> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let m = v.get("message")?;
    Some(Meta {
        title: m
            .get("title")
            .and_then(|t| t.get(0))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        container: m
            .get("container-title")
            .and_then(|t| t.get(0))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        year: m
            .get("issued")
            .and_then(|i| i.get("date-parts"))
            .and_then(|d| d.get(0))
            .and_then(|d| d.get(0))
            .and_then(|y| y.as_i64())
            .map(|y| y as i32),
    })
}

/// Парсер ответа DataCite (`api.datacite.org/dois/{doi}`).
pub fn parse_datacite(body: &str) -> Option<Meta> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let a = v.get("data")?.get("attributes")?;
    let title = a
        .get("titles")
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.first())
        .and_then(|t| t.get("title"))
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if title.is_empty() {
        return None;
    }
    let container = a
        .get("publisher")
        .and_then(|p| p.as_str())
        .unwrap_or("DataCite")
        .to_string();
    let year = a
        .get("publicationYear")
        .and_then(|y| y.as_i64())
        .map(|y| y as i32);
    Some(Meta {
        title,
        container,
        year,
    })
}

pub fn parse_arxiv(body: &str) -> Option<Meta> {
    // Atom: <entry><title>…</title>…
    let entry_start = body.find("<entry")?;
    let tail = &body[entry_start..];
    let title = extract_tag(tail, "title")?;
    let year =
        extract_tag(tail, "published").and_then(|p| p.get(..4).and_then(|y| y.parse::<i32>().ok()));
    Some(Meta {
        title: title.split_whitespace().collect::<Vec<_>>().join(" "),
        container: "arXiv".to_string(),
        year,
    })
}

fn extract_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let i = s.find(&open)?;
    let gt = s[i..].find('>')? + i + 1;
    let end = s[gt..].find(&format!("</{tag}>"))? + gt;
    Some(s[gt..end].trim().to_string())
}

// ── резолв с кэшем ───────────────────────────────────────────────────────────

/// Кэш-первый резолв. Ошибка сети → `Unverified` (НЕ «подделка»).
pub fn resolve<F: Fetcher>(id: &Id, cache: &mut Cache, f: &F) -> Entry {
    let key = id.key();
    if let Some(e) = cache.get_fresh(&key) {
        return e.clone();
    }
    let e = match f.fetch(id) {
        Ok(Some(meta)) => Entry::new(&key, Status::Resolved, Some(meta), None),
        Ok(None) => Entry::new(&key, Status::NotFound, None, None),
        Err(err) => Entry::new(&key, Status::Unverified, None, Some(err)),
    };
    cache.put(e.clone());
    e
}

pub fn resolve_many<F: Fetcher>(ids: &[Id], cache: &mut Cache, f: &F) -> Vec<Entry> {
    ids.iter().map(|id| resolve(id, cache, f)).collect()
}

/// Отображение на вердикт леджера: любое `NotFound` → Fabricated (гейт падает),
/// любое `Unverified` без подделок → Unresolved (свидетельства нет, но и обвинения нет).
pub fn to_ledger_verifier(entries: &[Entry]) -> crate::ledger::VerifyResult {
    use crate::ledger::VerifyResult;
    if entries.is_empty() {
        return VerifyResult::Unresolved;
    }
    if entries.iter().any(|e| e.status == Status::NotFound) {
        return VerifyResult::Fabricated;
    }
    if entries.iter().any(|e| e.status == Status::Unverified) {
        return VerifyResult::Unresolved;
    }
    VerifyResult::Resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct FakeFetcher {
        calls: RefCell<Vec<String>>,
        found: Vec<String>,
        error: bool,
    }

    impl Fetcher for FakeFetcher {
        fn fetch(&self, id: &Id) -> Result<Option<Meta>, String> {
            self.calls.borrow_mut().push(id.key());
            if self.error {
                return Err("network down".into());
            }
            if self.found.contains(&id.key()) {
                Ok(Some(Meta {
                    title: "fake".into(),
                    container: "J".into(),
                    year: Some(2020),
                }))
            } else {
                Ok(None)
            }
        }
    }

    fn cache_tmp(name: &str) -> Cache {
        let dir =
            std::env::temp_dir().join(format!("argus-verify-test-{}-{}", name, std::process::id()));
        Cache::load(&dir)
    }

    #[test]
    fn parses_three_id_kinds() {
        assert_eq!(Id::parse("PMID:23663408").unwrap().kind, IdKind::Pmid);
        assert_eq!(
            Id::parse("doi:10.1017/S0140525X19000049").unwrap().kind,
            IdKind::Doi
        );
        assert_eq!(Id::parse("arXiv:2504.13171").unwrap().kind, IdKind::Arxiv);
        assert!(Id::parse("PMID:12").is_none(), "слишком короткий PMID");
        assert!(
            Id::parse("10.1017/S0140525X19000049").is_none(),
            "без префикса — не ID"
        );
        assert_eq!(
            Id::parse("arXiv:2504.13171").unwrap().key(),
            "arXiv:2504.13171"
        );
    }

    #[test]
    fn parses_epmc_crossref_arxiv_bodies() {
        let epmc = r#"{"hitCount":1,"resultList":{"result":[{"title":"Mind-wandering as spontaneous thought","journalTitle":"Nat Rev Neurosci","pubYear":"2016"}]}}"#;
        let m = parse_epmc_hit(epmc).unwrap();
        assert_eq!(m.year, Some(2016));
        assert!(m.title.starts_with("Mind-wandering"));
        assert!(parse_epmc_hit(r#"{"hitCount":0,"resultList":{"result":[]}}"#).is_none());

        let cr = r#"{"message":{"title":["Is coding a relevant metaphor for the brain?"],"container-title":["Behavioral and Brain Sciences"],"issued":{"date-parts":[[2019]]}}}"#;
        let m = parse_crossref(cr).unwrap();
        assert_eq!(m.year, Some(2019));
        assert_eq!(m.container, "Behavioral and Brain Sciences");

        let ax = r#"<feed><entry><title>Sleep-time Compute: Beyond Inference
Scaling</title><published>2025-04-16T00:00:00Z</published></entry></feed>"#;
        let m = parse_arxiv(ax).unwrap();
        assert_eq!(m.year, Some(2025));
        assert!(m.title.contains("Sleep-time Compute"));
        assert!(parse_arxiv("<feed></feed>").is_none());
    }

    #[test]
    fn cache_hit_avoids_second_fetch() {
        let mut c = cache_tmp("hit");
        let f = FakeFetcher {
            calls: RefCell::new(vec![]),
            found: vec!["PMID:23663408".into()],
            error: false,
        };
        let id = Id::parse("PMID:23663408").unwrap();
        let e1 = resolve(&id, &mut c, &f);
        assert_eq!(e1.status, Status::Resolved);
        let e2 = resolve(&id, &mut c, &f);
        assert_eq!(e2.status, Status::Resolved);
        assert_eq!(
            f.calls.borrow().len(),
            1,
            "второй вызов должен прийти из кэша"
        );
    }

    #[test]
    fn notfound_is_fabrication_error_is_not() {
        let mut c = cache_tmp("cls");
        let f = FakeFetcher {
            calls: RefCell::new(vec![]),
            found: vec![],
            error: false,
        };
        let missing = Id::parse("doi:10.9999/nope").unwrap();
        let e = resolve(&missing, &mut c, &f);
        assert_eq!(e.status, Status::NotFound);
        assert_eq!(
            to_ledger_verifier(&[e]),
            crate::ledger::VerifyResult::Fabricated
        );

        let mut c2 = cache_tmp("cls2");
        let f2 = FakeFetcher {
            calls: RefCell::new(vec![]),
            found: vec![],
            error: true,
        };
        let id = Id::parse("PMID:23663408").unwrap();
        let e2 = resolve(&id, &mut c2, &f2);
        assert_eq!(e2.status, Status::Unverified);
        assert_eq!(
            to_ledger_verifier(&[e2]),
            crate::ledger::VerifyResult::Unresolved
        );
    }

    #[test]
    fn mixed_batch_with_one_notfound_fails_gate() {
        let mut c = cache_tmp("mix");
        let f = FakeFetcher {
            calls: RefCell::new(vec![]),
            found: vec!["PMID:23663408".into()],
            error: false,
        };
        let ids = vec![
            Id::parse("PMID:23663408").unwrap(),
            Id::parse("doi:10.9999/nope").unwrap(),
        ];
        let es = resolve_many(&ids, &mut c, &f);
        assert_eq!(
            to_ledger_verifier(&es),
            crate::ledger::VerifyResult::Fabricated
        );
    }

    #[test]
    fn retry_policy_and_backoff_are_bounded() {
        assert!(should_retry(429));
        assert!(should_retry(503));
        assert!(
            !should_retry(404),
            "404 — это не временная ошибка, а «не существует»"
        );
        assert!(!should_retry(200));
        assert_eq!(backoff_ms(0), 500);
        assert_eq!(backoff_ms(1), 1000);
        assert!(backoff_ms(10) <= 8_000, "потолок паузы");
    }

    #[test]
    fn offline_fetcher_never_hits_network() {
        let f = HttpFetcher::new(true);
        let id = Id::parse("arXiv:2504.13171").unwrap();
        let r = f.fetch(&id);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), "offline");
    }

    #[test]
    fn cache_survives_reload() {
        let dir = std::env::temp_dir().join(format!("argus-verify-persist-{}", std::process::id()));
        let mut c = Cache::load(&dir);
        c.put(Entry::new(
            "PMID:1",
            Status::Resolved,
            Some(Meta {
                title: "t".into(),
                container: "c".into(),
                year: Some(2001),
            }),
            None,
        ));
        c.save().unwrap();
        let c2 = Cache::load(&dir);
        assert_eq!(c2.get_fresh("PMID:1").unwrap().status, Status::Resolved);
        assert_eq!(c2.stats().0, 1);
    }
}

#[cfg(test)]
mod datacite_tests {
    use super::*;

    #[test]
    fn parses_datacite_zenodo_record() {
        let body = r#"{"data":{"attributes":{"titles":[{"title":"Adaptive Cognitive System Ze"}],
                       "publisher":"Longevity Horizon","publicationYear":2025}}}"#;
        let m = parse_datacite(body).expect("должен распарситься");
        assert_eq!(m.title, "Adaptive Cognitive System Ze");
        assert_eq!(m.container, "Longevity Horizon");
        assert_eq!(m.year, Some(2025));
    }

    #[test]
    fn datacite_without_title_is_none() {
        let body = r#"{"data":{"attributes":{"titles":[]}}}"#;
        assert!(parse_datacite(body).is_none());
    }
}
