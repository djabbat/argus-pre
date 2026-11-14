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

/// Версия логики резолва. Записи кэша, сделанные более старой версией `NotFound`,
/// не переиспользуются: «не нашли» — вывод конкретного набора источников, а не факт.
pub const CACHE_ENGINE: u32 = 2;

fn engine_default() -> u32 {
    1
}
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
    /// какой версией резолвера получен вердикт (см. CACHE_ENGINE)
    #[serde(default = "engine_default")]
    pub engine: u32,
}

impl Entry {
    fn new(key: &str, status: Status, meta: Option<Meta>, err: Option<String>) -> Entry {
        Entry {
            id: key.to_string(),
            status,
            meta,
            checked_at: Local::now().to_rfc3339(),
            err,
            engine: CACHE_ENGINE,
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
        // `Resolved` (ID существует) не протухает никогда и не зависит от версии логики;
        // `NotFound`/`Unverified` от старого резолвера перепроверяем — иначе один сетевой
        // сбой навсегда закрепляет ложное обвинение в подделке и валит гейт.
        if e.status == Status::Resolved {
            return true;
        }
        if e.engine < CACHE_ENGINE {
            return false;
        }
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

    /// Один GET с повторами на 429/5xx и сетевые ошибки. Возвращает (код, тело).
    fn get(&self, url: &str, accept: &str) -> Result<(u16, String), String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;
        let mut attempt = 0u32;
        let resp = loop {
            match client.get(url).header("Accept", accept).send() {
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
        let body = resp.text().unwrap_or_default();
        Ok((code, body))
    }

    /// PMID: два эндпоинта Europe PMC + независимый второй источник NCBI eutils.
    /// «Нет в EPMC» — не доказательство несуществования: PMID мог быть только что
    /// выдан и ещё не проиндексирован EPMC (13–14 ноя три настоящих PMID попали в «подделку»).
    fn fetch_pmid(&self, id: &Id) -> Result<Option<Meta>, String> {
        let epmc = [
            format!(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A{}%20AND%20SRC%3AMED&format=json&pageSize=1",
                id.value
            ),
            format!(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID%3A{}&format=json&pageSize=1",
                id.value
            ),
        ];
        let mut clean_negative = false;
        let mut transport_err: Option<String> = None;
        for url in epmc {
            match self.get(&url, "application/json") {
                Ok((code, body)) if (200..300).contains(&code) => {
                    if let Some(m) = parse_epmc_hit(&body) {
                        return Ok(Some(m));
                    }
                    clean_negative = true;
                }
                Ok((code, _)) => {
                    clean_negative = true;
                    transport_err.get_or_insert(format!("EPMC HTTP {code}"));
                }
                Err(e) => {
                    transport_err.get_or_insert(e);
                }
            }
        }
        // второй источник: NCBI esummary (тот же PubMed, другой сервис)
        let url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={}",
            id.value
        );
        match self.get(&url, "application/json") {
            Ok((code, body)) if (200..300).contains(&code) => {
                if let Some(m) = parse_ncbi_esummary(&body, &id.value) {
                    return Ok(Some(m));
                }
                clean_negative = true;
            }
            Ok((code, _)) => {
                clean_negative = true;
                transport_err.get_or_insert(format!("NCBI HTTP {code}"));
            }
            Err(e) => {
                transport_err.get_or_insert(e);
            }
        }
        if let Some(e) = transport_err {
            return Err(format!(
                "источники не дали полного ответа ({e}) — статус неизвестен, это не «подделка»"
            ));
        }
        if clean_negative {
            return Ok(None);
        }
        Err("ни один источник не ответил".to_string())
    }

    /// DOI: Crossref → DataCite → arXiv (для 10.48550/arxiv.*) → реестр handle doi.org.
    /// Только чистое «нет» у всех четырёх даёт NotFound.
    fn fetch_doi(&self, id: &Id) -> Result<Option<Meta>, String> {
        let mut clean_negative = false;
        let mut transport_err: Option<String> = None;
        match self.get(
            &format!("https://api.crossref.org/works/{}", id.value),
            "application/json",
        ) {
            Ok((code, body)) if (200..300).contains(&code) => {
                if let Some(m) = parse_crossref(&body) {
                    return Ok(Some(m));
                }
                clean_negative = true;
            }
            Ok(_) => clean_negative = true,
            Err(e) => {
                transport_err.get_or_insert(e);
            }
        }
        match self.fetch_datacite(&id.value) {
            Ok(Some(m)) => return Ok(Some(m)),
            Ok(None) => clean_negative = true,
            Err(e) => {
                transport_err.get_or_insert(e);
            }
        }
        if let Some(ax) = arxiv_id_from_doi(&id.value) {
            let axid = Id {
                kind: IdKind::Arxiv,
                value: ax,
            };
            match self.get(&axid.api_url(), "application/atom+xml") {
                Ok((code, body)) if (200..300).contains(&code) => {
                    if let Some(m) = parse_arxiv(&body) {
                        return Ok(Some(m));
                    }
                    clean_negative = true;
                }
                Ok(_) => clean_negative = true,
                Err(e) => {
                    transport_err.get_or_insert(e);
                }
            }
        }
        // последняя инстанция: существует ли handle вообще (метаданных может не быть)
        match self.get(
            &format!("https://doi.org/api/handles/{}", id.value),
            "application/json",
        ) {
            Ok((code, body)) if (200..300).contains(&code) => {
                if doi_handle_exists(&body) {
                    return Ok(Some(Meta {
                        title: String::new(),
                        container: "doi.org (handle существует)".to_string(),
                        year: None,
                    }));
                }
                clean_negative = true;
            }
            Ok(_) => clean_negative = true,
            Err(e) => {
                transport_err.get_or_insert(e);
            }
        }
        if let Some(e) = transport_err {
            return Err(format!(
                "DOI-источники ответили не полностью ({e}) — статус неизвестен, это не «подделка»"
            ));
        }
        if clean_negative {
            return Ok(None);
        }
        Err("ни один DOI-источник не ответил".to_string())
    }

    fn fetch_arxiv(&self, id: &Id) -> Result<Option<Meta>, String> {
        match self.get(&id.api_url(), "application/atom+xml") {
            Ok((code, body)) if (200..300).contains(&code) => Ok(parse_arxiv(&body)),
            Ok((404, _)) => Ok(None),
            Ok((code, _)) => Err(format!("arXiv HTTP {code}")),
            Err(e) => Err(e),
        }
    }
}

/// DOI вида `10.48550/arxiv.1507.07385` → `1507.07385` (arXiv-ID).
pub fn arxiv_id_from_doi(doi: &str) -> Option<String> {
    let d = doi.trim().to_ascii_lowercase();
    let rest = d.strip_prefix("10.48550/arxiv.")?;
    let v = rest.trim().trim_end_matches('/').to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// Ответ `doi.org/api/handles/{doi}`: `responseCode == 1` — handle существует.
pub fn doi_handle_exists(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("responseCode").and_then(|c| c.as_i64()))
        .map(|c| c == 1)
        .unwrap_or(false)
}

/// Ответ NCBI esummary (`db=pubmed&retmode=json`): uid в списке — запись существует.
pub fn parse_ncbi_esummary(body: &str, pmid: &str) -> Option<Meta> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let res = v.get("result")?;
    let uids: Vec<String> = res
        .get("uids")
        .and_then(|u| u.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    if !uids.iter().any(|u| u == pmid) {
        return None;
    }
    let rec = res.get(pmid)?;
    if rec.get("error").is_some() {
        return None;
    }
    let title = rec
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let pubdate = rec.get("pubdate").and_then(|t| t.as_str()).unwrap_or("");
    let year = pubdate
        .split_whitespace()
        .next()
        .and_then(|y| y.parse::<i32>().ok())
        .or_else(|| pubdate.chars().take(4).collect::<String>().parse().ok());
    Some(Meta {
        title,
        container: rec
            .get("fulljournalname")
            .and_then(|t| t.as_str())
            .unwrap_or("PubMed")
            .to_string(),
        year,
    })
}

impl Fetcher for HttpFetcher {
    fn fetch(&self, id: &Id) -> Result<Option<Meta>, String> {
        if self.offline {
            return Err("offline".to_string());
        }
        match id.kind {
            IdKind::Pmid => self.fetch_pmid(id),
            IdKind::Doi => self.fetch_doi(id),
            IdKind::Arxiv => self.fetch_arxiv(id),
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
    fn arxiv_doi_fallbacks_recognise_real_ids() {
        // факты 14.11.2026: DOI существует (DataCite + doi.org handle), Crossref его не знает
        assert_eq!(
            arxiv_id_from_doi("10.48550/arxiv.1507.07385").as_deref(),
            Some("1507.07385")
        );
        assert_eq!(arxiv_id_from_doi("doi:10.48550/ArXiv.2512.23554").as_deref(), None, "без префикса — не DOI");
        assert_eq!(
            arxiv_id_from_doi("10.48550/arxiv.2512.23554").as_deref(),
            Some("2512.23554")
        );
        assert!(arxiv_id_from_doi("10.1038/nature12373").is_none());
        assert!(doi_handle_exists(r#"{"responseCode":1,"handle":"..."}"#));
        assert!(!doi_handle_exists(r#"{"responseCode":100}"#), "100 — handle не найден");
        assert!(!doi_handle_exists("not json"));
    }

    #[test]
    fn ncbi_esummary_confirms_pmid() {
        let body = r#"{"header":{"type":"esummary","version":"0.3"},
            "result":{"uids":["43026037"],
            "43026037":{"uid":"43026037","pubdate":"2026 Nov 12",
            "title":"A Triboelectric-Hydrogel System","fulljournalname":"Advanced science"}}}"#;
        let m = parse_ncbi_esummary(body, "43026037").expect("должен подтвердить PMID");
        assert_eq!(m.year, Some(2026));
        assert!(m.title.starts_with("A Triboelectric"));
        assert!(parse_ncbi_esummary(body, "99999999").is_none());
        let err = r#"{"result":{"uids":[],"9999":{"error":"cannot get document summary"}}}"#;
        assert!(parse_ncbi_esummary(err, "9999").is_none());
    }

    #[test]
    fn stale_notfound_from_old_engine_is_rechecked() {
        let mut c = cache_tmp("engine");
        // запись, полученная старым резолвером (engine=1) и объявившая настоящий ID подделкой
        let mut old = Entry::new("PMID:43026037", Status::NotFound, None, None);
        old.engine = 1;
        c.put(old);
        assert!(
            c.get_fresh("PMID:43026037").is_none(),
            "старый NotFound не должен переиспользоваться"
        );
        // а Resolved — вечен, даже от старого резолвера
        let mut res = Entry::new("PMID:1", Status::Resolved, None, None);
        res.engine = 1;
        c.put(res);
        assert!(c.get_fresh("PMID:1").is_some());
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
