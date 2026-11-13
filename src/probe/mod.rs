//! P9/M2 — probe: read-only проба для idle-цикла.
//!
//! Три канала, все строго на чтение:
//!   * `epmc`   — поиск по Europe PMC (научный «web.search»: сразу отдаёт PMID/DOI);
//!   * `corpus` — поиск по локальному корпусу (*.md/*.txt): что уже написано у нас;
//!   * `none`   — проба не выполняется (M1-поведение).
//!
//! Ключевое: поиск НЕ выносит суждение о вопросе. Он возвращает кандидатов с идентификаторами,
//! которые затем обязан резолвить `verify`. Проба без суждения = `Outcome::Evidence`.

use std::fs;
use std::path::{Path, PathBuf};

pub const USER_AGENT: &str = "argus-pre/0.3.6 (idle-inference probe; mailto:jaba@longevity.ge)";
pub const MAX_QUERY_WORDS: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// `PMID:…` / `doi:…` / `arXiv:…` — формат, который понимает `verify::Id::parse`
    pub id: String,
    pub title: String,
    pub source: String,
    /// абстракт (Europe PMC `resultType=core`): нужен для оценки позиции свидетельства
    /// (`crate::stance`). `None` — заголовок есть, текста нет.
    #[allow(clippy::struct_field_names)]
    pub abstract_text: Option<String>,
}

pub trait Searcher {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<Candidate>, String>;
}

/// Короткий запрос из текста вопроса: убрать знаки, служебные слова, обрезать.
pub fn build_query(question: &str) -> String {
    let stop = [
        "держит",
        "ли",
        "падает",
        "ниже",
        "сходится",
        "достаточно",
        "нужен",
        "нужна",
        "есть",
        "было",
        "были",
        "будет",
        "чем",
        "что",
        "как",
        "для",
        "при",
        "или",
        "the",
        "is",
        "are",
        "does",
        "do",
        "can",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
    ];
    let cleaned: String = question
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' || c == '/' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<String> = cleaned
        .split_whitespace()
        .filter(|w| {
            let lw = w.to_lowercase();
            w.chars().count() > 2 && !stop.contains(&lw.as_str())
        })
        .take(MAX_QUERY_WORDS)
        .map(|w| w.to_string())
        .collect();
    words.join(" ")
}

// ── Europe PMC ────────────────────────────────────────────────────────────────

pub struct EpMcSearcher {
    pub offline: bool,
    pub retries: u32,
}

impl EpMcSearcher {
    pub fn new(offline: bool) -> Self {
        let env_offline = std::env::var("ARGUS_OFFLINE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        EpMcSearcher {
            offline: offline || env_offline,
            retries: crate::verify::http_retries(),
        }
    }
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

impl Searcher for EpMcSearcher {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<Candidate>, String> {
        if self.offline {
            return Err("offline".to_string());
        }
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(25))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;
        let mut last_err: Option<String> = None;
        for variant in query_ladder(query) {
            let url = format!(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={}&format=json&pageSize={}&resultType=core",
                urlencode(&variant),
                limit.clamp(1, 10)
            );
            let mut attempt = 0u32;
            let resp = loop {
                match client.get(&url).send() {
                    Ok(r)
                        if crate::verify::should_retry(r.status().as_u16())
                            && attempt < self.retries =>
                    {
                        std::thread::sleep(std::time::Duration::from_millis(
                            crate::verify::backoff_ms(attempt),
                        ));
                        attempt += 1;
                    }
                    Ok(r) => break Some(r),
                    Err(_e) if attempt < self.retries => {
                        std::thread::sleep(std::time::Duration::from_millis(
                            crate::verify::backoff_ms(attempt),
                        ));
                        attempt += 1;
                    }
                    Err(e) => {
                        last_err = Some(e.to_string());
                        break None;
                    }
                }
            };
            let Some(resp) = resp else { continue };
            if resp.status().as_u16() == 429 {
                last_err = Some("rate limited (429) после ретраев".to_string());
                continue;
            }
            if !resp.status().is_success() {
                last_err = Some(format!("HTTP {}", resp.status().as_u16()));
                continue;
            }
            let body = resp.text().map_err(|e| e.to_string())?;
            let cands = parse_epmc_candidates(&body, &format!("europepmc:{variant}"));
            if !cands.is_empty() {
                return Ok(cands);
            }
        }
        if let Some(e) = last_err {
            return Err(e);
        }
        Ok(vec![])
    }
}

/// Чистый парсер ответа Europe PMC (тестируется без сети).
pub fn parse_epmc_candidates(body: &str, source: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return out,
    };
    let arr = v
        .get("resultList")
        .and_then(|r| r.get("result"))
        .and_then(|r| r.as_array());
    let Some(arr) = arr else { return out };
    for r in arr {
        let title = r
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let pmid = r.get("pmid").and_then(|t| t.as_str()).unwrap_or("");
        let doi = r.get("doi").and_then(|t| t.as_str()).unwrap_or("");
        let id = if !pmid.is_empty() {
            format!("PMID:{pmid}")
        } else if !doi.is_empty() {
            format!("doi:{doi}")
        } else {
            continue;
        };
        if out.iter().any(|c: &Candidate| c.id == id) {
            continue;
        }
        out.push(Candidate {
            id,
            title,
            source: source.to_string(),
            abstract_text: r
                .get("abstractText")
                .and_then(|t| t.as_str())
                .map(|t| t.to_string())
                .filter(|t| !t.trim().is_empty()),
        });
    }
    out
}

// ── локальный корпус ──────────────────────────────────────────────────────────

pub struct CorpusSearcher {
    pub dir: PathBuf,
    pub max_files: usize,
}

impl CorpusSearcher {
    pub fn new(dir: &Path) -> Self {
        CorpusSearcher {
            dir: dir.to_path_buf(),
            max_files: 400,
        }
    }

    fn walk(&self, dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if out.len() >= self.max_files || depth > 4 {
            return;
        }
        let Ok(rd) = fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                self.walk(&p, out, depth + 1);
            } else {
                let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
                if matches!(ext, "md" | "txt") {
                    out.push(p);
                    if out.len() >= self.max_files {
                        return;
                    }
                }
            }
        }
    }
}

impl Searcher for CorpusSearcher {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<Candidate>, String> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let mut files = Vec::new();
        self.walk(&self.dir, &mut files, 0);
        // делим составные токены: «fiducial-autofocus» в файле обычно два слова
        let wanted: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() > 2)
            .map(|w| w.to_string())
            .collect();
        if wanted.is_empty() {
            return Ok(vec![]);
        }
        let mut out: Vec<(usize, Candidate)> = Vec::new();
        for f in files {
            let Ok(text) = fs::read_to_string(&f) else {
                continue;
            };
            let low = text.to_lowercase();
            let hits = wanted.iter().filter(|w| low.contains(w.as_str())).count();
            if hits == 0 {
                continue;
            }
            let ids = crate::ledger::extract_identifiers(&text);
            for id in ids.into_iter().take(2) {
                let title = f
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file")
                    .to_string();
                out.push((
                    hits,
                    Candidate {
                        id,
                        title,
                        source: format!("corpus:{}", f.display()),
                        abstract_text: None,
                    },
                ));
            }
        }
        out.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(out
            .into_iter()
            .take(limit.clamp(1, 10))
            .map(|(_, c)| c)
            .collect())
    }
}

/// Запрос для англоязычной базы (Europe PMC): оставляем только латиницу/цифры,
/// потому что русские слова дают 0 попаданий и цикл пишет «пробел» там, где пробел — артефакт запроса.
pub fn build_query_en(question: &str) -> String {
    let stop = [
        "the", "is", "are", "does", "do", "can", "a", "an", "of", "in", "on", "to", "and", "or",
        "for", "with", "px", "nm",
    ];
    let mut words: Vec<String> = Vec::new();
    for raw in question.split_whitespace() {
        let cleaned: String = raw
            .chars()
            .filter(|c| {
                c.is_ascii_alphanumeric() || *c == '-' || *c == '/' || *c == '.' || *c == ':'
            })
            .collect();
        // «fiducial-autofocus» → два слова: база ищет по AND и составное слово не находит
        for part in cleaned.split(['-', '/', '.', ':']) {
            let w = part
                .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .to_string();
            if w.chars().count() < 3 {
                continue;
            }
            let low = w.to_lowercase();
            if stop.contains(&low.as_str()) {
                continue;
            }
            // годы и «P0/P1»-маркеры в запрос не идут
            if (low.len() == 4 && (low.starts_with("19") || low.starts_with("20")))
                || (low.len() == 2 && low.starts_with('p'))
            {
                continue;
            }
            if !words.iter().any(|x: &String| x.eq_ignore_ascii_case(&w)) {
                words.push(w);
            }
        }
    }
    words.sort_by_key(|w| {
        let tech = w.chars().any(|c| c.is_ascii_digit()) || w.contains('-') || w.contains('/');
        if tech {
            0
        } else {
            1
        }
    });
    words.truncate(MAX_QUERY_WORDS);
    words.join(" ")
}

/// Выбор запроса под канал: `epmc` — латиница, остальные — общий cleaner.
/// Мини-глоссарий темы → английские термины для запроса.
///
/// Зачем: литература по нашим темам английская, а вопросы — русские. `build_query_en`
/// вытаскивает только латиницу, и вопрос «Показано ли, что fiducial-автофокус удерживает
/// дрейф < 1 px за 24 ч…» превращался в запрос из одного слова `fiducial` — Europe PMC
/// возвращал работы про рентгеновские фидуциалы в лучевой терапии. Ключ — префикс
/// (стем), значение — термин, который реально ищется в базе.
const GLOSSARY: [(&str, &str); 46] = [
    ("автофокус", "autofocus"),
    ("фокус", "focus"),
    ("дрейф", "drift"),
    ("центриол", "centriole"),
    ("фибробласт", "fibroblast"),
    ("делен", "division"),
    ("деления", "division"),
    ("интервал", "interval"),
    ("фототоксич", "phototoxicity"),
    ("съёмк", "imaging"),
    ("съемк", "imaging"),
    ("изображен", "imaging"),
    ("микроскоп", "microscope"),
    ("точность", "accuracy"),
    ("разрешен", "resolution"),
    ("локализац", "localization"),
    ("асимметр", "asymmetry"),
    ("дочерн", "daughter"),
    ("размер", "size"),
    ("расстоян", "distance"),
    ("электронн", "electron"),
    ("иммортализ", "immortalized"),
    ("вариабельн", "variability"),
    ("распределен", "distribution"),
    ("логнормальн", "lognormal"),
    ("экспоненциальн", "exponential"),
    ("трекер", "tracker"),
    ("линий", "lineage"),
    ("линия", "line"),
    ("ошибка", "error"),
    ("кадр", "frame"),
    ("доза", "dose"),
    ("сутки", "day"),
    ("конфокальн", "confocal"),
    ("контрол", "control"),
    ("скорост", "rate"),
    ("крупн", "size"),
    ("зрел", "mature"),
    ("стар", "old"),
    ("метк", "marker"),
    ("меток", "marker"),
    ("длительн", "long-term"),
    ("непрерывн", "continuous"),
    ("числ", "number"),
    ("моноклональн", "clonal"),
    ("пролиферац", "proliferation"),
];

/// Перевести русские термины вопроса в английские ключи (для литературного канала).
pub fn glossary_terms(question: &str) -> Vec<String> {
    let low = question.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    for (ru, en) in GLOSSARY {
        if low.contains(ru) && !out.iter().any(|w| w == en) {
            out.push(en.to_string());
        }
    }
    out
}

pub fn build_query_for(channel: &str, question: &str) -> String {
    if channel == "epmc" {
        let mut words: Vec<String> = build_query_en(question)
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();
        // латиницы мало (русский вопрос) → добавляем глоссарий темы
        if words.len() < 3 {
            for t in glossary_terms(question) {
                if !words.iter().any(|w| w.eq_ignore_ascii_case(&t)) {
                    words.push(t);
                }
            }
        }
        let q = words.join(" ");
        if !q.trim().is_empty() {
            return q;
        }
    }
    build_query(question)
}

/// Отсев нерелевантного: поиск отдаёт «что угодно, лишь бы совпало слово».
/// Без этого фильтра `identifier_resolution_rate` можно получить 1.00 на мусоре —
/// ровно та ошибка «приёмник выдумывает сигнал», против которой построен гейт.
/// Длина общего префикса (для грубого стемминга без внешних крейтов).
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

pub fn filter_relevant(query: &str, cands: Vec<Candidate>) -> (Vec<Candidate>, Vec<Candidate>) {
    let terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.chars().count() > 3)
        .collect();
    if terms.is_empty() {
        return (cands, vec![]);
    }
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for c in cands {
        let t = c.title.to_lowercase();
        let title_words: Vec<&str> = t
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 4)
            .collect();
        // точное вхождение ИЛИ общий префикс ≥5 символов: «Centrin1» ↔ «Centriole»
        let hit = terms.iter().any(|w| {
            t.contains(w.as_str()) || title_words.iter().any(|tw| common_prefix_len(w, tw) >= 5)
        });
        if hit {
            kept.push(c);
        } else {
            dropped.push(c);
        }
    }
    (kept, dropped)
}

/// Лестница запросов: база склеивает слова через AND, поэтому узкий запрос даёт 0 попаданий.
/// Идём от полного к короткому и берём первый непустой результат.
pub fn query_ladder(query: &str) -> Vec<String> {
    let words: Vec<&str> = query
        .split_whitespace()
        .filter(|w| w.chars().count() > 2)
        .collect();
    let mut out: Vec<String> = Vec::new();
    for n in (1..=words.len()).rev() {
        let q = words[..n].join(" ");
        if !q.is_empty() && !out.contains(&q) {
            out.push(q);
        }
        if out.len() >= 4 {
            break;
        }
    }
    out
}

/// Пересечение стемов как **доля запроса**: |Q∩T| / |Q|, стем = первые 5 символов.
/// Доля запроса, а не косинус: длинный заголовок иначе «разбавляет» единственное совпадение
/// (калибровка 12 ноя: релевантный заголовок давал 0.144 по косинусу → отсекался).
pub fn stem_overlap(query: &str, title: &str) -> f32 {
    const STOP: [&str; 14] = [
        "the", "and", "for", "with", "from", "this", "that", "are", "was", "has", "its", "not",
        "can", "than",
    ];
    fn stems(t: &str) -> Vec<String> {
        t.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 3)
            .filter(|w| !STOP.contains(w))
            .map(|w| w.chars().take(5).collect::<String>())
            .collect()
    }
    let q = stems(query);
    let t = stems(title);
    if q.is_empty() || t.is_empty() {
        return 0.0;
    }
    let hit = q.iter().filter(|w| t.iter().any(|x| x == *w)).count();
    hit as f32 / q.len() as f32
}

/// Релевантность кандидата: заголовок + абстракт (если есть).
///
/// Зачем абстракт: доля пересечения стемов считается от длины ЗАПРОСА, поэтому на
/// длинных вопросах («Is phototoxicity a limiting factor for multi-day GFP time-lapse
/// imaging?») заголовок почти никогда не даёт 0.20 и все кандидаты отсеивались — цикл
/// честно писал «0 после фильтра» и не доходил до оценки позиции.
pub fn relevance_score_candidate(query: &str, c: &Candidate) -> f32 {
    match c.abstract_text.as_deref() {
        Some(a) if !a.trim().is_empty() => {
            let text = format!("{} {}", c.title, a);
            relevance_score(query, &text)
        }
        _ => relevance_score(query, &c.title),
    }
}

/// Гибрид: доля пересечения стемов, либо половинный вес символьной близости (триграммы).
pub fn relevance_score(query: &str, title: &str) -> f32 {
    const DIM: usize = 512;
    let q = crate::memory::ngram_embed(query, DIM);
    let t = crate::memory::ngram_embed(title, DIM);
    let char_cos = crate::memory::cosine(&q, &t);
    stem_overlap(query, title).max(char_cos * 0.5)
}

/// Отсев нерелевантного. Порог по умолчанию 0.20 (калибровка 12 ноя: релевантный заголовок
/// 0.25, мусор 0.00–0.13).
pub fn filter_relevant_scored(
    query: &str,
    cands: Vec<Candidate>,
    min_score: f32,
) -> (Vec<Candidate>, Vec<(Candidate, f32)>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for c in cands {
        let sc = relevance_score_candidate(query, &c);
        if sc >= min_score {
            kept.push(c);
        } else {
            dropped.push((c, sc));
        }
    }
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_query_strips_question_marks_and_stopwords() {
        let q =
            build_query("Держит ли fiducial-autofocus дрейф < 1 px за 72 ч на OpenFlexure (P0)?");
        assert!(!q.contains('?'));
        assert!(!q.to_lowercase().contains("ли "));
        assert!(q.contains("fiducial-autofocus"), "{q}");
        assert!(q.split_whitespace().count() <= MAX_QUERY_WORDS);
    }

    #[test]
    fn glossary_turns_russian_question_into_english_query() {
        let q = build_query_for(
            "epmc",
            "Показано ли, что fiducial-автофокус удерживает дрейф фокуса < 1 px за 24 ч на открытом микроскопе?",
        );
        for term in ["fiducial", "autofocus", "drift", "focus", "microscope"] {
            assert!(q.contains(term), "нет «{term}» в запросе «{q}»");
        }
        assert!(
            q.split_whitespace().count() > 1,
            "запрос не должен схлопываться в одно слово: «{q}»"
        );
    }

    #[test]
    fn glossary_does_not_override_english_question() {
        let q = build_query_for(
            "epmc",
            "centriole pair distance electron microscopy human fibroblasts",
        );
        assert!(q.contains("centriole"));
        assert!(
            !q.contains("daughter"),
            "глоссарий не должен примешиваться: «{q}»"
        );
    }

    #[test]
    fn english_query_keeps_only_latin_technical_terms() {
        let q = build_query_for(
            "epmc",
            "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч на OpenFlexure (P0, дедлайн 15 янв 2027)?",
        );
        assert!(q.contains("fiducial"), "{q}");
        assert!(q.contains("autofocus"), "{q}");
        assert!(q.contains("OpenFlexure"), "{q}");
        assert!(!q.contains("2027"), "год не должен попадать в запрос: {q}");
        assert!(
            !q.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)),
            "в запрос просочилась кириллица: {q}"
        );
        // без латиницы — откат к общему cleaner
        let q2 = build_query_for("epmc", "Нужен ли Amendment 3 к протоколу?");
        assert!(q2.contains("Amendment"), "{q2}");
    }

    #[test]
    fn parses_epmc_candidates_prefers_pmid() {
        let body = r#"{"resultList":{"result":[
            {"title":"A","pmid":"23663408","doi":"10.1017/s0140525x12000477"},
            {"title":"B","doi":"10.1038/nrn.2016.113"},
            {"title":"C"}]}}"#;
        let c = parse_epmc_candidates(body, "europepmc");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].id, "PMID:23663408");
        assert_eq!(c[1].id, "doi:10.1038/nrn.2016.113");
        assert!(parse_epmc_candidates("not json", "x").is_empty());
    }

    #[test]
    fn corpus_search_finds_ids_in_local_files() {
        let dir = std::env::temp_dir().join(format!("argus-probe-corpus-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("note.md"),
            "Центриольный дрейф фокуса: см. PMID: 23663408 и doi:10.1038/nrn.2016.113",
        )
        .unwrap();
        fs::write(dir.join("other.md"), "ничего про маркеры").unwrap();
        let s = CorpusSearcher::new(&dir);
        let res = s.search("дрейф фокуса Центриольный", 5).unwrap();
        assert!(!res.is_empty());
        assert!(res.iter().any(|c| c.id == "PMID:23663408"));
        assert!(res.iter().all(|c| c.source.starts_with("corpus:")));
        // без совпадений — пусто, а не ошибка
        assert!(s.search("совсем другое слово", 5).unwrap().is_empty());
    }

    #[test]
    fn ladder_goes_from_full_query_to_single_term() {
        let l = query_ladder("fiducial autofocus drift OpenFlexure");
        assert_eq!(l.first().unwrap(), "fiducial autofocus drift OpenFlexure");
        assert!(l.contains(&"fiducial autofocus drift".to_string()));
        assert!(l.contains(&"fiducial".to_string()));
        assert!(l.len() <= 4, "{l:?}");
    }

    #[test]
    fn relevance_filter_drops_titles_without_query_terms() {
        let q = "fiducial-autofocus drift OpenFlexure";
        let cands = vec![
            Candidate {
                id: "PMID:1".into(),
                title: "Fiducial markers for autofocus drift correction".into(),
                source: "e".into(),
                abstract_text: None,
            },
            Candidate {
                id: "PMID:2".into(),
                title: "Biochar improves the nutrient cycle in soils".into(),
                source: "e".into(),
                abstract_text: None,
            },
        ];
        let (kept, dropped) = filter_relevant(q, cands);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].id, "PMID:2");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "PMID:1");
    }

    #[test]
    fn abstract_text_raises_relevance_for_long_query() {
        let q = "phototoxicity limiting factor multi day GFP time lapse imaging";
        let bare = Candidate {
            id: "PMID:10".into(),
            title: "From Static to Dynamic: Fluorescence Imaging Advances".into(),
            source: "e".into(),
            abstract_text: None,
        };
        let with_abs = Candidate {
            abstract_text: Some(
                "Phototoxicity is a limiting factor for multi-day GFP time-lapse imaging;                  we quantify dose budgets and show that fluorescence imaging of live cells                  requires careful light management."
                    .into(),
            ),
            ..bare.clone()
        };
        assert!(
            relevance_score_candidate(q, &with_abs) > relevance_score_candidate(q, &bare),
            "абстракт должен повышать релевантность"
        );
        assert!(relevance_score_candidate(q, &with_abs) >= 0.20);
    }

    #[test]
    fn relevance_filter_accepts_shared_prefix() {
        let cands = vec![Candidate {
            id: "PMID:33443571".into(),
            title: "Centriole and PCM cooperatively recruit CEP192 to spindle poles".into(),
            source: "e".into(),
            abstract_text: None,
        }];
        let (kept, dropped) = filter_relevant("Centrin1 GFP SAS mCherry", cands);
        assert_eq!(dropped.len(), 0, "префикс «Centr» должен засчитываться");
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn relevance_scores_separate_topics() {
        let q = "Centrin1 GFP SAS mCherry";
        let cases = [
            (
                "Centriole and PCM cooperatively recruit CEP192 to spindle poles",
                true,
            ),
            (
                "An extended DNA-free intranuclear compartment organizes centrosome microtubules",
                true,
            ),
            (
                "Ciliogenesis is regulated by a huntingtin-HAP1-PCM1 pathway",
                false,
            ),
            (
                "NuMA assemblies organize microtubule asters to establish spindle bipolarity",
                false,
            ),
            ("Biochar improves the nutrient cycle in sandy soils", false),
        ];
        for (title, want) in cases {
            let sc = relevance_score(q, title);
            assert_eq!(sc >= 0.20, want, "score={sc:.3} title={title}");
        }
    }

    #[test]
    fn offline_epmc_makes_no_request() {
        let s = EpMcSearcher::new(true);
        assert_eq!(s.search("dummy", 3).unwrap_err(), "offline");
    }
}
