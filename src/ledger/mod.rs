//! P9 — idle inference (M1): леджер открытых вопросов + байесовское обновление.
//!
//! Спека: `_gh/ARGUS-OS1/docs/ARGUS_Internal_Process_Idle_Inference_2026-11-12.md` §2–§5.
//! Идея: у агента нет «часов» — цикл запускается только по промпту. Здесь появляется
//! состояние между запросами: prior (вопросы с вероятностями), фильтр (контекст),
//! posterior (сдвиг вероятности только по резолвнутому идентификатору) и бюджет.
//!
//! M1 намеренно не делает проб: `think --once --dry-run` только выбирает вопрос и печатает план.
//! Жёсткий гейт из спеки §6: выдуманный идентификатор → вопрос `Blocked`, вероятность не двигается.

use chrono::{DateTime, Local, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Сдерживание: одно свидетельство не переводит 0.5 в 0.95 (спека §2).
pub const LAMBDA: f64 = 0.5;
/// Правдоподобие независимого подтверждения с резолвнутым идентификатором.
pub const LR_SUPPORT: f64 = 3.0;
/// Правдоподобие противоречащего свидетельства.
pub const LR_CONTRA: f64 = 1.0 / 3.0;
/// Клип вероятности: не бывает 0 и 1 (иначе logit взрывается и вопрос «закрыт навсегда»).
pub const P_MIN: f64 = 0.02;
pub const P_MAX: f64 = 0.98;
/// Порог закрытия вопроса.
pub const P_CLOSE_SUPPORT: f64 = 0.90;
pub const P_CLOSE_REFUTE: f64 = 0.10;

/// Бюджет (спека §4).
pub const MAX_CYCLES_PER_HOUR: u32 = 20;
pub const MAX_TOKENS_PER_DAY: u64 = 200_000;
pub const MAX_PROBES_PER_QUESTION: u32 = 12;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QStatus {
    Open,
    Supported,
    Refuted,
    Blocked,
    Gap,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VerifyResult {
    /// идентификатор(ы) резолвнуты
    Resolved,
    /// инструмент ничего не дал / идентификатора нет
    Unresolved,
    /// идентификатор выглядит валидным, но не резолвится — hard fail
    Fabricated,
}

/// Что принесла проба (только для Resolved).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Support,
    Contradict,
    /// ничего: «искал, не нашёл»
    None,
    /// выдуманный идентификатор
    Fabricated,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Question {
    pub id: String,
    pub text: String,
    pub p_true: f64,
    /// вероятность на прошлом цикле (нужна для surprise)
    #[serde(default = "half")]
    pub p_prev: f64,
    pub created: String,
    #[serde(default)]
    pub last_probed: Option<String>,
    #[serde(default)]
    pub probes: u32,
    /// только резолвнутые идентификаторы
    #[serde(default)]
    pub identifiers: Vec<String>,
    #[serde(default)]
    pub refutations: Vec<String>,
    /// «искал, не нашёл» — легитимный результат, не ошибка
    #[serde(default)]
    pub gaps: Vec<String>,
    pub status: QStatus,
}

fn half() -> f64 {
    0.5
}

impl Question {
    pub fn new(text: &str) -> Self {
        Question {
            id: qid(text),
            text: text.to_string(),
            p_true: 0.5,
            p_prev: 0.5,
            created: now_rfc3339(),
            last_probed: None,
            probes: 0,
            identifiers: Vec::new(),
            refutations: Vec::new(),
            gaps: Vec::new(),
            status: QStatus::Open,
        }
    }
}

impl Question {
    pub fn status_str(&self) -> &'static str {
        match self.status {
            QStatus::Open => "open",
            QStatus::Supported => "supported",
            QStatus::Refuted => "refuted",
            QStatus::Blocked => "blocked",
            QStatus::Gap => "gap",
        }
    }
    /// порядок для вывода списка: открытые — первыми
    pub fn status_key(&self) -> u8 {
        match self.status {
            QStatus::Open => 0,
            QStatus::Gap => 1,
            QStatus::Blocked => 2,
            QStatus::Supported => 3,
            QStatus::Refuted => 4,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cycle {
    pub qid: String,
    pub before: f64,
    pub after: f64,
    /// что делали: tool+args digest либо "none"
    pub probe: String,
    #[serde(default)]
    pub identifiers: Vec<String>,
    pub verifier: VerifyResult,
    pub note: String,
    #[serde(default)]
    pub tokens: u64,
    pub at: String,
}

/// Оценка вопроса для очереди (спека §3.1).
#[derive(Debug, Clone)]
pub struct Scored {
    pub id: String,
    pub text: String,
    pub score: f64,
    pub surprise: f64,
    pub staleness_h: f64,
    pub cost: f64,
}

#[derive(Default)]
pub struct Ledger {
    pub questions: Vec<Question>,
    pub cycles: Vec<Cycle>,
}

fn now_rfc3339() -> String {
    Local::now().to_rfc3339()
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

/// Короткий стабильный id вопроса (не криптография, нужна воспроизводимость).
pub fn qid(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64
    for b in text.trim().as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("q-{:08x}", (h ^ (h >> 32)) as u32)
}

pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Байесовский сдвиг в log-odds с сдерживанием λ.
pub fn shift(p: f64, lr: f64) -> f64 {
    sigmoid(logit(p) + LAMBDA * lr.ln()).clamp(P_MIN, P_MAX)
}

/// Детерминированный RNG (SplitMix64): seeded выбор без новых зависимостей.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Идентификаторы из текста: PMID / DOI / arXiv. Только нормализация, без сети.
pub fn extract_identifiers(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let low = text.to_lowercase();
    let bytes: Vec<char> = text.chars().collect();
    let low_chars: Vec<char> = low.chars().collect();

    // PMID
    let mut i = 0usize;
    while i + 4 <= low_chars.len() {
        if low_chars[i..i + 4].iter().collect::<String>() == "pmid" {
            let mut j = i + 4;
            while j < low_chars.len() && (low_chars[j] == ':' || low_chars[j].is_whitespace()) {
                j += 1;
            }
            let start = j;
            while j < low_chars.len() && low_chars[j].is_ascii_digit() {
                j += 1;
            }
            if j - start >= 4 {
                let digits: String = bytes[start..j].iter().collect();
                let id = format!("PMID:{digits}");
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            i = j.max(i + 4);
        } else {
            i += 1;
        }
    }

    // DOI: 10.<4-9 digits>/<tail> (скан по байтам: срезы str паникуют на UTF-8)
    let s: String = text.to_string();
    let sb = s.as_bytes();
    let mut k = 0usize;
    while k + 4 < sb.len() {
        if &sb[k..k + 3] == b"10." {
            let mut j = k + 3;
            let dstart = j;
            while j < sb.len() && sb[j].is_ascii_digit() {
                j += 1;
            }
            let ndigits = j - dstart;
            if (4..=9).contains(&ndigits) && j < sb.len() && sb[j] == b'/' {
                let mut e = j + 1;
                while e < sb.len() {
                    let c = sb[e] as char;
                    let ok = c.is_ascii_alphanumeric() || ".-_/:;()<>".contains(c);
                    if !ok {
                        break;
                    }
                    e += 1;
                }
                let mut tail = s[j + 1..e].to_string();
                while tail.ends_with(['.', ',', ';', ')', ']', '>']) {
                    tail.pop();
                }
                if tail.len() >= 3 {
                    let id = format!("doi:{}{}", &s[k..j + 1], tail);
                    if !out.contains(&id) {
                        out.push(id);
                    }
                }
                k = e.max(k + 3);
                continue;
            }
        }
        k += 1;
    }

    // arXiv: NNNN.NNNN(N)
    let chars: Vec<char> = s.chars().collect();
    let mut p = 0usize;
    while p + 8 <= chars.len() {
        let is_pre_boundary = p == 0
            || !chars[p - 1].is_ascii_alphanumeric() && chars[p - 1] != '.' && chars[p - 1] != '-';
        if is_pre_boundary
            && chars[p].is_ascii_digit()
            && chars[p + 1].is_ascii_digit()
            && chars[p + 2].is_ascii_digit()
            && chars[p + 3].is_ascii_digit()
            && chars[p + 4] == '.'
            && chars[p + 5].is_ascii_digit()
        {
            let mut q = p + 5;
            while q < chars.len() && chars[q].is_ascii_digit() {
                q += 1;
            }
            let tail_len = q - (p + 5);
            let post_ok = q >= chars.len() || !chars[q].is_ascii_alphanumeric();
            // 4 цифры до точки + 4–5 после; исключаем «10.1234/…» (это DOI, уже поймали)
            if (4..=5).contains(&tail_len) && post_ok && !(p > 0 && chars[p - 1] == '.') {
                let id: String = format!("arXiv:{}", chars[p..q].iter().collect::<String>());
                if !out.contains(&id) {
                    out.push(id);
                }
                p = q;
                continue;
            }
        }
        p += 1;
    }
    out
}

/// Свежесть + удивление + стоимость (спека §3.1).
pub fn score_question(q: &Question, now: DateTime<Utc>) -> Option<Scored> {
    if q.status != QStatus::Open || q.probes >= MAX_PROBES_PER_QUESTION {
        return None;
    }
    let surprise = (logit(q.p_true) - logit(q.p_prev)).abs()
        + if q.gaps.is_empty() { 0.0 } else { 0.5 };
    let ref_ts = parse_ts(q.last_probed.as_deref().unwrap_or(&q.created))?;
    let staleness_h = ((now - ref_ts).num_seconds().max(0) as f64) / 3600.0;
    let cost = 1.0 + 0.25 * q.probes as f64;
    // пол свежести 0.5: иначе только что засеянный леджер даёт score=0 и выбор вырождается
    let score = (0.1 + surprise) * (0.5 + (1.0 + staleness_h).ln()) / cost;
    Some(Scored {
        id: q.id.clone(),
        text: q.text.clone(),
        score,
        surprise,
        staleness_h,
        cost,
    })
}

impl Ledger {
    fn dir(state_dir: &Path) -> PathBuf {
        state_dir.join("ledger")
    }

    pub fn load(state_dir: &Path) -> Self {
        let d = Ledger::dir(state_dir);
        fs::create_dir_all(&d).ok();
        let questions: Vec<Question> =
            fs::read_to_string(d.join("questions.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
        let cycles: Vec<Cycle> = fs::read_to_string(d.join("cycles.jsonl"))
            .ok()
            .map(|s| {
                s.lines()
                    .filter(|l| !l.trim().is_empty())
                    .filter_map(|l| serde_json::from_str::<Cycle>(l).ok())
                    .collect()
            })
            .unwrap_or_default();
        Ledger { questions, cycles }
    }

    pub fn save(&self, state_dir: &Path) -> std::io::Result<()> {
        let d = Ledger::dir(state_dir);
        fs::create_dir_all(&d)?;
        let s = serde_json::to_string_pretty(&self.questions)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(d.join("questions.json"), s)
    }

    pub fn append_cycle(&self, state_dir: &Path, c: &Cycle) -> std::io::Result<()> {
        let d = Ledger::dir(state_dir);
        fs::create_dir_all(&d)?;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(d.join("cycles.jsonl"))?;
        let line = serde_json::to_string(c)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writeln!(f, "{line}")
    }

    /// Добавляет вопрос (или «переоткрывает» закрытый) и возвращает его id.
    pub fn add(&mut self, text: &str, p: Option<f64>) -> String {
        let mut q = Question::new(text);
        if let Some(p) = p {
            q.p_true = p.clamp(P_MIN, P_MAX);
            q.p_prev = q.p_true;
        }
        let id = q.id.clone();
        if let Some(existing) = self.questions.iter_mut().find(|x| x.id == id) {
            if existing.status == QStatus::Refuted {
                existing.status = QStatus::Open;
                existing.gaps.clear();
            }
            return id;
        }
        self.questions.push(q);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Question> {
        self.questions.iter().find(|q| q.id == id)
    }

    /// Топ-K кандидатов по score (детерминированно).
    pub fn ranked(&self, now: DateTime<Utc>) -> Vec<Scored> {
        let mut v: Vec<Scored> = self
            .questions
            .iter()
            .filter_map(|q| score_question(q, now))
            .collect();
        v.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        v
    }

    /// Выбор вопроса: взвешенная рулетка по score среди топ-K, seeded (воспроизводимо).
    pub fn select_next(&self, seed: u64, k: usize) -> Option<Scored> {
        let ranked = self.ranked(Utc::now());
        if ranked.is_empty() {
            return None;
        }
        let pool: Vec<&Scored> = ranked.iter().take(k.max(1)).collect();
        let total: f64 = pool.iter().map(|s| s.score.max(0.0)).sum();
        if total <= 0.0 {
            return pool.first().map(|s| (*s).clone());
        }
        let mut rng = SplitMix64::new(seed);
        let mut pick = rng.next_f64() * total;
        for s in &pool {
            pick -= s.score.max(0.0);
            if pick <= 0.0 {
                return Some((*s).clone());
            }
        }
        pool.last().map(|s| (*s).clone())
    }

    /// Применить исход пробы к вопросу и вернуть запись цикла (спека §3.4–3.5).
    pub fn apply(
        &mut self,
        id: &str,
        outcome: Outcome,
        probe: &str,
        identifiers: Vec<String>,
        note: &str,
        tokens: u64,
    ) -> Option<Cycle> {
        let now = now_rfc3339();
        let q = self.questions.iter_mut().find(|q| q.id == id)?;
        let before = q.p_true;
        q.probes += 1;
        q.last_probed = Some(now.clone());

        let (verifier, after) = match outcome {
            // выдуманный идентификатор: hard fail, вероятность НЕ двигается
            Outcome::Fabricated => {
                q.status = QStatus::Blocked;
                (VerifyResult::Fabricated, before)
            }
            Outcome::None => {
                q.gaps.push(note.to_string());
                (VerifyResult::Unresolved, before)
            }
            Outcome::Support | Outcome::Contradict => {
                if identifiers.is_empty() {
                    // свидетельство без идентификатора не засчитывается
                    q.gaps.push(format!("{note} [no identifier]"));
                    (VerifyResult::Unresolved, before)
                } else {
                    let lr = if outcome == Outcome::Support {
                        LR_SUPPORT
                    } else {
                        LR_CONTRA
                    };
                    let p = shift(before, lr);
                    q.p_prev = before;
                    q.p_true = p;
                    for i in &identifiers {
                        if !q.identifiers.contains(i) {
                            q.identifiers.push(i.clone());
                        }
                    }
                    if outcome == Outcome::Contradict {
                        q.refutations.push(note.to_string());
                    }
                    if p >= P_CLOSE_SUPPORT {
                        q.status = QStatus::Supported;
                    } else if p <= P_CLOSE_REFUTE {
                        q.status = QStatus::Refuted;
                    }
                    (VerifyResult::Resolved, p)
                }
            }
        };

        let cycle = Cycle {
            qid: q.id.clone(),
            before,
            after,
            probe: probe.to_string(),
            identifiers,
            verifier,
            note: note.to_string(),
            tokens,
            at: now,
        };
        self.cycles.push(cycle.clone());
        Some(cycle)
    }

    /// Метрики по записанным циклам (спека §5).
    pub fn metrics(&self) -> Metrics {
        metrics(&self.cycles)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub cycles: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub fabricated: usize,
    pub identifier_resolution_rate: f64,
    pub fabricated_identifiers: usize,
    pub refutation_share: f64,
    pub mean_abs_logit_shift: f64,
    pub gaps_recorded: usize,
}

pub fn metrics(cycles: &[Cycle]) -> Metrics {
    let mut m = Metrics {
        cycles: cycles.len(),
        ..Default::default()
    };
    let mut shifts = Vec::new();
    for c in cycles {
        match c.verifier {
            VerifyResult::Resolved => m.resolved += 1,
            VerifyResult::Unresolved => m.unresolved += 1,
            VerifyResult::Fabricated => {
                m.fabricated += 1;
                m.fabricated_identifiers += c.identifiers.len().max(1);
            }
        }
        if (c.after - c.before).abs() > f64::EPSILON {
            shifts.push((logit(c.after) - logit(c.before)).abs());
        }
        if c.verifier == VerifyResult::Unresolved && c.note.contains("[no identifier]") {
            m.gaps_recorded += 1;
        }
    }
    let claimed = m.resolved + m.unresolved + m.fabricated;
    m.identifier_resolution_rate = if claimed == 0 {
        0.0
    } else {
        m.resolved as f64 / claimed as f64
    };
    let closed = cycles
        .iter()
        .filter(|c| c.after <= P_CLOSE_REFUTE)
        .count();
    m.refutation_share = if claimed == 0 {
        0.0
    } else {
        closed as f64 / claimed as f64
    };
    m.mean_abs_logit_shift = if shifts.is_empty() {
        0.0
    } else {
        shifts.iter().sum::<f64>() / shifts.len() as f64
    };
    m
}

/// Гейт спеки §6: провал, если есть выдуманные идентификаторы или резолв ниже порога.
pub fn gate_failed(m: &Metrics) -> bool {
    m.fabricated_identifiers > 0 || (m.cycles >= 5 && m.identifier_resolution_rate < 0.7)
}

/// Реальные открытые вопросы ARGUS-OS1 для засева (не выдуманные: из STATE/TODO/TRD).
pub fn argus_default_questions() -> Vec<&'static str> {
    vec![
        "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч на OpenFlexure (P0, дедлайн 15 янв 2027)?",
        "Падает ли recall ниже 0.5 при 250 нм/px на фантоме readout (бенчмарк, DOI 10.5281/zenodo.24148459)?",
        "Нужен ли Amendment 3 к протоколу v0.3 при переходе на выборку ≤100 нм/px без бининга >2×2?",
        "Сходится ли Centrin1-GFP и SAS-4::mCherry на одном центриоле (пилот P4, кросс-валидация маркеров)?",
        "Достаточно ли 3 ч imaging без фототоксического потолка для SAS-4::GFP (пилот P2/P3)?",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn logit_sigmoid_roundtrip() {
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            assert!((sigmoid(logit(p)) - p).abs() < 1e-9, "p={p}");
        }
    }

    #[test]
    fn support_raises_but_damped() {
        let p = shift(0.5, LR_SUPPORT);
        assert!(p > 0.5 && p < 0.9, "p={p}");
        let p2 = shift(p, LR_SUPPORT);
        assert!(p2 > p);
    }

    #[test]
    fn contradict_lowers() {
        let p = shift(0.5, LR_CONTRA);
        assert!(p < 0.5 && p > 0.1, "p={p}");
    }

    #[test]
    fn identifiers_extracted() {
        let ids = extract_identifiers(
            "см. PMID: 23663408, doi:10.1007/s13194-023-00533-w и arXiv:2504.13171.",
        );
        assert!(ids.contains(&"PMID:23663408".to_string()), "{ids:?}");
        assert!(ids
            .contains(&"doi:10.1007/s13194-023-00533-w".to_string()));
        assert!(ids.contains(&"arXiv:2504.13171".to_string()));
        assert_eq!(ids.len(), 3, "{ids:?}");
    }

    #[test]
    fn fabricated_identifier_blocks_and_freezes_p() {
        let mut l = Ledger::default();
        let id = l.add("тестовый вопрос", None);
        let c = l
            .apply(
                &id,
                Outcome::Fabricated,
                "verify",
                vec!["doi:10.9999/fake".into()],
                "не резолвится",
                0,
            )
            .unwrap();
        assert_eq!(c.verifier, VerifyResult::Fabricated);
        assert!((c.after - c.before).abs() < 1e-9);
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Blocked);
        assert!(gate_failed(&l.metrics()));
    }

    #[test]
    fn evidence_without_identifier_is_only_a_gap() {
        let mut l = Ledger::default();
        let id = l.add("вопрос", None);
        l.apply(&id, Outcome::Support, "web.search", vec![], "нашёл текст", 10);
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Open);
        assert!((q.p_true - 0.5).abs() < 1e-9);
        assert_eq!(q.gaps.len(), 1);
        assert!(q.gaps[0].contains("[no identifier]"));
    }

    #[test]
    fn resolved_evidence_moves_p_and_records_id() {
        let mut l = Ledger::default();
        let id = l.add("вопрос", None);
        let c = l
            .apply(
                &id,
                Outcome::Support,
                "web.search",
                vec!["PMID:23663408".into()],
                "Clark 2013",
                1234,
            )
            .unwrap();
        assert_eq!(c.verifier, VerifyResult::Resolved);
        assert!(c.after > c.before);
        let q = l.get(&id).unwrap();
        assert_eq!(q.identifiers, vec!["PMID:23663408".to_string()]);
        assert_eq!(q.probes, 1);
        assert!(q.last_probed.is_some());
        let m = l.metrics();
        assert_eq!(m.resolved, 1);
        assert!((m.identifier_resolution_rate - 1.0).abs() < 1e-9);
        assert!(m.mean_abs_logit_shift > 0.0);
    }

    #[test]
    fn fresh_questions_still_have_positive_score() {
        let mut l = Ledger::default();
        let q1 = l.add("свежий1", None);
        let q2 = l.add("свежий2", None);
        let r = l.ranked(Utc::now());
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|s| s.score > 0.0), "пол свежести не работает: {r:?}");
        // после одного обновления у q2 появляется surprise → он впереди
        l.apply(&q2, Outcome::Support, "p", vec!["PMID:23663408".into()], "n", 0);
        let r2 = l.ranked(Utc::now());
        assert_eq!(r2[0].id, q2);
        assert!(r2[0].surprise > r2[1].surprise);
        assert!(r2.iter().all(|s| s.id == q1 || s.id == q2));
    }

    #[test]
    fn selection_is_seeded_and_skips_blocks() {
        let mut l = Ledger::default();
        let a = l.add("вопрос A", None);
        let b = l.add("вопрос B", Some(0.2));
        let c = l.add("вопрос C", Some(0.8));
        // C и B получают историю: у B был сдвиг → surprise выше
        l.apply(&b, Outcome::Support, "p", vec!["PMID:1234567".into()], "n", 0);
        let s1 = l.select_next(42, 3).unwrap();
        let s2 = l.select_next(42, 3).unwrap();
        assert_eq!(s1.id, s2.id, "seed должен давать один и тот же вопрос");
        // разные seed могут дать разные вопросы, но каждый — из пула
        let picks: Vec<String> = (0..20).map(|s| l.select_next(s, 3).unwrap().id).collect();
        assert!(picks.iter().all(|p| [&a, &b, &c].contains(&p)));
        // Blocked выпадает из очереди
        l.apply(&a, Outcome::Fabricated, "p", vec!["doi:10.0/x".into()], "bad", 0);
        let after: Vec<String> = (0..20).map(|s| l.select_next(s, 3).unwrap().id).collect();
        assert!(!after.contains(&a), "Blocked-вопрос не должен выбираться");
    }

    #[test]
    fn closed_question_leaves_queue() {
        let mut l = Ledger::default();
        let id = l.add("закрываемый", None);
        for _ in 0..8 {
            l.apply(&id, Outcome::Contradict, "p", vec!["PMID:7654321".into()], "no", 0);
        }
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Refuted, "p={}", q.p_true);
        assert!(l.select_next(1, 3).is_none());
        assert!(l.metrics().refutation_share > 0.0);
    }

    #[test]
    fn default_questions_are_real_and_unique() {
        let qs = argus_default_questions();
        assert_eq!(qs.len(), 5);
        let ids: std::collections::HashSet<String> = qs.iter().map(|t| qid(t)).collect();
        assert_eq!(ids.len(), 5);
    }
}
