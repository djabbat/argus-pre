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
/// Консилиум бесплатных моделей (`crate::dialogue`) — мнение, а не измерение:
/// сдвиг слабее литературного свидетельства и НЕ может закрыть вопрос.
pub const LR_COUNCIL_SUPPORT: f64 = 1.5;
pub const LR_COUNCIL_CONTRADICT: f64 = 1.0 / 1.5;
/// Не больше одного консилиумного сдвига на вопрос в сутки (защита от «накрутки»).
pub const COUNCIL_MAX_SHIFTS_PER_DAY: u32 = 1;
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
/// После стольких проб подряд без идентификаторов вопрос уходит в `Gap` и освобождает очередь.
pub const MAX_CONSECUTIVE_GAPS: u32 = 3;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QStatus {
    Open,
    Supported,
    Refuted,
    Blocked,
    Gap,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VerifyResult {
    /// идентификатор(ы) резолвнуты
    Resolved,
    /// инструмент ничего не дал / идентификатора нет
    Unresolved,
    /// идентификатор выглядит валидным, но не резолвится — hard fail
    Fabricated,
    /// свидетельства-идентификатора нет: применён консилиум бесплатных моделей
    Council,
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
    /// консилиум бесплатных чатов сказал «да»/«нет» — слабый сдвиг приора (не доказательство)
    CouncilSupport,
    CouncilContradict,
    /// найдено свидетельство, но пригодность его к вопросу НЕ оценена (M2: проба без суждения).
    /// Идентификаторы записываются, вероятность НЕ двигается — иначе агент набирает уверенность из шума.
    Evidence,
}

/// M4-фикс: лабораторный вопрос («держит ли фокус дрейф <1 px за 72 ч») литературой не отвечается.
/// Проба по Europe PMC имеет смысл только для `Lit`; `Lab` отвечается экспериментом и файлами.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QKind {
    Lit,
    Lab,
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
    /// литературный вопрос или лабораторный (влияет на выбор канала пробы)
    #[serde(default = "default_kind")]
    pub kind: QKind,
    /// сколько проб подряд не дали ни одного идентификатора (защита от «залипания»)
    #[serde(default)]
    pub consecutive_gaps: u32,
    /// что система ОЖИДАЕТ найти (прогноз); внешнее даёт разницу с этим — а не «смысл»
    #[serde(default)]
    pub expected: Option<String>,
    /// измеренная ошибка предсказания 0..1 на прошлой пробе: |Δlogit|/3 для суждённого
    /// свидетельства, 0.35 для найденного но не суждённого, 0.1 для «ничего не нашли»
    #[serde(default)]
    pub last_error: f32,
    /// форма вопроса (бинарная претензия / количественная оценка / поисковая тема).
    /// `None` в старых леджерах → вычисляется при загрузке из текста.
    #[serde(default)]
    pub form: Option<crate::stance::QForm>,
    /// id вопроса, которым этот заменён при переформулировке (13 ноя). Заменённый вопрос
    /// закрыт навсегда: `reopen_stale_gaps` его не воскрешает, иначе фразы/оценки
    /// возвращались бы в очередь каждые 6 часов.
    #[serde(default)]
    pub superseded_by: Option<String>,
    /// сколько раз консилиум сдвигал вероятность (ограничено в сутки)
    #[serde(default)]
    pub council_shifts: u32,
    /// когда консилиум последний раз сдвигал p (RFC3339)
    #[serde(default)]
    pub last_council_at: Option<String>,
}

fn default_kind() -> QKind {
    QKind::Lit
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
            kind: QKind::Lit,
            consecutive_gaps: 0,
            expected: None,
            last_error: 0.0,
            form: Some(crate::stance::detect_form(text)),
            superseded_by: None,
            council_shifts: 0,
            last_council_at: None,
        }
    }

    /// Форма вопроса: сохранённая либо (для старых записей) выведенная из текста.
    pub fn form(&self) -> crate::stance::QForm {
        self.form
            .unwrap_or_else(|| crate::stance::detect_form(&self.text))
    }

    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            QKind::Lit => "lit",
            QKind::Lab => "lab",
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
    /// позиция свидетельства относительно вопроса (`support`/`contradict`/`neutral`),
    /// оценённая `crate::stance`; `None` — старые циклы и ветки без оценки
    #[serde(default)]
    pub stance: Option<String>,
    /// уверенность классификатора 0..1
    #[serde(default)]
    pub stance_conf: f32,
    /// откуда позиция: `lexical:title+abstract` и т.п.
    #[serde(default)]
    pub stance_source: Option<String>,
    /// форма вопроса на момент цикла: `proposition`/`quantitative`/`topic`
    #[serde(default)]
    pub form: Option<String>,
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
    score_question_for(q, now, true)
}

/// То же, но с учётом канала: литературная проба (`epmc`) не имеет права брать `Lab`-вопросы.
pub fn score_question_for(q: &Question, now: DateTime<Utc>, lit_channel: bool) -> Option<Scored> {
    if q.status != QStatus::Open || q.probes >= MAX_PROBES_PER_QUESTION {
        return None;
    }
    if lit_channel && q.kind == QKind::Lab {
        return None;
    }
    // Внимание привлекает РАЗНИЦА между прогнозом и реальностью (ошибка предсказания),
    // а не «сколько раз не нашлось»: провал поиска — это низкая точность канала, а не сигнал.
    // Затухание: 24 ч после пробы ошибка падает вдвое — иначе одна сенсация держит очередь вечно.
    let ref_hours = parse_ts(q.last_probed.as_deref().unwrap_or(&q.created))
        .map(|t| ((now - t).num_seconds().max(0) as f64) / 3600.0)
        .unwrap_or(0.0);
    let decay = (-ref_hours / 24.0).exp() as f64;
    let surprise = (q.last_error as f64) * decay;
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
        let questions: Vec<Question> = fs::read_to_string(d.join("questions.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let mut questions = questions;
        // старые леджеры без поля `form`: достраиваем при загрузке (не переписывая файл сами по себе)
        for q in questions.iter_mut() {
            if q.form.is_none() {
                q.form = Some(crate::stance::detect_form(&q.text));
            }
            // Израсходованный бюджет проб при `open` = вопрос невидим в очереди (не выбирается)
            // и при этом не закрыт. Это ложное состояние: переводим в `gap` (переоткроется
            // через reopen_stale_gaps, если человек не решит иначе).
            if q.probes >= MAX_PROBES_PER_QUESTION && q.status == QStatus::Open {
                q.status = QStatus::Gap;
                q.gaps.push(format!(
                    "исчерпан бюджет {MAX_PROBES_PER_QUESTION} проб без закрытия"
                ));
            }
        }
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
        self.add_kind(text, p, QKind::Lit)
    }

    pub fn add_kind(&mut self, text: &str, p: Option<f64>, kind: QKind) -> String {
        let mut q = Question::new(text);
        q.kind = kind;
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
        self.ranked_for(now, false)
    }

    /// `lit_channel = true` — литературная проба: `Lab`-вопросы исключены.
    pub fn ranked_for(&self, now: DateTime<Utc>, lit_channel: bool) -> Vec<Scored> {
        let mut v: Vec<Scored> = self
            .questions
            .iter()
            .filter_map(|q| score_question_for(q, now, lit_channel))
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
        self.select_next_for(seed, k, false)
    }

    pub fn select_next_for(&self, seed: u64, k: usize, lit_channel: bool) -> Option<Scored> {
        let now = Utc::now();
        // Честная очередь (фикс 13 ноя): без штрафа за недавний повтор один и тот же вопрос
        // выигрывал рулетку 5 раз в час (q-b6e15701), а 14 других не опрашивались вообще.
        // Штраф 1/4 за каждый прогон за последний час: после третьего повтора вопрос
        // уступает место тем, кого давно не трогали — возникает round-robin вместо залипания.
        let mut ranked: Vec<Scored> = self
            .ranked_for(now, lit_channel)
            .into_iter()
            .map(|mut s| {
                let hits = self.recent_hits(&s.id, now, 1);
                s.score *= 0.25_f64.powi(hits.min(8) as i32);
                s
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
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

    /// Сколько раз вопрос опрашивался за последние `hours` часов (по журналу циклов).
    pub fn recent_hits(&self, id: &str, now: DateTime<Utc>, hours: i64) -> usize {
        let cutoff = now - chrono::Duration::hours(hours.max(1));
        self.cycles
            .iter()
            .filter(|c| c.qid == id)
            .filter(|c| parse_ts(&c.at).map(|t| t >= cutoff).unwrap_or(false))
            .count()
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
        self.apply_judged(id, outcome, probe, identifiers, note, tokens, None)
    }

    /// То же, но с записью позиции свидетельства (`crate::stance::Decision`) в цикл.
    ///
    /// Гейт спецификации §2: вероятность двигается **только** у вопроса-претензии
    /// (`QForm::Proposition`). Для количественного вопроса или поисковой темы
    /// исход `Support`/`Contradict` деградирует до `Evidence`: идентификаторы пишутся,
    /// `p` не меняется. Причина — у темы нет истинностного значения, у оценки нет «да/нет».
    pub fn apply_judged(
        &mut self,
        id: &str,
        outcome: Outcome,
        probe: &str,
        identifiers: Vec<String>,
        note: &str,
        tokens: u64,
        stance: Option<&crate::stance::Decision>,
    ) -> Option<Cycle> {
        let now = now_rfc3339();
        let q = self.questions.iter_mut().find(|q| q.id == id)?;
        let before = q.p_true;
        let form = q.form();
        q.probes += 1;
        q.last_probed = Some(now.clone());
        // бюджет проб исчерпан, а вопрос не закрыт → он не должен висеть «open» невидимым
        if q.probes == MAX_PROBES_PER_QUESTION {
            if q.status == QStatus::Open {
                q.status = QStatus::Gap;
            }
            q.gaps.push(format!(
                "исчерпан бюджет {MAX_PROBES_PER_QUESTION} проб без закрытия"
            ));
        }

        let (verifier, after) = match outcome {
            // выдуманный идентификатор: hard fail, вероятность НЕ двигается
            Outcome::Fabricated => {
                q.status = QStatus::Blocked;
                q.consecutive_gaps = 0;
                q.last_error = 1.0;
                (VerifyResult::Fabricated, before)
            }
            Outcome::None => {
                q.gaps.push(note.to_string());
                q.consecutive_gaps += 1;
                q.last_error = 0.1;
                if q.consecutive_gaps >= MAX_CONSECUTIVE_GAPS {
                    q.status = QStatus::Gap;
                }
                (VerifyResult::Unresolved, before)
            }
            Outcome::Evidence => {
                if identifiers.is_empty() {
                    q.consecutive_gaps += 1;
                    q.last_error = 0.1;
                    (VerifyResult::Unresolved, before)
                } else {
                    // «новых идентификаторов нет» — это пробел, а не прогресс (спека §3.6):
                    // повторный прогон того же запроса возвращает те же PMID, и без этого
                    // счётчика вопрос висел в очереди до потолка из 12 проб.
                    let novel = identifiers.iter().any(|i| !q.identifiers.contains(i));
                    for i in &identifiers {
                        if !q.identifiers.contains(i) {
                            q.identifiers.push(i.clone());
                        }
                    }
                    if novel {
                        q.consecutive_gaps = 0;
                        q.last_error = 0.35;
                    } else {
                        q.consecutive_gaps += 1;
                        q.last_error = 0.1;
                        q.gaps.push(format!("{note} [новых идентификаторов нет]"));
                        if q.consecutive_gaps >= MAX_CONSECUTIVE_GAPS {
                            q.status = QStatus::Gap;
                        }
                    }
                    (VerifyResult::Resolved, before)
                }
            }
            Outcome::CouncilSupport | Outcome::CouncilContradict => {
                let recent = q
                    .last_council_at
                    .as_deref()
                    .and_then(parse_ts)
                    .map(|t| (Utc::now() - t).num_hours() < 24)
                    .unwrap_or(false);
                if recent {
                    q.gaps.push(format!(
                        "{note} [консилиум уже сдвигал p за последние 24 ч — пропуск]"
                    ));
                    (VerifyResult::Council, before)
                } else if !form.allows_lr() {
                    q.gaps.push(format!(
                        "{note} [form={}: консилиум не применяется]",
                        form.as_str()
                    ));
                    (VerifyResult::Council, before)
                } else {
                    let lr = if outcome == Outcome::CouncilSupport {
                        LR_COUNCIL_SUPPORT
                    } else {
                        LR_COUNCIL_CONTRADICT
                    };
                    let p = shift(before, lr);
                    // консилиум НЕ закрывает вопрос: держим p ниже порога закрытия
                    let capped = p.min(P_CLOSE_SUPPORT - 0.005).max(P_MIN);
                    q.p_prev = before;
                    q.p_true = capped;
                    q.last_error = (((logit(capped) - logit(before)).abs() / 3.0) as f32)
                        .clamp(0.0, 1.0);
                    q.council_shifts += 1;
                    q.last_council_at = Some(now.clone());
                    (VerifyResult::Council, capped)
                }
            }
            Outcome::Support | Outcome::Contradict => {
                if identifiers.is_empty() {
                    // свидетельство без идентификатора не засчитывается
                    q.gaps.push(format!("{note} [no identifier]"));
                    (VerifyResult::Unresolved, before)
                } else if !form.allows_lr() {
                    // форма вопроса не допускает байесовского сдвига: пишем ID, не трогаем p
                    q.gaps
                        .push(format!("{note} [form={}: p не определена] ", form.as_str()));
                    q.consecutive_gaps = 0;
                    q.last_error = 0.35;
                    for i in &identifiers {
                        if !q.identifiers.contains(i) {
                            q.identifiers.push(i.clone());
                        }
                    }
                    (VerifyResult::Resolved, before)
                } else if identifiers.iter().all(|i| q.identifiers.contains(i)) {
                    // Те же идентификаторы, что уже учтены: повторный прогон той же пробы
                    // не должен начислять LR второй раз (иначе одна статья двигает веру
                    // бесконечно — цикл с тем же запросом возвращает те же PMID).
                    q.gaps.push(format!(
                        "{note} [повтор: идентификаторы уже учтены, LR не начислен]"
                    ));
                    q.consecutive_gaps += 1;
                    q.last_error = 0.1;
                    (VerifyResult::Resolved, before)
                } else {
                    let lr = if outcome == Outcome::Support {
                        LR_SUPPORT
                    } else {
                        LR_CONTRA
                    };
                    let p = shift(before, lr);
                    q.p_prev = before;
                    q.p_true = p;
                    // суждённое свидетельство = реальное обновление веры → его величина и есть ошибка
                    q.last_error =
                        (((logit(p) - logit(before)).abs() / 3.0) as f32).clamp(0.0, 1.0);
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
            stance: stance.map(|s| s.stance.as_str().to_string()),
            stance_conf: stance.map(|s| s.confidence).unwrap_or(0.0),
            stance_source: stance.map(|s| s.source.to_string()),
            form: Some(form.as_str().to_string()),
        };
        self.cycles.push(cycle.clone());
        Some(cycle)
    }

    /// Закрыть вопрос человеком: статус ставится явно (не выводится из LR), причина пишется
    /// в `gaps`/`refutations`, в журнал добавляется цикл `human:close` — для аудита.
    /// Нужно потому, что `close` без идентификатора иначе не мог ни подтвердить, ни опровергнуть
    /// вопрос (ветка `Support` с пустым списком ID = «не засчитывается»).
    pub fn close_by_human(&mut self, id: &str, how: &str, note: &str) -> Option<Cycle> {
        let now = now_rfc3339();
        let q = self.questions.iter_mut().find(|q| q.id == id)?;
        let p = q.p_true;
        let form = q.form();
        match how {
            "supported" => {
                q.status = QStatus::Supported;
                q.gaps.push(format!("закрыт человеком (supported): {note}"));
            }
            "refuted" => {
                q.status = QStatus::Refuted;
                q.refutations.push(format!("закрыт человеком (refuted): {note}"));
            }
            _ => {
                q.status = QStatus::Gap;
                q.gaps.push(format!("закрыт человеком (gap): {note}"));
            }
        }
        let cycle = Cycle {
            qid: q.id.clone(),
            before: p,
            after: p,
            probe: "human:close".to_string(),
            identifiers: Vec::new(),
            verifier: VerifyResult::Resolved,
            note: format!("{how}: {note}"),
            tokens: 0,
            at: now,
            stance: None,
            stance_conf: 0.0,
            stance_source: Some("human".to_string()),
            form: Some(form.as_str().to_string()),
        };
        self.cycles.push(cycle.clone());
        Some(cycle)
    }

    /// Пометить вопрос заменённым на другой (переформулировка): закрыт навсегда, но виден в
    /// журнале с указателем. Возвращает запись цикла для аудита.
    pub fn supersede(&mut self, old_id: &str, new_id: &str) -> Option<Cycle> {
        let now = now_rfc3339();
        let q = self.questions.iter_mut().find(|q| q.id == old_id)?;
        let p = q.p_true;
        let form = q.form();
        q.status = QStatus::Gap;
        q.superseded_by = Some(new_id.to_string());
        q.gaps.push(format!(
            "заменён на {new_id}: форма «{}» — не претензия, p не определена",
            form.as_str()
        ));
        let cycle = Cycle {
            qid: q.id.clone(),
            before: p,
            after: p,
            probe: "human:supersede".to_string(),
            identifiers: Vec::new(),
            verifier: VerifyResult::Resolved,
            note: format!("заменён на {new_id}"),
            tokens: 0,
            at: now,
            stance: None,
            stance_conf: 0.0,
            stance_source: Some("human".to_string()),
            form: Some(form.as_str().to_string()),
        };
        self.cycles.push(cycle.clone());
        Some(cycle)
    }

    /// Возвращает в очередь вопросы, которые ушли в `Gap`, если с последней пробы прошло
    /// больше `hours` часов: «не нашлось сегодня» ≠ «не найдётся никогда».
    pub fn reopen_stale_gaps(&mut self, now: DateTime<Utc>, hours: i64) -> usize {
        let mut n = 0;
        for q in self.questions.iter_mut() {
            if q.status != QStatus::Gap || q.superseded_by.is_some() {
                continue;
            }
            let ts = q.last_probed.clone().unwrap_or_else(|| q.created.clone());
            let old = parse_ts(&ts)
                .map(|t| (now - t).num_hours() >= hours)
                .unwrap_or(true);
            if old {
                q.status = QStatus::Open;
                q.consecutive_gaps = 0;
                n += 1;
            }
        }
        n
    }

    /// Метрики по записанным циклам (спека §5).
    pub fn metrics(&self) -> Metrics {
        metrics(&self.cycles)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub cycles: usize,
    /// циклы, в которых заявлялся хотя бы один идентификатор (только по ним считается резолв)
    pub id_cycles: usize,
    /// циклы без идентификаторов: честный «пробел» (поиск ничего не дал)
    pub gap_cycles: usize,
    pub resolved: usize,
    pub unresolved: usize,
    pub fabricated: usize,
    /// resolved / id_cycles — НЕ по всем циклам: иначе «ничего не нашёл» выглядит как провал резолва
    pub identifier_resolution_rate: f64,
    /// доля циклов, давших хоть один идентификатор: отдельная метрика выхода, не гейт
    pub evidence_yield: f64,
    pub fabricated_identifiers: usize,
    pub refutation_share: f64,
    pub mean_abs_logit_shift: f64,
    pub gaps_recorded: usize,
    /// циклы, где вероятность двигал консилиум бесплатных моделей (не идентификатор)
    pub council_cycles: usize,
}

pub fn metrics(cycles: &[Cycle]) -> Metrics {
    let mut m = Metrics {
        cycles: cycles.len(),
        ..Default::default()
    };
    let mut shifts = Vec::new();
    for c in cycles {
        if c.verifier == VerifyResult::Council {
            m.council_cycles += 1;
        } else if c.verifier == VerifyResult::Fabricated {
            m.fabricated += 1;
            m.fabricated_identifiers += c.identifiers.len().max(1);
        } else if !c.identifiers.is_empty() {
            m.id_cycles += 1;
            match c.verifier {
                VerifyResult::Resolved => m.resolved += 1,
                VerifyResult::Unresolved => m.unresolved += 1,
                VerifyResult::Fabricated | VerifyResult::Council => {}
            }
        } else {
            m.gap_cycles += 1;
            m.gaps_recorded += 1;
        }
        if (c.after - c.before).abs() > f64::EPSILON {
            shifts.push((logit(c.after) - logit(c.before)).abs());
        }
    }
    m.identifier_resolution_rate = if m.id_cycles == 0 {
        0.0
    } else {
        m.resolved as f64 / m.id_cycles as f64
    };
    m.evidence_yield = if m.cycles == 0 {
        0.0
    } else {
        m.id_cycles as f64 / m.cycles as f64
    };
    let closed = cycles.iter().filter(|c| c.after <= P_CLOSE_REFUTE).count();
    m.refutation_share = if m.id_cycles == 0 {
        0.0
    } else {
        closed as f64 / m.id_cycles as f64
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
    m.fabricated_identifiers > 0 || (m.id_cycles >= 5 && m.identifier_resolution_rate < 0.7)
}

/// Реальные открытые вопросы ARGUS-OS1 для засева (не выдуманные: из STATE/TODO/TRD).
pub fn argus_default_questions() -> Vec<(&'static str, QKind)> {
    vec![
        // литературные: на них может ответить Europe PMC
        ("Какая точность fiducial-based autofocus достигнута в опубликованных открытых микроскопах?", QKind::Lit),
        ("Сходится ли Centrin1-GFP и SAS-4::mCherry на одном центриоле (пилот P4, кросс-валидация маркеров)?", QKind::Lit),
        ("Достаточно ли 3 ч imaging без фототоксического потолка для SAS-4::GFP (пилот P2/P3)?", QKind::Lit),
        // лабораторные: литературой не отвечаются, нужен эксперимент/файлы
        ("Держит ли fiducial-autofocus дрейф < 1 px за 72 ч на OpenFlexure (P0, дедлайн 15 янв 2027)?", QKind::Lab),
        ("Падает ли recall ниже 0.5 при 250 нм/px на фантоме readout (бенчмарк, DOI 10.5281/zenodo.24148459)?", QKind::Lab),
        ("Нужен ли Amendment 3 к протоколу v0.3 при переходе на выборку ≤100 нм/px без бининга >2×2?", QKind::Lab),
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
        assert!(ids.contains(&"doi:10.1007/s13194-023-00533-w".to_string()));
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
        l.apply(
            &id,
            Outcome::Support,
            "web.search",
            vec![],
            "нашёл текст",
            10,
        );
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Open);
        assert!((q.p_true - 0.5).abs() < 1e-9);
        assert_eq!(q.gaps.len(), 1);
        assert!(q.gaps[0].contains("[no identifier]"));
    }

    #[test]
    fn resolved_evidence_moves_p_and_records_id() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли тестовый стенд дрейф < 1 px за 72 ч?", None);
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
        let q1 = l.add("Держит ли свежий стенд A дрейф < 1 px?", None);
        let q2 = l.add("Держит ли свежий стенд B дрейф < 1 px?", None);
        let r = l.ranked(Utc::now());
        assert_eq!(r.len(), 2);
        assert!(
            r.iter().all(|s| s.score > 0.0),
            "пол свежести не работает: {r:?}"
        );
        // после одного обновления у q2 появляется surprise → он впереди
        l.apply(
            &q2,
            Outcome::Support,
            "p",
            vec!["PMID:23663408".into()],
            "n",
            0,
        );
        let r2 = l.ranked(Utc::now());
        assert_eq!(r2[0].id, q2);
        assert!(r2[0].surprise > r2[1].surprise);
        assert!(r2.iter().all(|s| s.id == q1 || s.id == q2));
    }

    #[test]
    fn selection_is_seeded_and_skips_blocks() {
        let mut l = Ledger::default();
        let a = l.add("Держит ли стенд A дрейф < 1 px за 72 ч?", None);
        let b = l.add("Держит ли стенд B дрейф < 1 px за 72 ч?", Some(0.2));
        let c = l.add("Держит ли стенд C дрейф < 1 px за 72 ч?", Some(0.8));
        // C и B получают историю: у B был сдвиг → surprise выше
        l.apply(
            &b,
            Outcome::Support,
            "p",
            vec!["PMID:1234567".into()],
            "n",
            0,
        );
        let s1 = l.select_next(42, 3).unwrap();
        let s2 = l.select_next(42, 3).unwrap();
        assert_eq!(s1.id, s2.id, "seed должен давать один и тот же вопрос");
        // разные seed могут дать разные вопросы, но каждый — из пула
        let picks: Vec<String> = (0..20).map(|s| l.select_next(s, 3).unwrap().id).collect();
        assert!(picks.iter().all(|p| [&a, &b, &c].contains(&p)));
        // Blocked выпадает из очереди
        l.apply(
            &a,
            Outcome::Fabricated,
            "p",
            vec!["doi:10.0/x".into()],
            "bad",
            0,
        );
        let after: Vec<String> = (0..20).map(|s| l.select_next(s, 3).unwrap().id).collect();
        assert!(!after.contains(&a), "Blocked-вопрос не должен выбираться");
    }

    #[test]
    fn closed_question_leaves_queue() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли закрываемый стенд дрейф < 1 px?", None);
        // разные идентификаторы: одна и та же статья не может опровергать вопрос 8 раз
        for i in 0..8 {
            l.apply(
                &id,
                Outcome::Contradict,
                "p",
                vec![format!("PMID:765432{i}")],
                "no",
                0,
            );
        }
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Refuted, "p={}", q.p_true);
        assert!(l.select_next(1, 3).is_none());
        assert!(l.metrics().refutation_share > 0.0);
    }

    #[test]
    fn default_questions_are_real_and_unique() {
        let qs = argus_default_questions();
        assert_eq!(qs.len(), 6);
        let ids: std::collections::HashSet<String> = qs.iter().map(|(t, _)| qid(t)).collect();
        assert_eq!(ids.len(), 6);
        assert_eq!(
            qs.iter().filter(|(_, k)| *k == QKind::Lit).count(),
            3,
            "должно быть 3 литературных вопроса"
        );
    }

    #[test]
    fn failing_question_goes_to_gap_and_frees_queue() {
        let mut l = Ledger::default();
        let bad = l.add("плохо ищущийся вопрос", None);
        let good = l.add("нормальный вопрос", None);
        for _ in 0..MAX_CONSECUTIVE_GAPS {
            l.apply(&bad, Outcome::None, "probe:epmc", vec![], "0 попаданий", 0);
        }
        assert_eq!(l.get(&bad).unwrap().status, QStatus::Gap);
        // очередь обязана переключиться на другой вопрос
        for seed in 0..10 {
            assert_eq!(l.select_next(seed, 3).unwrap().id, good, "seed={seed}");
        }
    }

    #[test]
    fn prediction_error_drives_attention() {
        let mut l = Ledger::default();
        let revised = l.add("Держит ли стенд со свидетельством дрейф < 1 px?", None);
        let empty = l.add("Держит ли стенд без свидетельства дрейф < 1 px?", None);
        // свидетельство → ошибка предсказания высокая
        l.apply(
            &revised,
            Outcome::Support,
            "probe",
            vec!["PMID:23663408".into()],
            "ok",
            0,
        );
        // пустая проба → ошибка низкая (это свойство канала, а не мира)
        l.apply(&empty, Outcome::None, "probe", vec![], "0 попаданий", 0);
        assert!(l.get(&revised).unwrap().last_error > l.get(&empty).unwrap().last_error);
        // порядок в очереди: вопрос с большей ошибкой предсказания идёт первым
        let ranked = l.ranked(Utc::now());
        assert_eq!(
            ranked[0].id,
            revised,
            "score={:?}",
            ranked
                .iter()
                .map(|r| (r.id.clone(), r.score))
                .collect::<Vec<_>>()
        );
        // и выигрывает рулетку в большинстве случаев (не всегда: рулетка взвешена, а не argmax)
        let picks = (0..20)
            .filter(|s| l.select_next(*s, 2).unwrap().id == revised)
            .count();
        assert!(
            picks >= 12,
            "выбран {picks}/20 раз — вес ошибки слишком слаб"
        );
    }

    #[test]
    fn unverified_probe_does_not_raise_error() {
        let mut l = Ledger::default();
        let id = l.add("вопрос", None);
        l.apply(
            &id,
            Outcome::None,
            "probe",
            vec![],
            "проба не удалась: offline",
            0,
        );
        assert!(l.get(&id).unwrap().last_error <= 0.1);
    }

    #[test]
    fn stale_gaps_are_reopened() {
        let mut l = Ledger::default();
        let id = l.add("вопрос", None);
        for _ in 0..MAX_CONSECUTIVE_GAPS {
            l.apply(&id, Outcome::None, "p", vec![], "0", 0);
        }
        assert_eq!(l.get(&id).unwrap().status, QStatus::Gap);
        // сразу — не переоткрываем
        assert_eq!(l.reopen_stale_gaps(Utc::now(), 24), 0);
        // через 25 часов — переоткрываем
        assert_eq!(
            l.reopen_stale_gaps(Utc::now() + chrono::Duration::hours(25), 24),
            1
        );
        assert_eq!(l.get(&id).unwrap().status, QStatus::Open);
        assert_eq!(l.get(&id).unwrap().consecutive_gaps, 0);
    }

    // ── фикс 13 ноя: форма вопроса, повтор в очереди, исчерпание проб ──────

    #[test]
    fn topic_question_never_moves_p_even_with_identifiers() {
        let mut l = Ledger::default();
        let id = l.add(
            "fiducial autofocus drift long-term time-lapse open microscope",
            None,
        );
        let c = l
            .apply(
                &id,
                Outcome::Support,
                "probe:epmc",
                vec!["PMID:23663408".into()],
                "нашли",
                0,
            )
            .unwrap();
        assert_eq!(c.before, c.after, "у темы нет истинностного значения");
        assert_eq!(c.form.as_deref(), Some("topic"));
        let q = l.get(&id).unwrap();
        assert_eq!(q.p_true, 0.5);
        assert_eq!(
            q.identifiers,
            vec!["PMID:23663408".to_string()],
            "ID всё равно фиксируем"
        );
        assert!(
            q.gaps.iter().any(|g| g.contains("form=topic")),
            "{:?}",
            q.gaps
        );
    }

    #[test]
    fn quantitative_question_never_moves_p() {
        let mut l = Ledger::default();
        let id = l.add("Насколько дрейфует фокус за 8 ч без автофокуса?", None);
        let c = l
            .apply(
                &id,
                Outcome::Contradict,
                "probe:epmc",
                vec!["PMID:23663408".into()],
                "",
                0,
            )
            .unwrap();
        assert_eq!(c.before, c.after);
        assert_eq!(c.form.as_deref(), Some("quantitative"));
    }

    #[test]
    fn proposition_moves_p_and_records_stance() {
        use crate::stance::{Decision, Stance};
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px за 72 ч?", None);
        let d = Decision {
            stance: Stance::Support,
            confidence: 0.85,
            source: "lexical:title+abstract",
            reason: "we demonstrated … 72 h".into(),
        };
        let c = l
            .apply_judged(
                &id,
                Outcome::Support,
                "probe:epmc",
                vec!["PMID:23663408".into()],
                "суждено",
                0,
                Some(&d),
            )
            .unwrap();
        assert!(c.after > c.before);
        assert_eq!(c.stance.as_deref(), Some("support"));
        assert!((c.stance_conf - 0.85).abs() < 1e-6);
        assert_eq!(c.stance_source.as_deref(), Some("lexical:title+abstract"));
        assert_eq!(c.form.as_deref(), Some("proposition"));
    }

    #[test]
    fn same_identifier_does_not_move_p_twice() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px за 72 ч?", None);
        let ids = vec!["PMID:23663408".to_string()];
        let c1 = l
            .apply(
                &id,
                Outcome::Support,
                "probe:epmc",
                ids.clone(),
                "первый раз",
                0,
            )
            .unwrap();
        let p1 = c1.after;
        assert!(p1 > c1.before);
        let c2 = l
            .apply(&id, Outcome::Support, "probe:epmc", ids, "тот же PMID", 0)
            .unwrap();
        assert_eq!(c2.before, c2.after, "одна статья — одно свидетельство");
        let q = l.get(&id).unwrap();
        assert!(
            q.gaps.iter().any(|g| g.contains("повтор")),
            "причина отказа от LR должна быть записана: {:?}",
            q.gaps
        );
    }

    #[test]
    fn superseded_question_never_returns_to_queue() {
        let mut l = Ledger::default();
        let old = l.add("fiducial autofocus drift long-term time-lapse open microscope", None);
        let new = l.add("Показано ли, что fiducial-автофокус удерживает дрейф < 1 px за 24 ч?", None);
        l.supersede(&old, &new).unwrap();
        assert_eq!(l.get(&old).unwrap().status, QStatus::Gap);
        assert_eq!(l.get(&old).unwrap().superseded_by.as_deref(), Some(new.as_str()));
        // даже через 100 часов заменённый вопрос не воскресает
        assert_eq!(l.reopen_stale_gaps(Utc::now() + chrono::Duration::hours(100), 6), 0);
        let picks: Vec<String> = (0..10).map(|s| l.select_next(s, 5).unwrap().id).collect();
        assert!(picks.iter().all(|p| *p == new), "выбран заменённый: {picks:?}");
    }

    #[test]
    fn human_close_sets_status_without_touching_p() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px?", None);
        let c = l
            .close_by_human(&id, "supported", "протокол §4b.2: n ≈ 115 на группу")
            .unwrap();
        let q = l.get(&id).unwrap();
        assert_eq!(q.status, QStatus::Supported);
        assert_eq!(q.p_true, 0.5, "закрытие человеком не выдумывает уверенность");
        assert_eq!(c.probe, "human:close");
        assert!(q.gaps.iter().any(|g| g.contains("n ≈ 115")), "{:?}", q.gaps);
        // Gap-закрытие убирает вопрос из очереди
        let id2 = l.add("Держит ли стенд 2 дрейф < 1 px?", None);
        l.close_by_human(&id2, "gap", "не претензия");
        assert_eq!(l.get(&id2).unwrap().status, QStatus::Gap);
        assert!(l.select_next(1, 5).is_none(), "очередь должна быть пуста");
    }

    // ── консилиум бесплатных моделей (13 ноя) ──────────────────────────────

    #[test]
    fn council_shifts_weakly_and_never_closes() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px за 72 ч?", None);
        let c = l
            .apply(&id, Outcome::CouncilSupport, "probe:dialogue", vec![], "3/3 да", 0)
            .unwrap();
        let q = l.get(&id).unwrap();
        assert!(q.p_true > 0.5, "консилиум должен слабо поднять p: {}", q.p_true);
        assert!(q.p_true < 0.70, "консилиум не должен быть убедительным: {}", q.p_true);
        assert_eq!(q.status, QStatus::Open, "консилиум не закрывает вопрос");
        assert_eq!(c.verifier, VerifyResult::Council);
        // даже многократно (разные сутки) не пробивает порог закрытия
        for _ in 0..5 {
            // имитируем «прошли сутки»: ограничение по 24 ч снимается, потолок остаётся
            if let Some(q) = l.questions.iter_mut().find(|q| q.id == id) {
                q.last_council_at = None;
            }
            l.apply(&id, Outcome::CouncilSupport, "probe:dialogue", vec![], "", 0);
        }
        let q = l.get(&id).unwrap();
        assert!(q.p_true < P_CLOSE_SUPPORT, "потолок консилиума: p={}", q.p_true);
        assert!(
            !matches!(q.status, QStatus::Supported | QStatus::Refuted),
            "консилиум не имеет права закрыть вопрос: {:?}",
            q.status
        );
    }

    #[test]
    fn council_cannot_shift_twice_within_a_day() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px за 72 ч?", None);
        l.apply(&id, Outcome::CouncilSupport, "probe:dialogue", vec![], "да", 0);
        let after_first = l.get(&id).unwrap().p_true;
        let c = l
            .apply(&id, Outcome::CouncilSupport, "probe:dialogue", vec![], "да снова", 0)
            .unwrap();
        assert_eq!(c.before, c.after, "второй сдвиг за сутки запрещён");
        assert_eq!(l.get(&id).unwrap().p_true, after_first);
        assert!(l
            .get(&id)
            .unwrap()
            .gaps
            .iter()
            .any(|g| g.contains("за последние 24 ч")));
    }

    #[test]
    fn council_is_ignored_for_non_propositions() {
        let mut l = Ledger::default();
        let id = l.add("fiducial autofocus drift long-term imaging", None);
        let c = l
            .apply(&id, Outcome::CouncilSupport, "probe:dialogue", vec![], "да", 0)
            .unwrap();
        assert_eq!(c.before, c.after);
        assert_eq!(l.get(&id).unwrap().p_true, 0.5);
    }

    #[test]
    fn council_cycles_are_counted_separately_in_metrics() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px?", None);
        l.apply(&id, Outcome::CouncilContradict, "probe:dialogue", vec![], "нет", 0);
        let m = l.metrics();
        assert_eq!(m.council_cycles, 1);
        assert_eq!(m.gap_cycles, 0, "консилиум не должен считаться пробелом");
        assert_eq!(m.id_cycles, 0);
    }

    #[test]
    fn exhausted_open_question_is_reaped_on_load() {
        let dir = std::env::temp_dir().join(format!("argus-reap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("ledger")).unwrap();
        let mut q = Question::new("Держит ли стенд дрейф < 1 px?");
        q.probes = MAX_PROBES_PER_QUESTION;
        q.status = QStatus::Open;
        std::fs::write(
            dir.join("ledger/questions.json"),
            serde_json::to_string_pretty(&vec![q]).unwrap(),
        )
        .unwrap();
        let l = Ledger::load(&dir);
        assert_eq!(l.questions[0].status, QStatus::Gap);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_evidence_three_times_moves_question_to_gap() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли стенд дрейф < 1 px?", None);
        // первый цикл — прогресс (ID новый), дальше MAX_CONSECUTIVE_GAPS прогонов без новизны
        for _ in 0..MAX_CONSECUTIVE_GAPS + 1 {
            l.apply(
                &id,
                Outcome::Evidence,
                "probe:epmc",
                vec!["PMID:23663408".into()],
                "то же самое",
                0,
            );
        }
        let q = l.get(&id).unwrap();
        assert_eq!(
            q.status,
            QStatus::Gap,
            "повтор без новых ID должен освобождать очередь"
        );
        assert!(q
            .gaps
            .iter()
            .any(|g| g.contains("новых идентификаторов нет")));
    }

    #[test]
    fn repeated_question_yields_to_unprobed_ones() {
        let mut l = Ledger::default();
        let hot = l.add("Держит ли горячий стенд дрейф < 1 px за 72 ч?", None);
        let cold: Vec<String> = (0..4)
            .map(|i| l.add(&format!("Держит ли холодный стенд {i} дрейф < 1 px?"), None))
            .collect();
        for _ in 0..4 {
            l.apply(&hot, Outcome::None, "probe:epmc", vec![], "0 попаданий", 0);
        }
        // K=5: холодных четверо — все должны попадать в пул, иначе тест проверяет не то
        let picks: Vec<String> = (0..20).map(|s| l.select_next(s, 5).unwrap().id).collect();
        let hot_picks = picks.iter().filter(|p| **p == hot).count();
        assert!(hot_picks <= 5, "повтор доминирует: {hot_picks} /20");
        for c in &cold {
            assert!(picks.contains(c), "вопрос {c} ни разу не выбран: {picks:?}");
        }
    }

    #[test]
    fn exhausted_probe_budget_never_hides_open_question() {
        let mut l = Ledger::default();
        let id = l.add("Держит ли исчерпывающийся стенд дрейф < 1 px?", None);
        for _ in 0..MAX_PROBES_PER_QUESTION {
            l.apply(&id, Outcome::None, "probe:epmc", vec![], "0 попаданий", 0);
        }
        let q = l.get(&id).unwrap();
        assert_eq!(q.probes, MAX_PROBES_PER_QUESTION);
        if q.status == QStatus::Open {
            // если остался open — обязан быть выбираемым, иначе он невидим в очереди
            assert!(l.select_next(1, 3).is_some());
        } else {
            assert_eq!(q.status, QStatus::Gap);
            assert!(q.gaps.iter().any(|g| g.contains("исчерпан бюджет")));
        }
    }

    #[test]
    fn literature_channel_skips_lab_questions() {
        let mut l = Ledger::default();
        let lit = l.add_kind("литературный вопрос", None, QKind::Lit);
        l.add_kind("лабораторный вопрос", None, QKind::Lab);
        for seed in 0..5 {
            assert_eq!(
                l.select_next_for(seed, 3, true).unwrap().id,
                lit,
                "seed={seed}"
            );
        }
        // без ограничения канала доступны оба
        let ids: std::collections::HashSet<String> = (0..20)
            .map(|s| l.select_next_for(s, 3, false).unwrap().id)
            .collect();
        assert_eq!(ids.len(), 2);
    }
}
