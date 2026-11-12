//! Лабораторный канал (M4): ответ на `lab`-вопрос даёт эксперимент, а не литература.
//!
//! Вопросы вида «держит ли fiducial-автофокус дрейф < 1 px за 72 ч на нашем стенде»
//! литературой не отвечаются в принципе (§8.4 спеки). До сих пор такие вопросы висели
//! в `open` и не могли закрыться ничем. Здесь появляется носитель ответа:
//!
//! `$ARGUS_STATE_DIR/lab/evidence/<qid>.json`
//! ```json
//! {
//!   "qid": "q-8c6d5868",
//!   "verdict": "support",            // support | contradict | inconclusive
//!   "value": "0.6 px за 72 ч",
//!   "source": "lab/drift/2027-01-10_72h.csv",
//!   "note": "три прогона, дрейф 0.4–0.8 px",
//!   "at": "2027-01-10"
//! }
//! ```
//!
//! Правила (чтобы файл в каталоге не стал способом «подкрутить» леджер):
//! 1) `source` обязателен и должен существовать как файл относительно state-dir или
//!    абсолютным путём — цифра без артефакта не принимается;
//! 2) `verdict = inconclusive` → вероятность не двигается (честный «данные есть, ответа нет»);
//! 3) свежесть: по умолчанию данные старше 400 дней не применяются;
//! 4) по решению Джабы (13.11) перед применением лабораторных данных вопрос уходит в
//!    консилиум бесплатных моделей (`crate::dialogue`): расхождение консилиума блокирует
//!    применение, а его сомнения записываются в журнал.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Сколько дней лабораторные данные считаются актуальными по умолчанию.
pub const DEFAULT_MAX_AGE_DAYS: i64 = 400;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LabVerdict {
    Support,
    Contradict,
    Inconclusive,
}

impl LabVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            LabVerdict::Support => "support",
            LabVerdict::Contradict => "contradict",
            LabVerdict::Inconclusive => "inconclusive",
        }
    }
    pub fn parse(s: &str) -> Option<LabVerdict> {
        match s.trim().to_lowercase().as_str() {
            "support" | "да" | "yes" | "pass" => Some(LabVerdict::Support),
            "contradict" | "нет" | "no" | "fail" => Some(LabVerdict::Contradict),
            "inconclusive" | "неопределённо" | "unknown" => Some(LabVerdict::Inconclusive),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabEvidence {
    pub qid: String,
    pub verdict: LabVerdict,
    #[serde(default)]
    pub value: String,
    /// путь к артефакту (CSV/лог/картинка) — обязателен
    pub source: String,
    #[serde(default)]
    pub note: String,
    /// RFC3339 либо YYYY-MM-DD
    #[serde(default)]
    pub at: Option<String>,
}

impl LabEvidence {
    /// Возраст записи в днях (None — дата не указана).
    pub fn age_days(&self, now: chrono::DateTime<chrono::Utc>) -> Option<i64> {
        let raw = self.at.as_deref()?;
        let ts = chrono::DateTime::parse_from_rfc3339(raw)
            .map(|d| d.with_timezone(&chrono::Utc))
            .ok()
            .or_else(|| {
                chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                    .ok()
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
                    .map(|dt| chrono::DateTime::from_naive_utc_and_offset(dt, chrono::Utc))
            })?;
        Some((now - ts).num_days())
    }
}

pub fn evidence_dir(state_dir: &Path) -> PathBuf {
    state_dir.join("lab").join("evidence")
}

pub fn path_for(state_dir: &Path, qid: &str) -> PathBuf {
    evidence_dir(state_dir).join(format!("{qid}.json"))
}

pub fn load(state_dir: &Path, qid: &str) -> Option<LabEvidence> {
    std::fs::read_to_string(path_for(state_dir, qid))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

pub fn record(state_dir: &Path, ev: &LabEvidence) -> std::io::Result<PathBuf> {
    let p = path_for(state_dir, &ev.qid);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let s = serde_json::to_string_pretty(ev)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&p, s)?;
    Ok(p)
}

/// Можно ли применять запись: артефакт существует, вердикт определён, данные не устарели.
/// Числовой разбор: порог из текста вопроса и число из значения.
///
/// Зачем: консилиум из малых моделей путается в сравнении чисел (13 ноя: qwen сказала
/// «0.58 > 1 px» → `contradict`). Детерминированная проверка ловит такие расхождения
/// раньше, чем они попадут в вероятность: цифра обязана соответствовать вердикту.
pub fn numeric_consistency(question: &str, value: &str, verdict: &LabVerdict) -> Result<(), String> {
    let qnum = first_number(question);
    let vnum = first_number(value);
    let (Some(threshold), Some(measured)) = (qnum, vnum) else {
        return Ok(()); // сравнивать нечего — проверка не применяется
    };
    if matches!(verdict, LabVerdict::Inconclusive) {
        return Ok(());
    }
    // направление порога: «< 1», «ниже 1», «≤ 1» — меньше значит «да»
    let low = question.to_lowercase();
    let less = low.contains('<') || low.contains('≤') || low.contains("ниже") || low.contains("меньше");
    let satisfied = if less {
        measured <= threshold
    } else {
        measured >= threshold
    };
    match verdict {
        LabVerdict::Support if !satisfied => Err(format!(
            "значение {measured} не удовлетворяет порогу {threshold} из вопроса — вердикт support не подтверждён цифрой"
        )),
        LabVerdict::Contradict if satisfied => Err(format!(
            "значение {measured} удовлетворяет порогу {threshold}, а вердикт contradict — цифра противоречит вердикту"
        )),
        _ => Ok(()),
    }
}

fn first_number(s: &str) -> Option<f64> {
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || (ch == '.' && !cur.is_empty()) {
            cur.push(ch);
        } else if !cur.is_empty() {
            return cur.trim_matches('.').parse().ok();
        }
    }
    if cur.is_empty() {
        None
    } else {
        cur.trim_matches('.').parse().ok()
    }
}

pub fn validate(
    state_dir: &Path,
    ev: &LabEvidence,
    now: chrono::DateTime<chrono::Utc>,
    max_age_days: i64,
    question: &str,
) -> Result<(), String> {
    if ev.verdict == LabVerdict::Inconclusive {
        return Err("вердикт inconclusive: данные есть, ответа нет — вероятность не двигается".into());
    }
    if ev.source.trim().is_empty() {
        return Err("не указан артефакт (source) — цифра без источника не принимается".into());
    }
    let cand = PathBuf::from(&ev.source);
    let resolved = if cand.is_absolute() {
        cand
    } else {
        state_dir.join(&cand)
    };
    if !resolved.exists() {
        return Err(format!("артефакт не найден: {}", resolved.display()));
    }
    numeric_consistency(question, &ev.value, &ev.verdict)?;
    if let Some(age) = ev.age_days(now) {
        if age > max_age_days {
            return Err(format!(
                "данные устарели: {age} дней > {max_age_days} — нужен повторный прогон"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "argus-lab-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn ev(verdict: LabVerdict, source: &str, at: Option<&str>) -> LabEvidence {
        LabEvidence {
            qid: "q-8c6d5868".into(),
            verdict,
            value: "0.6 px".into(),
            source: source.into(),
            note: "три прогона".into(),
            at: at.map(|s| s.to_string()),
        }
    }

    #[test]
    fn record_and_load_roundtrip() {
        let dir = tmp();
        let e = ev(LabVerdict::Support, "lab/drift.csv", Some("2027-01-10"));
        record(&dir, &e).unwrap();
        let back = load(&dir, "q-8c6d5868").unwrap();
        assert_eq!(back.verdict, LabVerdict::Support);
        assert_eq!(back.value, "0.6 px");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_artifact_is_rejected() {
        let dir = tmp();
        let e = ev(LabVerdict::Support, "lab/нет-такого.csv", Some("2027-01-10"));
        let err = validate(&dir, &e, Utc::now(), DEFAULT_MAX_AGE_DAYS, "").unwrap_err();
        assert!(err.contains("артефакт не найден"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_artifact_passes() {
        let dir = tmp();
        std::fs::create_dir_all(dir.join("lab")).unwrap();
        std::fs::write(dir.join("lab/drift.csv"), "t,drift_px\n0,0.1\n").unwrap();
        let e = ev(LabVerdict::Support, "lab/drift.csv", Some("2026-11-13"));
        assert!(validate(&dir, &e, Utc::now(), DEFAULT_MAX_AGE_DAYS, "").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inconclusive_never_moves_p() {
        let dir = tmp();
        std::fs::write(dir.join("x.csv"), "a").unwrap();
        let e = ev(LabVerdict::Inconclusive, "x.csv", Some("2026-11-13"));
        let err = validate(&dir, &e, Utc::now(), DEFAULT_MAX_AGE_DAYS, "").unwrap_err();
        assert!(err.contains("inconclusive"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_data_is_rejected() {
        let dir = tmp();
        std::fs::write(dir.join("x.csv"), "a").unwrap();
        let e = ev(LabVerdict::Support, "x.csv", Some("2024-01-01"));
        let err = validate(&dir, &e, Utc::now(), 400, "").unwrap_err();
        assert!(err.contains("устарели"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_source_is_rejected() {
        let dir = tmp();
        let e = ev(LabVerdict::Contradict, "", None);
        let err = validate(&dir, &e, Utc::now(), DEFAULT_MAX_AGE_DAYS, "").unwrap_err();
        assert!(err.contains("source"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn numeric_gate_catches_sign_error() {
        let q = "Держит ли fiducial-autofocus дрейф < 1 px за 72 ч?";
        // 0.58 < 1 → support согласован
        assert!(numeric_consistency(q, "0.58 px за 72 ч", &LabVerdict::Support).is_ok());
        // 1.7 > 1 → support НЕ согласован
        assert!(numeric_consistency(q, "1.7 px за 72 ч", &LabVerdict::Support).is_err());
        // 1.7 > 1 → contradict согласован
        assert!(numeric_consistency(q, "1.7 px", &LabVerdict::Contradict).is_ok());
        // 0.58 < 1 → contradict противоречит цифре
        assert!(numeric_consistency(q, "0.58 px", &LabVerdict::Contradict).is_err());
    }

    #[test]
    fn numeric_gate_is_skipped_without_numbers() {
        let q = "Нужен ли Amendment 3 к протоколу?";
        assert!(numeric_consistency(q, "утверждён", &LabVerdict::Support).is_ok());
    }

    #[test]
    fn verdict_parsing_from_cli_values() {
        assert_eq!(LabVerdict::parse("да"), Some(LabVerdict::Support));
        assert_eq!(LabVerdict::parse("FAIL"), Some(LabVerdict::Contradict));
        assert_eq!(LabVerdict::parse("неопределённо"), Some(LabVerdict::Inconclusive));
        assert_eq!(LabVerdict::parse("?!"), None);
    }
}
