//! P9/M3 — бюджет idle-цикла (спека §4): 20 циклов/час и 200k токенов/сутки.
//!
//! Почему отдельный модуль: без него «постоянный внутренний процесс» превращается в
//! фоновый генератор счёта. Гейт проверяется **до** цикла, запись — **после**.

use chrono::{DateTime, Duration, Local, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_MAX_CYCLES_PER_HOUR: u32 = 20;
pub const DEFAULT_MAX_TOKENS_PER_DAY: u64 = 200_000;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Budget {
    pub max_cycles_per_hour: u32,
    pub max_tokens_per_day: u64,
    /// день в формате YYYY-MM-DD (локальная дата) — при смене суток счётчики обнуляются
    pub day: String,
    pub cycles_today: u32,
    pub tokens_today: u64,
    /// метки времени циклов внутри скользящего часа
    #[serde(default)]
    pub recent: Vec<String>,
    #[serde(skip)]
    path: PathBuf,
}

fn today() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            max_cycles_per_hour: DEFAULT_MAX_CYCLES_PER_HOUR,
            max_tokens_per_day: DEFAULT_MAX_TOKENS_PER_DAY,
            day: today(),
            cycles_today: 0,
            tokens_today: 0,
            recent: Vec::new(),
            path: PathBuf::new(),
        }
    }
}

impl Budget {
    pub fn load(state_dir: &Path) -> Self {
        let path = state_dir.join("ledger").join("budget.json");
        fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).ok();
        let mut b: Budget = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        b.path = path;
        b.roll_over(Utc::now());
        b
    }

    pub fn save(&self) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&self.path, s)
    }

    /// Смена суток обнуляет счётчики; окно часа чистится всегда.
    pub fn roll_over(&mut self, now: DateTime<Utc>) {
        let t = today();
        if self.day != t {
            self.day = t;
            self.cycles_today = 0;
            self.tokens_today = 0;
        }
        let cutoff = now - Duration::hours(1);
        self.recent.retain(|ts| {
            DateTime::parse_from_rfc3339(ts)
                .map(|d| d.with_timezone(&Utc) > cutoff)
                .unwrap_or(false)
        });
    }

    /// Гейт перед циклом. `Err` — причина отказа (цикл не должен выполняться).
    pub fn gate(&mut self, now: DateTime<Utc>) -> Result<(), String> {
        self.roll_over(now);
        if self.recent.len() as u32 >= self.max_cycles_per_hour {
            return Err(format!(
                "бюджет: {} циклов за последний час исчерпан",
                self.max_cycles_per_hour
            ));
        }
        if self.tokens_today >= self.max_tokens_per_day {
            return Err(format!(
                "бюджет: {}/{} токенов за сутки исчерпано",
                self.tokens_today, self.max_tokens_per_day
            ));
        }
        Ok(())
    }

    pub fn record(&mut self, now: DateTime<Utc>, tokens: u64) {
        self.roll_over(now);
        self.recent.push(now.to_rfc3339());
        self.cycles_today += 1;
        self.tokens_today += tokens;
    }

    pub fn hour_remaining(&mut self, now: DateTime<Utc>) -> u32 {
        self.roll_over(now);
        self.max_cycles_per_hour
            .saturating_sub(self.recent.len() as u32)
    }

    pub fn tokens_remaining(&mut self) -> u64 {
        self.max_tokens_per_day.saturating_sub(self.tokens_today)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_budget(name: &str) -> Budget {
        let dir = std::env::temp_dir().join(format!("argus-budget-{name}-{}", std::process::id()));
        let mut b = Budget::load(&dir);
        b.max_cycles_per_hour = 2;
        b.max_tokens_per_day = 100;
        b
    }

    #[test]
    fn hourly_gate_blocks_after_limit() {
        let mut b = tmp_budget("hour");
        let now = Utc::now();
        assert!(b.gate(now).is_ok());
        b.record(now, 10);
        b.record(now, 10);
        assert!(
            b.gate(now).is_err(),
            "третий цикл в час должен быть заблокирован"
        );
        assert_eq!(b.hour_remaining(now), 0);
    }

    #[test]
    fn daily_gate_blocks_on_tokens() {
        let mut b = tmp_budget("tokens");
        let now = Utc::now();
        b.record(now, 99);
        b.record(now, 1);
        assert!(b.gate(now).is_err());
        assert_eq!(b.tokens_remaining(), 0);
    }

    #[test]
    fn window_older_than_hour_is_forgotten() {
        let mut b = tmp_budget("window");
        let now = Utc::now();
        b.recent.push((now - Duration::minutes(90)).to_rfc3339());
        b.recent.push((now - Duration::minutes(30)).to_rfc3339());
        b.roll_over(now);
        assert_eq!(
            b.recent.len(),
            1,
            "старая метка должна выпасть из окна часа"
        );
    }

    #[test]
    fn new_day_resets_counters() {
        let mut b = tmp_budget("day");
        b.day = "2000-01-01".to_string();
        b.cycles_today = 5;
        b.tokens_today = 50;
        b.roll_over(Utc::now());
        assert_eq!(b.cycles_today, 0);
        assert_eq!(b.tokens_today, 0);
        assert_eq!(b.day, today());
    }

    #[test]
    fn state_survives_reload() {
        let dir = std::env::temp_dir().join(format!("argus-budget-persist-{}", std::process::id()));
        let mut b = Budget::load(&dir);
        b.record(Utc::now(), 42);
        b.save().unwrap();
        let b2 = Budget::load(&dir);
        assert_eq!(b2.tokens_today, 42);
        assert_eq!(b2.recent.len(), 1);
    }
}
