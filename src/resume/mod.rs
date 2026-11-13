//! P9/M4 — выживание фонового цикла: сон и выключение.
//!
//! Две вещи, которых не хватало:
//!   1. **Сон.** Пока цикл жив, сон системы должен блокироваться (`systemd-inhibit`), иначе
//!      «постоянный процесс» останавливается, как только машина решит уснуть.
//!   2. **Выключение.** Состояние леджера на диске, значит после включения цикл продолжается
//!      не «с нуля», а **с моментом выключения**: считаем пропущенные тики и догоняем их
//!      (в пределах бюджета).
//!
//! Здесь — только арифметика догоняния и персист «последнего тика»; блокировку сна делает
//! systemd-юнит (`systemd-inhibit --what=idle:sleep`).

use chrono::{DateTime, Local, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Больше 10 догоняющих циклов за раз не делаем: после суток выключения не надо выжигать бюджет.
pub const MAX_CATCHUP_CYCLES: u32 = 10;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Runtime {
    /// когда цикл последний раз «тикал» (RFC3339)
    #[serde(default)]
    pub last_tick: Option<String>,
    /// сколько секунд система простояла выключенной/спящей (суммарно)
    #[serde(default)]
    pub downtime_secs: u64,
    /// сколько догоняющих циклов сделано всего
    #[serde(default)]
    pub catchup_cycles: u64,
    /// сколько раз цикл поднимался после простоя
    #[serde(default)]
    pub resumes: u64,
    #[serde(skip)]
    path: PathBuf,
}

impl Runtime {
    pub fn load(state_dir: &Path) -> Runtime {
        let path = state_dir.join("ledger").join("runtime.json");
        if let Some(p) = path.parent() {
            let _ = fs::create_dir_all(p);
        }
        let mut r: Runtime = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        r.path = path;
        r
    }

    pub fn save(&self) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(&self.path, s)
    }

    pub fn tick(&mut self) {
        self.last_tick = Some(Local::now().to_rfc3339());
    }

    /// Сколько секунд прошло с последнего тика (None — если тика ещё не было).
    pub fn seconds_since_tick(&self) -> Option<i64> {
        let last = self.last_tick.as_deref()?;
        let t = DateTime::parse_from_rfc3339(last).ok()?.with_timezone(&Utc);
        Some((Utc::now() - t).num_seconds())
    }
}

/// План догоняния: сколько циклов запустить сразу после простоя.
///
/// * `last_tick` — когда цикл тикал в прошлый раз (None — первый запуск, догонять нечего);
/// * `interval_secs` — штатный интервал между циклами;
/// * `budget_left` — сколько циклов осталось по бюджету в этом часе;
/// * `max_catchup` — потолок (по умолчанию `MAX_CATCHUP_CYCLES`).
///
/// Возвращает `(сколько_догнать, пропущено_тиков, простой_сек)`. Пропущенные тики считаем
/// как `простой / интервал`, но догоняем не больше бюджета и не больше потолка.
pub fn plan_catchup(
    last_tick: Option<&str>,
    interval_secs: u64,
    budget_left: u32,
    max_catchup: u32,
) -> (u32, u64, u64) {
    let Some(last) = last_tick else {
        return (0, 0, 0);
    };
    let Ok(t) = DateTime::parse_from_rfc3339(last) else {
        return (0, 0, 0);
    };
    let interval = interval_secs.max(1) as i64;
    let gap = (Utc::now() - t.with_timezone(&Utc)).num_seconds().max(0);
    // меньше 1.5 интервала — это не простой, а нормальная работа
    if gap < interval * 3 / 2 {
        return (0, 0, gap as u64);
    }
    let missed = (gap / interval) as u64;
    let want = missed.min(budget_left as u64).min(max_catchup as u64) as u32;
    (want, missed, gap as u64)
}

/// Человекочитаемая длительность.
pub fn human(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    if h >= 24 {
        format!("{} сут {} ч", h / 24, h % 24)
    } else if h > 0 {
        format!("{h} ч {m} мин")
    } else {
        format!("{m} мин")
    }
}

/// Подсказка для отчёта: была ли машина выключена/спала и что делаем.
pub fn resume_note(plan: (u32, u64, u64)) -> Option<String> {
    let (do_now, missed, gap) = plan;
    if missed == 0 {
        return None;
    }
    Some(format!(
        "простой {}: пропущено {} тиков, догоняю {} (потолок {} / бюджет)",
        human(gap),
        missed,
        do_now,
        MAX_CATCHUP_CYCLES
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn ago(secs: i64) -> String {
        (Utc::now() - Duration::seconds(secs)).to_rfc3339()
    }

    #[test]
    fn first_run_has_nothing_to_catch_up() {
        assert_eq!(plan_catchup(None, 60, 20, 10), (0, 0, 0));
    }

    #[test]
    fn short_gap_is_not_downtime() {
        let (do_now, missed, _) = plan_catchup(Some(&ago(30)), 60, 20, 10);
        assert_eq!(
            (do_now, missed),
            (0, 0),
            "30 с при интервале 60 — норма, не простой"
        );
    }

    #[test]
    fn hours_of_downtime_are_caught_up_within_budget() {
        // 6 часов простоя при интервале 60 с = 360 пропущенных тиков
        let (do_now, missed, gap) = plan_catchup(Some(&ago(6 * 3600)), 60, 7, 10);
        assert_eq!(missed, 360);
        assert_eq!(do_now, 7, "упираемся в бюджет часа");
        assert_eq!(gap, 6 * 3600);
    }

    #[test]
    fn catchup_is_capped() {
        let (do_now, missed, _) = plan_catchup(Some(&ago(48 * 3600)), 60, 20, 10);
        assert!(missed > 1000);
        assert_eq!(do_now, 10, "потолок MAX_CATCHUP");
    }

    #[test]
    fn runtime_roundtrip() {
        let dir = std::env::temp_dir().join(format!("argus-runtime-{}", std::process::id()));
        let mut r = Runtime::load(&dir);
        r.tick();
        r.downtime_secs = 123;
        r.resumes = 2;
        r.save().unwrap();
        let r2 = Runtime::load(&dir);
        assert_eq!(r2.downtime_secs, 123);
        assert_eq!(r2.resumes, 2);
        assert!(r2.seconds_since_tick().unwrap() < 5);
    }

    #[test]
    fn human_readable() {
        assert_eq!(human(90), "1 мин");
        assert_eq!(human(3700), "1 ч 1 мин");
        assert_eq!(human(2 * 86400 + 3600), "2 сут 1 ч");
    }
}
