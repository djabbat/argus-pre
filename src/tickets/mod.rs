//! ARGUS-OS1 V2 — P8a: лёгкий слой Ticket + Mandate (авторизация работы)
//!
//! Из DeepSeek Super-Agent (001-domain.md, 002-authorization.md) в лёгкой форме:
//! - **Policy** — версионированные правила допуска (кто может инициировать работу)
//! - **Gate** — детерминированный слой (без LLM): кто и с какой силой (explicit/inferred)
//! - **Mandate** — что accept дал: версия Policy, сила, скоуп, matched rules. Ревалидируется
//!   при resume; только человек отзывает
//! - **Ticket** — единица авторизованной работы, создаётся только Eval-ом/accept
//! - **Revalidate** — если Policy версия изменилась, мандат переоценивается
//!
//! Файловая модель: <stateDir>/tickets/tickets.json + policy.json
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Accept mode (как DeepSeek accept mode / ARGUS P4 режимы).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum AcceptMode {
    /// Ничего не принимается без явного разрешения
    #[default]
    Off,
    /// Inferred-запросы записываются как «would accept», но не применяются
    Observe,
    /// Inferred-запросы принимаются по правилам
    Auto,
}

impl AcceptMode {
    pub fn name(&self) -> &'static str {
        match self {
            AcceptMode::Off => "off",
            AcceptMode::Observe => "observe",
            AcceptMode::Auto => "auto",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "off" => Some(AcceptMode::Off),
            "observe" => Some(AcceptMode::Observe),
            "auto" => Some(AcceptMode::Auto),
            _ => None,
        }
    }
}

/// Сила допуска (как Strength в DeepSeek).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Strength {
    /// Человек из allow-list адресовал агента
    Explicit,
    /// Бот / фоллов-ап не из списка
    Inferred,
}

impl Strength {
    pub fn name(&self) -> &'static str {
        match self {
            Strength::Explicit => "explicit",
            Strength::Inferred => "inferred",
        }
    }
}

/// Одно правило допуска: кому разрешено инициировать работу.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AllowRule {
    /// id правила (для matched rules)
    pub id: String,
    /// паттерн отправителя (подстрока/glob): "jaba", "bot:*", "dependabot"
    pub pattern: String,
    /// сила, которую даёт это правило
    pub strength: Strength,
    /// скоуп работы, который разрешён (напр. "workspace", "read-only", "*")
    pub scope: String,
    /// включено ли
    pub enabled: bool,
}

/// Policy — версионированные правила допуска.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Policy {
    pub version: String,
    pub accept_mode: AcceptMode,
    pub rules: Vec<AllowRule>,
    pub updated_at: String,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            version: "v0.1.0".into(),
            accept_mode: AcceptMode::Observe,
            rules: vec![AllowRule {
                id: "owner".into(),
                pattern: "jaba".into(),
                strength: Strength::Explicit,
                scope: "*".into(),
                enabled: true,
            }],
            updated_at: Local::now().to_rfc3339(),
        }
    }
}

impl Policy {
    /// Версия = хэш(accept_mode + rules) — меняется при любом изменении.
    pub fn recompute_version(&mut self) {
        let mut h: u64 = 0xcbf29ce484222325;
        let feed = |h: &mut u64, b: &[u8]| {
            for x in b {
                *h ^= *x as u64;
                *h = h.wrapping_mul(0x100000001b3);
            }
        };
        feed(&mut h, self.accept_mode.name().as_bytes());
        for r in &self.rules {
            feed(&mut h, r.id.as_bytes());
            feed(&mut h, r.pattern.as_bytes());
            feed(&mut h, r.strength.name().as_bytes());
            feed(&mut h, r.scope.as_bytes());
            feed(&mut h, if r.enabled { b"1" } else { b"0" });
        }
        self.version = format!("v{:#x}", h);
    }

    /// Загрузить Policy из <stateDir>/policy.json (default, если нет).
    pub fn load_policy(state_dir: &Path) -> Policy {
        read_json(&state_dir.join("policy.json")).unwrap_or_default()
    }

    /// Сохранить Policy.
    pub fn save_policy(state_dir: &Path, p: &Policy) {
        if let Ok(s) = serde_json::to_string_pretty(p) {
            let _ = fs::write(state_dir.join("policy.json"), s);
        }
    }
}

/// Результат допуска (gate).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Admission {
    pub admitted: bool,
    pub strength: Option<Strength>,
    pub scope: Option<String>,
    pub matched_rules: Vec<String>,
    pub reason: String,
}

/// Mandate — что accept дал.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Mandate {
    pub policy_version: String,
    pub strength: Strength,
    pub scope: String,
    pub matched_rules: Vec<String>,
    /// кто инициировал (origin)
    pub origin: String,
    pub granted_at: String,
}

impl Mandate {
    /// Ревалидация: мандат действителен, только если policy_version совпадает.
    pub fn revalidates(&self, policy: &Policy) -> bool {
        self.policy_version == policy.version
    }
}

/// Статус Ticket.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TicketStatus {
    Open,
    Ended,
}

/// Ticket — единица авторизованной работы.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ticket {
    pub id: u64,
    pub goal: String,
    pub mandate: Mandate,
    pub status: TicketStatus,
    pub outcome: Option<String>,
    pub created_at: String,
    pub ended_at: Option<String>,
}

/// Gate (детерминированный, без LLM): кто и с какой силой.
/// sender — кто просит (имя/роль), goal — текст запроса (для скоупа).
pub fn gate(policy: &Policy, sender: &str, goal: &str) -> Admission {
    // 1. собрать совпавшие правила
    let mut matched: Vec<&AllowRule> = Vec::new();
    let low = sender.to_lowercase();
    for r in &policy.rules {
        if !r.enabled {
            continue;
        }
        let pat = r.pattern.to_lowercase();
        // простой glob: "bot:*" → prefix; иначе подстрока
        if let Some(prefix) = pat.strip_suffix('*') {
            if low.starts_with(prefix) {
                matched.push(r);
            }
        } else if low.contains(&pat) {
            matched.push(r);
        }
    }
    if matched.is_empty() {
        return Admission {
            admitted: false,
            strength: None,
            scope: None,
            matched_rules: vec![],
            reason: format!("отправитель '{sender}' не в allow-списке"),
        };
    }
    // 2. сильнейшее совпадение (explicit > inferred)
    matched.sort_by_key(|r| match r.strength {
        Strength::Explicit => 0,
        Strength::Inferred => 1,
    });
    let best = matched[0];
    let scope = best.scope.clone();
    let matched_ids: Vec<String> = matched.iter().map(|r| r.id.clone()).collect();
    let strength = best.strength;

    match strength {
        Strength::Explicit => Admission {
            admitted: true,
            strength: Some(strength),
            scope: Some(scope),
            matched_rules: matched_ids,
            reason: "явный запрос из allow-списка".into(),
        },
        Strength::Inferred => match policy.accept_mode {
            AcceptMode::Auto => Admission {
                admitted: true,
                strength: Some(strength),
                scope: Some(scope),
                matched_rules: matched_ids,
                reason: format!("inferred-запрос принят в auto-режиме (goal: {goal})"),
            },
            AcceptMode::Observe => Admission {
                // В observe — admit, но пометка inferred (вызывающий решает записать "would")
                admitted: true,
                strength: Some(strength),
                scope: Some(scope),
                matched_rules: matched_ids,
                reason: "inferred-запрос: observe-режим — только записать 'would accept'".into(),
            },
            AcceptMode::Off => Admission {
                admitted: false,
                strength: Some(strength),
                scope: None,
                matched_rules: matched_ids,
                reason: "inferred-запрос отклонён в off-режиме".into(),
            },
        },
    }
}

/// Проверить, разрешён ли скоуп действия внутри мандата.
/// action_scope — напр. "read", "write", "shell", "network".
pub fn allows_scope(mandate: &Mandate, action_scope: &str) -> bool {
    mandate.scope == "*"
        || mandate.scope == action_scope
        || mandate.scope == "all"
        || (mandate.scope == "workspace" && matches!(action_scope, "read" | "write"))
}

/// Хранилище tickets (файловое, как memory).
pub struct TicketStore {
    pub tickets: Vec<Ticket>,
    pub next_id: u64,
}

impl TicketStore {
    pub fn load(state_dir: &Path) -> Self {
        let tdir = state_dir.join("tickets");
        fs::create_dir_all(&tdir).ok();
        let tickets: Vec<Ticket> =
            read_json(&tdir.join("tickets.json")).unwrap_or_default();
        let next_id = tickets.iter().map(|t| t.id).max().unwrap_or(0) + 1;
        TicketStore { tickets, next_id }
    }

    pub fn save(&self, state_dir: &Path) {
        let tdir = state_dir.join("tickets");
        fs::create_dir_all(&tdir).ok();
        write_json(&tdir.join("tickets.json"), &self.tickets);
    }

    /// Создать Ticket из admit-результата (gate). Только Eval/accept создаёт.
    pub fn open_ticket(
        &mut self,
        policy: &Policy,
        sender: &str,
        goal: &str,
        admission: &Admission,
    ) -> Result<Ticket, String> {
        if !admission.admitted {
            return Err(admission.reason.clone());
        }
        let id = self.next_id;
        self.next_id += 1;
        let mandate = Mandate {
            policy_version: policy.version.clone(),
            strength: admission.strength.unwrap_or(Strength::Inferred),
            scope: admission.scope.clone().unwrap_or_else(|| "read".into()),
            matched_rules: admission.matched_rules.clone(),
            origin: sender.to_string(),
            granted_at: Local::now().to_rfc3339(),
        };
        let t = Ticket {
            id,
            goal: goal.to_string(),
            mandate,
            status: TicketStatus::Open,
            outcome: None,
            created_at: Local::now().to_rfc3339(),
            ended_at: None,
        };
        self.tickets.push(t.clone());
        Ok(t)
    }

    /// Закрыть Ticket с outcome.
    pub fn end_ticket(&mut self, id: u64, outcome: &str) -> Result<(), String> {
        let t = self
            .tickets
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or_else(|| format!("ticket #{id} не найден"))?;
        if t.status == TicketStatus::Ended {
            return Err(format!("ticket #{id} уже закрыт"));
        }
        t.status = TicketStatus::Ended;
        t.outcome = Some(outcome.to_string());
        t.ended_at = Some(Local::now().to_rfc3339());
        Ok(())
    }

    /// Открытые tickets (текущая работа).
    pub fn open(&self) -> Vec<&Ticket> {
        self.tickets
            .iter()
            .filter(|t| t.status == TicketStatus::Open)
            .collect()
    }

    /// P8c: открытые tickets, чей Mandate не ревалидируется против текущей Policy
    /// (мандат устарел — версия Policy изменилась). Возвращает их id+goal.
    pub fn open_stale_mandates(&self, policy: &Policy) -> Vec<(u64, String)> {
        self.tickets
            .iter()
            .filter(|t| t.status == TicketStatus::Open && !t.mandate.revalidates(policy))
            .map(|t| (t.id, t.goal.clone()))
            .collect()
    }

    pub fn get(&self, id: u64) -> Option<&Ticket> {
        self.tickets.iter().find(|t| t.id == id)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> Option<T> {
    let raw = fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_json<T: Serialize>(p: &Path, v: &T) {
    if let Ok(s) = serde_json::to_string_pretty(v) {
        let _ = fs::write(p, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir(name: &str) -> PathBuf {
        let d = PathBuf::from(format!("/tmp/argus-tickets-test/{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn explicit_owner_admitted() {
        let p = Policy::default(); // owner: jaba explicit *
        let a = gate(&p, "Jaba", "проверь гипотезу");
        assert!(a.admitted);
        assert_eq!(a.strength, Some(Strength::Explicit));
        assert_eq!(a.scope.as_deref(), Some("*"));
        assert_eq!(a.matched_rules, vec!["owner"]);
    }

    #[test]
    fn unknown_sender_rejected_in_observe() {
        let p = Policy::default(); // observe по умолч.
        let a = gate(&p, "random_person", "сделай X");
        assert!(!a.admitted, "никто вне allow-list — отказ");
        assert!(a.reason.contains("allow-списке"));
    }

    #[test]
    fn inferred_bot_in_observe_admitted_as_would() {
        let mut p = Policy::default();
        p.rules.push(AllowRule {
            id: "bot".into(),
            pattern: "dependabot*".into(),
            strength: Strength::Inferred,
            scope: "workspace".into(),
            enabled: true,
        });
        p.accept_mode = AcceptMode::Observe;
        p.recompute_version();
        let a = gate(&p, "dependabot[bot]", "update deps");
        assert!(a.admitted);
        assert_eq!(a.strength, Some(Strength::Inferred));
        assert!(a.reason.contains("observe"));
    }

    #[test]
    fn inferred_rejected_in_off() {
        let mut p = Policy::default();
        p.rules.push(AllowRule {
            id: "bot".into(),
            pattern: "dependabot*".into(),
            strength: Strength::Inferred,
            scope: "workspace".into(),
            enabled: true,
        });
        p.accept_mode = AcceptMode::Off;
        p.recompute_version();
        let a = gate(&p, "dependabot[bot]", "update deps");
        assert!(!a.admitted);
    }

    #[test]
    fn mandate_revalidates_on_policy_change() {
        let mut p = Policy::default();
        p.recompute_version();
        let v1 = p.version.clone();
        let a = gate(&p, "Jaba", "работа");
        let mut store = TicketStore::load(&dir("a"));
        let t = store.open_ticket(&p, "Jaba", "работа", &a).unwrap();
        assert!(t.mandate.revalidates(&p));
        // меняем правила → версия меняется → мандат не ревалидируется
        p.rules.push(AllowRule {
            id: "new".into(),
            pattern: "alice".into(),
            strength: Strength::Explicit,
            scope: "read".into(),
            enabled: true,
        });
        p.recompute_version();
        assert_ne!(v1, p.version);
        assert!(!t.mandate.revalidates(&p), "мандат аннулируется при смене Policy");
    }

    #[test]
    fn scope_check_works() {
        let mut p = Policy::default();
        p.rules[0].scope = "workspace".into();
        p.recompute_version();
        let a = gate(&p, "Jaba", "работа");
        let mut store = TicketStore::load(&dir("a"));
        let t = store.open_ticket(&p, "Jaba", "работа", &a).unwrap();
        assert!(allows_scope(&t.mandate, "read"));
        assert!(allows_scope(&t.mandate, "write"));
        assert!(!allows_scope(&t.mandate, "shell"));
    }

    #[test]
    fn ticket_lifecycle_open_end() {
        let d = dir("lifecycle");
        let p = Policy::default();
        let a = gate(&p, "Jaba", "работа");
        let mut store = TicketStore::load(&d);
        let t = store.open_ticket(&p, "Jaba", "работа", &a).unwrap();
        assert_eq!(t.status, TicketStatus::Open);
        assert_eq!(store.open().len(), 1);
        store.end_ticket(t.id, "успешно").unwrap();
        assert_eq!(store.open().len(), 0);
        assert_eq!(store.get(t.id).unwrap().status, TicketStatus::Ended);
        // повторное закрытие — ошибка
        assert!(store.end_ticket(t.id, "снова").is_err());
        // персистентность
        store.save(&d);
        let store2 = TicketStore::load(&d);
        assert_eq!(store2.tickets.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod p8c_tests {
    use super::*;
    use std::path::PathBuf;

    /// P8c: открытые tickets с устаревшим мандатом после смены Policy.
    #[test]
    fn open_stale_mandates_found_after_policy_change() {
        let d = PathBuf::from("/tmp/argus-tickets-test/p8c");
        let _ = std::fs::remove_dir_all(&d);
        let mut p = Policy::default();
        p.recompute_version();
        let a = gate(&p, "Jaba", "работа1");
        let mut store = TicketStore::load(&d);
        let t1 = store.open_ticket(&p, "Jaba", "работа1", &a).unwrap();
        assert!(store.open_stale_mandates(&p).is_empty(), "до смены — всё валидно");

        // меняем Policy (добавляем правило) → версия изменится → мандат t1 устарел
        p.rules.push(AllowRule {
            id: "alice".into(),
            pattern: "alice".into(),
            strength: Strength::Explicit,
            scope: "read".into(),
            enabled: true,
        });
        p.recompute_version();
        let stale = store.open_stale_mandates(&p);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0, t1.id);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// P8c: закрытый ticket не считается устаревшим (не «живой»).
    #[test]
    fn closed_ticket_not_stale() {
        let d = PathBuf::from("/tmp/argus-tickets-test/p8c2");
        let _ = std::fs::remove_dir_all(&d);
        let mut p = Policy::default();
        p.recompute_version();
        let a = gate(&p, "Jaba", "работа");
        let mut store = TicketStore::load(&d);
        let t = store.open_ticket(&p, "Jaba", "работа", &a).unwrap();
        store.end_ticket(t.id, "готово").unwrap();
        p.rules.push(AllowRule {
            id: "bob".into(),
            pattern: "bob".into(),
            strength: Strength::Explicit,
            scope: "read".into(),
            enabled: true,
        });
        p.recompute_version();
        assert!(store.open_stale_mandates(&p).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
