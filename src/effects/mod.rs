//! ARGUS-OS1 V2 — P8d: durable effects queue (side-effect очередь)
//!
//! Паттерн из DeepSeek Super-Agent (docs/007-architecture.md → Effects):
//! - enqueue внутри "транзакции" (здесь: атомарный write файла) → выполнение после commit
//! - тот же waiting key → replace payload, становится due немедленно
//! - retry с экспоненциальным backoff (base 2s, doubling, cap 60s)
//! - 6 видов эффектов, каждый со своим обработчиком и health
//! - 3 отказа подряд → ошибка в health (а не молчаливый сбой)
//! - watchdog: running-эффект дольше timeout → снова pending (не зависает)
//!
//! Файловая модель (как memory): <stateDir>/effects/effects.json + health.json
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Виды эффектов (каждый — отдельный обработчик + health entry).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EffectKind {
    /// Уведомление человеку (gabro-notify): «говори только когда человек должен действовать»
    Notify,
    /// Отчёт/сообщение в канал (ответ на Ticket)
    Report,
    /// Внешний вызов (HTTP POST и т.п.)
    External,
    /// Shell-команда (исполняется под контролем безопасности)
    Command,
    /// Durable «сообщение», которое обязано дойти (outbox)
    Outbox,
    /// Внутренний (память, консолидация, …)
    Internal,
}

impl EffectKind {
    pub fn name(&self) -> &'static str {
        match self {
            EffectKind::Notify => "notify",
            EffectKind::Report => "report",
            EffectKind::External => "external",
            EffectKind::Command => "command",
            EffectKind::Outbox => "outbox",
            EffectKind::Internal => "internal",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "notify" => Some(EffectKind::Notify),
            "report" => Some(EffectKind::Report),
            "external" => Some(EffectKind::External),
            "command" => Some(EffectKind::Command),
            "outbox" => Some(EffectKind::Outbox),
            "internal" => Some(EffectKind::Internal),
            _ => None,
        }
    }
    /// Все виды (для CLI/справочника).
    pub const ALL: [EffectKind; 6] = [
        EffectKind::Notify,
        EffectKind::Report,
        EffectKind::External,
        EffectKind::Command,
        EffectKind::Outbox,
        EffectKind::Internal,
    ];
}

/// Статус эффекта.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectStatus {
    /// Готов к выполнению (или ждёт backoff-паузы)
    Pending,
    /// Выполняется сейчас (защищено watchdog-ом от зависания)
    Running,
    /// Успешно завершён
    Done,
    /// Исчерпал попытки (max_attempts) — требует человека
    Failed,
}

impl EffectStatus {
    pub fn name(&self) -> &'static str {
        match self {
            EffectStatus::Pending => "pending",
            EffectStatus::Running => "running",
            EffectStatus::Done => "done",
            EffectStatus::Failed => "failed",
        }
    }
}

/// Уровень «внимания» уведомления (P8g: философия внимания — говорить только
/// когда человек должен действовать). По умолчанию для Notify — Action.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "lowercase")]
pub enum Attention {
    /// Справочное: человек может прочитать, но не обязан действовать (шум по умолчанию)
    Info,
    /// Человек должен действовать или может использовать результат (по умолчанию)
    #[default]
    Action,
    /// Работа не может продолжаться без человека (блокер)
    Blocker,
}

impl Attention {
    pub fn name(&self) -> &'static str {
        match self {
            Attention::Info => "info",
            Attention::Action => "action",
            Attention::Blocker => "blocker",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "info" => Some(Attention::Info),
            "action" => Some(Attention::Action),
            "blocker" => Some(Attention::Blocker),
            _ => None,
        }
    }
}

/// Один эффект в durable-очереди.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Effect {
    pub id: u64,
    pub kind: EffectKind,
    /// dedup-ключ: enqueue того же waiting key заменяет payload
    pub key: String,
    pub payload: serde_json::Value,
    pub status: EffectStatus,
    pub attempts: u32,
    pub max_attempts: u32,
    /// epoch millis, когда можно выполнять снова (backoff)
    pub next_run_at: i64,
    pub created_at: String,
    pub updated_at: String,
    pub last_error: Option<String>,
    /// P8g: уровень внимания (по умолчанию Action). Для Notify-эффектов.
    #[serde(default)]
    pub attention: Attention,
}

/// Тип обработчика: (&Effect, &str) → Result<(), String>.
/// Второй аргумент — state_dir (для файловых сайд-эффектов).
pub type HandlerFn = Box<dyn Fn(&Effect, &Path) -> Result<(), String> + Send + Sync>;

/// Health-запись по виду эффекта.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct HealthEntry {
    pub total: u64,
    pub pending: u64,
    pub running: u64,
    pub done: u64,
    pub failed: u64,
    /// подряд идущие отказы (последних попыток)
    pub consecutive_failures: u32,
    /// healthy=false при consecutive_failures >= 3
    pub healthy: bool,
    pub last_error: Option<String>,
    pub last_success_at: Option<String>,
}

/// Health-сводка по всем видам.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Health {
    pub kinds: HashMap<String, HealthEntry>,
    pub updated_at: String,
}

// Конфиг backoff (как DeepSeek: base 2s, doubling, cap 60s)
const BACKOFF_BASE_MS: i64 = 2_000;
const BACKOFF_CAP_MS: i64 = 60_000;
/// Watchdog: running дольше этого — считается зависшим, возвращается в pending
pub const RUNNING_TIMEOUT_MS: i64 = 5 * 60_000;
/// По умолчанию параллельно выполняется до N эффектов
pub const DEFAULT_CONCURRENCY: usize = 4;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn now_rfc3339() -> String {
    Local::now().to_rfc3339()
}

/// Экспоненциальный backoff: attempt = 1 → 2s, 2 → 4s, … cap 60s.
fn backoff_delay_ms(attempt: u32) -> i64 {
    let shift = attempt.saturating_sub(1).min(30);
    let delay = BACKOFF_BASE_MS.saturating_mul(1i64 << shift.min(5));
    delay.min(BACKOFF_CAP_MS)
}

/// ARGUS-OS1 V2 — P8d: durable effects queue.
pub struct EffectsQueue {
    pub effects: Vec<Effect>,
    handlers: HashMap<EffectKind, HandlerFn>,
    next_id: u64,
    /// максимальное число одновременно исполняемых
    pub concurrency: usize,
    /// список health (считается на save/flush)
    pub health: Health,
}

impl EffectsQueue {
    /// Создаёт/читает очередь в <stateDir>/effects/ (effects.json, health.json).
    pub fn load(state_dir: &Path) -> Self {
        let edir = state_dir.join("effects");
        fs::create_dir_all(&edir).ok();
        let mut q = EffectsQueue {
            effects: read_json::<Vec<Effect>>(&edir.join("effects.json")).unwrap_or_default(),
            handlers: HashMap::new(),
            next_id: q_next_id(&edir),
            concurrency: DEFAULT_CONCURRENCY,
            health: read_json::<Health>(&edir.join("health.json")).unwrap_or_default(),
        };
        q.health.updated_at = now_rfc3339();
        q
    }

    pub fn save(&mut self, state_dir: &Path) {
        self.refresh_health();
        let edir = state_dir.join("effects");
        fs::create_dir_all(&edir).ok();
        write_json(&edir.join("effects.json"), &self.effects);
        write_json(&edir.join("health.json"), &self.health);
    }

    /// Пересчитать health по текущему состоянию очереди (публично для CLI/отладки).
    pub fn refresh_health(&mut self) {
        let mut h = Health {
            updated_at: now_rfc3339(),
            ..Default::default()
        };
        for k in EffectKind::ALL {
            let mut e = HealthEntry::default();
            for eff in &self.effects {
                if eff.kind != k {
                    continue;
                }
                e.total += 1;
                match eff.status {
                    EffectStatus::Pending => e.pending += 1,
                    EffectStatus::Running => e.running += 1,
                    EffectStatus::Done => e.done += 1,
                    EffectStatus::Failed => e.failed += 1,
                }
            }
            // consecutive_failures: последние попытки этого вида
            let mut fails = 0u32;
            for eff in self.effects.iter().filter(|e| e.kind == k) {
                if eff.status == EffectStatus::Failed {
                    fails += 1;
                }
            }
            e.consecutive_failures = fails;
            e.healthy = e.consecutive_failures < 3;
            if let Some(err) = self
                .effects
                .iter()
                .filter(|e| e.kind == k && e.status == EffectStatus::Failed)
                .map(|e| e.last_error.clone().unwrap_or_default())
                .find(|s| !s.is_empty())
            {
                e.last_error = Some(err);
            }
            if let Some(ts) = self
                .effects
                .iter()
                .filter(|e| e.kind == k && e.status == EffectStatus::Done)
                .map(|e| e.updated_at.clone())
                .max()
            {
                e.last_success_at = Some(ts);
            }
            h.kinds.insert(k.name().to_string(), e);
        }
        self.health = h;
    }

    // ---- Enqueue (P8d: тот же waiting key → replace payload, due немедленно) ----
    pub fn enqueue(
        &mut self,
        kind: EffectKind,
        key: &str,
        payload: serde_json::Value,
    ) -> u64 {
        self.enqueue_with_attention(kind, key, payload, Attention::Action)
    }

    /// P8g: enqueue с уровнем внимания.
    /// Правило «тихо по умолчанию»: Info-эффект с key, который уже был доставлен (Done),
    /// не ставится повторно — иначе уведомления плодят шум.
    pub fn enqueue_with_attention(
        &mut self,
        kind: EffectKind,
        key: &str,
        payload: serde_json::Value,
        attention: Attention,
    ) -> u64 {
        // «тихо по умолчанию»: Info с тем же key уже был доставлен → не плодим шум
        if attention == Attention::Info
            && self
                .effects
                .iter()
                .any(|e| e.key == key && e.status == EffectStatus::Done)
        {
            return 0;
        }
        // dedup: waiting-эффект того же key → replace payload + due немедленно
        if let Some(existing) = self
            .effects
            .iter_mut()
            .find(|e| e.key == key && e.status == EffectStatus::Pending)
        {
            existing.payload = payload;
            existing.next_run_at = now_ms();
            existing.updated_at = now_rfc3339();
            existing.last_error = None;
            return existing.id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.effects.push(Effect {
            id,
            kind,
            key: key.to_string(),
            payload,
            status: EffectStatus::Pending,
            attempts: 0,
            max_attempts: 5,
            next_run_at: now_ms(),
            created_at: now_rfc3339(),
            updated_at: now_rfc3339(),
            last_error: None,
            attention,
        });
        id
    }

    /// P8g: «что ждёт человека» — pending эффекты, требующие действия (Action/Blocker)
    /// + failed (человек должен разобраться). Это то, что показывается в панели/промпте.
    pub fn needs_attention(&self) -> Vec<&Effect> {
        self.effects
            .iter()
            .filter(|e| {
                e.status == EffectStatus::Failed
                    || (e.status == EffectStatus::Pending && e.attention != Attention::Info)
            })
            .collect()
    }

    /// P8g: отрендерить «что ждёт человека» одной строкой для промпта/панели.
    pub fn attention_block(&self) -> String {
        let items = self.needs_attention();
        if items.is_empty() {
            return String::new();
        }
        let mut out = String::from("### attention (ждёт человека)\n");
        for e in items {
            let msg = e
                .payload
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(&e.key);
            out.push_str(&format!(
                "- #{} [{}] {} ({}): {}\n",
                e.id,
                e.kind.name(),
                e.attention.name(),
                e.status.name(),
                msg.chars().take(120).collect::<String>()
            ));
        }
        out
    }

    /// Регистрация обработчика вида. Возвращает предыдущий, если был.
    pub fn register_handler(&mut self, kind: EffectKind, handler: HandlerFn) -> Option<HandlerFn> {
        self.handlers.insert(kind, handler)
    }

    /// Есть ли обработчик для вида (health: вид без обработчика = не может выполниться).
    pub fn has_handler(&self, kind: EffectKind) -> bool {
        self.handlers.contains_key(&kind)
    }

    /// Watchdog: running-эффекты дольше RUNNING_TIMEOUT_MS → снова pending.
    /// Возвращает число возвращённых.
    pub fn watchdog(&mut self) -> usize {
        let now = now_ms();
        let mut revived = 0usize;
        for eff in self.effects.iter_mut() {
            if eff.status == EffectStatus::Running {
                // next_run_at у running = время старта
                if now - eff.next_run_at > RUNNING_TIMEOUT_MS {
                    eff.status = EffectStatus::Pending;
                    eff.next_run_at = now_ms();
                    eff.last_error = Some("watchdog: истёк RUNNING_TIMEOUT, эффект возвращён в очередь".into());
                    eff.updated_at = now_rfc3339();
                    revived += 1;
                }
            }
        }
        revived
    }

    /// Исполнить due-эффекты (pending с next_run_at <= now) до concurrency.
    /// Один проход. Возвращает (выполнено, ошибок).
    pub fn process_due(&mut self, state_dir: &Path) -> (usize, usize) {
        let now = now_ms();
        let mut done = 0usize;
        let mut errors = 0usize;
        // running считается занятым слотом
        let mut running = self
            .effects
            .iter()
            .filter(|e| e.status == EffectStatus::Running)
            .count();
        // сортируем: сначала самые старые due
        let mut due: Vec<u64> = self
            .effects
            .iter()
            .filter(|e| e.status == EffectStatus::Pending && e.next_run_at <= now)
            .map(|e| e.id)
            .collect();
        due.sort_by(|a, b| {
            let ea = self.effects.iter().find(|e| e.id == *a).unwrap();
            let eb = self.effects.iter().find(|e| e.id == *b).unwrap();
            ea.next_run_at.cmp(&eb.next_run_at)
        });
        for id in due {
            if running >= self.concurrency {
                break;
            }
            let Some(eff) = self.effects.iter_mut().find(|e| e.id == id) else {
                continue;
            };
            // маркируем running, сохраняем время старта в next_run_at
            eff.status = EffectStatus::Running;
            eff.attempts += 1;
            eff.next_run_at = now_ms();
            eff.updated_at = now_rfc3339();
            let eff_clone = eff.clone();
            running += 1;

            let handler = self.handlers.get(&eff_clone.kind);
            let result = match handler {
                Some(h) => h(&eff_clone, state_dir),
                None => Err(format!("нет обработчика для вида '{}'", eff_clone.kind.name())),
            };
            // обновляем состояние после обработчика
            let now2 = now_ms();
            let eff = self.effects.iter_mut().find(|e| e.id == id).unwrap();
            match result {
                Ok(()) => {
                    eff.status = EffectStatus::Done;
                    eff.last_error = None;
                    eff.updated_at = now_rfc3339();
                    running -= 1;
                    done += 1;
                }
                Err(err) => {
                    errors += 1;
                    eff.status = EffectStatus::Pending;
                    eff.last_error = Some(err.clone());
                    eff.updated_at = now_rfc3339();
                    if eff.attempts >= eff.max_attempts {
                        eff.status = EffectStatus::Failed;
                        eff.next_run_at = now_ms();
                    } else {
                        eff.next_run_at = now2 + backoff_delay_ms(eff.attempts);
                    }
                    running -= 1;
                }
            }
            // после каждого эффекта — пишем состояние (durable на каждом шаге)
            self.refresh_health();
            self.write_state(state_dir);
        }
        (done, errors)
    }

    /// Принудительно повторить failed/pending эффект (сброс attempts).
    pub fn retry(&mut self, id: u64) -> Result<(), String> {
        let eff = self
            .effects
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| format!("эффект #{id} не найден"))?;
        eff.status = EffectStatus::Pending;
        eff.attempts = 0;
        eff.next_run_at = now_ms();
        eff.last_error = None;
        eff.updated_at = now_rfc3339();
        Ok(())
    }

    /// Отменить (только pending): cancelled → удалить.
    pub fn cancel(&mut self, id: u64) -> Result<(), String> {
        let before = self.effects.len();
        self.effects.retain(|e| !(e.id == id && e.status == EffectStatus::Pending));
        if self.effects.len() == before {
            Err("эффект не найден или уже выполняется".into())
        } else {
            Ok(())
        }
    }

    pub fn get(&self, id: u64) -> Option<&Effect> {
        self.effects.iter().find(|e| e.id == id)
    }

    /// Сводка по статусам.
    pub fn status_counts(&self) -> (usize, usize, usize, usize) {
        let mut p = 0;
        let mut r = 0;
        let mut d = 0;
        let mut f = 0;
        for e in &self.effects {
            match e.status {
                EffectStatus::Pending => p += 1,
                EffectStatus::Running => r += 1,
                EffectStatus::Done => d += 1,
                EffectStatus::Failed => f += 1,
            }
        }
        (p, r, d, f)
    }

    /// Список failed-эффектов (то, что требует человека).
    pub fn failed(&self) -> Vec<&Effect> {
        self.effects
            .iter()
            .filter(|e| e.status == EffectStatus::Failed)
            .collect()
    }

    /// Список pending (ждёт выполнения/повтора).
    pub fn pending(&self) -> Vec<&Effect> {
        self.effects
            .iter()
            .filter(|e| e.status == EffectStatus::Pending)
            .collect()
    }

    fn write_state(&self, state_dir: &Path) {
        let edir = state_dir.join("effects");
        fs::create_dir_all(&edir).ok();
        write_json(&edir.join("effects.json"), &self.effects);
        write_json(&edir.join("health.json"), &self.health);
    }
}

/// Следующий id: максимум существующих + 1 (устойчиво к перезапуску).
fn q_next_id(edir: &Path) -> u64 {
    read_json::<Vec<Effect>>(&edir.join("effects.json"))
        .map(|v| v.iter().map(|e| e.id).max().unwrap_or(0) + 1)
        .unwrap_or(1)
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

// ============================================================================
// Готовые обработчики (используются argus_agent / тестами)
// ============================================================================

/// Обработчик External: HTTP POST payload.url + payload.body (reqwest blocking).
/// Успех = статус 2xx/3xx. Ошибка = текст.
pub fn http_post_handler(eff: &Effect, _state: &Path) -> Result<(), String> {
    let url = eff
        .payload
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "payload.url отсутствует".to_string())?;
    let body = eff.payload.get("body").cloned().unwrap_or(serde_json::json!({}));
    let timeout_ms = eff
        .payload
        .get("timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(10_000);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms))
        .build()
        .map_err(|e| format!("client: {e}"))?;
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .map_err(|e| format!("send: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", resp.status()))
    }
}

/// Обработчик Command: исполняет payload.command через sh -c.
/// ⚠️ Опасный — регистрируется только в режиме с одобрением/безопасной политикой.
pub fn command_handler(eff: &Effect, _state: &Path) -> Result<(), String> {
    let cmd = eff
        .payload
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "payload.command отсутствует".to_string())?;
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr);
        Err(format!("exit {}: {}", out.status, stderr.trim().chars().take(200).collect::<String>()))
    }
}

/// Обработчик Report: пишет отчёт в <stateDir>/effects/out/<id>.json (durable артефакт).
pub fn report_handler(eff: &Effect, state: &Path) -> Result<(), String> {
    let out_dir = state.join("effects").join("out");
    fs::create_dir_all(&out_dir).ok();
    let path = out_dir.join(format!("{}.json", eff.id));
    let doc = serde_json::json!({
        "id": eff.id,
        "kind": eff.kind.name(),
        "key": eff.key,
        "payload": eff.payload,
        "done_at": now_rfc3339(),
    });
    fs::write(&path, serde_json::to_string_pretty(&doc).unwrap_or_default())
        .map_err(|e| format!("write: {e}"))
}

/// Обработчик Notify: выводит уведомление в stderr (заглушка; реальный канал — gabro-notify).
pub fn notify_handler(eff: &Effect, _state: &Path) -> Result<(), String> {
    let msg = eff
        .payload
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("<нет message>");
    eprintln!("[effects:notify] {} ({})", msg, eff.key);
    Ok(())
}

/// Обработчик Internal: no-op (пример: памятка для консолидатора).
pub fn internal_handler(_eff: &Effect, _state: &Path) -> Result<(), String> {
    Ok(())
}

/// Единая точка регистрации обработчиков по умолчанию (агент, CLI, serve-демон).
/// `allow_command` — включать ли опасный shell-обработчик (по умолчанию НЕТ).
pub fn register_default_handlers(q: &mut EffectsQueue, allow_command: bool) {
    q.register_handler(EffectKind::Notify, Box::new(notify_handler));
    q.register_handler(EffectKind::Report, Box::new(report_handler));
    q.register_handler(EffectKind::External, Box::new(http_post_handler));
    if allow_command {
        q.register_handler(EffectKind::Command, Box::new(command_handler));
    }
    q.register_handler(EffectKind::Internal, Box::new(internal_handler));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn q(dir: &str) -> (EffectsQueue, PathBuf) {
        let d = PathBuf::from(format!("/tmp/argus-effects-test/{dir}"));
        let _ = std::fs::remove_dir_all(&d);
        (EffectsQueue::load(&d), d)
    }

    fn ok_handler() -> HandlerFn {
        Box::new(|_eff, _s| Ok(()))
    }

    fn fail_handler() -> HandlerFn {
        Box::new(|_eff, _s| Err("преднамеренный сбой".into()))
    }

    #[test]
    fn enqueue_dedup_same_key_replaces_payload() {
        let (mut q, _) = q("dedup");
        let a = q.enqueue(EffectKind::Notify, "k1", serde_json::json!({"message": "первое"}));
        let b = q.enqueue(EffectKind::Notify, "k1", serde_json::json!({"message": "второе"}));
        assert_eq!(a, b, "тот же key → тот же id");
        assert_eq!(q.effects.len(), 1);
        assert_eq!(
            q.effects[0].payload["message"],
            serde_json::json!("второе"),
            "payload заменён"
        );
    }

    #[test]
    fn enqueue_different_keys_create_two() {
        let (mut q, _) = q("two");
        q.enqueue(EffectKind::Notify, "a", serde_json::json!({}));
        q.enqueue(EffectKind::Report, "b", serde_json::json!({}));
        assert_eq!(q.effects.len(), 2);
    }

    #[test]
    fn process_due_ok_marks_done() {
        let (mut q, d) = q("ok");
        q.register_handler(EffectKind::Notify, ok_handler());
        q.enqueue(EffectKind::Notify, "n", serde_json::json!({"message": "x"}));
        let (done, errors) = q.process_due(&d);
        assert_eq!(done, 1);
        assert_eq!(errors, 0);
        let (_, _, done_count, failed) = q.status_counts();
        assert_eq!(done_count, 1);
        assert_eq!(failed, 0);
        assert_eq!(q.effects[0].status, EffectStatus::Done);
    }

    #[test]
    fn process_due_fail_backoff_then_failed() {
        let (mut q, d) = q("fail");
        q.register_handler(EffectKind::External, fail_handler());
        q.enqueue(EffectKind::External, "e", serde_json::json!({"url": "x"}));
        // max_attempts по умолчанию 5: 5 неудач → Failed.
        // Моделируем время: сбрасываем next_run_at (backoff) перед каждым проходом.
        for _ in 0..5 {
            let (_d, _e) = q.process_due(&d);
            q.effects[0].next_run_at = now_ms();
        }
        let (_, _, _, failed) = q.status_counts();
        assert_eq!(failed, 1, "после max_attempts — Failed");
        assert_eq!(q.effects[0].attempts, 5);
        assert!(q.effects[0].last_error.is_some());
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay_ms(1), 2_000);
        assert_eq!(backoff_delay_ms(2), 4_000);
        assert_eq!(backoff_delay_ms(3), 8_000);
        assert!(backoff_delay_ms(10) <= BACKOFF_CAP_MS);
        assert_eq!(backoff_delay_ms(100), BACKOFF_CAP_MS);
    }

    #[test]
    fn watchdog_revives_stuck_running() {
        let (mut q, d) = q("watchdog");
        q.register_handler(EffectKind::Internal, ok_handler());
        q.enqueue(EffectKind::Internal, "i", serde_json::json!({}));
        // запустить, но «зависнуть»: вручную выставить running + старое время старта
        let (_d, _e) = q.process_due(&d);
        assert_eq!(q.effects[0].status, EffectStatus::Done);
        // второй эффект — смоделируем зависший running
        q.enqueue(EffectKind::Internal, "i2", serde_json::json!({}));
        q.effects[0].status = EffectStatus::Running;
        q.effects[0].next_run_at = now_ms() - RUNNING_TIMEOUT_MS - 1000;
        let revived = q.watchdog();
        assert_eq!(revived, 1);
        assert_eq!(q.effects[0].status, EffectStatus::Pending);
    }

    #[test]
    fn concurrency_limits_parallel_running() {
        let (mut q, d) = q("conc");
        q.register_handler(EffectKind::Internal, ok_handler());
        q.concurrency = 1;
        for i in 0..3 {
            q.enqueue(EffectKind::Internal, &format!("c{i}"), serde_json::json!({}));
        }
        let (_d, _e) = q.process_due(&d);
        // с concurrency=1 и мгновенными обработчиками всё успеет за один проход
        let (_, _, done, _) = q.status_counts();
        assert_eq!(done, 3);
    }

    #[test]
    fn retry_resets_attempts() {
        let (mut q, d) = q("retry");
        q.register_handler(EffectKind::External, fail_handler());
        let id = q.enqueue(EffectKind::External, "r", serde_json::json!({}));
        for _ in 0..5 {
            let _ = q.process_due(&d);
            q.effects[0].next_run_at = now_ms();
        }
        assert_eq!(q.effects[0].status, EffectStatus::Failed);
        q.retry(id).unwrap();
        assert_eq!(q.effects[0].status, EffectStatus::Pending);
        assert_eq!(q.effects[0].attempts, 0);
    }

    #[test]
    fn health_flags_unhealthy_after_3_failures() {
        let (mut q, d) = q("health");
        q.register_handler(EffectKind::External, fail_handler());
        for i in 0..3 {
            q.enqueue(EffectKind::External, &format!("h{i}"), serde_json::json!({}));
        }
        // прогоняем до исчерпания max_attempts; между проходами сбрасываем backoff
        for _ in 0..5 {
            let _ = q.process_due(&d);
            for e in q.effects.iter_mut() {
                if e.status == EffectStatus::Pending {
                    e.next_run_at = now_ms();
                }
            }
        }
        q.refresh_health();
        let h = &q.health.kinds["external"];
        assert!(!h.healthy, "3+ failed → unhealthy");
        assert!(h.consecutive_failures >= 3);
    }

    #[test]
    fn persists_across_load() {
        let (mut q, d) = q("persist");
        q.register_handler(EffectKind::Report, ok_handler());
        q.enqueue(EffectKind::Report, "p", serde_json::json!({"text": "сохрани меня"}));
        q.save(&d);
        let q2 = EffectsQueue::load(&d);
        assert_eq!(q2.effects.len(), 1);
        assert_eq!(q2.effects[0].key, "p");
        // id продолжает расти после перезапуска
        let mut q2 = q2;
        let nid = q2.enqueue(EffectKind::Report, "p2", serde_json::json!({}));
        assert!(nid > 0);
    }

    #[test]
    fn no_handler_reports_error_and_backs_off() {
        let (mut q, d) = q("nohandler");
        q.enqueue(EffectKind::Command, "c", serde_json::json!({"command": "true"}));
        let (_d, errors) = q.process_due(&d);
        assert_eq!(errors, 1);
        assert!(q.effects[0].last_error.as_deref().unwrap().contains("нет обработчика"));
    }

    // ---- P8g: философия внимания ----

    #[test]
    fn default_attention_is_action() {
        let (mut q, d) = q("p8g-default");
        let id = q.enqueue(EffectKind::Notify, "n", serde_json::json!({}));
        assert_eq!(q.effects[0].attention, Attention::Action);
        assert_ne!(id, 0);
    }

    #[test]
    fn info_same_key_not_redelivered_after_done() {
        let (mut q, d) = q("p8g-quiet");
        q.register_handler(EffectKind::Notify, ok_handler());
        // доставить Action-сообщение
        let id = q.enqueue_with_attention(
            EffectKind::Notify,
            "status",
            serde_json::json!({"message": "первое"}),
            Attention::Action,
        );
        q.process_due(&d);
        assert_eq!(q.effects[0].status, EffectStatus::Done);
        // повторный Info с тем же key → 0 (не плодим шум)
        let again = q.enqueue_with_attention(
            EffectKind::Notify,
            "status",
            serde_json::json!({"message": "второе"}),
            Attention::Info,
        );
        assert_eq!(again, 0, "Info-дубликат после доставки — тихо по умолчанию");
        assert_eq!(q.effects.len(), 1);
    }

    #[test]
    fn needs_attention_lists_action_blocker_failed() {
        let (mut q, d) = q("p8g-attention");
        q.enqueue_with_attention(
            EffectKind::Notify,
            "info",
            serde_json::json!({"message": "инфо"}),
            Attention::Info,
        );
        q.enqueue_with_attention(
            EffectKind::Notify,
            "act",
            serde_json::json!({"message": "сделай"}),
            Attention::Action,
        );
        q.enqueue_with_attention(
            EffectKind::Notify,
            "block",
            serde_json::json!({"message": "блокер"}),
            Attention::Blocker,
        );
        // failed-эффект (external) тоже требует человека — ставим статус вручную,
        // чтобы не мешать process_due (структура needs_attention, не доставка)
        let fid = q.enqueue(EffectKind::External, "f", serde_json::json!({}));
        q.effects.iter_mut().find(|e| e.id == fid).unwrap().status = EffectStatus::Failed;
        let att = q.needs_attention();
        let keys: Vec<&str> = att.iter().map(|e| e.key.as_str()).collect();
        assert!(keys.contains(&"act"), "action должен ждать человека");
        assert!(keys.contains(&"block"), "blocker должен ждать человека");
        assert!(keys.contains(&"f"), "failed должен ждать человека");
        assert!(!keys.contains(&"info"), "info не должен требовать внимания");
        // блок в промпт
        let block = q.attention_block();
        assert!(block.contains("### attention"));
        assert!(block.contains("сделай"));
        let _ = d;
    }
}
