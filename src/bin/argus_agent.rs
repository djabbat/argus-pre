//! argus-agent V3.6 — CLI-вход: init / run / memory / skill / session / task / plugin / bench / eval / fusion / serve
//! Паттерны: P1 (цикл + реальная LLM), P2 (файловая память), P4 (режимы-сессия).
use argus_pre::agent::{AgentLoop, LlmConfig, LlmModel, LoopConfig, Mode, StubModel};
use argus_pre::ledger::{self, Ledger, Outcome};
use argus_pre::memory::{Memory, NoteKind};
use argus_pre::probe::{self, CorpusSearcher, EpMcSearcher, Searcher};
use argus_pre::dialogue;
use argus_pre::lab;
use argus_pre::stance;
use argus_pre::thirdparty;
use argus_pre::verify::{self, Cache, HttpFetcher};
use std::path::PathBuf;

fn state_dir() -> PathBuf {
    std::env::var("ARGUS_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".argus-agent"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let sdir = state_dir();

    match cmd {
        "init" => {
            let mut mem = Memory::load(&sdir);
            mem.save(&sdir);
            let n = make_starter_skills(&sdir);
            println!(
                "✅ argus-agent V3.6 инициализирован: {} (скиллов: {})",
                sdir.display(),
                n
            );
        }
        "skill" => cmd_skill(&args, &sdir),
        "memory" => cmd_memory(&args, &sdir),
        "run" => cmd_run(&args, &sdir),
        "bench" => cmd_bench(&args, &sdir),
        "eval" => cmd_eval(&args, &sdir),
        "fusion" => cmd_fusion(&args, &sdir),
        "serve" => cmd_serve(&args, &sdir),
        "session" => cmd_session(&args, &sdir),
        "task" => cmd_task(&args, &sdir),
        "plugin" => cmd_plugin(&args, &sdir),
        "mcp" => cmd_mcp(&args, &sdir),
        "effects" => cmd_effects(&args, &sdir),
        "think" => cmd_think(&args, &sdir),
        "ledger" => cmd_ledger(&args, &sdir),
        "lab" => cmd_lab(&args, &sdir),
        "council" => cmd_council(&args, &sdir),
        "external" => cmd_external(&args, &sdir),
        "ticket" => cmd_ticket(&args, &sdir),
        "policy" => cmd_policy(&args, &sdir),
        "plan" => println!("План-режим (read-only): запусти run --mode plan."),
        _ => {
            println!(
                "argus-agent V3.6 — локально-первый агент
\nКоманды:
  init                     создать память + стартовые скиллы
  run \"цель\" [опции]       запустить агентский цикл с LLM (--session для продолжения)
  memory add/list/recall/export/import/embed/lessons
  skill list/search/install/show/tap/push/browse
  session list/show/tail/search/rm/tui
  task add/list/run         durable задачи (cron)
  plugin add/list/remove    hooks: before_step|after_step|on_done|on_error
  effects enqueue/list/attention/retry/process/health
                        durable side-effect очередь (P8d): notify/report/external/command/outbox/internal
  ticket list/show          P8a: единицы авторизованной работы (Ticket+Mandate)
  policy show/set           P8a: версионированные правила допуска (accept mode off/observe/auto)
  think [--once|--loop|--dry-run] [--probe epmc|corpus|none] [--corpus DIR]
        [--limit N] [--min-score 0.20] [--offline] [--judge auto|lexical|hybrid|llm|support|contradict|off]
        [--qid <id>] [--seed N] [--k N]
        [--loop --max-cycles N --interval-secs N --budget-cycles N --budget-tokens N --quiet]
        [--report [--write] [--hours N]]
                        P9: внутренний цикл (idle inference): выбор вопроса → проба →
                        резолв PMID/DOI/arXiv → запись цикла + суточный отчёт
  (отдельный бинарник) argus-cka --corpus DIR [--a ngram] [--b wordhash|ratings:<model>]
                        [--n N] [--perms N] — M3: сходимость семантики (CKA + перестановочный нуль)
  ledger list|add|show|seed|close|report
                        P9: леджер открытых вопросов (prior с вероятностями)
  bench                    smoke-бенчмарк (3 задачи)
  eval --evals <file>       GAIA-lite / GAIA-L1 subset
  fusion \"задача\" --workers N
  serve --port N --tick S   skill hub + планировщик задач
\nОпции run:
  --url <http://host:port>   LLM-сервер (Ollama 11434 / LM Studio 1234)
  --model <name>             модель (по умолч. qwen2.5:7b)
  --mode plan|auto|bypass    режим сессии (default — одобрения)
  --yes                      автоодобрение опасных вызовов
  --cwd <dir>                рабочая директория
\nФайловая память: {}/memory/ (profile.json, notes.json, lessons.json, procedures.json)
\nПолное руководство: argus-pre/DOCS.md",
                sdir.display()
            );
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn cmd_run(args: &[String], sdir: &std::path::Path) {
    let goal = args.get(2).cloned().unwrap_or_default();
    if goal.is_empty() {
        eprintln!("использование: argus-agent run \"цель\" [--url --model --mode --yes --cwd]");
        std::process::exit(1);
    }
    let mode = parse_mode(args);
    let yes = has(args, "--yes");
    let cwd = flag(args, "--cwd").unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .display()
            .to_string()
    });

    let mut mem = Memory::load(sdir);

    // V2.0: сессии — продолжить по --session <id> (seed = прошлый транскрипт)
    let session_id = flag(args, "--session");
    let mut seed = Vec::new();
    if let Some(sid) = &session_id {
        if let Some(s) = load_session(sdir, sid) {
            seed = s.transcript;
        } else {
            eprintln!("сессия '{sid}' не найдена — начинаю новую");
        }
    }

    let stub = has(args, "--stub");
    let model: Box<dyn argus_pre::agent::Model> = if stub {
        Box::new(StubModel)
    } else {
        let cfg = LlmConfig {
            base_url: flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:11434".into()),
            model: flag(args, "--model").unwrap_or_else(|| "qwen2.5:7b".into()),
            ..Default::default()
        };
        println!("LLM: {} · {}\n", cfg.base_url, cfg.model);
        Box::new(LlmModel::new(cfg))
    };

    // P8a: gate → Ticket (авторизация работы). Sender по умолчанию — владелец (Jaba).
    let sender = flag(args, "--sender").unwrap_or_else(|| "jaba".into());
    let policy = argus_pre::tickets::Policy::load_policy(sdir);
    let admission = argus_pre::tickets::gate(&policy, &sender, &goal);
    let mut tickets = argus_pre::tickets::TicketStore::load(sdir);
    // P8c: мандат, который пойдёт в цикл (скоуп-гейт опасных действий)
    let mut run_mandate: Option<argus_pre::tickets::Mandate> = None;
    let ticket_id = if admission.admitted {
        match tickets.open_ticket(&policy, &sender, &goal, &admission) {
            Ok(t) => {
                println!(
                    "🎫 Ticket #{} [{}] scope={} rules={} ({})",
                    t.id,
                    t.mandate.strength.name(),
                    t.mandate.scope,
                    t.mandate.matched_rules.join(","),
                    admission.reason
                );
                // P8c: мандат передаётся в цикл для scope-гейта опасных действий
                run_mandate = Some(t.mandate.clone());
                Some(t.id)
            }
            Err(e) => {
                eprintln!("⚠️ ticket не создан: {e}");
                None
            }
        }
    } else {
        eprintln!("🔒 отказ допуска: {}", admission.reason);
        eprintln!("    (проверь policy.json / --sender; in off-режиме inferred не принимается)");
        None
    };
    tickets.save(sdir);

    let mut loop_ = AgentLoop {
        model: model.as_ref(),
        mode,
        cfg: LoopConfig::default(),
        approve: yes,
        cwd,
        last_transcript: Vec::new(),
        goal: goal.clone(),
        skills: argus_pre::skills::load_skills(sdir),
        seed,
        plugins: argus_pre::plugins::load(sdir),
        state_dir: Some(sdir.to_path_buf()),
        mandate: run_mandate,
    };
    println!(
        "Режим: {mode:?} · одобрение опасных: {}\n",
        if yes { "ДА" } else { "нет (отказ)" }
    );
    let out = loop_.run(&goal, &mut mem);
    mem.save(sdir);
    // V2.8: персист last_fire таймер-хуков
    argus_pre::plugins::save_pub(sdir, &loop_.plugins);
    // V3.2: авто-синк памяти в Obsidian-ваулт (ARGUS_VAULT_SYNC=<path>)
    if let Ok(vault) = std::env::var("ARGUS_VAULT_SYNC") {
        if !vault.is_empty() {
            let n = mem.export_vault(std::path::Path::new(&vault));
            println!("🔄 auto-sync: экспортировано в ваулт: {n}");
        }
    }
    println!("{out}");
    // P8a: закрыть Ticket по завершении
    if let Some(tid) = ticket_id {
        let mut tickets = argus_pre::tickets::TicketStore::load(sdir);
        match tickets.end_ticket(tid, &out) {
            Ok(()) => println!("🎫 Ticket #{tid} закрыт"),
            Err(e) => eprintln!("⚠️ {e}"),
        }
        tickets.save(sdir);
    }
    if !loop_.last_transcript.is_empty() {
        println!("\n── Транскрипт ──");
        for t in &loop_.last_transcript {
            println!("{}", t.chars().take(400).collect::<String>());
        }
    }
    // V2.0: сохранить/обновить сессию
    let sid = session_id.unwrap_or_else(|| {
        format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        )
    });
    save_session(sdir, &sid, &goal, &loop_.last_transcript);
    println!("\n📁 сессия: {sid} (продолжить: run \"…\" --session {sid})");
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Session {
    id: String,
    goal: String,
    created: String,
    updated: String,
    transcript: Vec<String>,
}

fn sessions_dir(sdir: &std::path::Path) -> std::path::PathBuf {
    sdir.join("sessions")
}

fn load_session(sdir: &std::path::Path, id: &str) -> Option<Session> {
    let p = sessions_dir(sdir).join(format!("{id}.json"));
    std::fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn save_session(sdir: &std::path::Path, id: &str, goal: &str, transcript: &[String]) {
    let dir = sessions_dir(sdir);
    std::fs::create_dir_all(&dir).ok();
    let now = chrono::Local::now().to_rfc3339();
    // V3.4: dedup — если транскрипт не изменился, не трогаем updated (не плодим «свежие» дубли)
    if let Some(old) = load_session(sdir, id) {
        if old.transcript == transcript && old.goal == goal {
            return;
        }
    }
    let s = Session {
        id: id.to_string(),
        goal: goal.to_string(),
        created: load_session(sdir, id)
            .map(|s| s.created)
            .unwrap_or_else(|| now.clone()),
        updated: now,
        transcript: transcript.to_vec(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&s) {
        let _ = std::fs::write(dir.join(format!("{id}.json")), json);
    }
}

/// Durable задачи (V2.0, cron-like): add/list/run.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct TaskItem {
    id: u64,
    desc: String,
    every_sec: u64,
    next_due: u64,
    last_run: Option<String>,
    model: String,
}

fn tasks_path(sdir: &std::path::Path) -> std::path::PathBuf {
    sdir.join("tasks.json")
}

fn load_tasks(sdir: &std::path::Path) -> Vec<TaskItem> {
    std::fs::read_to_string(tasks_path(sdir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_tasks(sdir: &std::path::Path, tasks: &[TaskItem]) {
    if let Ok(s) = serde_json::to_string_pretty(tasks) {
        let _ = std::fs::write(tasks_path(sdir), s);
    }
}

fn now_sec() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cmd_task(args: &[String], sdir: &std::path::Path) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    let mut tasks = load_tasks(sdir);
    match sub {
        "add" => {
            let desc = args.get(3).cloned().unwrap_or_default();
            if desc.is_empty() {
                println!("использование: task add \"описание\" --every <сек> [--model M]");
                return;
            }
            let every = flag(args, "--every")
                .and_then(|s| s.parse().ok())
                .unwrap_or(3600);
            let model = flag(args, "--model").unwrap_or_else(|| "qwen2.5:7b".into());
            let id = tasks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
            tasks.push(TaskItem {
                id,
                desc,
                every_sec: every,
                next_due: now_sec() + every,
                last_run: None,
                model,
            });
            save_tasks(sdir, &tasks);
            println!("✅ задача #{id} создана (каждые {every}с)");
        }
        "list" => {
            if tasks.is_empty() {
                println!("(задач нет — task add \"…\" --every N)");
            }
            let now = now_sec();
            for t in &tasks {
                let due = if t.next_due <= now {
                    "⚠️ к запуску".to_string()
                } else {
                    format!("через {}с", t.next_due - now)
                };
                println!(
                    "- #{} [{}] {}\n    {} · last: {}",
                    t.id,
                    t.model,
                    t.desc.chars().take(70).collect::<String>(),
                    due,
                    t.last_run.as_deref().unwrap_or("—")
                );
            }
        }
        "run" => {
            let n = run_due_tasks(sdir, true);
            if n == 0 {
                println!("(нет просроченных задач)");
            }
        }
        "tui" => {
            // V2.5: интерактивный список задач: [i] показать, r <i> выполнить, d <i> удалить, a <описание> добавить
            use std::io::Write;
            loop {
                tasks = load_tasks(sdir);
                if tasks.is_empty() {
                    println!("(задач нет — a <описание> чтобы добавить, q выход)");
                } else {
                    println!(
                        "\n══ Tasks TUI ══ (r <i> run · d <i> delete · a <описание> add · q выход)"
                    );
                    for (i, t) in tasks.iter().enumerate() {
                        let due = if t.next_due <= now_sec() {
                            "⚠️ DUE"
                        } else {
                            "ok"
                        };
                        println!(
                            "  [{i}] #{} [{due}] {}",
                            t.id,
                            t.desc.chars().take(50).collect::<String>()
                        );
                    }
                }
                print!("> ");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line).is_err() {
                    return;
                }
                let line = line.trim().to_string();
                if line == "q" {
                    return;
                }
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                let cmd = parts[0];
                let arg = parts.get(1).unwrap_or(&"").to_string();
                match cmd {
                    "r" => {
                        if let Ok(i) = arg.parse::<usize>() {
                            if let Some(t) = tasks.get(i).cloned() {
                                let tdir = sdir.join(format!("task-{}", t.id));
                                std::fs::create_dir_all(&tdir).ok();
                                let mut mem = Memory::load(&tdir);
                                let llm = LlmModel::new(LlmConfig {
                                    model: t.model.clone(),
                                    ..Default::default()
                                });
                                let mut loop_ = AgentLoop {
                                    model: &llm,
                                    mode: Mode::Default,
                                    cfg: LoopConfig::default(),
                                    approve: true,
                                    cwd: tdir.display().to_string(),
                                    last_transcript: Vec::new(),
                                    goal: t.desc.clone(),
                                    skills: argus_pre::skills::load_skills(sdir),
                                    seed: Vec::new(),
                                    plugins: argus_pre::plugins::load(sdir),
                                    state_dir: Some(sdir.to_path_buf()),
                                    mandate: None,
                                };
                                let out = loop_.run(&t.desc, &mut mem);
                                mem.save(&tdir);
                                let mut ts = load_tasks(sdir);
                                if let Some(tt) = ts.iter_mut().find(|x| x.id == t.id) {
                                    tt.next_due = now_sec() + t.every_sec;
                                    tt.last_run = Some(format!(
                                        "{} · {}",
                                        chrono::Local::now().to_rfc3339(),
                                        out.chars().take(120).collect::<String>()
                                    ));
                                }
                                save_tasks(sdir, &ts);
                                println!(
                                    "#{}: {}",
                                    t.id,
                                    out.chars().take(200).collect::<String>()
                                );
                            }
                        }
                    }
                    "d" => {
                        if let Ok(i) = arg.parse::<usize>() {
                            if let Some(t) = tasks.get(i) {
                                let mut ts = load_tasks(sdir);
                                ts.retain(|x| x.id != t.id);
                                save_tasks(sdir, &ts);
                                println!("✅ задача #{} удалена", t.id);
                            }
                        }
                    }
                    "a" => {
                        if !arg.is_empty() {
                            let mut ts = load_tasks(sdir);
                            let id = ts.iter().map(|x| x.id).max().unwrap_or(0) + 1;
                            ts.push(TaskItem {
                                id,
                                desc: arg,
                                every_sec: 3600,
                                next_due: now_sec() + 3600,
                                last_run: None,
                                model: "qwen2.5:7b".into(),
                            });
                            save_tasks(sdir, &ts);
                            println!("✅ задача #{id} создана");
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => println!("task add | list | run | tui"),
    }
}

/// Выполняет просроченные задачи (общее для task run и планировщика serve).
fn run_due_tasks(sdir: &std::path::Path, verbose: bool) -> usize {
    let now = now_sec();
    let tasks = load_tasks(sdir);
    let due: Vec<TaskItem> = tasks
        .iter()
        .filter(|t| t.next_due <= now)
        .cloned()
        .collect();
    if due.is_empty() {
        return 0;
    }
    let total = due.len();
    // V3.3: multi-session concurrency — due-задачи выполняются параллельно (пул потоков)
    let results: Vec<(u64, String)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in &due {
            let t = t.clone();
            let sdir = sdir.to_path_buf();
            handles.push(scope.spawn(move || {
                let tdir = sdir.join(format!("task-{}", t.id));
                std::fs::create_dir_all(&tdir).ok();
                let mut mem = Memory::load(&tdir);
                let llm = LlmModel::new(LlmConfig {
                    model: t.model.clone(),
                    ..Default::default()
                });
                let mut loop_ = AgentLoop {
                    model: &llm,
                    mode: Mode::Default,
                    cfg: LoopConfig::default(),
                    approve: true,
                    cwd: tdir.display().to_string(),
                    last_transcript: Vec::new(),
                    goal: t.desc.clone(),
                    skills: argus_pre::skills::load_skills(&sdir),
                    seed: Vec::new(),
                    plugins: argus_pre::plugins::load(&sdir),
                    state_dir: Some(sdir.to_path_buf()),
                    mandate: None,
                };
                let out = loop_.run(&t.desc, &mut mem);
                mem.save(&tdir);
                (t.id, out)
            }));
        }
        // дождаться всех (не ограничиваем пул — задач обычно немного; при необходимости — батчами)
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or((0, "(воркер упал)".into())))
            .collect::<Vec<_>>()
    });
    // обновить next_due/last_run
    let mut tasks = load_tasks(sdir);
    for (id, out) in &results {
        if let Some(t) = tasks.iter_mut().find(|t| t.id == *id) {
            t.next_due = now + t.every_sec;
            t.last_run = Some(format!(
                "{} · {}",
                chrono::Local::now().to_rfc3339(),
                out.chars().take(120).collect::<String>()
            ));
        }
        if verbose {
            println!("#{id}: {}", out.chars().take(200).collect::<String>());
        }
    }
    save_tasks(sdir, &tasks);
    // P8d: enqueue из task-раннера — durable отчёт о каждой выполненной задаче
    if !results.is_empty() {
        let mut q = argus_pre::effects::EffectsQueue::load(sdir);
        for (id, out) in &results {
            q.enqueue(
                argus_pre::effects::EffectKind::Report,
                &format!("task:{id}"),
                serde_json::json!({
                    "task_id": id,
                    "result": out.chars().take(500).collect::<String>(),
                    "at": chrono::Local::now().to_rfc3339(),
                }),
            );
        }
        q.save(sdir);
    }
    total
}

/// Skill hub + планировщик задач (V2.1): hub в фоне, каждые 30с — просроченные задачи.
fn cmd_serve(args: &[String], sdir: &std::path::Path) {
    let port = flag(args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8797);
    let tick = flag(args, "--tick")
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // фоновый поток: skill hub
    let sdir2 = sdir.to_path_buf();
    std::thread::spawn(move || {
        let _ = argus_pre::skills::serve(&sdir2, port);
    });
    // основной поток: планировщик задач + периодический тик effects (P8d)
    println!(
        "🟢 daemon: skill hub :{port} + задачи каждые {tick}с + effects тик (Ctrl+C для выхода)"
    );
    loop {
        let n = run_due_tasks(sdir, true);
        if n > 0 {
            println!("[daemon] выполнено задач: {n}");
        }
        // P8d: тик effects queue (watchdog + process_due) в serve-демоне
        let mut q = argus_pre::effects::EffectsQueue::load(sdir);
        argus_pre::effects::register_default_handlers(&mut q, false);
        let revived = q.watchdog();
        let (done, errors) = q.process_due(sdir);
        if done > 0 || errors > 0 {
            println!("[daemon] effects: done={done} errors={errors} watchdog={revived}");
        }
        q.save(sdir);
        std::thread::sleep(std::time::Duration::from_secs(tick));
    }
}

/// MCP (V3.6): list / add / remove / tools <server> / call <server> <tool> <json-args>.
fn cmd_mcp(args: &[String], sdir: &std::path::Path) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    match sub {
        "add" => {
            let name = flag(args, "--name").unwrap_or_default();
            let command = flag(args, "--cmd").unwrap_or_default();
            let a: Vec<String> = flag(args, "--args")
                .map(|x| x.split_whitespace().map(String::from).collect())
                .unwrap_or_default();
            match argus_pre::mcp::add_server(sdir, &name, &command, &a) {
                Ok(_) => println!("✅ MCP-сервер '{name}' добавлен"),
                Err(e) => println!("❌ {e}"),
            }
        }
        "remove" => {
            let name = args.get(3).map(String::as_str).unwrap_or("");
            match argus_pre::mcp::remove_server(sdir, name) {
                Ok(_) => println!("✅ MCP-сервер '{name}' удалён"),
                Err(e) => println!("❌ {e}"),
            }
        }
        "tools" => {
            let name = args.get(3).map(String::as_str).unwrap_or("");
            let servers = argus_pre::mcp::load_servers(sdir);
            let Some(srv) = servers.iter().find(|s| s.name == name).cloned() else {
                println!("сервер '{name}' не найден");
                return;
            };
            match argus_pre::mcp::McpClient::connect(&srv) {
                Ok(mut c) => match c.tools() {
                    Ok(tools) => {
                        for t in &tools {
                            println!(
                                "- {}: {}",
                                t.name,
                                t.description.chars().take(80).collect::<String>()
                            );
                        }
                    }
                    Err(e) => println!("❌ {e}"),
                },
                Err(e) => println!("❌ {e}"),
            }
        }
        "call" => {
            let name = args.get(3).map(String::as_str).unwrap_or("");
            let tool = args.get(4).map(String::as_str).unwrap_or("");
            let j = args.get(5).cloned().unwrap_or_else(|| "{}".into());
            let servers = argus_pre::mcp::load_servers(sdir);
            let Some(srv) = servers.iter().find(|s| s.name == name).cloned() else {
                println!("сервер '{name}' не найден");
                return;
            };
            let jv: serde_json::Value =
                serde_json::from_str(&j).unwrap_or(serde_json::Value::Object(Default::default()));
            match argus_pre::mcp::McpClient::connect(&srv) {
                Ok(mut c) => match c.call_tool(tool, &jv) {
                    Ok(out) => println!("{out}"),
                    Err(e) => println!("❌ {e}"),
                },
                Err(e) => println!("❌ {e}"),
            }
        }
        _ => {
            let servers = argus_pre::mcp::load_servers(sdir);
            if servers.is_empty() {
                println!("(MCP-серверов нет — mcp add --name X --cmd 'cmd' [--args 'a b'])");
            }
            for s in &servers {
                println!("- {}: {} {:?}", s.name, s.command, s.args);
            }
        }
    }
}

/// Plugin hooks (V2.3): add/list/remove; события before_step|after_step|on_done|on_error.
fn cmd_plugin(args: &[String], sdir: &std::path::Path) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    match sub {
        "add" => {
            let event = flag(args, "--event").unwrap_or_default();
            let cmd = flag(args, "--cmd").unwrap_or_default();
            let interval = flag(args, "--interval").and_then(|s| s.parse::<u64>().ok());
            match argus_pre::plugins::add(sdir, &event, &cmd, interval) {
                Ok(id) => println!("✅ хук #{id} [{event}] добавился: {cmd}"),
                Err(e) => println!("❌ {e}"),
            }
        }
        "remove" => {
            let id = args.get(3).and_then(|s| s.parse().ok());
            match id {
                Some(i) => match argus_pre::plugins::remove(sdir, i) {
                    Ok(_) => println!("✅ хук #{i} удалён"),
                    Err(e) => println!("❌ {e}"),
                },
                None => println!("использование: plugin remove <id>"),
            }
        }
        _ => {
            let plugins = argus_pre::plugins::load(sdir);
            if plugins.is_empty() {
                println!("(хуков нет — plugin add --event on_done --cmd \"…\")");
            }
            for p in &plugins {
                println!("- #{} [{}] {}", p.id, p.event, p.command);
            }
            println!("\nсобытия: {}", argus_pre::plugins::EVENTS.join(" | "));
            println!("env: ARGUS_GOAL ARGUS_STEP ARGUS_TOOLS ARGUS_SUMMARY");
        }
    }
}

/// P8d: durable side-effect очередь.
/// sub: enqueue | list | retry | process | health
fn cmd_effects(args: &[String], sdir: &std::path::Path) {
    use argus_pre::effects::{EffectKind, EffectsQueue};
    let sub = args.get(2).map(String::as_str).unwrap_or("list");

    match sub {
        "enqueue" => {
            // effects enqueue <kind> <key> [--message "…"] [--command "…"] [--url …] [--body 'json']
            let kind = args.get(3).and_then(|s| EffectKind::from_name(s)).unwrap_or(EffectKind::Notify);
            let key = args.get(4).cloned().unwrap_or_else(|| format!("{}", std::process::id()));
            let mut payload = serde_json::json!({});
            if let Some(m) = flag(args, "--message") {
                payload["message"] = serde_json::json!(m);
            }
            if let Some(c) = flag(args, "--command") {
                payload["command"] = serde_json::json!(c);
            }
            if let Some(u) = flag(args, "--url") {
                payload["url"] = serde_json::json!(u);
            }
            if let Some(b) = flag(args, "--body") {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                    payload["body"] = v;
                }
            }
            if let Some(m) = flag(args, "--payload") {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&m) {
                    payload = v;
                }
            }
            let mut q = EffectsQueue::load(sdir);
            // P8g: уровень внимания (--attention info|action|blocker), по умолч. action
            let att = args
                .iter()
                .position(|a| a == "--attention")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| argus_pre::effects::Attention::from_name(s))
                .unwrap_or(argus_pre::effects::Attention::Action);
            let id = q.enqueue_with_attention(kind, &key, payload, att);
            q.save(sdir);
            if id == 0 {
                println!("🔇 [{}] key='{key}' уже доставлен (info — тихо по умолчанию)", kind.name());
            } else {
                println!("✅ эффект #{id} [{}] attention={} key='{key}' — enqueued", kind.name(), att.name());
            }
        }
        "attention" => {
            // P8g: «что ждёт человека» — Action/Blocker pending + failed
            let q = EffectsQueue::load(sdir);
            let block = q.attention_block();
            if block.is_empty() {
                println!("(ничего не ждёт человека)");
            } else {
                print!("{}", block);
            }
        }
        "list" => {
            let q = EffectsQueue::load(sdir);
            if q.effects.is_empty() {
                println!("(очередь пуста — effects enqueue <kind> <key> …)");
            }
            for e in &q.effects {
                println!(
                    "#{} [{}] {:8} att={} key={} attempts={}/{} err={}",
                    e.id,
                    e.kind.name(),
                    e.status.name(),
                    e.attention.name(),
                    e.key,
                    e.attempts,
                    e.max_attempts,
                    e.last_error.as_deref().unwrap_or("-")
                );
            }
        }
        "retry" => {
            let id = args.get(3).and_then(|s| s.parse().ok());
            match id {
                Some(i) => {
                    let mut q = EffectsQueue::load(sdir);
                    match q.retry(i) {
                        Ok(()) => {
                            q.save(sdir);
                            println!("✅ эффект #{i} — снова pending");
                        }
                        Err(e) => println!("❌ {e}"),
                    }
                }
                None => println!("использование: effects retry <id>"),
            }
        }
        "process" => {
            let mut q = EffectsQueue::load(sdir);
            argus_pre::effects::register_default_handlers(&mut q, has(args, "--allow-command"));
            let revived = q.watchdog();
            let (done, errors) = q.process_due(sdir);
            q.save(sdir);
            let (p, r, d, f) = q.status_counts();
            println!(
                "⚙️ watchdog вернул {revived} зависших; выполнено {done}, ошибок {errors} | pending={p} running={r} done={d} failed={f}"
            );
        }
        "health" => {
            let mut q = EffectsQueue::load(sdir);
            q.refresh_health();
            for k in EffectKind::ALL {
                let h = q.health.kinds.get(k.name());
                let line = match h {
                    Some(h) => format!(
                        "{}: total={} pending={} done={} failed={} healthy={}{}",
                        k.name(),
                        h.total,
                        h.pending,
                        h.done,
                        h.failed,
                        h.healthy,
                        h.last_error
                            .as_deref()
                            .map(|e| format!(" last_err={e}"))
                            .unwrap_or_default()
                    ),
                    None => format!("{}: (нет данных)", k.name()),
                };
                println!("{line}");
            }
        }
        _ => println!(
            "effects: enqueue <kind> <key> [--message/--command/--url/--body/--payload/--attention info|action|blocker] | list | attention | retry <id> | process [--allow-command] | health\nвиды: notify report external command outbox internal"
        ),
    }
}

/// P8a: ticket list/show — единицы авторизованной работы.
fn cmd_ticket(args: &[String], sdir: &std::path::Path) {
    use argus_pre::tickets::{TicketStatus, TicketStore};
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    let store = TicketStore::load(sdir);
    match sub {
        "list" => {
            if store.tickets.is_empty() {
                println!("(tickets нет — run создаёт Ticket через gate)");
            }
            for t in &store.tickets {
                let st = if t.status == TicketStatus::Open {
                    "🟡 open"
                } else {
                    "✅ ended"
                };
                println!(
                    "#{:<4} {} [{}] scope={} rules={} origin={} — {}",
                    t.id,
                    st,
                    t.mandate.strength.name(),
                    t.mandate.scope,
                    t.mandate.matched_rules.join(","),
                    t.mandate.origin,
                    t.goal.chars().take(60).collect::<String>()
                );
            }
        }
        "show" => {
            let id = args.get(3).and_then(|s| s.parse().ok());
            match id.and_then(|i| store.get(i)) {
                Some(t) => {
                    println!("Ticket #{}", t.id);
                    println!("  goal: {}", t.goal);
                    println!("  статус: {:?}", t.status);
                    println!(
                        "  mandate: policy={} strength={} scope={} rules={}",
                        t.mandate.policy_version,
                        t.mandate.strength.name(),
                        t.mandate.scope,
                        t.mandate.matched_rules.join(",")
                    );
                    println!("  origin: {}", t.mandate.origin);
                    println!("  created: {}", t.created_at);
                    if let Some(o) = &t.outcome {
                        println!("  outcome: {}", o.chars().take(300).collect::<String>());
                    }
                    if let Some(e) = &t.ended_at {
                        println!("  ended: {e}");
                    }
                }
                None => println!("ticket #{:?} не найден", id),
            }
        }
        _ => println!("ticket: list | show <id>"),
    }
}

/// P8a: policy show/set — версионированные правила допуска.
fn cmd_policy(args: &[String], sdir: &std::path::Path) {
    use argus_pre::tickets::{AcceptMode, Policy};
    let sub = args.get(2).map(String::as_str).unwrap_or("show");
    match sub {
        "show" => {
            let p = Policy::load_policy(sdir);
            println!("Policy {}", p.version);
            println!("  accept mode: {} (off/observe/auto)", p.accept_mode.name());
            for r in &p.rules {
                println!(
                    "  rule {}: pattern={} strength={} scope={} {}",
                    r.id,
                    r.pattern,
                    r.strength.name(),
                    r.scope,
                    if r.enabled { "✅" } else { "⛔" }
                );
            }
            println!("  updated: {}", p.updated_at);
        }
        "set" => {
            // policy set --mode observe|off|auto  |  policy set --rule <id> --enable|--disable
            let mut p = Policy::load_policy(sdir);
            if let Some(m) = flag(args, "--mode") {
                match AcceptMode::from_name(&m) {
                    Some(mode) => {
                        p.accept_mode = mode;
                        println!("✅ accept mode → {}", mode.name());
                    }
                    None => {
                        println!("❌ режим '{m}' не из off/observe/auto");
                        return;
                    }
                }
            }
            if let Some(rid) = flag(args, "--rule") {
                let enable = has(args, "--enable");
                let disable = has(args, "--disable");
                let Some(r) = p.rules.iter_mut().find(|r| r.id == rid) else {
                    println!("❌ правило '{rid}' не найдено");
                    return;
                };
                if enable {
                    r.enabled = true;
                } else if disable {
                    r.enabled = false;
                }
                println!(
                    "✅ правило '{rid}' → {}",
                    if enable {
                        "включено"
                    } else {
                        "отключено"
                    }
                );
            }
            p.recompute_version();
            p.updated_at = chrono::Local::now().to_rfc3339();
            Policy::save_policy(sdir, &p);
            println!("💾 Policy пересохранена: {}", p.version);
            // P8c: доставка смены Policy в живые Tickets (уведомление, НЕ отмена)
            let store = argus_pre::tickets::TicketStore::load(sdir);
            let stale = store.open_stale_mandates(&p);
            if stale.is_empty() {
                println!("ℹ️ открытых tickets с устаревшим мандатом нет");
            } else {
                println!(
                    "🔔 смена Policy затронула {} открытых ticket(ов):",
                    stale.len()
                );
                for (id, goal) in &stale {
                    println!("   - #{id}: {}", goal.chars().take(60).collect::<String>());
                }
                // durable-уведомление через effects queue (переживёт рестарт, retry+backoff)
                let mut q = argus_pre::effects::EffectsQueue::load(sdir);
                let ids: Vec<String> = stale.iter().map(|(id, _)| id.to_string()).collect();
                q.enqueue(
                    argus_pre::effects::EffectKind::Notify,
                    &format!("policy-changed:{}", p.version),
                    serde_json::json!({
                        "message": format!(
                            "Policy сменилась → {}; открытые tickets с устаревшим мандатом: {}",
                            p.version,
                            ids.join(", ")
                        ),
                        "policy_version": p.version,
                        "affected_tickets": ids,
                    }),
                );
                q.save(sdir);
                println!(
                    "📨 notify-эффект поставлен в очередь effects (key=policy-changed:{})",
                    p.version
                );
            }
        }
        _ => {
            println!("policy: show | set --mode off|observe|auto [--rule <id> --enable|--disable]")
        }
    }
}

fn cmd_session(args: &[String], sdir: &std::path::Path) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    let mut sessions = list_sessions(sdir);
    match sub {
        "list" => {
            if sessions.is_empty() {
                println!("(сессий нет)");
            }
            for s in &sessions {
                println!(
                    "- {} · шагов {} · {}\n    goal: {}",
                    s.id,
                    s.transcript.len(),
                    s.updated,
                    s.goal.chars().take(70).collect::<String>()
                );
            }
        }
        "show" | "tail" => {
            let id = args.get(3).map(String::as_str).unwrap_or("");
            let Some(s) = load_session(sdir, id) else {
                println!("сессия '{id}' не найдена");
                return;
            };
            println!("╔══ сессия {} ══\ngoal: {}\n", s.id, s.goal);
            let n: usize = flag(args, "--n")
                .and_then(|x| x.parse().ok())
                .unwrap_or(usize::MAX);
            let steps: Vec<&String> = if sub == "tail" {
                s.transcript.iter().rev().take(n).collect()
            } else {
                s.transcript.iter().collect()
            };
            for t in steps.into_iter().rev() {
                println!("  {}", t.chars().take(300).collect::<String>());
            }
        }
        "search" => {
            let q = args.get(3).map(String::as_str).unwrap_or("");
            if q.is_empty() {
                println!("использование: session search \"текст\" [--n N]");
                return;
            }
            let n = flag(args, "--n").and_then(|x| x.parse().ok()).unwrap_or(5);
            let ql = q.to_lowercase();
            // V2.8: ранжирование — score = совпадения в транскрипте×10 + goal(5) + свежесть
            let mut scored: Vec<(i64, &Session)> = sessions
                .iter()
                .filter_map(|s| {
                    let hits = s
                        .transcript
                        .iter()
                        .filter(|t| t.to_lowercase().contains(&ql))
                        .count() as i64;
                    let goal_hit = if s.goal.to_lowercase().contains(&ql) {
                        5
                    } else {
                        0
                    };
                    if hits + goal_hit == 0 {
                        None
                    } else {
                        Some((hits * 10 + goal_hit + s.updated.len() as i64, s))
                    }
                })
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let mut found = 0usize;
            for (_, s) in scored.iter().take(n) {
                let ctx = s.transcript.iter().find(|t| t.to_lowercase().contains(&ql));
                println!(
                    "- {} · {}\n    goal: {}",
                    s.id,
                    s.updated,
                    s.goal.chars().take(70).collect::<String>()
                );
                if let Some(c) = ctx {
                    println!("    ↳ {}", c.chars().take(160).collect::<String>());
                }
                found += 1;
            }
            if found == 0 {
                println!("ничего не найдено по '{q}'");
            }
        }
        "dedup" => {
            // V3.4: удалить дубликаты (goal+transcript) и пустые сессии
            let all = list_sessions(sdir);
            let mut seen: std::collections::HashSet<(String, Vec<String>)> =
                std::collections::HashSet::new();
            let mut removed = 0usize;
            for s in &all {
                if s.transcript.is_empty() || !seen.insert((s.goal.clone(), s.transcript.clone())) {
                    let _ = std::fs::remove_file(sessions_dir(sdir).join(format!("{}.json", s.id)));
                    removed += 1;
                }
            }
            println!(
                "✅ удалено дублей/пустых: {removed} (осталось: {})",
                all.len() - removed
            );
        }
        "rm" => {
            let id = args.get(3).map(String::as_str).unwrap_or("");
            let p = sessions_dir(sdir).join(format!("{id}.json"));
            if std::fs::remove_file(&p).is_ok() {
                println!("✅ сессия {id} удалена");
            } else {
                println!("❌ не удалено (нет такой сессии)");
            }
        }
        "tui" => {
            // интерактивный браузер: номер — просмотр; t <i> — tail; d <i> — удалить; q — выход
            use std::io::Write;
            loop {
                sessions = list_sessions(sdir);
                if sessions.is_empty() {
                    println!("(сессий нет)");
                    return;
                }
                println!("\n══ Sessions TUI ══ (номер — показать · t <i> tail · d <i> удалить · q выход)");
                for (i, s) in sessions.iter().enumerate() {
                    println!(
                        "  [{i}] {} · шагов {} · {}",
                        s.goal.chars().take(50).collect::<String>(),
                        s.transcript.len(),
                        s.id.chars().take(8).collect::<String>()
                    );
                }
                print!("> ");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line).is_err() {
                    return;
                }
                let line = line.trim().to_string();
                if line == "q" {
                    return;
                }
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                let cmd = parts[0];
                let arg = parts.get(1).unwrap_or(&"").to_string();
                let show = |s: &Session, n: Option<usize>| {
                    println!("\n══ {} ══\ngoal: {}", s.id, s.goal);
                    let steps: Vec<&String> = match n {
                        Some(x) => s.transcript.iter().rev().take(x).collect(),
                        None => s.transcript.iter().collect(),
                    };
                    for t in steps.into_iter().rev() {
                        println!("  {}", t.chars().take(300).collect::<String>());
                    }
                };
                match cmd {
                    "t" => {
                        if let Ok(i) = arg.parse::<usize>() {
                            if let Some(s) = sessions.get(i) {
                                show(s, Some(5));
                            }
                        }
                    }
                    "d" => {
                        if let Ok(i) = arg.parse::<usize>() {
                            if let Some(s) = sessions.get(i) {
                                let p = sessions_dir(sdir).join(format!("{}.json", s.id));
                                let _ = std::fs::remove_file(&p);
                                println!(
                                    "✅ удалена: {}",
                                    s.id.chars().take(8).collect::<String>()
                                );
                            }
                        }
                    }
                    _ => {
                        if let Ok(i) = line.parse::<usize>() {
                            if let Some(s) = sessions.get(i) {
                                show(s, None);
                            }
                        } else if let Some(s) = sessions.iter().find(|x| x.id.starts_with(&line)) {
                            show(s, None);
                        }
                    }
                }
            }
        }
        _ => println!("session list | show <id> | tail <id> [--n N] | search <q> | rm <id> | tui"),
    }
}

fn list_sessions(sdir: &std::path::Path) -> Vec<Session> {
    let dir = sessions_dir(sdir);
    let mut sessions: Vec<Session> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                .filter_map(|e| std::fs::read_to_string(e.path()).ok())
                .filter_map(|s| serde_json::from_str::<Session>(&s).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    sessions.sort_by(|a, b| b.updated.cmp(&a.updated));
    sessions
}

fn parse_mode(args: &[String]) -> Mode {
    match flag(args, "--mode").as_deref() {
        Some("plan") => Mode::Plan,
        Some("auto") => Mode::Auto,
        Some("bypass") => Mode::Bypass,
        _ => Mode::Default,
    }
}

/// Self-contained бенчмарк: N простых задач, замер accuracy + wall-time.
/// GAIA-lite eval: читает задачи из markdown (evals/*.md), прогоняет, считает accuracy.
fn cmd_eval(args: &[String], sdir: &std::path::Path) {
    use std::time::Instant;

    let model = flag(args, "--model").unwrap_or_else(|| "qwen2.5:7b".into());
    let url = flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:11434".into());
    let evals_arg = flag(args, "--evals").unwrap_or_else(|| "evals/gaia_lite.md".into());
    // V2.9: multi-file — список через запятую ИЛИ директория (все *.md)
    let files = resolve_eval_files(&evals_arg);
    if files.is_empty() {
        eprintln!("eval: не найдено файлов по '{evals_arg}'");
        std::process::exit(1);
    }
    let mut all: Vec<(String, EvalTask)> = Vec::new();
    for f in &files {
        for t in parse_evals(f) {
            all.push((f.display().to_string(), t));
        }
    }
    if all.is_empty() {
        eprintln!("eval: не найдено задач в файлах");
        std::process::exit(1);
    }
    let cfg = LlmConfig {
        base_url: url.clone(),
        model: model.clone(),
        ..Default::default()
    };

    println!(
        "\n═══ ARGUS-OS1 eval · {model} · {} файлов · {} задач ═══\n",
        files.len(),
        all.len()
    );
    let mut passed = 0usize;
    let mut total_time = std::time::Duration::ZERO;
    let mut rows: Vec<(String, String, bool, String)> = Vec::new();

    for (i, (fname, t)) in all.iter().enumerate() {
        let eval_dir = sdir.join(format!("eval/task{i}"));
        std::fs::create_dir_all(&eval_dir).ok();
        let mut mem = Memory::load(&eval_dir);
        let llm = LlmModel::new(cfg.clone());
        let mut loop_ = AgentLoop {
            model: &llm,
            mode: Mode::Default,
            cfg: LoopConfig::default(),
            approve: true, // eval-задачи включают shell — одобряем автоматически
            cwd: eval_dir.display().to_string(),
            last_transcript: Vec::new(),
            goal: t.task.clone(),
            skills: Vec::new(),
            seed: Vec::new(),
            plugins: Vec::new(),
            state_dir: Some(sdir.to_path_buf()),
            mandate: None,
        };
        let t0 = Instant::now();
        let out = loop_.run(&t.task, &mut mem);
        mem.save(&eval_dir);
        let dt = t0.elapsed();
        total_time += dt;

        let ok = check_eval(t, &out, &mem, &eval_dir);
        if ok {
            passed += 1;
        }
        let base = std::path::Path::new(fname)
            .file_name()
            .map(|x| x.to_string_lossy().to_string())
            .unwrap_or_default();
        rows.push((
            base,
            t.name.clone(),
            ok,
            format!("{:.1}s", dt.as_secs_f64()),
        ));
    }

    let mut by_file: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    for (fname, _, ok, _) in &rows {
        let e = by_file.entry(fname.clone()).or_insert((0, 0));
        e.1 += 1;
        if *ok {
            e.0 += 1;
        }
    }
    for (fname, (p, t)) in &by_file {
        println!("─ {fname}: {p}/{t}");
    }
    println!("─ Задачи ─");
    for (_, name, ok, tm) in &rows {
        println!("  {} {} · {}", if *ok { "✅" } else { "❌" }, name, tm);
    }
    let total = all.len();
    let acc = passed as f64 / total as f64 * 100.0;
    let avg = total_time.as_secs_f64() / total as f64;
    println!("\nAccuracy: {passed}/{total} = {acc:.1}% · Avg wall: {avg:.1}s/task");
}

/// V2.9: резолв --evals (список через запятую | директория | файл).
fn resolve_eval_files(arg: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for part in arg.split(',') {
        let p = std::path::PathBuf::from(part.trim());
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(&p)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|x| x.extension().map(|e| e == "md").unwrap_or(false))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            v.sort();
            out.extend(v);
        } else if p.exists() {
            out.push(p);
        }
    }
    out
}

struct EvalTask {
    name: String,
    task: String,
    answer: String,
    ctype: String,
}

/// Парсер задач: `## name` + `task:` / `answer:` / `type:`.
fn parse_evals(path: &std::path::Path) -> Vec<EvalTask> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut cur: Option<EvalTask> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix("## ") {
            if let Some(t) = cur.take() {
                out.push(t);
            }
            cur = Some(EvalTask {
                name: name.to_string(),
                task: String::new(),
                answer: String::new(),
                ctype: String::new(),
            });
        } else if let Some(t) = cur.as_mut() {
            if let Some(v) = line.strip_prefix("task: ") {
                t.task = v.to_string();
            } else if let Some(v) = line.strip_prefix("answer: ") {
                t.answer = v.to_string();
            } else if let Some(v) = line.strip_prefix("type: ") {
                t.ctype = v.trim().to_string();
            }
        }
    }
    if let Some(t) = cur {
        out.push(t);
    }
    out.into_iter().filter(|t| !t.task.is_empty()).collect()
}

fn check_eval(t: &EvalTask, out: &str, mem: &Memory, dir: &std::path::Path) -> bool {
    match t.ctype.as_str() {
        "contains" => out.contains(&t.answer),
        "file" => dir.join(&t.answer).exists(),
        "file_contains" => {
            let parts: Vec<&str> = t.answer.split(',').collect();
            let file = parts.first().unwrap_or(&"").trim();
            let needle = parts.get(1).unwrap_or(&"").trim();
            std::fs::read_to_string(dir.join(file))
                .map(|c| c.contains(needle))
                .unwrap_or(false)
        }
        "memory" => mem.profile.facts.contains_key(&t.answer),
        _ => false,
    }
}

fn cmd_bench(args: &[String], sdir: &std::path::Path) {
    use std::time::Instant;

    let model = flag(args, "--model").unwrap_or_else(|| "qwen2.5:7b".into());
    let url = flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:11434".into());
    let cfg = LlmConfig {
        base_url: url.clone(),
        model: model.clone(),
        ..Default::default()
    };

    struct Task {
        name: &'static str,
        goal: &'static str,
        check: (&'static str, &'static str),
    }
    let tasks = [
        Task {
            name: "T1 remember-fact",
            goal: "Запомни: bench_marker=true и подтверди",
            check: ("profile", "bench_marker"),
        },
        Task {
            name: "T2 read-file",
            goal: "Прочитай файл /etc/hostname и назови содержимое",
            check: ("out", "hostname"),
        },
        Task {
            name: "T3 remember-note",
            goal: "Запомни заметку: бенчмарк ARGUS прошёл успешно",
            check: ("notes", "бенчмарк"),
        },
    ];

    println!("\n═══ ARGUS-OS1 bench · {model} @ {url} ═══\n");
    let mut passed = 0;
    let mut total_time = std::time::Duration::ZERO;
    let mut rows: Vec<(String, bool, String)> = Vec::new();

    for (i, t) in tasks.iter().enumerate() {
        let bench_dir = sdir.join(format!("bench/task{i}"));
        std::fs::create_dir_all(&bench_dir).ok();
        let mut mem = Memory::load(&bench_dir);
        let llm = LlmModel::new(cfg.clone());
        let mut loop_ = AgentLoop {
            model: &llm,
            mode: Mode::Default,
            cfg: LoopConfig::default(),
            approve: true, // bench-задачи безопасны (read/memory)
            cwd: bench_dir.display().to_string(),
            last_transcript: Vec::new(),
            goal: t.goal.to_string(),
            skills: Vec::new(),
            seed: Vec::new(),
            plugins: Vec::new(),
            state_dir: Some(sdir.to_path_buf()),
            mandate: None,
        };
        let t0 = Instant::now();
        let out = loop_.run(t.goal, &mut mem);
        mem.save(&bench_dir);
        let dt = t0.elapsed();
        total_time += dt;

        let ok = match t.check.0 {
            "profile" => mem.profile.facts.contains_key(t.check.1),
            "notes" => mem.notes.iter().any(|n| n.content.contains(t.check.1)),
            _ => out.contains(t.check.1),
        };
        if ok {
            passed += 1;
        }
        rows.push((t.name.to_string(), ok, format!("{:.1}s", dt.as_secs_f64())));
    }

    println!("─ Результаты ─");
    for (name, ok, tm) in &rows {
        println!("  {} {} · {}", if *ok { "✅" } else { "❌" }, name, tm);
    }
    let total = tasks.len();
    let acc = passed as f64 / total as f64 * 100.0;
    let avg = total_time.as_secs_f64() / total as f64;
    println!("\nAccuracy: {passed}/{total} = {acc:.1}% · Avg wall: {avg:.1}s/task");
}

/// Skill hub-сервер: отдаёт /skills и /skills/<name>.
/// Fusion: планировщик разбивает задачу, пул воркеров выполняет параллельно.
fn cmd_fusion(args: &[String], sdir: &std::path::Path) {
    use std::time::Instant;

    let goal = args.get(2).cloned().unwrap_or_default();
    if goal.is_empty() {
        eprintln!(
            "использование: argus-agent fusion \"задача\" [--workers N] [--model M] [--url U]"
        );
        std::process::exit(1);
    }
    let workers = flag(args, "--workers")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);
    let model = flag(args, "--model").unwrap_or_else(|| "qwen2.5:7b".into());
    let url = flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:11434".into());
    let cfg = LlmConfig {
        base_url: url.clone(),
        model: model.clone(),
        ..Default::default()
    };

    // 1. планировщик
    let planner = LlmModel::new(cfg.clone());
    let subtasks = planner.plan_subtasks(&goal, workers);
    if subtasks.is_empty() {
        println!("планировщик не разбил задачу — выполню целиком");
        cmd_run(args, sdir);
        return;
    }
    println!("\n═══ Fusion · {workers} воркера ═══");
    for (i, s) in subtasks.iter().enumerate() {
        println!("  [{i}] {}", s.chars().take(100).collect::<String>());
    }

    // 2. воркеры (параллельно, каждый со своей памятью)
    let t0 = Instant::now();
    let results: Vec<(usize, String)> = std::thread::scope(|scope| {
        let handles: Vec<_> = subtasks
            .iter()
            .enumerate()
            .map(|(i, sub)| {
                let sub = sub.clone();
                let cfg = cfg.clone();
                let sdir = sdir.to_path_buf();
                scope.spawn(move || {
                    let wdir = sdir.join(format!("fusion/w{i}"));
                    std::fs::create_dir_all(&wdir).ok();
                    let mut mem = Memory::load(&wdir);
                    let llm = LlmModel::new(cfg);
                    let mut loop_ = AgentLoop {
                        model: &llm,
                        mode: Mode::Default,
                        cfg: LoopConfig::default(),
                        approve: true,
                        cwd: wdir.display().to_string(),
                        last_transcript: Vec::new(),
                        goal: sub.clone(),
                        skills: Vec::new(),
                        seed: Vec::new(),
                        plugins: Vec::new(),
                        state_dir: Some(sdir.to_path_buf()),
                        mandate: None,
                    };
                    let out = loop_.run(&sub, &mut mem);
                    mem.save(&wdir);
                    (i, out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or((0, "(воркер упал)".into())))
            .collect()
    });
    let dt = t0.elapsed();

    // 3. сводка
    println!("\n─ Результаты воркеров ─");
    let mut merged = String::new();
    for (i, out) in &results {
        println!("  [{i}] {}", out.chars().take(200).collect::<String>());
        merged.push_str(out);
        merged.push('\n');
    }
    println!(
        "\nFusion завершён за {:.1}s · воркеров: {}",
        dt.as_secs_f64(),
        results.len()
    );
}

fn cmd_skill(args: &[String], sdir: &std::path::Path) {
    let skills = argus_pre::skills::load_skills(sdir);
    match args.get(2).map(String::as_str) {
        Some("list") => {
            for s in &skills {
                println!(
                    "- {} v{} · {}{}",
                    s.name,
                    s.version,
                    s.description,
                    if s.dangerous { " ⚠️" } else { "" }
                );
            }
            if skills.is_empty() {
                println!("(скиллов нет — запусти init)");
            }
        }
        Some("search") => {
            let q = args.get(3).map(String::as_str).unwrap_or("");
            let hits = argus_pre::skills::search_registry(q);
            if hits.is_empty() {
                println!("в реестре ничего по '{q}'");
            }
            for (n, d) in hits {
                println!("- {n}: {d}");
            }
        }
        Some("browse") => {
            // локальный реестр ИЛИ удалённый hub
            match flag(args, "--remote") {
                Some(url) => {
                    let body = reqwest::blocking::get(format!("{}/skills", url.trim_end_matches('/')))
                        .and_then(|r| r.error_for_status())
                        .map_err(|e| {
                            println!("❌ hub недоступен: {e}");
                        })
                        .and_then(|r| r.text().map_err(|e| {
                            println!("❌ чтение: {e}");
                        }))
                        .unwrap_or_default();
                    if !body.is_empty() {
                        if let Ok(list) = serde_json::from_str::<Vec<serde_json::Value>>(&body) {
                            for s in list {
                                let name = s["name"].as_str().unwrap_or("?");
                                let desc = s["description"].as_str().unwrap_or("");
                                println!("- {name}: {desc}");
                            }
                        } else {
                            println!("{body}");
                        }
                    }
                }
                None => {
                    for (n, d) in argus_pre::skills::REGISTRY {
                        println!("- {n}: {d}");
                    }
                    println!("\n(удалённый: skill browse --remote http://host:port)");
                }
            }
        }
        Some("install") => {
            let name = args.get(3).map(String::as_str).unwrap_or("");
            // формат owner/repo[/path] → GitHub-тап; иначе встроенный реестр
            if name.contains('/') && name.matches('/').count() >= 1 && !name.starts_with('@') {
                match argus_pre::skills::install_from_github(sdir, name) {
                    Ok(n) => println!("✅ скилл '{n}' установлен из GitHub ({name})"),
                    Err(e) => println!("❌ {e}"),
                }
            } else {
                match argus_pre::skills::install_from_registry(sdir, name) {
                    Ok(_) => println!("✅ скилл '{name}' установлен из реестра"),
                    Err(e) => println!("❌ {e}"),
                }
            }
        }
        Some("push") => {
            let remote = flag(args, "--remote").unwrap_or_default();
            if remote.is_empty() {
                println!("использование: skill push --remote <url> [--skill <name>|all] [--token <tok>]");
            } else {
                // V3.3: токен из --token или env ARGUS_HUB_TOKEN
                let token = flag(args, "--token").or_else(|| std::env::var("ARGUS_HUB_TOKEN").ok()).unwrap_or_default();
                let skill = flag(args, "--skill").unwrap_or_else(|| "all".into());
                let names: Vec<String> = if skill == "all" {
                    skills.iter().map(|s| s.name.clone()).collect()
                } else {
                    vec![skill]
                };
                for n in names {
                    match argus_pre::skills::push_skill(&remote, sdir, &n, &token) {
                        Ok(r) => println!("{r}"),
                        Err(e) => println!("❌ {e}"),
                    }
                }
            }
        }
        Some("tap") => {
            let sub = args.get(3).map(String::as_str).unwrap_or("list");
            match sub {
                "list" => {
                    let taps = argus_pre::skills::load_taps(sdir);
                    if taps.is_empty() {
                        println!("(тапов нет — skill tap add owner/repo)");
                    }
                    for t in taps {
                        println!("- {t}");
                    }
                }
                "add" => {
                    let spec = args.get(4).map(String::as_str).unwrap_or("");
                    match argus_pre::skills::tap_add(sdir, spec) {
                        Ok(_) => println!("✅ тап '{spec}' добавлен"),
                        Err(e) => println!("❌ {e}"),
                    }
                }
                "remove" => {
                    let spec = args.get(4).map(String::as_str).unwrap_or("");
                    match argus_pre::skills::tap_remove(sdir, spec) {
                        Ok(_) => println!("✅ тап '{spec}' удалён"),
                        Err(e) => println!("❌ {e}"),
                    }
                }
                _ => println!("skill tap list | add <owner/repo> | remove <owner/repo>"),
            }
        }
        Some("show") => match argus_pre::skills::find_skill(
            &skills,
            args.get(3).map(String::as_str).unwrap_or(""),
        ) {
            Some(s) => println!("## {} v{}\n\n{}", s.name, s.version, s.body),
            None => println!("не найден"),
        },
        Some("info") => match argus_pre::skills::find_skill(
            &skills,
            args.get(3).map(String::as_str).unwrap_or(""),
        ) {
            Some(s) => {
                println!("name: {}", s.name);
                println!("version: {}", s.version);
                println!("description: {}", s.description);
                println!("requires_tools: {:?}", s.requires_tools);
                println!("requires_scripts: {:?}", s.requires_scripts);
                println!("dangerous: {}", s.dangerous);
                println!("dir: {}", s.dir.display());
                if !s.body.is_empty() {
                    println!("\n--- body (первые 300) ---\n{}", s.body.chars().take(300).collect::<String>());
                }
            }
            None => println!("не найден"),
        },
        Some("update") => {
            // V2.9: переустановка из реестра (force) — обновление версии
            let name = args.get(3).map(String::as_str).unwrap_or("");
            if name.is_empty() {
                println!("использование: skill update <name>");
            } else {
                let dir = sdir.join("skills").join(name);
                let _ = std::fs::remove_dir_all(&dir);
                match argus_pre::skills::install_from_registry(sdir, name) {
                    Ok(_) => {
                        let sk = argus_pre::skills::load_skills(sdir);
                        let v = sk.iter().find(|x| x.name == name).map(|x| x.version.as_str()).unwrap_or("?");
                        println!("✅ скилл '{name}' обновлён (v{v})");
                    }
                    Err(e) => println!("❌ {e}"),
                }
            }
        },
        Some("import-pi") => {
            // V3.6: импорт скиллов из pi (~/.pi/agent/skills)
            let src = flag(args, "--source")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".pi/agent"));
            let n = argus_pre::skills::import_from_pi(sdir, &src);
            println!("✅ импортировано скиллов из pi: {n} ← {}", src.display());
        }
        Some("recipes") => {
            // V3.2: галерея browser-рецептов (готовые browser.flow сценарии)
            let name = args.get(3).map(String::as_str).unwrap_or("");
            if name.is_empty() {
                println!("═══ Browser-рецепты ═══ (install <recipe> — установить)\n");
                for (n, d) in argus_pre::skills::RECIPES {
                    println!("- {n}: {d}");
                }
                println!(
                    "\n  (установленные: {:?})",
                    skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
                );
            } else {
                match argus_pre::skills::RECIPES.iter().find(|(n, _)| *n == name) {
                    Some((n, d)) => println!("## {n}\n\n{d}\n\nУстановка: skill install {n}"),
                    None => println!("рецепт '{name}' не найден (см. skill recipes)"),
                }
            }
        },
        _ => println!("skill list | search <q> | install <name|owner/repo[/path]> | show <name> | info <name> | update <name> | recipes [name] | tap list|add|remove <owner/repo>"),
    }
}

/// Создаёт стартовые скиллы (docker с скриптом, pdf, obsidian) при init.
fn make_starter_skills(sdir: &std::path::Path) -> usize {
    use std::fs;
    let dir = sdir.join("skills");
    let _ = fs::create_dir_all(&dir);
    let mut n = 0;

    let docker = dir.join("docker");
    if !docker.join("SKILL.md").exists() {
        let _ = fs::create_dir_all(docker.join("scripts"));
        let _ = fs::write(
            docker.join("SKILL.md"),
            "---\nname: docker\ndescription: \"Управление Docker: образы, контейнеры, логи (docker ps, images, logs)\"\nversion: 0.1.0\nrequires_tools: [shell]\nrequires_scripts: [ps.sh]\ndangerous: true\n---\n\nПросмотр Docker-окружения через скрипт `ps.sh` (не удалять ничего без подтверждения).\n",
        );
        let _ = fs::write(
            docker.join("scripts/ps.sh"),
            "#!/bin/sh\ndocker ps -a --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}'\n",
        );
        n += 1;
    }

    for (name, desc) in [
        (
            "pdf",
            "Извлечение текста из PDF локально (pdftotext, если установлен)",
        ),
        (
            "obsidian",
            "Чтение/поиск в Obsidian-ваулте (OBSIDIAN_VAULT_PATH)",
        ),
    ] {
        let d = dir.join(name);
        if !d.join("SKILL.md").exists() {
            let _ = fs::create_dir_all(&d);
            let _ = fs::write(
                d.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: \"{desc}\"\nversion: 0.1.0\nrequires_tools: []\nrequires_scripts: []\ndangerous: false\n---\n\nИнструкция по использованию: см. описание.\n"),
            );
            n += 1;
        }
    }
    n
}

// ───────────────────────────── P9: idle inference ─────────────────────────────
// Спека: _gh/ARGUS-OS1/docs/ARGUS_Internal_Process_Idle_Inference_2026-11-12.md
// M1: леджер + выбор вопроса + запись цикла. Проб нет: инструменты появятся в M2.

fn arg_num(args: &[String], key: &str) -> Option<u64> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn arg_str(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

// ── M4-фикс: защита от двух одновременных циклов (последняя запись затирала состояние) ──
struct LoopLock {
    path: PathBuf,
}

impl LoopLock {
    fn acquire(sdir: &PathBuf) -> Result<LoopLock, String> {
        let path = sdir.join("ledger").join("think.lock");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(prev) = std::fs::read_to_string(&path) {
            let pid: u32 = prev.trim().parse().unwrap_or(0);
            let alive = pid > 0 && std::path::Path::new(&format!("/proc/{pid}")).exists();
            if alive && pid != std::process::id() {
                return Err(format!(
                    "уже запущен другой idle-цикл (pid {pid}); файл {} — остановите его или удалите lock",
                    path.display()
                ));
            }
        }
        std::fs::write(&path, std::process::id().to_string())
            .map_err(|e| format!("lock не создан: {e}"))?;
        Ok(LoopLock { path })
    }
}

impl Drop for LoopLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct ThinkCfg {
    dry: bool,
    seed: u64,
    k: usize,
    limit: usize,
    channel: String,
    offline: bool,
    judge: Option<String>,
    /// перед применением лабораторных данных обязательно спросить консилиум
    require_council: bool,
    corpus: PathBuf,
    qid: Option<String>,
    min_cos: f32,
    quiet: bool,
    budget_cycles: u32,
    budget_tokens: u64,
    interval_secs: u64,
}

fn parse_think_cfg(args: &[String], sdir: &PathBuf) -> ThinkCfg {
    ThinkCfg {
        dry: args.iter().any(|a| a == "--dry-run")
            || !args.iter().any(|a| a == "--once" || a == "--loop"),
        seed: arg_num(args, "--seed").unwrap_or(42),
        k: arg_num(args, "--k").unwrap_or(3) as usize,
        limit: arg_num(args, "--limit").unwrap_or(3) as usize,
        channel: arg_str(args, "--probe").unwrap_or_else(|| "none".to_string()),
        offline: args.iter().any(|a| a == "--offline"),
        judge: arg_str(args, "--judge"),
        require_council: args.iter().any(|a| a == "--require-council"),
        corpus: arg_str(args, "--corpus")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdir.join("corpus")),
        qid: arg_str(args, "--qid"),
        min_cos: arg_str(args, "--min-score")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.20),
        quiet: args.iter().any(|a| a == "--quiet"),
        budget_cycles: arg_num(args, "--budget-cycles")
            .unwrap_or(argus_pre::budget::DEFAULT_MAX_CYCLES_PER_HOUR as u64)
            as u32,
        budget_tokens: arg_num(args, "--budget-tokens")
            .unwrap_or(argus_pre::budget::DEFAULT_MAX_TOKENS_PER_DAY),
        interval_secs: arg_num(args, "--interval-secs").unwrap_or(300),
    }
}

fn select_question(cfg: &ThinkCfg, led: &Ledger) -> Result<ledger::Scored, String> {
    if let Some(id) = &cfg.qid {
        let q = led
            .get(id)
            .ok_or_else(|| format!("нет такого вопроса: {id}"))?;
        return Ok(ledger::Scored {
            id: q.id.clone(),
            text: q.text.clone(),
            score: f64::NAN,
            surprise: (ledger::logit(q.p_true) - ledger::logit(q.p_prev)).abs(),
            staleness_h: 0.0,
            cost: 1.0,
        });
    }
    if cfg.channel == "lab" || cfg.channel == "external" {
        // лабораторный канал: только `lab`-вопросы, детерминированно по score
        let now = chrono::Utc::now();
        let mut best: Option<ledger::Scored> = None;
        for q in &led.questions {
            if q.kind != ledger::QKind::Lab {
                continue;
            }
            if let Some(sc) = ledger::score_question_for(q, now, false) {
                if best.as_ref().map(|b| sc.score > b.score).unwrap_or(true) {
                    best = Some(sc);
                }
            }
        }
        return best.ok_or_else(|| "нет открытых лабораторных вопросов".to_string());
    }
    let lit = cfg.channel == "epmc";
    led.select_next_for(cfg.seed, cfg.k, lit)
        .ok_or_else(|| "нет открытых вопросов, подходящих под канал пробы".to_string())
}

/// Один цикл: выбор вопроса → проба → резолв → запись. Возвращает (qid, note, tokens).
fn run_cycle(
    cfg: &ThinkCfg,
    sdir: &PathBuf,
    led: &mut Ledger,
    cache: &mut Cache,
    fetch: &HttpFetcher,
) -> Result<(String, String, u64), String> {
    let pick = select_question(cfg, led)?;
    let q = probe::build_query_for(&cfg.channel, &pick.text);
    if !cfg.quiet {
        println!("\nвопрос: {} (seed={}) — {}", pick.id, cfg.seed, pick.text);
        println!("  канал: {} · запрос: «{}»", cfg.channel, q);
    }

    // расход токенов LLM-судьи (идёт в дневной бюджет) и его заметка для журнала
    let mut judge_tokens: u64 = 0;
    let mut judge_note: Option<String> = None;
    let (outcome, ids, note, stance_dec) = match cfg.channel.as_str() {
        "epmc" | "corpus" => {
            let found = if cfg.channel == "epmc" {
                EpMcSearcher::new(cfg.offline).search(&q, cfg.limit)
            } else {
                CorpusSearcher::new(&cfg.corpus).search(&q, cfg.limit)
            };
            match found {
                Err(e) => (
                    Outcome::None,
                    vec![],
                    format!("проба не удалась ({}): {e}", cfg.channel),
                    None,
                ),
                Ok(cands) if cands.is_empty() => (
                    Outcome::None,
                    vec![],
                    format!("{}: 0 попаданий по «{q}»", cfg.channel),
                    None,
                ),
                Ok(cands0) => {
                    let total = cands0.len();
                    // M3: семантика (word-level эмбеддинги) вместо грубого совпадения термов
                    let (cands, dropped_v) = if cfg.channel == "epmc" {
                        probe::filter_relevant_scored(&q, cands0, cfg.min_cos)
                    } else {
                        (cands0, vec![])
                    };
                    if !cfg.quiet {
                        for (c, cos) in &dropped_v {
                            println!("    {} отсеян (score={cos:.3}) — {}", c.id, c.title);
                        }
                    }
                    let dropped = dropped_v.len();
                    if cands.is_empty() {
                        (
                            Outcome::None,
                            vec![],
                            format!(
                                "{}: найдено {total}, после семантического фильтра 0 (отсеяно {dropped})",
                                cfg.channel
                            ),
                            None,
                        )
                    } else {
                        let parsed: Vec<verify::Id> = cands
                            .iter()
                            .filter_map(|c| verify::Id::parse(&c.id))
                            .collect();
                        let entries = verify::resolve_many(&parsed, cache, fetch);
                        if !cfg.quiet {
                            for (c, e) in cands.iter().zip(entries.iter()) {
                                let meta = e
                                    .meta
                                    .as_ref()
                                    .map(|m| {
                                        let y = m.year.map(|y| y.to_string()).unwrap_or_default();
                                        format!("{} {} · {}", m.container, y, m.title)
                                    })
                                    .unwrap_or_else(|| e.err.clone().unwrap_or_default());
                                println!("    {} {:?} — {}", c.id, e.status, meta);
                            }
                        }
                        let verdict = verify::to_ledger_verifier(&entries);
                        let resolved: Vec<String> = entries
                            .iter()
                            .filter(|e| e.status == verify::Status::Resolved)
                            .map(|e| e.id.clone())
                            .collect();
                        let bad: Vec<String> = entries
                            .iter()
                            .filter(|e| e.status == verify::Status::NotFound)
                            .map(|e| e.id.clone())
                            .collect();
                        let note = format!(
                            "{}: найдено {total}, релевантных {} (отсеяно {dropped}), резолвнуто {} ({verdict:?})",
                            cfg.channel,
                            cands.len(),
                            resolved.len()
                        );
                        // ── позиция свидетельства (фикс 13 ноя) ──────────────────
                        // Релевантность ≠ подтверждение: без этого шага Resolved давал
                        // 0.500 → 0.500 в каждом цикле (15/15 за ночь). Классификатор
                        // молчит, если нет якорей/маркеров или они конфликтуют;
                        // тогда исход остаётся Evidence и p не двигается.
                        let stance_dec: Option<stance::Decision> =
                            match (verdict, cfg.judge.as_deref()) {
                                (ledger::VerifyResult::Resolved, Some("off")) => None,
                                (
                                    ledger::VerifyResult::Resolved,
                                    None | Some("auto") | Some("lexical"),
                                ) => {
                                    let ds: Vec<stance::Decision> = cands
                                        .iter()
                                        .map(|c| {
                                            stance::classify(
                                                &pick.text,
                                                &c.title,
                                                c.abstract_text.as_deref().unwrap_or(""),
                                            )
                                        })
                                        .collect();
                                    if !cfg.quiet {
                                        for (c, d) in cands.iter().zip(ds.iter()) {
                                            println!(
                                                "    stance {} {} ({:.2}, {}) — {}",
                                                c.id,
                                                d.stance.as_str(),
                                                d.confidence,
                                                d.source,
                                                d.reason
                                            );
                                        }
                                    }
                                    Some(stance::aggregate(&ds))
                                }
                                (ledger::VerifyResult::Resolved, mode @ (Some("hybrid") | Some("llm"))) => {
                                    // лексика считается всегда: она бесплатна и служит опорой
                                    let ds: Vec<stance::Decision> = cands
                                        .iter()
                                        .map(|c| {
                                            stance::classify(
                                                &pick.text,
                                                &c.title,
                                                c.abstract_text.as_deref().unwrap_or(""),
                                            )
                                        })
                                        .collect();
                                    let lexical = stance::aggregate(&ds);
                                    let is_prop = led
                                        .get(&pick.id)
                                        .map(|q| q.form().allows_lr())
                                        .unwrap_or(false);
                                    let want_llm = is_prop
                                        && stance::llm::JudgeConfig::enabled()
                                        && (mode == Some("llm") || !lexical.acts());
                                    if !want_llm {
                                        if !cfg.quiet {
                                            println!(
                                                "    stance lexical {} ({:.2}) — LLM не нужен{}",
                                                lexical.stance.as_str(),
                                                lexical.confidence,
                                                if is_prop { "" } else { " (не претензия)" }
                                            );
                                        }
                                        Some(lexical)
                                    } else {
                                        let mut cache = stance::llm::JudgeCache::load(sdir);
                                        let items: Vec<stance::llm::Item<'_>> = cands
                                            .iter()
                                            .map(|c| stance::llm::Item {
                                                id: c.id.as_str(),
                                                title: c.title.as_str(),
                                                abstract_text: c.abstract_text.as_deref().unwrap_or(""),
                                            })
                                            .collect();
                                        let jcfg = stance::llm::JudgeConfig::from_env();
                                        let out = stance::llm::judge_batch(
                                            &jcfg,
                                            &pick.id,
                                            &pick.text,
                                            &items,
                                            &mut cache,
                                        );
                                        if let Err(e) = cache.save(sdir) {
                                            eprintln!("⚠️ кэш судьи не сохранён: {e}");
                                        }
                                        judge_tokens = out.tokens;
                                        judge_note = Some(out.note.clone());
                                        if !cfg.quiet {
                                            println!("    {}", out.note);
                                            for (c, d) in cands.iter().zip(out.decisions.iter()) {
                                                println!(
                                                    "    judge {} {} ({:.2}) — {}",
                                                    c.id,
                                                    d.stance.as_str(),
                                                    d.confidence,
                                                    d.reason
                                                );
                                            }
                                            if stance::stance_conflict(&out.decisions) {
                                                println!(
                                                    "    ⚠️ свидетельства расходятся — воздержание"
                                                );
                                            }
                                        }
                                        Some(stance::aggregate_with(&lexical, &out.decisions))
                                    }
                                }
                                _ => None,
                            };
                        let note = match &stance_dec {
                            Some(d) => format!(
                                "{note} · stance {}/{}: {}",
                                d.stance.as_str(),
                                d.confidence,
                                d.reason
                            ),
                            None => note,
                        };
                        let note = match &judge_note {
                            Some(j) => format!("{note} · {j}"),
                            None => note,
                        };
                        let outcome = match verdict {
                            ledger::VerifyResult::Fabricated => Outcome::Fabricated,
                            ledger::VerifyResult::Resolved => match cfg.judge.as_deref() {
                                Some("support") => Outcome::Support,
                                Some("contradict") => Outcome::Contradict,
                                _ => match stance_dec.as_ref().filter(|d| d.acts()) {
                                    Some(d) if d.stance == stance::Stance::Support => {
                                        Outcome::Support
                                    }
                                    Some(d) if d.stance == stance::Stance::Contradict => {
                                        Outcome::Contradict
                                    }
                                    _ => Outcome::Evidence,
                                },
                            },
                            ledger::VerifyResult::Unresolved | ledger::VerifyResult::Council => {
                                Outcome::None
                            }
                        };
                        let ids_out = if verdict == ledger::VerifyResult::Fabricated {
                            bad
                        } else {
                            resolved
                        };
                        (outcome, ids_out, note, stance_dec)
                    }
                }
            }
        }
        // ── чужие эксперименты и данные: наборы данных/репозитории сообщества ──
        "external" => {
            let q = probe::build_query_for("epmc", &pick.text);
            let (cands, errors) = thirdparty::gather(&q, thirdparty::PER_SOURCE);
            if !cfg.quiet {
                for e in &errors {
                    println!("    ⚠️ {e}");
                }
                for c in &cands {
                    println!("    {} [{}] {}", c.id, c.kind.as_str(), c.title);
                }
            }
            if cands.is_empty() {
                (
                    Outcome::None,
                    vec![],
                    format!("external: ничего не найдено по «{q}»{}", if errors.is_empty() { String::new() } else { format!(" ({})", errors.join("; ")) }),
                    None,
                )
            } else {
                // судья: есть ли в описании измерение по нашей претензии (с цитатой)
                let mut cache = stance::llm::JudgeCache::load(sdir);
                let items: Vec<stance::llm::Item<'_>> = cands
                    .iter()
                    .map(|c| stance::llm::Item {
                        id: c.id.as_str(),
                        title: c.title.as_str(),
                        abstract_text: c.summary.as_str(),
                    })
                    .collect();
                let jcfg = stance::llm::JudgeConfig::from_env();
                let out = stance::llm::judge_batch(&jcfg, &pick.id, &pick.text, &items, &mut cache);
                let _ = cache.save(sdir);
                if !cfg.quiet {
                    for (c, d) in cands.iter().zip(out.decisions.iter()) {
                        println!(
                            "    judge {} {} ({:.2}) — {}",
                            c.id,
                            d.stance.as_str(),
                            d.confidence,
                            d.reason
                        );
                    }
                }
                let acting: Vec<(usize, &stance::Decision)> = out
                    .decisions
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.acts())
                    .collect();
                match acting.first().map(|(i, d)| (*i, (*d).clone())) {
                    None => (
                        Outcome::None,
                        vec![],
                        format!("external: {} источников, измерения по претензии не найдено", cands.len()),
                        Some(stance::Decision::neutral(
                            "external",
                            "чужие данные не подтвердили и не опровергли претензию",
                        )),
                    ),
                    Some((idx, d)) => {
                        let c = &cands[idx];
                        // источник обязан быть живым
                        if let Err(why) = thirdparty::source_alive(&c.id) {
                            (
                                Outcome::None,
                                vec![],
                                format!("external: {} — {why}", c.id),
                                None,
                            )
                        } else {
                            let ev = thirdparty::ExternalEvidence {
                                qid: pick.id.clone(),
                                verdict: d.stance.as_str().to_string(),
                                value: d.reason.clone(),
                                source: c.id.clone(),
                                title: c.title.clone(),
                                note: format!("источник {}, {}", c.kind.as_str(), c.url),
                                at: Some(chrono::Local::now().format("%Y-%m-%d").to_string()),
                                third_party: true,
                            };
                            if let Err(e) = thirdparty::record(sdir, &ev) {
                                eprintln!("⚠️ запись чужого свидетельства не сохранена: {e}");
                            }
                            let outcome = if d.stance == stance::Stance::Support {
                                Outcome::Support
                            } else {
                                Outcome::Contradict
                            };
                            (
                                outcome,
                                vec![c.id.clone()],
                                format!(
                                    "external/{}: {} — «{}» ({})",
                                    c.kind.as_str(),
                                    d.stance.as_str(),
                                    c.title,
                                    d.reason
                                ),
                                Some(d),
                            )
                        }
                    }
                }
            }
        }
        // ── лабораторный канал (M4): ответ даёт эксперимент, а не литература ──
        "lab" => match lab::load(sdir, &pick.id) {
            None => (
                Outcome::None,
                vec![],
                format!(
                    "lab: данных нет — нужен эксперимент; положить файл {}",
                    lab::path_for(sdir, &pick.id).display()
                ),
                None,
            ),
            Some(ev) => {
                let now = chrono::Utc::now();
                match lab::validate(sdir, &ev, now, lab::DEFAULT_MAX_AGE_DAYS, &pick.text) {
                    Err(why) => (
                        Outcome::None,
                        vec![],
                        format!("lab: {why}"),
                        None,
                    ),
                    Ok(()) => {
                        let stance_wanted = match ev.verdict {
                            lab::LabVerdict::Support => stance::Stance::Support,
                            lab::LabVerdict::Contradict => stance::Stance::Contradict,
                            lab::LabVerdict::Inconclusive => stance::Stance::Neutral,
                        };
                        // решение Джабы 13.11: до применения лабораторных данных — консилиум
                        let mut council_note = String::new();
                        let mut blocked = None;
                        if cfg.require_council && dialogue::CouncilConfig::enabled() {
                            let cc = dialogue::CouncilConfig::from_env();
                            let problem = led
                                .get(&pick.id)
                                .and_then(|q| q.expected.clone())
                                .unwrap_or_default();
                            let ctx = format!(
                                "лабораторные данные: вердикт {}, значение «{}», источник {}, примечание «{}»",
                                ev.verdict.as_str(),
                                ev.value,
                                ev.source,
                                ev.note
                            );
                            let cons = dialogue::consult(&cc, &pick.text, &problem, &ctx);
                            if !cfg.quiet {
                                println!("    {}", cons.reason);
                                for o in &cons.opinions {
                                    println!(
                                        "    council {} {} ({:.2}) — {}",
                                        o.model,
                                        o.stance_str(),
                                        o.confidence,
                                        o.key_point
                                    );
                                }
                            }
                            if cons.acts() && cons.stance != stance_wanted {
                                blocked = Some(format!(
                                    "консилиум против данных: {} — нужен ручной разбор",
                                    cons.reason
                                ));
                            }
                            council_note = format!(" · {}", cons.reason);
                        }
                        match blocked {
                            Some(why) => (
                                Outcome::None,
                                vec![],
                                format!("lab: {why}{council_note}"),
                                None,
                            ),
                            None => {
                                let d = stance::Decision {
                                    stance: stance_wanted,
                                    confidence: 0.95,
                                    source: "lab:file",
                                    reason: format!(
                                        "эксперимент: {} — «{}» ({}){}",
                                        ev.verdict.as_str(),
                                        ev.value,
                                        ev.source,
                                        council_note
                                    ),
                                };
                                let outcome = match ev.verdict {
                                    lab::LabVerdict::Support => Outcome::Support,
                                    lab::LabVerdict::Contradict => Outcome::Contradict,
                                    lab::LabVerdict::Inconclusive => Outcome::None,
                                };
                                (
                                    outcome,
                                    vec![format!("lab:{}", ev.source)],
                                    format!(
                                        "lab: {} ({}), источник {}{council_note}",
                                        ev.verdict.as_str(),
                                        ev.value,
                                        ev.source
                                    ),
                                    Some(d),
                                )
                            }
                        }
                    }
                }
            }
        },
        // ── канал диалога: консилиум бесплатных моделей ──
        "dialogue" => {
            let cc = dialogue::CouncilConfig::from_env();
            if !dialogue::CouncilConfig::enabled() || cc.members.is_empty() {
                (
                    Outcome::None,
                    vec![],
                    "сouncil: канал выключен (ARGUS_COUNCIL=0) или пустой состав".to_string(),
                    None,
                )
            } else {
                let q = led.get(&pick.id).cloned();
                let problem = q
                    .as_ref()
                    .and_then(|q| q.expected.clone())
                    .unwrap_or_default();
                let ctx = format!(
                    "форма вопроса: {}; проб сделано: {}; найдено идентификаторов: {}; известные данные: {}",
                    q.as_ref().map(|q| q.form().as_str()).unwrap_or("?"),
                    q.as_ref().map(|q| q.probes).unwrap_or(0),
                    q.as_ref().map(|q| q.identifiers.len()).unwrap_or(0),
                    q.as_ref()
                        .map(|q| q.identifiers.join(", "))
                        .unwrap_or_default()
                );
                let cons = dialogue::consult(&cc, &pick.text, &problem, &ctx);
                if !cfg.quiet {
                    println!("    {}", cons.reason);
                    for o in &cons.opinions {
                        println!(
                            "    council {} {} ({:.2}) — {} {}",
                            o.model,
                            o.stance_str(),
                            o.confidence,
                            o.key_point,
                            if o.missing.is_empty() {
                                String::new()
                            } else {
                                format!("| не хватает: {}", o.missing)
                            }
                        );
                    }
                }
                let outcome = if cons.acts() {
                    match cons.stance {
                        stance::Stance::Support => Outcome::CouncilSupport,
                        stance::Stance::Contradict => Outcome::CouncilContradict,
                        stance::Stance::Neutral => Outcome::None,
                    }
                } else {
                    Outcome::None
                };
                let d = stance::Decision {
                    stance: cons.stance,
                    confidence: cons.confidence,
                    source: "council",
                    reason: format!("{} [{}]", cons.reason, cons.label()),
                };
                (
                    outcome,
                    vec![],
                    format!(
                        "консилиум {}: {}{}",
                        cons.label(),
                        cons.reason,
                        cons.opinions
                            .iter()
                            .filter(|o| !o.key_point.is_empty())
                            .map(|o| format!("
    • {}: {}", o.model, o.key_point))
                            .collect::<String>()
                    ),
                    Some(d),
                )
            }
        }
        _ => (
            Outcome::None,
            vec![],
            "канал пробы не выбран (--probe epmc|corpus|lab|dialogue)".to_string(),
            None,
        ),
    };

    let cycle = led
        .apply_judged(
            &pick.id,
            outcome,
            &format!("probe:{}", cfg.channel),
            ids,
            &note,
            judge_tokens,
            stance_dec.as_ref(),
        )
        .ok_or_else(|| "вопрос исчез из леджера".to_string())?;
    let _ = led.save(sdir);
    let _ = led.append_cycle(sdir, &cycle);
    if let Err(e) = cache.save() {
        eprintln!("⚠️ кэш verify не сохранён: {e}");
    }
    if !cfg.quiet {
        println!(
            "  цикл: {:.3} → {:.3} ({:?}); probes={}",
            cycle.before,
            cycle.after,
            cycle.verifier,
            led.get(&pick.id).map(|q| q.probes).unwrap_or(0)
        );
        println!("  {note}");
    }
    Ok((pick.id.clone(), note, judge_tokens))
}

fn cmd_think(args: &[String], sdir: &PathBuf) {
    if args.iter().any(|a| a == "--report") {
        print_idle_report(sdir);
        if args.iter().any(|a| a == "--write") {
            let hours = arg_str(args, "--hours").and_then(|v| v.parse::<f64>().ok());
            match write_daily_report(sdir, hours) {
                Ok(p) => println!("\n📄 отчёт записан: {}", p.display()),
                Err(e) => println!("⚠️ отчёт не записан: {e}"),
            }
        }
        return;
    }

    let cfg = parse_think_cfg(args, sdir);
    let mut led = Ledger::load(sdir);
    if led.questions.is_empty() {
        println!("леджер пуст: `argus-agent ledger seed` (5 вопросов ARGUS-OS1) или `ledger add \"...\"`");
        return;
    }

    // план: пул + выбранный вопрос, без записи
    if cfg.dry {
        let ranked = led.ranked_for(chrono::Utc::now(), cfg.channel == "epmc");
        println!("пул (seed={}, k={}):", cfg.seed, cfg.k);
        for (i, s) in ranked.iter().take(cfg.k).enumerate() {
            println!(
                "  {}. {} score={:.3} surprise={:.3} staleness={:.1}ч cost={:.2}\n     {}",
                i + 1,
                s.id,
                s.score,
                s.surprise,
                s.staleness_h,
                s.cost,
                s.text
            );
        }
        match select_question(&cfg, &led) {
            Ok(p) => {
                let q = probe::build_query_for(&cfg.channel, &p.text);
                println!(
                    "\nвыбран: {} · канал {} · запрос «{}»",
                    p.id, cfg.channel, q
                );
                println!("--dry-run: ничего не записано. Записать: --once; фон: --loop");
            }
            Err(e) => println!("{e}"),
        }
        return;
    }

    let mut cache = Cache::load(sdir);
    let fetch = HttpFetcher::new(cfg.offline);
    let mut budget = argus_pre::budget::Budget::load(sdir);
    budget.max_cycles_per_hour = cfg.budget_cycles;
    budget.max_tokens_per_day = cfg.budget_tokens;

    if !args.iter().any(|a| a == "--loop") {
        match run_cycle(&cfg, sdir, &mut led, &mut cache, &fetch) {
            Ok((_qid, _note, tokens)) => {
                budget.record(chrono::Utc::now(), tokens);
                let _ = budget.save();
                print_idle_report(sdir);
            }
            Err(e) => println!("{e}"),
        }
        return;
    }

    // ── M3: непрерывный цикл с бюджетом ─────────────────────────────────────
    let _lock = match LoopLock::acquire(sdir) {
        Ok(l) => l,
        Err(e) => {
            println!("⛔ {e}");
            return;
        }
    };
    let max_cycles = arg_num(args, "--max-cycles");
    let interval = std::time::Duration::from_secs(cfg.interval_secs.max(10));

    // ── подъём после выключения/сна: продолжаем с момента остановки ──────────
    // Gap-вопросы живут не вечно: через сутки возвращаем их в очередь
    if led.reopen_stale_gaps(chrono::Utc::now(), 6) > 0 {
        let _ = led.save(sdir);
        println!("♻️ часть Gap-вопросов переоткрыта (прошло > 6 ч)");
    }
    let mut runtime = argus_pre::resume::Runtime::load(sdir);
    let plan = argus_pre::resume::plan_catchup(
        runtime.last_tick.as_deref(),
        cfg.interval_secs,
        budget.hour_remaining(chrono::Utc::now()),
        argus_pre::resume::MAX_CATCHUP_CYCLES,
    );
    let mut to_catchup = plan.0;
    if let Some(note) = argus_pre::resume::resume_note(plan) {
        runtime.resumes += 1;
        runtime.downtime_secs += plan.2;
        println!("⏯ {note}");
    }
    runtime.tick();
    let _ = runtime.save();
    println!(
        "idle-loop: бюджет {} циклов/ч · {} токенов/сутки · интервал {}с · channel={}",
        cfg.budget_cycles, cfg.budget_tokens, cfg.interval_secs, cfg.channel
    );
    let mut done: u64 = 0;
    loop {
        // сервис живёт часами: подхватываем изменения леджера, сделанные снаружи
        led = Ledger::load(sdir);
        let now = chrono::Utc::now();
        if let Err(why) = budget.gate(now) {
            if max_cycles.is_some() {
                println!("⏸ {why} — цикл остановлен");
                break;
            }
            // как сервис: не выходим, а ждём следующего окна бюджета (иначе systemd
            // перезапускает нас каждые 20 с и мы просто жжём циклы вхолостую)
            println!(
                "⏸ {why} — ждём окно бюджета ({} с)",
                cfg.interval_secs.max(10)
            );
            std::thread::sleep(interval);
            continue;
        }
        match run_cycle(&cfg, sdir, &mut led, &mut cache, &fetch) {
            Ok((qid, _note, tokens)) => {
                budget.record(chrono::Utc::now(), tokens);
                let _ = budget.save();
                done += 1;
                if to_catchup > 0 {
                    to_catchup -= 1;
                    runtime.catchup_cycles += 1;
                }
                runtime.tick();
                let _ = runtime.save();
                if let Ok(p) = write_daily_report(sdir, None) {
                    println!("  📄 {qid} → {}", p.display());
                }
            }
            Err(e) => {
                println!("⚠️ {e}");
                if max_cycles.is_some() {
                    break;
                }
                // как сервис: не умираем, если очередь опустела — ждём и пробуем снова
                // (Gap-вопросы переоткрываются через 24 ч, см. reopen_stale_gaps)
                println!("   ждём {} с и пробуем снова", cfg.interval_secs.max(10));
                std::thread::sleep(interval);
                continue;
            }
        }
        if let Some(m) = max_cycles {
            if done >= m {
                println!("остановка по --max-cycles {m}");
                break;
            }
        }
        if to_catchup > 0 {
            println!("  ⏩ догон: осталось {to_catchup}");
            continue; // догоняем без паузы
        }
        std::thread::sleep(interval);
    }
    println!("\nсделано циклов: {done}");
    print_idle_report(sdir);
    if let Ok(p) = write_daily_report(sdir, None) {
        println!("📄 суточный отчёт: {}", p.display());
    }
    if let Ok(p) = write_daily_report(sdir, Some(24.0)) {
        println!("📄 окно 24 ч: {}", p.display());
    }
}

/// Суточный отчёт: метрики, бюджет, состояние леджера, вердикт гейта.
fn write_daily_report(sdir: &PathBuf, hours: Option<f64>) -> std::io::Result<PathBuf> {
    use std::fmt::Write as _;
    let mut led = Ledger::load(sdir);
    // окно: только циклы за последние N часов (M4: отчёт за сутки)
    if let Some(h) = hours {
        let cutoff = chrono::Utc::now() - chrono::Duration::minutes((h * 60.0) as i64);
        led.cycles.retain(|c| {
            chrono::DateTime::parse_from_rfc3339(&c.at)
                .map(|t| t.with_timezone(&chrono::Utc) >= cutoff)
                .unwrap_or(false)
        });
    }
    let m = led.metrics();
    let mut budget = argus_pre::budget::Budget::load(sdir);
    let now = chrono::Utc::now();
    budget.roll_over(now);
    let day = chrono::Local::now().format("%Y-%m-%d").to_string();

    let verdict = if ledger::gate_failed(&m) {
        "❌ ГЕЙТ ПРОВАЛЕН (выдуманные ID или резолв < 0.7) → цикл в --dry-run"
    } else if m.id_cycles < 5 {
        "⏳ данных мало (<5 циклов с идентификаторами) — гейт не применяется"
    } else {
        "✅ гейт пройден"
    };

    let mut s = String::new();
    let _ = writeln!(s, "# idle-report — {day}\n");
    let _ = writeln!(s, "Сгенерировано: {}", now.to_rfc3339());
    let _ = writeln!(s, "Вердикт: **{verdict}**\n");
    let _ = writeln!(s, "## Метрики\n");
    let _ = writeln!(s, "| метрика | значение |\n|---|---|");
    let _ = writeln!(s, "| циклов | {} |", m.cycles);
    let _ = writeln!(s, "| с идентификаторами | {} |", m.id_cycles);
    let _ = writeln!(s, "| пробелов (ничего не найдено) | {} |", m.gap_cycles);
    let _ = writeln!(
        s,
        "| resolved / unresolved / fabricated | {} / {} / {} |",
        m.resolved, m.unresolved, m.fabricated
    );
    let _ = writeln!(
        s,
        "| identifier_resolution_rate | {:.2} |",
        m.identifier_resolution_rate
    );
    let _ = writeln!(s, "| evidence_yield | {:.2} |", m.evidence_yield);
    let _ = writeln!(
        s,
        "| fabricated_identifiers | {} |",
        m.fabricated_identifiers
    );
    let _ = writeln!(
        s,
        "| mean_abs_logit_shift | {:.3} |",
        m.mean_abs_logit_shift
    );
    let _ = writeln!(s, "| gaps_recorded | {} |", m.gaps_recorded);
    let (judged, acted) = stance_stats(&led.cycles);
    let _ = writeln!(s, "| свидетельств оценено позиционно | {judged} |");
    let _ = writeln!(s, "| из них p сдвинута | {acted} |");
    let _ = writeln!(s, "\n## Бюджет\n");
    let _ = writeln!(
        s,
        "- циклов за час: {} / {}",
        budget.recent.len(),
        budget.max_cycles_per_hour
    );
    let _ = writeln!(s, "- циклов за сутки: {}", budget.cycles_today);
    let _ = writeln!(
        s,
        "- токенов за сутки: {} / {}",
        budget.tokens_today, budget.max_tokens_per_day
    );

    let _ = writeln!(s, "\n## Леджер\n");
    let _ = writeln!(
        s,
        "| id | p | probes | status | форма | вопрос |\n|---|---|---|---|---|---|"
    );
    let mut qs = led.questions.clone();
    qs.sort_by(|a, b| a.status_key().cmp(&b.status_key()).then(a.id.cmp(&b.id)));
    for q in qs {
        let text: String = q.text.chars().take(90).collect();
        let _ = writeln!(
            s,
            "| {} | {:.2} | {} | {} | {} | {} |",
            q.id,
            q.p_true,
            q.probes,
            q.status_str(),
            q.form().as_str(),
            text.replace('|', "/")
        );
    }
    // Честная пометка: у темы/оценки p не определена — литература их не «закроет»
    let non_prop: Vec<_> = led
        .questions
        .iter()
        .filter(|q| !q.form().allows_lr() && q.status == ledger::QStatus::Open)
        .collect();
    if !non_prop.is_empty() {
        let _ = writeln!(
            s,
            "\n> ⚠️ {} открытых вопросов — не бинарные претензии, p у них не определена и двигаться не может.\n> Переформулировать в «да/нет» (`ledger form <id>` покажет форму):",
            non_prop.len()
        );
        for q in non_prop {
            let _ = writeln!(s, "> - {} [{}] {}", q.id, q.form().as_str(), q.text);
        }
    }

    let _ = writeln!(s, "\n## Последние циклы\n");
    let _ = writeln!(s, "| когда | вопрос | вердикт | stance | p до → после | что нашли |\n|---|---|---|---|---|---|");
    for c in led.cycles.iter().rev().take(15) {
        let _ = writeln!(
            s,
            "| {} | {} | {:?} | {}/{} | {:.3} → {:.3} | {} |",
            c.at,
            c.qid,
            c.verifier,
            c.stance.as_deref().unwrap_or("—"),
            c.stance_conf,
            c.before,
            c.after,
            c.note.replace('|', "/")
        );
    }

    let name = match hours {
        Some(h) => format!("report_{:.0}h_{day}.md", h),
        None => format!("report_{day}.md"),
    };
    let path = sdir.join("ledger").join(name);
    std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))?;
    std::fs::write(&path, s)?;
    Ok(path)
}

fn print_idle_report(sdir: &PathBuf) {
    let led = Ledger::load(sdir);
    let m = led.metrics();
    let verdict = if ledger::gate_failed(&m) {
        "❌ ГЕЙТ ПРОВАЛЕН (выдуманные ID или резолв < 0.7) → цикл в --dry-run"
    } else if m.id_cycles < 5 {
        "⏳ данных мало (<5 циклов с идентификаторами) — гейт не применяется"
    } else {
        "✅ гейт пройден"
    };
    println!("\n── метрики idle-цикла ──");
    println!(
        "циклов: {} (с идентификаторами {} · пробелов {})",
        m.cycles, m.id_cycles, m.gap_cycles
    );
    println!(
        "  resolved {} · unresolved {} · fabricated {}",
        m.resolved, m.unresolved, m.fabricated
    );
    println!(
        "identifier_resolution_rate: {:.2} (по циклам с ID, не по всем)",
        m.identifier_resolution_rate
    );
    println!(
        "evidence_yield:             {:.2} (доля циклов, давших хоть один ID)",
        m.evidence_yield
    );
    println!("fabricated_identifiers:     {}", m.fabricated_identifiers);
    println!("refutation_share:           {:.2}", m.refutation_share);
    println!("mean_abs_logit_shift:       {:.3}", m.mean_abs_logit_shift);
    println!("gaps_recorded:              {}", m.gaps_recorded);
    let (judged, acted) = stance_stats(&led.cycles);
    println!("stance: оценено {judged}, из них сдвинули p {acted}");
    println!("вердикт: {verdict}");
}

/// Сколько циклов получили позиционную оценку и сколько из них реально сдвинули p.
fn stance_stats(cycles: &[ledger::Cycle]) -> (usize, usize) {
    let judged = cycles.iter().filter(|c| c.stance.is_some()).count();
    let acted = cycles
        .iter()
        .filter(|c| (c.after - c.before).abs() > f64::EPSILON)
        .count();
    (judged, acted)
}

/// `lab` — лабораторные данные (ответ на lab-вопросы даёт эксперимент).
fn cmd_lab(args: &[String], sdir: &PathBuf) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    match sub {
        "list" => {
            let led = Ledger::load(sdir);
            let now = chrono::Utc::now();
            let labs: Vec<_> = led
                .questions
                .iter()
                .filter(|q| q.kind == ledger::QKind::Lab)
                .collect();
            if labs.is_empty() {
                println!("лабораторных вопросов нет: ledger add \"...\" --kind lab");
                return;
            }
            println!("{:<12} {:<9} {:<8} {}", "id", "status", "данные", "вопрос");
            for q in labs {
                match lab::load(sdir, &q.id) {
                    None => println!(
                        "{:<12} {:<9} {:<8} {}",
                        q.id,
                        q.status_str(),
                        "нет",
                        q.text
                    ),
                    Some(ev) => {
                        let ok = lab::validate(sdir, &ev, now, lab::DEFAULT_MAX_AGE_DAYS, &q.text);
                        let mark = if ok.is_ok() { "готово" } else { "непригодно" };
                        println!(
                            "{:<12} {:<9} {:<8} {} → {} ({})",
                            q.id,
                            q.status_str(),
                            mark,
                            q.text,
                            ev.verdict.as_str(),
                            ev.source
                        );
                        if let Err(why) = ok {
                            println!("{:>31}⚠️ {why}", "");
                        }
                    }
                }
            }
        }
        "record" => {
            let qid = args.get(3).cloned().unwrap_or_default();
            let verdict_raw = args.get(4).cloned().unwrap_or_default();
            let Some(verdict) = lab::LabVerdict::parse(&verdict_raw) else {
                println!("использование: lab record <qid> support|contradict|inconclusive --source ПУТЬ [--value \"...\"] [--note \"...\"] [--at YYYY-MM-DD]");
                return;
            };
            let Some(source) = arg_str(args, "--source") else {
                println!("⚠️ обязателен --source: цифра без артефакта не принимается");
                return;
            };
            let ev = lab::LabEvidence {
                qid: qid.clone(),
                verdict,
                value: arg_str(args, "--value").unwrap_or_default(),
                source,
                note: arg_str(args, "--note").unwrap_or_default(),
                at: arg_str(args, "--at").or_else(|| Some(chrono::Local::now().format("%Y-%m-%d").to_string())),
            };
            match lab::record(sdir, &ev) {
                Ok(p) => {
                    println!("✅ записано: {}", p.display());
                    match lab::validate(sdir, &ev, chrono::Utc::now(), lab::DEFAULT_MAX_AGE_DAYS, "") {
                        Ok(()) => println!("   проверка пройдена: данные применимы в канале `lab`"),
                        Err(why) => println!("   ⚠️ проверка не пройдена: {why}"),
                    }
                }
                Err(e) => println!("⚠️ не записано: {e}"),
            }
        }
        _ => println!("lab list | lab record <qid> support|contradict|inconclusive --source ПУТЬ [--value ...] [--note ...] [--at ДАТА]"),
    }
}

/// `external` — чужие эксперименты и данные (Zenodo, GitHub) по lab-вопросам.
fn cmd_external(args: &[String], sdir: &PathBuf) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    let led = Ledger::load(sdir);
    match sub {
        "list" => {
            let mut n = 0;
            for q in led.questions.iter().filter(|q| q.kind == ledger::QKind::Lab) {
                match thirdparty::load(sdir, &q.id) {
                    Some(ev) => {
                        n += 1;
                        println!(
                            "{:<12} {} [{}] {} — {}",
                            q.id, ev.verdict, ev.source, ev.title, ev.value
                        );
                    }
                    None => {}
                }
            }
            if n == 0 {
                println!("чужих свидетельств пока нет: набрать можно `external search <qid>`");
            }
        }
        "search" => {
            let qid = args.get(3).cloned().unwrap_or_default();
            let Some(q) = led.get(&qid) else {
                println!("нет такого вопроса: {qid}");
                return;
            };
            let query = probe::build_query_for("epmc", &q.text);
            println!("вопрос: {}", q.text);
            println!("запрос: «{query}»");
            let (cands, errors) = thirdparty::gather(&query, thirdparty::PER_SOURCE);
            for e in &errors {
                println!("  ⚠️ {e}");
            }
            if cands.is_empty() {
                println!("  ничего не найдено");
                return;
            }
            let items: Vec<stance::llm::Item<'_>> = cands
                .iter()
                .map(|c| stance::llm::Item {
                    id: c.id.as_str(),
                    title: c.title.as_str(),
                    abstract_text: c.summary.as_str(),
                })
                .collect();
            let mut cache = stance::llm::JudgeCache::load(sdir);
            let jcfg = stance::llm::JudgeConfig::from_env();
            let out = stance::llm::judge_batch(&jcfg, &qid, &q.text, &items, &mut cache);
            let _ = cache.save(sdir);
            for (c, d) in cands.iter().zip(out.decisions.iter()) {
                println!(
                    "  {} [{}] {} → {} ({:.2}) — {}",
                    c.id,
                    c.kind.as_str(),
                    c.title,
                    d.stance.as_str(),
                    d.confidence,
                    d.reason
                );
            }
            println!("  {}", out.note);
            println!("  (применить: `think --once --probe external --qid {qid}`)");
        }
        _ => println!("external list | external search <qid>"),
    }
}

/// `council` — консилиум бесплатных моделей (мнение, не доказательство).
fn cmd_council(args: &[String], sdir: &PathBuf) {
    let sub = args.get(2).map(String::as_str).unwrap_or("models");
    let cc = dialogue::CouncilConfig::from_env();
    match sub {
        "models" => {
            println!("состав ({}):", cc.members.len());
            for m in &cc.members {
                println!("  {} → {} [{:?}]", m.model, m.url, m.api);
            }
            println!("порог согласия: {} моделей, уверенность ≥ {:.2}", cc.min_agree, dialogue::ACT_CONF);
            println!("выключить: ARGUS_COUNCIL=0; сменить состав: ARGUS_COUNCIL_MODELS=\"m1,m2\"");
        }
        "ask" => {
            let qid = args.get(3).cloned().unwrap_or_default();
            let led = Ledger::load(sdir);
            let Some(q) = led.get(&qid) else {
                println!("нет такого вопроса: {qid}");
                return;
            };
            let problem = arg_str(args, "--problem")
                .or_else(|| q.expected.clone())
                .unwrap_or_default();
            let ctx = format!(
                "форма: {}; проб: {}; идентификаторов: {}",
                q.form().as_str(),
                q.probes,
                q.identifiers.len()
            );
            println!("вопрос: {}", q.text);
            if !problem.trim().is_empty() {
                println!("проблема: {problem}");
            }
            let cons = dialogue::consult(&cc, &q.text, &problem, &ctx);
            for o in &cons.opinions {
                println!(
                    "  {} {} ({:.2}) — {} {}",
                    o.model,
                    o.stance_str(),
                    o.confidence,
                    o.key_point,
                    if o.missing.is_empty() {
                        String::new()
                    } else {
                        format!("| не хватает: {}", o.missing)
                    }
                );
            }
            println!(
                "итог: {} · {} · {}",
                cons.label(),
                cons.reason,
                if cons.acts() {
                    "можно применить (слабый сдвиг приора)"
                } else {
                    "применять нельзя"
                }
            );
        }
        _ => println!("council models | council ask <qid> [--problem \"...\"]"),
    }
}

fn cmd_ledger(args: &[String], sdir: &PathBuf) {
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    let mut led = Ledger::load(sdir);
    match sub {
        "list" => {
            if led.questions.is_empty() {
                println!("леджер пуст: ledger seed | ledger add \"вопрос\"");
                return;
            }
            println!(
                "{:<12} {:>6} {:>3}  {:<9} {:<3} {:<12} {}",
                "id", "p", "n", "status", "вид", "форма", "вопрос"
            );
            let mut qs = led.questions.clone();
            qs.sort_by(|a, b| a.status_key().cmp(&b.status_key()).then(a.id.cmp(&b.id)));
            for q in qs {
                let form = q.form().as_str();
                let flag = if q.superseded_by.is_some() {
                    "→ заменён "
                } else if q.form().allows_lr() {
                    ""
                } else {
                    "⚠️ "
                };
                println!(
                    "{:<12} {:>6.2} {:>3}  {:<9} {:<3} {:<12} {flag}{}",
                    q.id,
                    q.p_true,
                    q.probes,
                    q.status_str(),
                    q.kind_str(),
                    form,
                    q.text
                );
            }
            let non_prop = led
                .questions
                .iter()
                .filter(|q| {
                    !q.form().allows_lr()
                        && q.status == ledger::QStatus::Open
                        && q.superseded_by.is_none()
                })
                .count();
            if non_prop > 0 {
                println!();
                println!(
                    "⚠️  {non_prop} открытых вопросов — не бинарные претензии: p у них не определена,
   свидетельство её сдвинуть не может. Переформулируй в «… ли …?»"
                );
            }
        }
        "add" => {
            let text = args.get(3).cloned().unwrap_or_default();
            if text.trim().is_empty() {
                println!("использование: ledger add \"текст вопроса\" [--p 0.4]");
                return;
            }
            let p = args
                .iter()
                .position(|a| a == "--p")
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse::<f64>().ok());
            let id = led.add(&text, p);
            let _ = led.save(sdir);
            println!("✅ добавлен {id}: {text}");
            let form = led.get(&id).map(|q| q.form()).unwrap_or(argus_pre::stance::QForm::Topic);
            if !form.allows_lr() {
                println!(
                    "⚠️ форма «{}»: p у такого вопроса не определена, свидетельство её не сдвинет.\n   Если нужна байесовская бухгалтерия — переформулируй как «… ли …?»",
                    form.as_str()
                );
            }
        }
        "seed" => {
            let mut n = 0;
            for (t, kind) in ledger::argus_default_questions() {
                if led.get(&ledger::qid(t)).is_some() {
                    continue;
                }
                led.add_kind(t, None, kind);
                n += 1;
            }
            let _ = led.save(sdir);
            println!("✅ засеяно вопросов: {n} (всего {})", led.questions.len());
        }
        "show" => {
            let id = match args.get(3) {
                Some(v) => v.clone(),
                None => {
                    println!("использование: ledger show <id>");
                    return;
                }
            };
            match led.get(&id) {
                Some(q) => {
                    println!("{}\n  {}", q.id, q.text);
                    println!(
                        "  p={:.3} (prev {:.3}) status={} probes={} форма={}",
                        q.p_true,
                        q.p_prev,
                        q.status_str(),
                        q.probes,
                        q.form().as_str()
                    );
                    if !q.form().allows_lr() {
                        println!("  ⚠️ форма не бинарная: p не определена, LR не применяется");
                    }
                    println!("  identifiers: {:?}", q.identifiers);
                    println!("  gaps: {:?}", q.gaps);
                    println!("  refutations: {:?}", q.refutations);
                    let hist: Vec<_> = led.cycles.iter().filter(|c| c.qid == id).collect();
                    for c in hist {
                        println!("  · {} {:?} {} → {} ({})", c.at, c.verifier, c.before, c.after, c.note);
                    }
                }
                None => println!("нет такого вопроса: {id}"),
            }
        }
        "close" => {
            let id = args.get(3).cloned().unwrap_or_default();
            let how = args.get(4).map(String::as_str).unwrap_or("gap");
            // --note "почему закрыт / какой ответ": без него причина закрытия терялась
            let note = arg_str(args, "--note").unwrap_or_else(|| match how {
                "refuted" => "опровергнут человеком".to_string(),
                "supported" => "подтверждён человеком".to_string(),
                _ => "пробел: ответа нет".to_string(),
            });
            match led.close_by_human(&id, how, &note) {
                Some(c) => {
                    let _ = led.save(sdir);
                    let _ = led.append_cycle(sdir, &c);
                    println!("✅ {id} закрыт как «{how}»");
                    println!("   причина: {note}");
                }
                None => println!("нет такого вопроса: {id}"),
            }
        }
        "supersede" => {
            let old_id = args.get(3).cloned().unwrap_or_default();
            let new_id = args.get(4).cloned().unwrap_or_default();
            if new_id.is_empty() {
                println!("использование: ledger supersede <старый id> <новый id>");
                return;
            }
            match led.supersede(&old_id, &new_id) {
                Some(c) => {
                    let _ = led.save(sdir);
                    let _ = led.append_cycle(sdir, &c);
                    println!("✅ {old_id} заменён на {new_id} (закрыт навсегда)");
                }
                None => println!("нет такого вопроса: {old_id}"),
            }
        }
        "set-kind" => {
            let id = args.get(3).cloned().unwrap_or_default();
            let kind = match args.get(4).map(String::as_str) {
                Some("lab") => ledger::QKind::Lab,
                Some("lit") => ledger::QKind::Lit,
                _ => {
                    println!("использование: ledger set-kind <id> lit|lab");
                    return;
                }
            };
            match led.questions.iter_mut().find(|q| q.id == id) {
                Some(q) => {
                    q.kind = kind;
                    let k = q.kind_str();
                    let _ = led.save(sdir);
                    println!("✅ {id} → {k}");
                }
                None => println!("нет такого вопроса: {id}"),
            }
        }
        "reopen" => {
            let what = args.get(3).cloned().unwrap_or_else(|| "all".to_string());
            let mut n = 0;
            for q in led.questions.iter_mut() {
                if (what == "all" || q.id == what) && q.status == ledger::QStatus::Gap {
                    q.status = ledger::QStatus::Open;
                    q.consecutive_gaps = 0;
                    n += 1;
                    println!("♻️ {} → open", q.id);
                }
            }
            let _ = led.save(sdir);
            println!("переоткрыто: {n}");
        }
        "report" => print_idle_report(sdir),
        _ => println!("ledger list | add \"...\" [--p 0.5] [--kind lit|lab] | show <id> | seed | set-kind <id> lit|lab | reopen [id|all] | close <id> gap|refuted|supported [--note \"...\"] | supersede <old> <new> | report"),
    }
}

fn cmd_memory(args: &[String], sdir: &std::path::Path) {
    let mut mem = Memory::load(sdir);
    let sub = args.get(2).map(String::as_str).unwrap_or("list");
    match sub {
        "add" => {
            let text = args.get(3).cloned().unwrap_or_default();
            let kind = flag(args, "--kind").unwrap_or_else(|| "knowledge".into());
            let kind = match kind.as_str() {
                "event" => NoteKind::Event,
                "behavior" => NoteKind::Behavior,
                "skill" => NoteKind::Skill,
                _ => NoteKind::Knowledge,
            };
            let id = mem.add_note(&text, vec!["cli".into()], kind.clone());
            mem.save(sdir);
            println!("✅ note #{id} сохранена (kind={kind:?})");
        }
        "list" => {
            print!("{}", mem.memory_index());
            print!("{}", mem.lessons_block(""));
            print!("{}", mem.procedures_block(""));
        }
        "lessons" => {
            for l in &mem.lessons {
                println!(
                    "*{} [{}] {}\n    {}",
                    l.id,
                    l.tags.join(","),
                    l.activation,
                    l.principle
                );
            }
            for p in &mem.procedures {
                println!(
                    ">{} [{}] {} · шагов: {}",
                    p.id,
                    p.tags.join(","),
                    p.activation,
                    p.steps.len()
                );
            }
        }
        "links" => {
            // V3.6: показать граф связей
            if mem.links.is_empty() {
                println!("(связей нет — они создаются автоматически по общим тегам)");
            }
            for l in &mem.links {
                println!("#{} --[{kind}]--> #{}", l.a, l.b, kind = l.kind);
            }
        }
        "link" => {
            // V3.6: ручная связь memory link <a> <b> [kind]
            let a = args.get(3).and_then(|x| x.parse::<u64>().ok());
            let b = args.get(4).and_then(|x| x.parse::<u64>().ok());
            let kind = args.get(5).map(String::as_str).unwrap_or("RELATES_TO");
            match (a, b) {
                (Some(aa), Some(bb)) => {
                    if mem.add_link(aa, bb, kind) {
                        mem.save(sdir);
                        println!("✅ связь #{aa} --[{kind}]--> #{bb}");
                    } else {
                        println!("ℹ️ связь уже существует");
                    }
                }
                _ => println!("использование: memory link <a> <b> [RELATES_TO]"),
            }
        }
        "procedures" => {
            // V3.5: показать процедуры отдельно (с голосами)
            if mem.procedures.is_empty() {
                println!("(процедур нет — консолидация создаёт их после 3+ заметок с общим тегом)");
            }
            for p in &mem.procedures {
                println!(
                    ">{} [{}] v{} · {} шагов:",
                    p.id,
                    p.tags.join(","),
                    p.votes,
                    p.steps.len()
                );
                for (i, st) in p.steps.iter().enumerate() {
                    println!("    {}. {}", i + 1, st.chars().take(90).collect::<String>());
                }
            }
        }
        "vote" => {
            // V3.5: голосование memory vote <note|lesson|procedure> <id> <+|->
            let kind = args.get(3).cloned().unwrap_or_default();
            let id = args.get(4).and_then(|x| x.parse::<u64>().ok());
            let delta = match args.get(5).map(String::as_str) {
                Some("-") | Some("down") => -1,
                _ => 1,
            };
            match (id, kind.as_str()) {
                (Some(i), k) => match mem.vote(k, i, delta) {
                    Ok(v) => {
                        mem.save(sdir);
                        println!("✅ {k} #{i} → votes: {v}");
                    }
                    Err(e) => println!("❌ {e}"),
                },
                _ => println!("использование: memory vote <note|lesson|procedure> <id> <+|->"),
            }
        }
        "recall" => {
            let q = args.get(3).cloned().unwrap_or_default();
            if q.is_empty() {
                println!("использование: memory recall \"запрос\" [--k N] [--vector local|ollama]");
            } else {
                let k = flag(args, "--k").and_then(|s| s.parse().ok()).unwrap_or(3);
                let vector = flag(args, "--vector").unwrap_or_else(|| "local".into());
                println!("── Гибридный recall (substring + векторы: {vector}) ──");
                let (hits, used_ollama) = if vector == "ollama" {
                    mem.recall_hybrid_ollama(&q, k)
                } else {
                    (mem.recall_hybrid(&q, k), false)
                };
                if vector == "ollama" && !used_ollama {
                    println!("⚠️ Ollama-эмбеддинги недоступны (сервер без --embeddings) — fallback на локальные");
                }
                for n in hits {
                    println!(
                        "#{} [{:?}] {}",
                        n.id,
                        n.kind,
                        n.content.chars().take(160).collect::<String>()
                    );
                }
            }
        }
        "embed" => {
            // V2.4: проверка доступности Ollama-эмбеддингов
            match argus_pre::memory::ollama_embed("проверка") {
                Some(v) => println!("✅ Ollama-эмбеддинги доступны (размерность {})", v.len()),
                None => println!("❌ Ollama-эмбеддинги недоступны (сервер без --embeddings) — используется локальный ngram"),
            }
        }
        "embed-build" => {
            // V2.7: пересборка ngram-кэша для всех заметок
            let n = mem.rebuild_ngram_cache();
            mem.save(sdir);
            println!("✅ ngram-кэш пересобран: {n} эмбеддингов");
        }
        "export-vectors" => {
            // V3.1: экспорт векторного индекса в JSON
            let path = flag(args, "--path")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| sdir.join("memory_vectors.json"));
            let n = mem.export_vectors(&path);
            println!(
                "✅ векторный индекс экспортирован: {n} векторов → {}",
                path.display()
            );
        }
        "sync" => {
            // V3.2: авто-синхронизация с Obsidian-ваултом (import|export|both)
            let vault = flag(args, "--vault").map(std::path::PathBuf::from);
            match vault {
                Some(v) => {
                    let mode = flag(args, "--mode").unwrap_or_else(|| "both".into());
                    let mut imp = 0usize;
                    let mut exp = 0usize;
                    if mode == "both" || mode == "import" {
                        imp = mem.import_vault(&v);
                        println!("  импорт: +{imp} заметок");
                    }
                    if mode == "both" || mode == "export" {
                        exp = mem.export_vault(&v);
                        println!("  экспорт: {exp} файлов");
                    }
                    mem.save(sdir);
                    println!(
                        "✅ sync [{mode}] с {} (import {imp} / export {exp})",
                        v.display()
                    );
                }
                None => println!(
                    "использование: memory sync --vault <path> [--mode import|export|both]"
                ),
            }
        }
        "export" => {
            let vault = flag(args, "--vault").map(std::path::PathBuf::from);
            match vault {
                Some(v) => {
                    let n = mem.export_vault(&v);
                    println!("✅ экспортировано файлов: {n} → {}", v.display());
                }
                None => println!("использование: memory export --vault <path>"),
            }
        }
        "import" => {
            let vault = flag(args, "--vault").map(std::path::PathBuf::from);
            match vault {
                Some(v) => {
                    let n = mem.import_vault(&v);
                    mem.save(sdir);
                    println!("✅ импортировано заметок: {n} ← {}", v.display());
                }
                None => println!("использование: memory import --vault <path>"),
            }
        }
        _ => println!("memory add|list|lessons|export --vault <path>"),
    }
}
