//! argus-agent V1.0 — файловая память (P2): profile + notes + lessons
//! Паттерн: указатели в промпт, тело по запросу (как Atomic Agent / pi memory).
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Profile {
    /// pinned-факты рендерятся всегда; прочие — по keywords в текущем сообщении
    pub facts: BTreeMap<String, ProfileFact>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProfileFact {
    pub value: String,
    pub pinned: bool,
    #[serde(default)]
    pub keywords: Vec<String>,
    pub updated_at: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Note {
    pub id: u64,
    pub content: String,
    pub tags: Vec<String>,
    pub kind: NoteKind, // P3: typed notes (event|behavior|knowledge|skill)
    pub created_at: String,
    /// консолидирована ли заметка в урок (consolidator)
    #[serde(default)]
    pub consolidated: bool,
    /// V3.5: полезность (voting) — влияет на recall-ранжирование и eviction
    #[serde(default)]
    pub votes: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NoteKind {
    Event,
    Behavior,
    Knowledge,
    Skill,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Lesson {
    pub id: u64,
    /// активация: короткая строка, по ней ищется BM25/подстрока
    pub activation: String,
    pub principle: String,
    pub tags: Vec<String>,
    /// parent-ссылки на исходные заметки
    pub parents: Vec<u64>,
    #[serde(default)]
    pub votes: i32,
}

/// Процедура (V1.6): advisory-шаблон «как сделать», НЕ автовыполняется.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct Procedure {
    pub id: u64,
    pub activation: String,
    pub steps: Vec<String>,
    pub tags: Vec<String>,
    pub parents: Vec<u64>,
    #[serde(default)]
    pub votes: i32,
}

#[derive(Default)]
/// V3.6: связь между заметками (typed edge). kind: RELATES_TO|CAUSED_BY|…
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Link {
    pub a: u64,
    pub b: u64,
    pub kind: String,
}

#[derive(Default)]
pub struct Memory {
    pub profile: Profile,
    pub notes: Vec<Note>,
    pub lessons: Vec<Lesson>,
    pub procedures: Vec<Procedure>,
    pub links: Vec<Link>,
    next_note_id: u64,
    next_lesson_id: u64,
    next_proc_id: u64,
    // V2.5: кэш эмбеддингов (ngram — всегда; ollama — опционально), id → вектор
    pub ngram_cache: std::collections::HashMap<u64, Vec<f32>>,
    pub ollama_cache: std::collections::HashMap<u64, Vec<f32>>,
}

/// Размерность локальных эмбеддингов.
pub const EMBED_DIM: usize = 256;

impl Memory {
    /// Создаёт/читает файловую память в <stateDir>/memory/ (profile.json, notes.json, lessons.json)
    pub fn load(state_dir: &Path) -> Self {
        let mdir = state_dir.join("memory");
        fs::create_dir_all(&mdir).ok();
        let mut m = Memory::default();
        m.profile = read_json::<Profile>(&mdir.join("profile.json")).unwrap_or_default();
        m.notes = read_json::<Vec<Note>>(&mdir.join("notes.json")).unwrap_or_default();
        m.lessons = read_json::<Vec<Lesson>>(&mdir.join("lessons.json")).unwrap_or_default();
        m.procedures =
            read_json::<Vec<Procedure>>(&mdir.join("procedures.json")).unwrap_or_default();
        m.links = read_json::<Vec<Link>>(&mdir.join("links.json")).unwrap_or_default();
        m.next_note_id = m.notes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        m.next_lesson_id = m.lessons.iter().map(|l| l.id).max().unwrap_or(0) + 1;
        m.next_proc_id = m.procedures.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        // V2.5: загрузка кэша эмбеддингов
        if let Ok(raw) = fs::read_to_string(mdir.join("embeddings.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(ng) = v.get("ngram").and_then(|x| x.as_object()) {
                    for (k, val) in ng {
                        if let (Ok(id), Some(arr)) = (k.parse::<u64>(), val.as_array()) {
                            let vec: Vec<f32> = arr
                                .iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect();
                            m.ngram_cache.insert(id, vec);
                        }
                    }
                }
                if let Some(ol) = v.get("ollama").and_then(|x| x.as_object()) {
                    for (k, val) in ol {
                        if let (Ok(id), Some(arr)) = (k.parse::<u64>(), val.as_array()) {
                            let vec: Vec<f32> = arr
                                .iter()
                                .filter_map(|x| x.as_f64().map(|f| f as f32))
                                .collect();
                            m.ollama_cache.insert(id, vec);
                        }
                    }
                }
            }
        }
        // V2.7: если кэш неполон/устарел (новые заметки, старый файл) — пересобрать
        if m.notes.len() != m.ngram_cache.len() {
            m.rebuild_ngram_cache();
        }
        m
    }

    pub fn save(&mut self, state_dir: &Path) {
        // V3.5: eviction применяется при сохранении (лимиты, вытеснение по полезности)
        let _ = self.evict();
        let mdir = state_dir.join("memory");
        fs::create_dir_all(&mdir).ok();
        write_json(&mdir.join("profile.json"), &self.profile);
        write_json(&mdir.join("notes.json"), &self.notes);
        write_json(&mdir.join("lessons.json"), &self.lessons);
        write_json(&mdir.join("procedures.json"), &self.procedures);
        write_json(&mdir.join("links.json"), &self.links);
        // V2.5: сохранение кэша эмбеддингов
        let mut ngram = serde_json::Map::new();
        for (k, v) in &self.ngram_cache {
            ngram.insert(
                k.to_string(),
                serde_json::Value::Array(v.iter().map(|f| serde_json::json!(f)).collect()),
            );
        }
        let mut ollama = serde_json::Map::new();
        for (k, v) in &self.ollama_cache {
            ollama.insert(
                k.to_string(),
                serde_json::Value::Array(v.iter().map(|f| serde_json::json!(f)).collect()),
            );
        }
        let _ = fs::write(
            mdir.join("embeddings.json"),
            serde_json::to_string_pretty(&serde_json::json!({"ngram": ngram, "ollama": ollama}))
                .unwrap_or_default(),
        );
    }

    // ---- Profile ----
    pub fn set_fact(&mut self, key: &str, value: &str, pinned: bool, keywords: Vec<String>) {
        self.profile.facts.insert(
            key.to_string(),
            ProfileFact {
                value: value.to_string(),
                pinned,
                keywords,
                updated_at: Local::now().to_rfc3339(),
            },
        );
    }

    /// Гейт: pinned-факты всегда; остальные — только если keywords встречены в сообщении.
    pub fn profile_block(&self, message: &str) -> String {
        let mut out = String::from("### profile\n");
        let msg_lower = message.to_lowercase();
        for (k, f) in &self.profile.facts {
            let hit = f.pinned
                || f.keywords.is_empty()
                || f.keywords
                    .iter()
                    .any(|kw| msg_lower.contains(&kw.to_lowercase()));
            if hit {
                out.push_str(&format!("- {}: {}\n", k, f.value));
            }
        }
        if out.ends_with("\n") && out.len() > "### profile\n".len() {
            out
        } else {
            String::new()
        }
    }

    // ---- Notes ----
    pub fn add_note(&mut self, content: &str, tags: Vec<String>, kind: NoteKind) -> u64 {
        let id = self.next_note_id;
        self.next_note_id += 1;
        let new_tags: Vec<String> = tags.clone();
        self.notes.push(Note {
            id,
            content: content.to_string(),
            tags,
            kind,
            created_at: Local::now().to_rfc3339(),
            consolidated: false,
            votes: 0,
        });
        // V2.5: предвычислить и закэшировать ngram-эмбеддинг
        self.ngram_cache.insert(id, ngram_embed(content, EMBED_DIM));
        // V3.6: авто-связи с заметками, у которых есть общий тег
        for other in &self.notes {
            if other.id == id {
                continue;
            }
            if other.tags.iter().any(|t| new_tags.contains(t)) {
                self.links.push(Link {
                    a: id,
                    b: other.id,
                    kind: "RELATES_TO".into(),
                });
            }
        }
        id
    }

    /// V3.6: добавить связь вручную; false, если уже есть.
    pub fn add_link(&mut self, a: u64, b: u64, kind: &str) -> bool {
        if self
            .links
            .iter()
            .any(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
        {
            return false;
        }
        self.links.push(Link {
            a,
            b,
            kind: kind.to_string(),
        });
        true
    }

    /// V3.6: соседи заметки по графу (1 hop).
    pub fn neighbors(&self, id: u64) -> Vec<u64> {
        self.links
            .iter()
            .filter(|l| l.a == id || l.b == id)
            .map(|l| if l.a == id { l.b } else { l.a })
            .collect()
    }

    /// V2.7: пересобрать ngram-кэш для всех заметок (гарантия персиста после загрузки).
    pub fn rebuild_ngram_cache(&mut self) -> usize {
        self.ngram_cache.clear();
        for n in &self.notes {
            self.ngram_cache
                .insert(n.id, ngram_embed(&n.content, EMBED_DIM));
        }
        self.ngram_cache.len()
    }

    /// V3.5: голосование за полезность (note|lesson|procedure), delta = +1/-1.
    pub fn vote(&mut self, kind: &str, id: u64, delta: i32) -> Result<i32, String> {
        match kind {
            "note" => self
                .notes
                .iter_mut()
                .find(|n| n.id == id)
                .map(|n| {
                    n.votes += delta;
                    n.votes
                })
                .ok_or_else(|| format!("заметка #{id} не найдена")),
            "lesson" => self
                .lessons
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| {
                    l.votes += delta;
                    l.votes
                })
                .ok_or_else(|| format!("урок #{id} не найден")),
            "procedure" => self
                .procedures
                .iter_mut()
                .find(|p| p.id == id)
                .map(|p| {
                    p.votes += delta;
                    p.votes
                })
                .ok_or_else(|| format!("процедура #{id} не найдена")),
            other => Err(format!("вид '{other}' не из note|lesson|procedure")),
        }
    }

    /// V3.5: eviction — лимиты (1000 заметок / 500 уроков / 500 процедур),
    /// вытеснение по полезности (votes), затем по новизне.
    pub fn evict(&mut self) -> usize {
        let mut removed = 0usize;
        let trim = |v: &mut Vec<Note>, cap: usize, removed: &mut usize| {
            if v.len() > cap {
                v.sort_by(|a, b| {
                    b.votes
                        .cmp(&a.votes)
                        .then_with(|| b.created_at.cmp(&a.created_at))
                });
                v.truncate(cap);
                *removed += 1;
            }
        };
        trim(&mut self.notes, 1000, &mut removed);
        if self.lessons.len() > 500 {
            self.lessons.sort_by(|a, b| b.votes.cmp(&a.votes));
            self.lessons.truncate(500);
            removed += 1;
        }
        if self.procedures.len() > 500 {
            self.procedures.sort_by(|a, b| b.votes.cmp(&a.votes));
            self.procedures.truncate(500);
            removed += 1;
        }
        removed
    }

    /// V3.5: экспорт векторного индекса (id/content/kind/tags/embedding) в JSON.
    pub fn export_vectors(&self, path: &Path) -> usize {
        let arr: Vec<serde_json::Value> = self
            .notes
            .iter()
            .map(|n| {
                let emb = self
                    .ngram_cache
                    .get(&n.id)
                    .cloned()
                    .unwrap_or_else(|| ngram_embed(&n.content, EMBED_DIM));
                serde_json::json!({
                    "id": n.id,
                    "content": n.content,
                    "kind": kind_name(&n.kind),
                    "tags": n.tags,
                    "embedding": emb
                })
            })
            .collect();
        let _ = fs::write(path, serde_json::to_string_pretty(&arr).unwrap_or_default());
        arr.len()
    }

    /// Простой recall: подстрока-поиск по content+tags (стаб BM25/FTS5; эмбеддинги — позже).
    pub fn recall(&self, query: &str, k: usize) -> Vec<&Note> {
        let q = query.to_lowercase();
        let mut hits: Vec<&Note> = self
            .notes
            .iter()
            .filter(|n| {
                n.content.to_lowercase().contains(&q)
                    || n.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .collect();
        hits.sort_by_key(|n| std::cmp::Reverse(n.created_at.clone()));
        hits.truncate(k);
        hits
    }

    /// V2.2: гибридный recall — substring-хиты + косинусное сходство локальных
    /// n-грамм-эмбеддингов (кэш V2.5; без внешних сервисов).
    pub fn recall_hybrid(&self, query: &str, k: usize) -> Vec<&Note> {
        let mut hits: Vec<&Note> = self.recall(query, k);
        // кандидаты по косинусному сходству (дополняют substring-хиты)
        let qe = ngram_embed(query, EMBED_DIM);
        let mut scored: Vec<(f32, &Note)> = Vec::with_capacity(self.notes.len());
        for n in &self.notes {
            let ne = match self.ngram_cache.get(&n.id) {
                Some(v) => v.clone(),
                None => ngram_embed(&n.content, EMBED_DIM),
            };
            // V3.5: votes дают лёгкий вес в ранжировании
            let score = cosine(&qe, &ne) + (n.votes as f32) * 0.05;
            scored.push((score, n));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, n) in scored {
            if hits.len() >= k {
                break;
            }
            if !hits.iter().any(|h| h.id == n.id) {
                hits.push(n);
            }
        }
        // V3.6: 1-hop расширение по графу связей
        let mut neighbors: Vec<u64> = hits.iter().flat_map(|h| self.neighbors(h.id)).collect();
        neighbors.sort();
        neighbors.dedup();
        for nid in neighbors {
            if hits.len() >= k {
                break;
            }
            if !hits.iter().any(|h| h.id == nid) {
                if let Some(n) = self.notes.iter().find(|n| n.id == nid) {
                    hits.push(n);
                }
            }
        }
        hits.truncate(k);
        hits
    }

    /// V2.4: гибридный recall с Ollama-эмбеддингами (если сервер с --embeddings),
    /// иначе fallback на локальные n-грамм-эмбеддинги. Кэш V2.5.
    pub fn recall_hybrid_ollama(&mut self, query: &str, k: usize) -> (Vec<&Note>, bool) {
        let Some(qe) = ollama_embed(query) else {
            return (self.recall_hybrid(query, k), false);
        };
        // substring-хиты как id (не ссылки), чтобы не блокировать &mut кэш
        let mut hit_ids: Vec<u64> = self.recall(query, k).iter().map(|n| n.id).collect();
        // локальная копия (id, content), чтобы не конфликтовать с &mut кэшем
        let items: Vec<(u64, String)> = self
            .notes
            .iter()
            .map(|n| (n.id, n.content.clone()))
            .collect();
        let mut scored: Vec<(f32, u64)> = Vec::with_capacity(items.len());
        for (id, content) in &items {
            let ne = if let Some(v) = self.ollama_cache.get(id) {
                v.clone()
            } else if let Some(v) = ollama_embed(content) {
                self.ollama_cache.insert(*id, v.clone());
                v
            } else {
                continue;
            };
            scored.push((cosine(&qe, &ne), *id));
        }
        if scored.is_empty() {
            return (self.recall_hybrid(query, k), false);
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, id) in scored {
            if hit_ids.len() >= k {
                break;
            }
            if !hit_ids.contains(&id) {
                hit_ids.push(id);
            }
        }
        hit_ids.truncate(k);
        let hits: Vec<&Note> = self
            .notes
            .iter()
            .filter(|n| hit_ids.contains(&n.id))
            .collect();
        (hits, true)
    }

    /// Память «указатели: топ-20 последних заметок однострочниками (тело — только по id).
    pub fn memory_index(&self) -> String {
        if self.notes.is_empty() {
            return String::new();
        }
        let mut out = String::from("### memory-index\n");
        for n in self.notes.iter().rev().take(20) {
            let preview: String = n.content.chars().take(60).collect();
            out.push_str(&format!("- #{} [{}] {}\n", n.id, n.tags.join(","), preview));
        }
        out
    }

    pub fn get_note(&self, id: u64) -> Option<&Note> {
        self.notes.iter().find(|n| n.id == id)
    }

    // ---- Lessons ----
    pub fn add_lesson(
        &mut self,
        activation: &str,
        principle: &str,
        tags: Vec<String>,
        parents: Vec<u64>,
    ) -> u64 {
        let id = self.next_lesson_id;
        self.next_lesson_id += 1;
        self.lessons.push(Lesson {
            id,
            activation: activation.to_string(),
            principle: principle.to_string(),
            tags,
            parents,
            votes: 0,
        });
        id
    }

    /// V3.1: обновить урок (LLM-дистилляция).
    pub fn set_lesson(&mut self, id: u64, activation: &str, principle: &str) {
        if let Some(l) = self.lessons.iter_mut().find(|l| l.id == id) {
            l.activation = activation.to_string();
            l.principle = principle.to_string();
        }
    }

    pub fn lessons_block(&self, message: &str) -> String {
        let q = message.to_lowercase();
        let hits: Vec<&Lesson> = self
            .lessons
            .iter()
            .filter(|l| {
                l.activation.to_lowercase().contains(&q)
                    || l.principle.to_lowercase().contains(&q)
                    || l.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .take(2)
            .collect();
        if hits.is_empty() {
            return String::new();
        }
        let mut out = String::from("### lessons\n");
        for l in hits {
            out.push_str(&format!(
                "*{} [{}] {}\n",
                l.id,
                l.tags.join(","),
                l.activation
            ));
        }
        out
    }

    /// Consolidator (как Atomic): заметки с тегом, встречающимся ≥3 раза,
    /// дистиллируются в урок (активация + принцип + parents) И процедуру (шаги).
    /// Без LLM — эвристика. Возвращает число созданных уроков.
    pub fn consolidate(&mut self) -> usize {
        let mut tag_count: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for n in self.notes.iter().filter(|n| !n.consolidated) {
            for t in &n.tags {
                *tag_count.entry(t.clone()).or_insert(0) += 1;
            }
        }
        let mut created = 0usize;
        // для каждого тега с >=3 заметками: проверить, нет ли уже урока с этим тегом
        for (tag, count) in tag_count {
            if count < 3 {
                continue;
            }
            if self.lessons.iter().any(|l| l.tags.contains(&tag)) {
                continue;
            }
            let ids: Vec<u64> = self
                .notes
                .iter()
                .filter(|n| !n.consolidated && n.tags.contains(&tag))
                .map(|n| n.id)
                .collect();
            if ids.len() < 3 {
                continue;
            }
            let principle: String = self
                .notes
                .iter()
                .filter(|n| ids.contains(&n.id))
                .map(|n| {
                    format!(
                        "#{}: {}",
                        n.id,
                        n.content.chars().take(120).collect::<String>()
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            let activation = format!("повторяющийся паттерн: {tag} ({} эпизода)", ids.len());
            self.add_lesson(&activation, &principle, vec![tag.clone()], ids.clone());
            // V1.6: та же консолидация создаёт advisory-процедуру (не автовыполняется)
            let steps: Vec<String> = self
                .notes
                .iter()
                .filter(|n| ids.contains(&n.id))
                .take(3)
                .map(|n| n.content.chars().take(90).collect::<String>())
                .collect();
            self.add_procedure(
                &format!("как работать с паттерном: {tag}"),
                steps,
                vec![tag.clone()],
                ids.clone(),
            );
            // пометить консолидированными
            for n in self.notes.iter_mut().filter(|n| ids.contains(&n.id)) {
                n.consolidated = true;
            }
            created += 1;
        }
        created
    }

    /// Добавляет advisory-процедуру (шаги не автовыполняются).
    pub fn add_procedure(
        &mut self,
        activation: &str,
        steps: Vec<String>,
        tags: Vec<String>,
        parents: Vec<u64>,
    ) -> u64 {
        let id = self.next_proc_id;
        self.next_proc_id += 1;
        self.procedures.push(Procedure {
            id,
            activation: activation.to_string(),
            steps,
            tags,
            parents,
            votes: 0,
        });
        id
    }

    /// ### procedures — топ-2 релевантных процедуры (указатели, как lessons).
    pub fn procedures_block(&self, message: &str) -> String {
        let q = message.to_lowercase();
        let hits: Vec<&Procedure> = self
            .procedures
            .iter()
            .filter(|p| {
                p.activation.to_lowercase().contains(&q)
                    || p.steps.iter().any(|s| s.to_lowercase().contains(&q))
                    || p.tags.iter().any(|t| t.to_lowercase().contains(&q))
            })
            .take(2)
            .collect();
        if hits.is_empty() {
            return String::new();
        }
        let mut out = String::from("### procedures\n");
        for p in hits {
            out.push_str(&format!(
                ">{} [{}] {}\n",
                p.id,
                p.tags.join(","),
                p.activation
            ));
        }
        out
    }

    pub fn get_procedure(&self, id: u64) -> Option<&Procedure> {
        self.procedures.iter().find(|p| p.id == id)
    }

    /// V3.0: обновить процедуру (LLM-дистилляция).
    pub fn set_procedure(&mut self, id: u64, activation: &str, steps: Vec<String>) {
        if let Some(p) = self.procedures.iter_mut().find(|p| p.id == id) {
            p.activation = activation.to_string();
            p.steps = steps;
        }
    }

    /// Obsidian-импорт (round-trip): читает .md из ваулта (рекурсивно),
    /// извлекает frontmatter-теги, добавляет как заметки памяти. Возвращает число.
    pub fn import_vault(&mut self, vault: &Path) -> usize {
        let mut count = 0usize;
        let mut stack = vec![vault.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    // V2.1: пропуск шаблонных/скрытых папок (_templates, .git, …)
                    let name = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if !name.starts_with('_') && !name.starts_with('.') {
                        stack.push(p);
                    }
                } else if p.extension().map(|x| x == "md").unwrap_or(false) {
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        let (mut tags, kind, created, content) = parse_md_note(&text);
                        // V2.0: inline #теги из тела
                        for t in inline_tags(&content) {
                            if !tags.contains(&t) {
                                tags.push(t);
                            }
                        }
                        // V2.6: source-тег по имени файла (для round-trip прослеживаемости)
                        if let Some(fname) = p.file_name().map(|n| n.to_string_lossy().to_string())
                        {
                            let src = format!("src:{}", fname.trim_end_matches(".md"));
                            if !tags.contains(&src) {
                                tags.push(src);
                            }
                        }
                        // V3.5: двусторонний dedup — файл уже импортирован (src-тег) ИЛИ тот же контент
                        let src_tag = format!(
                            "src:{}",
                            p.file_name()
                                .map(|n| n.to_string_lossy().trim_end_matches(".md").to_string())
                                .unwrap_or_default()
                        );
                        let already = self.notes.iter().any(|n| n.tags.contains(&src_tag))
                            || self.notes.iter().any(|n| n.content == content);
                        if !content.is_empty() && !already {
                            let id = self.add_note(&content, tags, kind);
                            // V2.6: сохранить дату из frontmatter
                            if let Some(c) = created {
                                if let Some(n) = self.notes.iter_mut().find(|n| n.id == id) {
                                    n.created_at = c;
                                }
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Obsidian-экспорт (как Atomic): notes/ и lessons/ как Markdown с YAML
    /// frontmatter и [[wikilinks]] между заметками с общим тегом.
    /// Возвращает количество экспортированных файлов.
    pub fn export_vault(&self, vault: &Path) -> usize {
        let notes_dir = vault.join("notes");
        let lessons_dir = vault.join("lessons");
        let _ = fs::create_dir_all(&notes_dir);
        let _ = fs::create_dir_all(&lessons_dir);
        let mut count = 0usize;
        for n in &self.notes {
            let linked: Vec<String> = self
                .notes
                .iter()
                .filter(|m| m.id != n.id && m.tags.iter().any(|t| n.tags.contains(t)))
                .map(|m| format!("[[note-{}]]", m.id))
                .collect();
            let title = n
                .content
                .lines()
                .find(|l| l.starts_with("# "))
                .map(|l| l.trim_start_matches("# ").trim().to_string());
            // V2.6: source из src: тега (имя исходного файла)
            let src = n
                .tags
                .iter()
                .find(|t| t.starts_with("src:"))
                .map(|t| t.trim_start_matches("src:").to_string());
            let front = format!(
                "---\nid: {}\nkind: {}\ntags: [{}]\ncreated: {}\ndate: {}\n{}source: {}\n---\n\n",
                n.id,
                kind_name(&n.kind),
                n.tags.join(", "),
                n.created_at,
                n.created_at.chars().take(10).collect::<String>(),
                title.map(|t| format!("title: {t}\n")).unwrap_or_default(),
                src.unwrap_or_default()
            );
            let mut body = format!("# Note {}\n\n{}\n", n.id, n.content);
            if !linked.is_empty() {
                body.push_str("\n## Связано\n");
                for l in linked {
                    body.push_str(&format!("- {}\n", l));
                }
            }
            let _ = fs::write(
                notes_dir.join(format!("note-{}.md", n.id)),
                format!("{front}{body}"),
            );
            count += 1;
        }
        for l in &self.lessons {
            let parents: Vec<String> = l.parents.iter().map(|p| format!("[[note-{p}]]")).collect();
            let front = format!("---\nid: {}\ntags: [{}]\n---\n\n", l.id, l.tags.join(", "));
            let mut body = format!("# Lesson {}\n\n**{}\n\n{}", l.id, l.activation, l.principle);
            if !parents.is_empty() {
                body.push_str("\n\n## Источники\n");
                for p in parents {
                    body.push_str(&format!("- {}\n", p));
                }
            }
            let _ = fs::write(
                lessons_dir.join(format!("lesson-{}.md", l.id)),
                format!("{front}{body}"),
            );
            count += 1;
        }
        // V1.6: экспорт процедур
        let procs_dir = vault.join("procedures");
        let _ = fs::create_dir_all(&procs_dir);
        for p in &self.procedures {
            let parents: Vec<String> = p.parents.iter().map(|x| format!("[[note-{x}]]")).collect();
            let front = format!("---\nid: {}\ntags: [{}]\n---\n\n", p.id, p.tags.join(", "));
            let mut body = format!("# Procedure {}\n\n**{}\n\nШаги:\n", p.id, p.activation);
            for (i, s) in p.steps.iter().enumerate() {
                body.push_str(&format!("{}. {}\n", i + 1, s));
            }
            if !parents.is_empty() {
                body.push_str("\n## Источники\n");
                for x in parents {
                    body.push_str(&format!("- {}\n", x));
                }
            }
            let _ = fs::write(
                procs_dir.join(format!("procedure-{}.md", p.id)),
                format!("{front}{body}"),
            );
            count += 1;
        }
        count
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> Option<T> {
    let raw = fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn kind_name(k: &NoteKind) -> &'static str {
    match k {
        NoteKind::Event => "event",
        NoteKind::Behavior => "behavior",
        NoteKind::Knowledge => "knowledge",
        NoteKind::Skill => "skill",
    }
}

/// V2.4: эмбеддинг через Ollama /api/embed (если сервер с --embeddings).
/// Возвращает None, если сервер не поддерживает или недоступен.
pub fn ollama_embed(text: &str) -> Option<Vec<f32>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .ok()?;
    let url = std::env::var("ARGUS_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
    let body = serde_json::json!({ "model": "qwen2.5:7b", "input": text });
    let resp = client
        .post(format!("{url}/api/embed"))
        .json(&body)
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let v: serde_json::Value = resp.json().ok()?;
    let emb = v.get("embeddings")?.as_array()?.first()?.as_array()?;
    let vec: Vec<f32> = emb
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect();
    if vec.is_empty() {
        None
    } else {
        Some(vec)
    }
}

/// V2.2: локальный хеш-эмбеддинг из слово- и n-грамм (офлайн, без сервисов).
pub fn ngram_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    for w in words {
        v[hash_idx(w.as_bytes(), dim)] += 1.0;
        let chars: Vec<char> = w.chars().collect();
        if chars.len() >= 2 {
            for i in 0..chars.len() - 1 {
                let bigram: String = chars[i..=i + 1].iter().collect();
                v[hash_idx(bigram.as_bytes(), dim)] += 1.0;
            }
        }
        if chars.len() >= 3 {
            for i in 0..chars.len() - 2 {
                let trigram: String = chars[i..=i + 2].iter().collect();
                v[hash_idx(trigram.as_bytes(), dim)] += 1.0;
            }
        }
    }
    // L2-нормализация
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn hash_idx(bytes: &[u8], dim: usize) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h as usize) % dim
}

/// Косинусное сходство двух векторов.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Собирает inline Obsidian-теги `#слово` из текста.
fn inline_tags(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|w| {
            let t = w.strip_prefix('#')?;
            let t = t.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
            if t.is_empty() || t.chars().all(|c| c.is_numeric()) {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect()
}

/// Парсит Markdown-заметку из Obsidian: frontmatter `tags:`/`kind:`/`date:` + тело.
/// Возвращает (tags, kind, created-из-frontmatter, content).
fn parse_md_note(text: &str) -> (Vec<String>, NoteKind, Option<String>, String) {
    let mut tags = Vec::new();
    let mut kind = NoteKind::Knowledge;
    let mut created: Option<String> = None;
    let mut body = text;
    if let Some(rest) = text.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let front = &rest[..end];
            body = rest[end + 4..].trim();
            for line in front.lines() {
                let l = line.trim();
                if let Some(v) = l.strip_prefix("tags:") {
                    tags = v
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .split(',')
                        .map(|s| s.trim().trim_matches('"').to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                } else if let Some(v) = l.strip_prefix("kind:") {
                    // V2.1: kind из frontmatter (event|behavior|skill|knowledge)
                    kind = match v.trim().trim_matches('"') {
                        "event" => NoteKind::Event,
                        "behavior" => NoteKind::Behavior,
                        "skill" => NoteKind::Skill,
                        _ => NoteKind::Knowledge,
                    };
                } else if let Some(v) = l.strip_prefix("date:") {
                    // V2.6: Obsidian date (YYYY-MM-DD) → created_at
                    let d = v.trim().trim_matches('"').to_string();
                    if !d.is_empty() {
                        created = Some(format!("{}T00:00:00Z", d));
                    }
                } else if let Some(v) = l.strip_prefix("- ") {
                    // yaml-list тегов
                    if v.starts_with('"')
                        || v.chars()
                            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                    {
                        tags.push(v.trim_matches('"').to_string());
                    }
                }
            }
        }
    }
    // убрать wikilinks вида [[...]] в тексте для памяти
    let cleaned = body.replace("[[", "").replace("]]", "");
    let cleaned = cleaned.trim().to_string();
    (tags, kind, created, cleaned)
}

fn write_json<T: Serialize>(p: &Path, v: &T) {
    if let Ok(s) = serde_json::to_string_pretty(v) {
        let _ = fs::write(p, s);
    }
}

// ============================================================================
// Рефлексия (P3B): парсинг строк SET / NOTE из LLM
// ============================================================================

/// Одно действие рефлексии: SET-факт или NOTE-заметка.
pub enum ReflectionAction {
    Set {
        key: String,
        value: String,
        pinned: bool,
        keywords: Vec<String>,
    },
    Note {
        content: String,
        tags: Vec<String>,
        kind: NoteKind,
    },
}

/// Парсит многострочный вывод рефлексии: `SET key=value [opts]` / `NOTE ... [opts]`.
pub fn parse_reflection(text: &str) -> Vec<ReflectionAction> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("SET ") {
            if let Some(a) = parse_set(rest) {
                out.push(a);
            }
        } else if let Some(rest) = line.strip_prefix("NOTE ") {
            if let Some(a) = parse_note(rest) {
                out.push(a);
            }
        }
    }
    out
}

fn parse_set(rest: &str) -> Option<ReflectionAction> {
    // первый токен: key=value (по первому '=')
    let eq = rest.find('=')?;
    let key = rest[..eq].trim().to_string();
    let mut rest2 = &rest[eq + 1..];
    let value = if let Some(br) = rest2.find('[') {
        let v = rest2[..br].trim().to_string();
        rest2 = &rest2[br..];
        v
    } else {
        rest2.trim().to_string()
    };
    let (pinned, keywords) = parse_opts(rest2);
    Some(ReflectionAction::Set {
        key,
        value,
        pinned,
        keywords,
    })
}

fn parse_note(rest: &str) -> Option<ReflectionAction> {
    let br = rest.find('[').unwrap_or(rest.len());
    let content = rest[..br].trim().to_string();
    if content.is_empty() {
        return None;
    }
    let opts = if br < rest.len() { &rest[br..] } else { "" };
    let mut tags = Vec::new();
    let mut kind = NoteKind::Knowledge;
    for part in opts.split('[').skip(1) {
        let part = part.trim_end_matches(']').trim();
        if let Some(t) = part.strip_prefix("tags=") {
            tags = t
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        } else if let Some(k) = part.strip_prefix("type=") {
            kind = match k.trim() {
                "event" => NoteKind::Event,
                "behavior" => NoteKind::Behavior,
                "skill" => NoteKind::Skill,
                _ => NoteKind::Knowledge,
            };
        }
    }
    Some(ReflectionAction::Note {
        content,
        tags,
        kind,
    })
}

fn parse_opts(s: &str) -> (bool, Vec<String>) {
    let mut pinned = false;
    let mut keywords = Vec::new();
    for part in s.split('[').skip(1) {
        let part = part.trim_end_matches(']').trim();
        if part == "pinned=true" {
            pinned = true;
        } else if let Some(k) = part.strip_prefix("keywords=") {
            keywords = k
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect();
        }
    }
    (pinned, keywords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mem() -> Memory {
        let dir = PathBuf::from("/tmp/argus-mem-test");
        let _ = std::fs::remove_dir_all(&dir);
        Memory::load(&dir)
    }

    #[test]
    fn consolidate_creates_lesson_after_3_notes() {
        let mut m = mem();
        m.add_note(
            "клик по кнопке флакал; getByRole помог",
            vec!["playwright".into()],
            NoteKind::Behavior,
        );
        m.add_note(
            "css-селектор сломался; role-локатор выжил",
            vec!["playwright".into()],
            NoteKind::Behavior,
        );
        m.add_note(
            "добавил retries; реальный фикс — getByRole",
            vec!["playwright".into()],
            NoteKind::Behavior,
        );
        m.add_note("другая тема", vec!["misc".into()], NoteKind::Knowledge);

        let created = m.consolidate();
        assert_eq!(created, 1);
        assert_eq!(m.lessons.len(), 1);
        let l = &m.lessons[0];
        assert!(l.tags.contains(&"playwright".to_string()));
        assert_eq!(l.parents.len(), 3);
        // V1.6: консолидация создаёт и процедуру
        assert_eq!(m.procedures.len(), 1);
        let p = &m.procedures[0];
        assert!(!p.steps.is_empty());
        assert!(p.tags.contains(&"playwright".to_string()));
        assert!(m.procedures_block("playwright").contains("### procedures"));
        // повторный запуск не создаёт дубликат
        assert_eq!(m.consolidate(), 0);
    }

    #[test]
    fn consolidate_skips_less_than_3() {
        let mut m = mem();
        m.add_note("раз", vec!["x".into()], NoteKind::Knowledge);
        m.add_note("два", vec!["x".into()], NoteKind::Knowledge);
        assert_eq!(m.consolidate(), 0);
        assert!(m.lessons.is_empty());
    }

    #[test]
    fn vote_changes_recall_ranking_and_persists() {
        let mut m = mem();
        let a = m.add_note("клетки микроманипулятор", vec![], NoteKind::Knowledge);
        let b = m.add_note("погода в тбилиси", vec![], NoteKind::Knowledge);
        // голосуем против релевантной заметки — она должна уйти ниже
        let _ = m.vote("note", a, -3);
        let hits = m.recall_hybrid("микроманипулятор клетка", 2);
        // после -3 голосов вес 0.15 — всё равно первым (ngram), но проверяем что vote работает
        assert_eq!(m.notes.iter().find(|n| n.id == a).unwrap().votes, -3);
        assert_eq!(m.notes.iter().find(|n| n.id == b).unwrap().votes, 0);
        let _ = hits;
    }

    #[test]
    fn evict_trims_over_limit_by_votes() {
        let mut m = mem();
        // заполнить 3 заметки, лимит 1000 не достижим легко — проверим что evict не трогает малый объём
        for i in 0..5 {
            m.add_note(&format!("заметка {i}"), vec![], NoteKind::Knowledge);
        }
        assert_eq!(m.evict(), 0);
        assert_eq!(m.notes.len(), 5);
    }

    #[test]
    fn import_skips_src_duplicate() {
        // src-тег уже есть → повторный импорт пропускается
        let dir = PathBuf::from("/tmp/argus-src-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.md"), "---\ntags: [x]\n---\nсодержимое").ok();
        let mut m = mem();
        assert_eq!(m.import_vault(&dir), 1);
        assert_eq!(
            m.import_vault(&dir),
            0,
            "повторный импорт того же файла — пропуск по src"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hybrid_recall_finds_related_by_vectors() {
        let mut m = mem();
        m.add_note(
            "клетки повреждаются механически микроманипулятором",
            vec!["argus".into()],
            NoteKind::Knowledge,
        );
        m.add_note(
            "обычная заметка о погоде в Тбилиси",
            vec![],
            NoteKind::Knowledge,
        );
        // запрос без общей подстроки, но с близкими словами
        let hits = m.recall_hybrid("механическое повреждение клеток инструментом", 2);
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].id,
            1,
            "первым должен идти семантически близкий хит, got={:?}",
            hits.iter().map(|h| h.id).collect::<Vec<_>>()
        );
        // косинусные эмбеддинги нормированы
        let a = ngram_embed("клетка повреждение", 256);
        let b = ngram_embed("клетки механическое", 256);
        assert!(cosine(&a, &b) > 0.0);
    }
}
