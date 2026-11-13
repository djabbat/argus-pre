//! M3/§2.6 — CKA: первый настоящий тест «поля смыслов» (сходимость семантики сетей).
//!
//! Что проверяем: если смысл «предсуществует» отдельному приёмнику (Huh et al., PRH,
//! arXiv:2405.07987), то две системы, обученные/построенные по-разному, должны давать
//! **согласованные** представления одних и тех же объектов — выше, чем при случайной
//! перестановке объектов.
//!
//! Метрика: linear CKA (Kornblith et al., arXiv:1905.00414) на грам-матрицах:
//!   CKA(X, Y) = <A, B>_F / (||A||_F · ||B||_F),  A = Xc Xcᵀ, B = Yc Ycᵀ.
//! Нулевая модель — перестановка строк/столбцов B (объекты перемешаны, структура та же).

use crate::memory::{cosine, ngram_embed};

/// Векторизатор: имя + функция. `Probe` — отдельный, независимый от `ngram` (для контроля).
pub trait Vectorizer {
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Штатный `ngram_embed` из памяти агента (символьные 1–3-граммы, hash, L2).
pub struct NgramVec {
    dim: usize,
}

impl NgramVec {
    pub fn new(dim: usize) -> Self {
        NgramVec { dim }
    }
}

impl Vectorizer for NgramVec {
    fn name(&self) -> &str {
        "ngram"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        ngram_embed(text, self.dim)
    }
}

/// Независимый векторизатор: bag-of-words (униграммы+биграммы слов), другой hash-seed.
/// Нужен как «вторая система» в офлайне, когда нет двух разных LLM.
pub struct WordHashVec {
    dim: usize,
    seed: u64,
}

impl WordHashVec {
    pub fn new(dim: usize, seed: u64) -> Self {
        WordHashVec { dim, seed }
    }

    fn idx(&self, s: &str) -> usize {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ self.seed;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        ((h ^ (h >> 32)) as usize) % self.dim
    }
}

impl Vectorizer for WordHashVec {
    fn name(&self) -> &str {
        "wordhash"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let low = text.to_lowercase();
        let words: Vec<&str> = low
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();
        for w in &words {
            v[self.idx(w)] += 1.0;
        }
        for pair in words.windows(2) {
            let bigram = format!("{}_{}", pair[0], pair[1]);
            v[self.idx(&bigram)] += 0.5;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

/// Векторизатор «второй модели» через Ollama (реальная проверка gemma vs qwen).
pub struct OllamaVec {
    model: String,
    name: String,
    dim_cache: std::cell::Cell<usize>,
}

impl OllamaVec {
    pub fn new(model: &str) -> Self {
        OllamaVec {
            model: model.to_string(),
            name: format!("ollama:{model}"),
            dim_cache: std::cell::Cell::new(0),
        }
    }
}

impl Vectorizer for OllamaVec {
    fn name(&self) -> &str {
        &self.name
    }
    fn dim(&self) -> usize {
        self.dim_cache.get()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        let v = crate::memory::ollama_embed_model(text, &self.model).unwrap_or_default();
        if !v.is_empty() {
            self.dim_cache.set(v.len());
        }
        v
    }
}

fn center(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let d = x[0].len();
    let mut mean = vec![0f64; d];
    for row in x {
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    x.iter()
        .map(|row| row.iter().enumerate().map(|(j, v)| v - mean[j]).collect())
        .collect()
}

/// Линейная грам-матрица A = Xc Xcᵀ.
pub fn gram(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let xc = center(x);
    let n = xc.len();
    let mut a = vec![vec![0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0f64;
            for (a, b) in xc[i].iter().zip(xc[j].iter()) {
                s += a * b;
            }
            a[i][j] = s;
            a[j][i] = s;
        }
    }
    a
}

fn frob(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

fn inner(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut s = 0f64;
    for i in 0..n {
        for j in 0..n {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

/// CKA по готовым грам-матрицам.
pub fn cka_from_grams(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let na = frob(a);
    let nb = frob(b);
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (inner(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// CKA по двум наборам представлений.
pub fn cka(x: &[Vec<f32>], y: &[Vec<f32>]) -> f64 {
    let xf: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().map(|v| *v as f64).collect())
        .collect();
    let yf: Vec<Vec<f64>> = y
        .iter()
        .map(|r| r.iter().map(|v| *v as f64).collect())
        .collect();
    cka_from_grams(&gram(&xf), &gram(&yf))
}

/// Детерминированный RNG (тот же SplitMix64, что в леджере) — без новых зависимостей.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermTest {
    pub observed: f64,
    pub null_mean: f64,
    pub null_p95: f64,
    pub p_value: f64,
    pub perms: usize,
    pub n: usize,
}

/// Перестановочный тест: нулевая гипотеза — соответствие объектов между X и Y случайно.
pub fn permutation_test(x: &[Vec<f32>], y: &[Vec<f32>], perms: usize, seed: u64) -> PermTest {
    let n = x.len().min(y.len());
    let xf: Vec<Vec<f64>> = x[..n]
        .iter()
        .map(|r| r.iter().map(|v| *v as f64).collect())
        .collect();
    let yf: Vec<Vec<f64>> = y[..n]
        .iter()
        .map(|r| r.iter().map(|v| *v as f64).collect())
        .collect();
    let a = gram(&xf);
    let b = gram(&yf);
    let observed = cka_from_grams(&a, &b);
    let na = frob(&a);
    let nb = frob(&b);

    let mut rng = Rng(seed);
    let mut null = Vec::with_capacity(perms);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut at_least = 0usize;
    for _ in 0..perms.max(1) {
        rng.shuffle(&mut idx);
        // перестановка строк и столбцов одной и той же матрицей
        let mut bp = vec![vec![0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                bp[i][j] = b[idx[i]][idx[j]];
            }
        }
        let val = if na > 0.0 && nb > 0.0 {
            (inner(&a, &bp) / (na * nb)).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        if val >= observed {
            at_least += 1;
        }
        null.push(val);
    }
    let mut sorted = null.clone();
    sorted.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let null_mean = null.iter().sum::<f64>() / null.len() as f64;
    let p95 = sorted
        .get(((sorted.len() as f64 * 0.95) as usize).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0);
    PermTest {
        observed,
        null_mean,
        null_p95: p95,
        p_value: (at_least as f64 + 1.0) / (null.len() as f64 + 1.0),
        perms: perms.max(1),
        n,
    }
}

/// Чанки корпуса: абзацы ≥ 60 символов, не более `max` штук.
pub fn chunks(text: &str, max: usize) -> Vec<String> {
    text.split("\n\n")
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|s| s.chars().count() >= 60)
        .take(max)
        .collect()
}

/// Средняя косинусная близость между ближайшими соседями (дополнительный сигнал: не вырождены ли представления).
pub fn mean_nn_cosine(x: &[Vec<f32>]) -> f64 {
    if x.len() < 3 {
        return 0.0;
    }
    let mut sum = 0f64;
    let mut cnt = 0usize;
    for i in 0..x.len() {
        let mut best = -2f32;
        for j in 0..x.len() {
            if i == j {
                continue;
            }
            let c = cosine(&x[i], &x[j]);
            if c > best {
                best = c;
            }
        }
        sum += best as f64;
        cnt += 1;
    }
    sum / cnt as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(n: usize) -> Vec<String> {
        let topics = [
            "centriole elimination in the early embryo requires careful timing and markers",
            "bayesian inference conditions a prior on sensory evidence in the brain",
            "microscope autofocus drift over seventy two hours needs fiducial markers",
            "sleep consolidates memory through replay of the previous day episodes",
            "language models compute only after a prompt and waste idle time between turns",
            "senescent cells accumulate damage and change nuclear morphology visibly",
        ];
        (0..n)
            .map(|i| format!("{} {}", topics[i % topics.len()], i))
            .collect()
    }

    #[test]
    fn same_representation_gives_one() {
        let v = WordHashVec::new(128, 7);
        let xs: Vec<Vec<f32>> = corpus(20).iter().map(|t| v.embed(t)).collect();
        let c = cka(&xs, &xs);
        assert!((c - 1.0).abs() < 1e-6, "CKA(X,X)={c}");
    }

    #[test]
    fn independent_vectorizers_beat_permutation() {
        let a = WordHashVec::new(256, 11);
        let b = NgramVec::new(256);
        let items = corpus(24);
        let x: Vec<Vec<f32>> = items.iter().map(|t| a.embed(t)).collect();
        let y: Vec<Vec<f32>> = items.iter().map(|t| b.embed(t)).collect();
        let t = permutation_test(&x, &y, 200, 42);
        assert!(
            t.observed > t.null_p95,
            "наблюдаемая CKA должна быть выше 95% нуля: {t:?}"
        );
        assert!(t.p_value <= 0.05, "p={} ({t:?})", t.p_value);
    }

    #[test]
    fn shuffled_pairs_fail_the_test() {
        let a = WordHashVec::new(256, 3);
        let b = NgramVec::new(256);
        let items = corpus(24);
        let x: Vec<Vec<f32>> = items.iter().map(|t| a.embed(t)).collect();
        // второй набор — те же тексты, но в другом порядке И про другой повод: связь разрушена
        let other: Vec<String> = (0..items.len())
            .map(|i| format!("совершенно посторонний текст номер {i} про погоду и курсы валют"))
            .collect();
        let y: Vec<Vec<f32>> = other.iter().map(|t| b.embed(t)).collect();
        let t = permutation_test(&x, &y, 200, 42);
        assert!(t.p_value > 0.05, "шум не должен проходить тест: {t:?}");
    }

    #[test]
    fn chunks_filters_short_paragraphs() {
        let text = "короткий\n\nэто достаточно длинный абзац, в котором точно больше шестидесяти символов, так и есть\n\nещё";
        let c = chunks(text, 10);
        assert_eq!(c.len(), 1);
    }
}

// ── M3: «две независимые обученные системы» без embeddings-эндпоинта ──────────
///
/// Семантическая проекция через оценки: модель просят оценить текст по 8 осям
/// (0–10, целые), получаем 8-мерное представление. Две модели из разных семейств
/// дают два независимых представления одного и того же корпуса → CKA между ними.
/// Ограничение честное: 8 осей — грубо; это проверка согласия семантической
/// геометрии, а не полная репликация PRH.
pub const RATING_AXES: [&str; 8] = [
    "biological_or_technical",
    "quantitative",
    "methodological",
    "clinical",
    "philosophical",
    "historical",
    "mathematical",
    "experimental",
];

pub struct RatingsVec {
    model: String,
    name: String,
    cache: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>>,
}

impl RatingsVec {
    pub fn new(model: &str) -> Self {
        RatingsVec {
            model: model.to_string(),
            name: format!("ratings:{model}"),
            cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn prompt(text: &str) -> String {
        format!(
            "Rate the following text on 8 dimensions, integers 0-10, separated by commas, nothing else.\n\
             Axes: {}\n\nTEXT: {}\n\nAnswer with exactly 8 integers.",
            RATING_AXES.join(", "),
            text.chars().take(600).collect::<String>()
        )
    }

    fn call(&self, text: &str) -> Vec<f32> {
        for attempt in 0..3u32 {
            let v = self.call_once(text, attempt);
            if v.len() == RATING_AXES.len() {
                return v;
            }
        }
        vec![]
    }

    fn call_once(&self, text: &str, attempt: u32) -> Vec<f32> {
        let url =
            std::env::var("ARGUS_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
        {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let body = serde_json::json!({
            "model": self.model,
            "prompt": if attempt == 0 {
                RatingsVec::prompt(text)
            } else {
                format!(
                    "{}\n\nONLY the 8 integers separated by commas. No words, no explanation.",
                    RatingsVec::prompt(text)
                )
            },
            "stream": false,
            "options": { "temperature": 0, "num_predict": 60 }
        });
        let resp = match client
            .post(format!("{url}/api/generate"))
            .json(&body)
            .send()
        {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        let v: serde_json::Value = match resp.json() {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let out = v.get("response").and_then(|r| r.as_str()).unwrap_or("");
        parse_ratings(out, RATING_AXES.len())
    }
}

/// Чистый парсер: первые `k` чисел в диапазоне 0..=10, нормированные в 0..1.
pub fn parse_ratings(text: &str, k: usize) -> Vec<f32> {
    let mut nums: Vec<f32> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(v) = cur.parse::<f32>() {
                if (0.0..=10.0).contains(&v) {
                    nums.push(v / 10.0);
                }
            }
            cur.clear();
        }
        if nums.len() == k {
            break;
        }
    }
    if nums.len() < k && !cur.is_empty() {
        if let Ok(v) = cur.parse::<f32>() {
            if (0.0..=10.0).contains(&v) {
                nums.push(v / 10.0);
            }
        }
    }
    if nums.len() == k {
        nums
    } else {
        vec![]
    }
}

impl Vectorizer for RatingsVec {
    fn name(&self) -> &str {
        &self.name
    }
    fn dim(&self) -> usize {
        RATING_AXES.len()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        if let Some(v) = self.cache.borrow().get(text) {
            return v.clone();
        }
        let v = self.call(text);
        if !v.is_empty() {
            self.cache.borrow_mut().insert(text.to_string(), v.clone());
        }
        v
    }
}

#[cfg(test)]
mod ratings_tests {
    use super::*;

    #[test]
    fn parses_eight_ratings() {
        let v = parse_ratings("[7, 9, 8, 2, 5, 1, 6, 10]", 8);
        assert_eq!(v.len(), 8);
        assert!((v[0] - 0.7).abs() < 1e-6);
        assert!((v[7] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_incomplete_or_out_of_range() {
        assert!(
            parse_ratings("7, 9, 8", 8).is_empty(),
            "меньше 8 чисел — брак"
        );
        assert!(
            parse_ratings("7,9,8,2,5,1,6,99", 8).is_empty(),
            "99 вне диапазона"
        );
    }

    #[test]
    fn tolerates_chatter() {
        let v = parse_ratings("Sure! Ratings: 7 9 8 2 5 1 6 10 — hope that helps", 8);
        assert_eq!(v.len(), 8);
    }
}

// ── M3++: настоящие энкодеры через OpenAI-совместимый /v1/embeddings ──────────
///
/// Зачем: у локального Ollama 0.32.9 нет `/api/embed`, поэтому пару «двух независимых
/// энкодеров» поднимаем через `scripts/embed_server.py` (sentence-transformers).
/// URL берётся из `ARGUS_EMBED_URL` (по умолчанию `http://127.0.0.1:8791/v1/embeddings`).
pub struct EmbedVec {
    model: String,
    name: String,
    dim_cache: std::cell::Cell<usize>,
}

impl EmbedVec {
    pub fn new(model: &str) -> Self {
        EmbedVec {
            model: model.to_string(),
            name: format!("emb:{model}"),
            dim_cache: std::cell::Cell::new(0),
        }
    }

    fn url() -> String {
        std::env::var("ARGUS_EMBED_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8791/v1/embeddings".to_string())
    }

    fn call(&self, text: &str) -> Vec<f32> {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
        {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let body = serde_json::json!({ "model": self.model, "input": text });
        let resp = match client.post(EmbedVec::url()).json(&body).send() {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        let txt = match resp.text() {
            Ok(t) => t,
            Err(_) => return vec![],
        };
        parse_embeddings_response(&txt).unwrap_or_default()
    }
}

/// Чистый парсер ответа OpenAI-совместимого `/v1/embeddings`.
pub fn parse_embeddings_response(body: &str) -> Option<Vec<f32>> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let emb = v.get("data")?.get(0)?.get("embedding")?.as_array()?;
    let out: Vec<f32> = emb
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

impl Vectorizer for EmbedVec {
    fn name(&self) -> &str {
        &self.name
    }
    fn dim(&self) -> usize {
        self.dim_cache.get()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        let v = self.call(text);
        if !v.is_empty() {
            self.dim_cache.set(v.len());
        }
        v
    }
}

#[cfg(test)]
mod embed_tests {
    use super::*;

    #[test]
    fn parses_openai_embedding_response() {
        let body = r#"{"object":"list","model":"x","data":[{"object":"embedding","index":0,"embedding":[0.1,-0.2,0.3]}]}"#;
        let v = parse_embeddings_response(body).unwrap();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 0.1).abs() < 1e-6);
        assert!(parse_embeddings_response(r#"{"data":[]}"#).is_none());
        assert!(parse_embeddings_response("not json").is_none());
    }

    #[test]
    fn cka_is_one_for_identical_encoder_output() {
        // два «энкодера» с одинаковыми векторами: CKA обязана быть 1.0
        let xs = vec![
            vec![0.1f32, 0.5, 0.2],
            vec![0.3, 0.1, 0.9],
            vec![0.7, 0.2, 0.4],
            vec![0.2, 0.8, 0.1],
        ];
        assert!((cka(&xs, &xs) - 1.0).abs() < 1e-6);
    }
}
