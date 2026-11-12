//! M3/§2.6 — `argus-cka`: первый локальный тест «поля смыслов».
//!
//! Берём корпус (абзацы из *.md/*.txt), считаем представления двумя независимыми
//! системами и меряем сходимость (linear CKA) против перестановочной нулевой модели.
//!
//! Смысл: если семантика «предсуществует» отдельному приёмнику (PRH, arXiv:2405.07987),
//! то независимые системы должны согласовываться по объектам лучше случайного.
//!
//! Примеры:
//!   argus-agent-cka --corpus ~/Desktop/Services/publications --n 60 --perms 500
//!   argus-agent-cka --corpus DIR --a ollama:gemma-4-e4b-it --b ollama:qwen2.5:7b
//!   argus-agent-cka --corpus DIR --a wordhash --b ngram --json

use argus_pre::cka::{
    chunks, cka, mean_nn_cosine, permutation_test, EmbedVec, NgramVec, OllamaVec, RatingsVec,
    Vectorizer, WordHashVec,
};
use std::fs;
use std::path::{Path, PathBuf};

fn arg_str(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg_num(args: &[String], key: &str) -> Option<u64> {
    arg_str(args, key).and_then(|v| v.parse().ok())
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>, depth: usize, max_files: usize) {
    if out.len() >= max_files || depth > 5 {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out, depth + 1, max_files);
        } else {
            let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("");
            if matches!(ext, "md" | "txt") {
                out.push(p);
                if out.len() >= max_files {
                    return;
                }
            }
        }
    }
}

fn loader(spec: &str) -> Box<dyn Vectorizer> {
    if let Some(model) = spec.strip_prefix("ollama:") {
        return Box::new(OllamaVec::new(model));
    }
    if let Some(model) = spec.strip_prefix("ratings:") {
        return Box::new(RatingsVec::new(model));
    }
    if let Some(model) = spec.strip_prefix("emb:") {
        return Box::new(EmbedVec::new(model));
    }
    match spec {
        "ngram" => Box::new(NgramVec::new(256)),
        "wordhash" => Box::new(WordHashVec::new(512, 0)),
        "wordhash2" => Box::new(WordHashVec::new(512, 0xDEAD_BEEF)),
        other => {
            eprintln!("неизвестная система «{other}»: ngram | wordhash | wordhash2 | ollama:<model> | ratings:<model> | emb:<model>");
            std::process::exit(2);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(corpus) = arg_str(&args, "--corpus") else {
        println!("argus-agent-cka --corpus <dir|file> [--corpus2 DIR (контроль)] [--n 60] [--perms 500] [--a ngram] [--b wordhash] [--seed 42] [--json]\n  системы: ngram | wordhash | wordhash2 | ollama:<model> | ratings:<model> | emb:<model>\n  emb: нужен ARGUS_EMBED_URL (по умолчанию http://127.0.0.1:8791/v1/embeddings)");
        return;
    };
    let n = arg_num(&args, "--n").unwrap_or(60) as usize;
    let perms = arg_num(&args, "--perms").unwrap_or(500) as usize;
    let seed = arg_num(&args, "--seed").unwrap_or(42);
    let corpus2 = arg_str(&args, "--corpus2");
    let spec_a = arg_str(&args, "--a").unwrap_or_else(|| "ngram".to_string());
    let spec_b = arg_str(&args, "--b").unwrap_or_else(|| "wordhash".to_string());
    let json_out = args.iter().any(|a| a == "--json");

    let root = PathBuf::from(&corpus);
    let mut files = Vec::new();
    if root.is_file() {
        files.push(root.clone());
    } else {
        collect(&root, &mut files, 0, 300);
    }
    if files.is_empty() {
        println!("в корпусе нет *.md/*.txt: {}", root.display());
        return;
    }

    let mut items: Vec<String> = Vec::new();
    for f in &files {
        if let Ok(text) = fs::read_to_string(f) {
            for c in chunks(&text, n) {
                if items.len() >= n {
                    break;
                }
                items.push(c);
            }
        }
        if items.len() >= n {
            break;
        }
    }
    // контроль: второй набор текстов из ДРУГОГО корпуса (соответствия объектов нет)
    if let Some(c2) = &corpus2 {
        let root2 = PathBuf::from(c2);
        let mut files2 = Vec::new();
        if root2.is_file() {
            files2.push(root2.clone());
        } else {
            collect(&root2, &mut files2, 0, 300);
        }
        let mut items2: Vec<String> = Vec::new();
        for f in &files2 {
            if let Ok(text) = fs::read_to_string(f) {
                for c in chunks(&text, n) {
                    if items2.len() >= n {
                        break;
                    }
                    items2.push(c);
                }
            }
            if items2.len() >= n {
                break;
            }
        }
        if items2.len() < items.len() {
            println!(
                "контроль невозможен: во втором корпусе {} абзацев, нужно {}",
                items2.len(),
                items.len()
            );
            return;
        }
        let half = items.len().min(items2.len());
        let kept: Vec<String> = items2.into_iter().take(half).collect();
        items = items.into_iter().take(half).collect();
        items2 = kept;
        // второй набор подменяем на тексты другого корпуса: pairing разрушен по содержанию
        let va = loader(&spec_a);
        let vb = loader(&spec_b);
        let x: Vec<Vec<f32>> = items.iter().map(|t| va.embed(t)).collect();
        let y: Vec<Vec<f32>> = items2.iter().map(|t| vb.embed(t)).collect();
        if x.iter().any(|r| r.is_empty()) || y.iter().any(|r| r.is_empty()) {
            println!("контроль: векторизатор вернул пустые векторы");
            return;
        }
        let t = permutation_test(&x, &y, perms, seed);
        println!("КОНТРОЛЬ «разные корпуса»: {} абзацев, B берёт текст из {}", t.n, c2);
        println!("CKA = {:.3} · нуль: среднее {:.3}, 95% {:.3} · p = {:.3}", t.observed, t.null_mean, t.null_p95, t.p_value);
        println!(
            "ожидание: сходимость на уровне нуля (p > 0.05). Факт: {}",
            if t.p_value > 0.05 { "✅ на уровне нуля" } else { "❌ выше нуля — метрика ловит не содержание, а форму" }
        );
        return;
    }

    if items.len() < 8 {
        println!(
            "слишком мало абзацев ({}); нужно ≥8 — возьмите корпус побольше",
            items.len()
        );
        return;
    }

    let va = loader(&spec_a);
    let vb = loader(&spec_b);
    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<Vec<f32>> = Vec::new();
    let mut failed = 0usize;
    for t in &items {
        let a = va.embed(t);
        let b = vb.embed(t);
        if a.is_empty() || b.is_empty() {
            failed += 1;
            continue;
        }
        x.push(a);
        y.push(b);
    }
    if x.len() < 8 {
        println!(
            "годных объектов {} (отброшено {failed}): нужно ≥8. Для ollama проверьте, что сервер отвечает на /api/generate",
            x.len()
        );
        return;
    }
    if failed > 0 {
        println!("предупреждение: {failed} объектов отброшено (модель не вернула 8 чисел)");
    }

    let t = permutation_test(&x, &y, perms, seed);
    let self_cka = cka(&x, &x);
    let nn_a = mean_nn_cosine(&x);
    let nn_b = mean_nn_cosine(&y);
    let verdict = if t.p_value <= 0.05 && t.observed > t.null_p95 {
        "сходимость выше перестановочного нуля → совместная семантическая структура (согласуется с PRH в слабой форме)"
    } else {
        "сходимость не отличима от случайной → общей семантики для этих двух систем не показано"
    };

    if json_out {
        println!(
            "{{\"n\":{},\"perms\":{},\"cka\":{:.4},\"null_mean\":{:.4},\"null_p95\":{:.4},\"p\":{:.4},\"self_cka\":{:.4},\"nn_a\":{:.4},\"nn_b\":{:.4},\"a\":\"{}\",\"b\":\"{}\"}}",
            t.n, t.perms, t.observed, t.null_mean, t.null_p95, t.p_value, self_cka, nn_a, nn_b, va.name(), vb.name()
        );
    } else {
        println!("корпус: {} абзацев из {} файлов (в тесте {})", items.len(), files.len(), x.len());
        println!("A: {} (dim {}) · B: {} (dim {})", va.name(), x[0].len(), vb.name(), y[0].len());
        println!(
            "CKA = {:.3} · нуль: среднее {:.3}, 95% {:.3} · p = {:.3} ({} перестановок)",
            t.observed, t.null_mean, t.null_p95, t.p_value, t.perms
        );
        println!("контроль: CKA(X,X) = {:.3} (должно быть 1.000)", self_cka);
        println!("ближайший сосед: A {:.3} · B {:.3} (вырожденность: чем ближе к 1, тем хуже)", nn_a, nn_b);
        println!("вердикт: {verdict}");
    }

    // запись результата в состояние агента (если задано)
    if let Ok(state) = std::env::var("ARGUS_STATE_DIR") {
        let dir = PathBuf::from(state).join("ledger");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join(format!(
            "cka_{}.json",
            chrono::Local::now().format("%Y-%m-%d_%H%M%S")
        ));
        let rec = format!(
            "{{\"when\":\"{}\",\"corpus\":\"{}\",\"files\":{},\"n\":{},\"perms\":{},\"a\":\"{}\",\"b\":\"{}\",\"cka\":{:.4},\"null_mean\":{:.4},\"null_p95\":{:.4},\"p\":{:.4},\"verdict\":\"{}\"}}",
            chrono::Local::now().to_rfc3339(),
            root.display(),
            files.len(),
            t.n,
            t.perms,
            va.name(),
            vb.name(),
            t.observed,
            t.null_mean,
            t.null_p95,
            t.p_value,
            if t.p_value <= 0.05 { "converge" } else { "no-convergence" }
        );
        let _ = fs::write(&path, rec);
        if !json_out {
            println!("запись: {}", path.display());
        }
    }
}
