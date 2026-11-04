//! CDP browser (V1.4): headless Chrome + Chrome DevTools Protocol (WebSocket).
//! Инструменты: navigate / click / type / snapshot — реальное управление браузером.
use std::process::{Child, Command};
use std::sync::{Mutex, OnceLock};

const PORT: u16 = 9222;
const CHROME_BIN: &str = "google-chrome";

static CHROME: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

/// Ленивый запуск headless Chrome с remote-debugging-портом.
pub fn ensure_chrome() -> Result<(), String> {
    let lock = CHROME.get_or_init(|| Mutex::new(None));
    let mut guard = lock
        .lock()
        .map_err(|_| "chrome: lock poisoned".to_string())?;
    let alive = guard
        .as_mut()
        .map(|c| c.try_wait().map(|o| o.is_none()).unwrap_or(false))
        .unwrap_or(false);
    if alive {
        return Ok(());
    }
    let profile = format!(
        "/tmp/argus-cdp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let child = Command::new(CHROME_BIN)
        .args([
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--user-data-dir",
            &profile,
            "--remote-debugging-port=9222",
            "about:blank",
        ])
        // chrome не должен держать stdout/stderr родителя (иначе shell/тест ждут его fd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("chrome: {e} (нужен {CHROME_BIN} в PATH)"))?;
    *guard = Some(child);
    // ждём, пока порт откроется
    for _ in 0..40 {
        if std::net::TcpStream::connect(("127.0.0.1", PORT)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    Err("chrome: порт 9222 не поднялся".into())
}

/// Берёт WebSocket-URL первой page-вкладки из /json.
fn page_ws() -> Option<String> {
    let body = reqwest::blocking::get(format!("http://127.0.0.1:{PORT}/json"))
        .ok()?
        .text()
        .ok()?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    v.as_array()?
        .iter()
        .find(|p| p.get("type").and_then(|t| t.as_str()) == Some("page"))
        .and_then(|p| {
            p.get("webSocketDebuggerUrl")
                .and_then(|u| u.as_str())
                .map(String::from)
        })
}

/// Runtime.evaluate через CDP (WebSocket), возвращает значение (строка).
pub fn cdp_eval(expr: &str) -> Result<String, String> {
    ensure_chrome()?;
    let ws_url = page_ws().ok_or("нет page-вкладки")?;
    let (mut sock, _) = tungstenite::connect(&ws_url).map_err(|e| format!("ws: {e}"))?;
    let id = 1u64;
    let msg = serde_json::json!({
        "id": id,
        "method": "Runtime.evaluate",
        "params": { "expression": expr, "returnByValue": true }
    })
    .to_string();
    sock.send(tungstenite::Message::Text(msg))
        .map_err(|e| format!("send: {e}"))?;
    loop {
        let m = sock.read().map_err(|e| format!("read: {e}"))?;
        if let tungstenite::Message::Text(t) = m {
            let v: serde_json::Value = serde_json::from_str(&t).unwrap_or_default();
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                let s = v
                    .get("result")
                    .and_then(|r| r.get("result"))
                    .and_then(|r| r.get("value"))
                    .and_then(|r| r.as_str())
                    .unwrap_or("(не строка)")
                    .to_string();
                return Ok(s);
            }
        }
    }
}

/// Открывает новую вкладку с URL (CDP /json/new?url) и возвращает заголовок.
pub fn navigate(url: &str) -> Result<String, String> {
    ensure_chrome()?;
    let client = reqwest::blocking::Client::new();
    let u = format!("http://127.0.0.1:{PORT}/json/new?{}", url);
    let resp = client
        .put(&u)
        .send()
        .map_err(|e| format!("open tab: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("open tab: HTTP {}", resp.status()));
    }
    std::thread::sleep(std::time::Duration::from_millis(1500));
    cdp_eval("document.title").or_else(|_| Ok("(страница открыта)".into()))
}

pub fn click(selector: &str) -> Result<String, String> {
    let expr = format!(
        "(()=>{{const e=document.querySelector({:?}); if(!e) return 'NOT_FOUND'; e.click(); return 'clicked';}})()",
        selector
    );
    cdp_eval(&expr)
}

pub fn type_text(selector: &str, text: &str) -> Result<String, String> {
    let expr = format!(
        "(()=>{{const e=document.querySelector({:?}); if(!e) return 'NOT_FOUND'; \
         e.focus(); e.value={:?}; e.dispatchEvent(new Event('input',{{bubbles:true}})); \
         e.dispatchEvent(new Event('change',{{bubbles:true}})); return 'typed';}})()",
        selector, text
    );
    cdp_eval(&expr)
}

pub fn snapshot() -> Result<String, String> {
    cdp_eval("document.title + '\\n' + (document.body ? document.body.innerText : '')")
}

/// Шаг сценария для browser.flow.
#[derive(serde::Deserialize, Debug, Default)]
#[serde(default)]
pub struct FlowStep {
    pub op: String, // navigate | click | type | eval | snapshot
    pub url: String,
    pub selector: String,
    pub text: String,
    pub expression: String,
}

/// browser.flow: выполняет последовательность шагов и возвращает итоговый снимок.
pub fn flow(steps: &[FlowStep]) -> Result<String, String> {
    for s in steps {
        match s.op.as_str() {
            "navigate" => {
                navigate(&s.url)?;
            }
            "click" => {
                click(&s.selector)?;
            }
            "type" => {
                type_text(&s.selector, &s.text)?;
            }
            "eval" => {
                cdp_eval(&s.expression)?;
            }
            "snapshot" => {}
            other => return Err(format!("browser.flow: неизвестный op '{other}'")),
        }
    }
    snapshot()
}

/// Снимает скриншот страницы (CDP Page.captureScreenshot) в PNG-файл.
pub fn screenshot(path: &str) -> Result<String, String> {
    use base64::Engine;
    ensure_chrome()?;
    let ws_url = page_ws().ok_or("нет page-вкладки")?;
    let (mut sock, _) = tungstenite::connect(&ws_url).map_err(|e| format!("ws: {e}"))?;
    let id = 1u64;
    let msg = serde_json::json!({
        "id": id,
        "method": "Page.captureScreenshot",
        "params": { "format": "png" }
    })
    .to_string();
    sock.send(tungstenite::Message::Text(msg))
        .map_err(|e| format!("send: {e}"))?;
    let data: String = loop {
        let m = sock.read().map_err(|e| format!("read: {e}"))?;
        if let tungstenite::Message::Text(t) = m {
            let v: serde_json::Value = serde_json::from_str(&t).unwrap_or_default();
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                break v
                    .get("result")
                    .and_then(|r| r.get("data"))
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
            }
        }
    };
    if data.is_empty() {
        return Err("screenshot: пустой результат".into());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| format!("base64: {e}"))?;
    std::fs::write(path, &bytes).map_err(|e| format!("write: {e}"))?;
    Ok(format!("screenshot saved: {path} ({} байт)", bytes.len()))
}

/// Завершает Chrome (вызывается при выходе CLI).
pub fn shutdown() {
    if let Some(lock) = CHROME.get() {
        if let Ok(mut g) = lock.lock() {
            if let Some(mut c) = g.take() {
                let _ = c.kill();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Реальный CDP-тест: требует headless Chrome в PATH.
    #[test]
    #[ignore = "требует google-chrome в PATH"]
    fn cdp_navigate_title() {
        let t = crate::browser::navigate("https://example.com").unwrap();
        assert!(t.contains("Example Domain"), "title = {t}");
        let s = crate::browser::snapshot().unwrap();
        assert!(s.contains("Example Domain"));
        crate::browser::shutdown();
    }

    #[test]
    #[ignore = "требует google-chrome в PATH"]
    fn cdp_flow_steps() {
        // V1.7: многошаговый сценарий navigate→eval→snapshot
        use crate::browser::FlowStep;
        let steps = vec![
            FlowStep {
                op: "navigate".into(),
                url: "https://example.com".into(),
                ..Default::default()
            },
            FlowStep {
                op: "eval".into(),
                expression: "document.title".into(),
                ..Default::default()
            },
        ];
        let snap = crate::browser::flow(&steps).unwrap();
        assert!(snap.contains("Example Domain"), "snap = {snap}");
        crate::browser::shutdown();
    }
}
