//! V3.6: минимальный MCP-клиент (Model Context Protocol, JSON-RPC 2.0 по stdio).
//! Серверы объявляются в <stateDir>/mcp_servers.json: [{name, command, args}].
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Stdio};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

pub fn load_servers(state_dir: &Path) -> Vec<McpServer> {
    std::fs::read_to_string(state_dir.join("mcp_servers.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_servers(state_dir: &Path, servers: &[McpServer]) {
    if let Ok(s) = serde_json::to_string_pretty(servers) {
        let _ = std::fs::write(state_dir.join("mcp_servers.json"), s);
    }
}

pub fn add_server(
    state_dir: &Path,
    name: &str,
    command: &str,
    args: &[String],
) -> Result<(), String> {
    if name.is_empty() || command.is_empty() {
        return Err("нужны name и command".into());
    }
    let mut servers = load_servers(state_dir);
    if servers.iter().any(|s| s.name == name) {
        return Err(format!("сервер '{name}' уже есть"));
    }
    servers.push(McpServer {
        name: name.to_string(),
        command: command.to_string(),
        args: args.to_vec(),
    });
    save_servers(state_dir, &servers);
    Ok(())
}

pub fn remove_server(state_dir: &Path, name: &str) -> Result<(), String> {
    let mut servers = load_servers(state_dir);
    let before = servers.len();
    servers.retain(|s| s.name != name);
    if servers.len() == before {
        return Err(format!("сервер '{name}' не найден"));
    }
    save_servers(state_dir, &servers);
    Ok(())
}

/// Подключённый MCP-сервер (процесс + stdin/stdout).
pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    id: u64,
    pub name: String,
}

impl McpClient {
    pub fn connect(server: &McpServer) -> Result<McpClient, String> {
        let mut child = std::process::Command::new(&server.command)
            .args(&server.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("запуск MCP '{}': {e}", server.name))?;
        let stdin = child.stdin.take().ok_or("нет stdin")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("нет stdout")?);
        let mut c = McpClient {
            child,
            stdin,
            stdout,
            id: 0,
            name: server.name.clone(),
        };
        // initialize
        let _ = c.request(
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "argus-agent", "version": "3.6"}
            })),
        )?;
        Ok(c)
    }

    fn request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        self.id += 1;
        let id = self.id;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params.unwrap_or(serde_json::Value::Null)
        });
        self.stdin
            .write_all(msg.to_string().as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("запись MCP: {e}"))?;
        loop {
            let mut line = String::new();
            let n = self
                .stdout
                .read_line(&mut line)
                .map_err(|e| format!("чтение MCP: {e}"))?;
            if n == 0 {
                return Err("MCP-сервер закрыл stdout".into());
            }
            let v: serde_json::Value =
                serde_json::from_str(line.trim()).map_err(|e| format!("JSON-RPC: {e}"))?;
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                if let Some(err) = v.get("error") {
                    return Err(format!("MCP error: {err}"));
                }
                return Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null));
            }
        }
    }

    /// Список инструментов сервера.
    pub fn tools(&mut self) -> Result<Vec<McpTool>, String> {
        let res = self.request("tools/list", None)?;
        let arr = res
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        let tools: Vec<McpTool> = arr
            .iter()
            .filter_map(|t| serde_json::from_value(t.clone()).ok())
            .collect();
        Ok(tools)
    }

    /// Вызвать инструмент.
    pub fn call_tool(&mut self, tool: &str, args: &serde_json::Value) -> Result<String, String> {
        let res = self.request(
            "tools/call",
            Some(serde_json::json!({ "name": tool, "arguments": args })),
        )?;
        let text: String = res
            .get("content")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| res.to_string());
        Ok(text)
    }

    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn mcp_server_management() {
        let dir = PathBuf::from("/tmp/argus-mcp-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        assert!(add_server(&dir, "echo", "echo", &["hello".into()]).is_ok());
        assert!(
            add_server(&dir, "echo", "echo", &[]).is_err(),
            "дубль имени запрещён"
        );
        assert_eq!(load_servers(&dir).len(), 1);
        assert!(remove_server(&dir, "echo").is_ok());
        assert_eq!(load_servers(&dir).len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
