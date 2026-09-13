//! Pre-execution guard for unsaved client buffers. Never writes editor contents to disk.
use super::ToolContext;
use crate::protocol::ServerEvent;
use anyhow::{Context, Result};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};
use tokio::sync::{mpsc, oneshot};

type Reply = std::result::Result<Option<String>, String>;
#[derive(Default)]
struct State {
    clients: HashMap<String, mpsc::UnboundedSender<ServerEvent>>,
    pending: HashMap<(String, String), oneshot::Sender<Reply>>,
}
fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}
pub fn install(session: &str, sender: Option<mpsc::UnboundedSender<ServerEvent>>) {
    let mut state = state().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(sender) = sender {
        state.clients.insert(session.into(), sender);
    } else {
        state.clients.remove(session);
    }
    state.pending.retain(|(id, _), _| id != session);
}
pub fn reply(session: &str, request: &str, content: Option<String>, error: Option<String>) {
    if let Some(sender) = state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pending
        .remove(&(session.into(), request.into()))
    {
        let _ = sender.send(match error {
            Some(error) => Err(error),
            None => Ok(content),
        });
    }
}
fn paths(name: &str, input: &Value, ctx: &ToolContext) -> Vec<PathBuf> {
    if !matches!(name, "write" | "edit" | "multiedit" | "apply_patch") {
        return Vec::new();
    }
    let mut paths = Vec::new();
    for key in ["path", "file_path", "filePath"] {
        if let Some(path) = input[key].as_str() {
            paths.push(ctx.resolve_path(std::path::Path::new(path)));
        }
    }
    if name == "apply_patch" {
        for key in ["patch", "patch_text", "input"] {
            if let Some(patch) = input[key].as_str() {
                for line in patch.lines() {
                    for prefix in [
                        "*** Add File: ",
                        "*** Update File: ",
                        "*** Delete File: ",
                        "*** Move to: ",
                    ] {
                        if let Some(path) = line.strip_prefix(prefix) {
                            paths.push(ctx.resolve_path(std::path::Path::new(path.trim())));
                        }
                    }
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}
// Zed exposes decoded, normalized text, including an empty buffer for new files.
// Only representation differences are normalized; never trim actual user content.
fn editor_matches_disk(content: Option<&str>, disk: Option<&[u8]>) -> bool {
    match (content, disk) {
        (None, None) | (Some(""), None) => true,
        (Some(editor), Some(bytes)) => std::str::from_utf8(bytes).is_ok_and(|disk| {
            editor
                == disk
                    .strip_prefix('\u{feff}')
                    .unwrap_or(disk)
                    .replace("\r\n", "\n")
        }),
        _ => false,
    }
}

pub async fn check(name: &str, input: &Value, ctx: &ToolContext) -> Result<()> {
    let sender = state()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clients
        .get(&ctx.session_id)
        .cloned();
    let Some(sender) = sender else {
        return Ok(());
    };
    for (index, path) in paths(name, input, ctx).into_iter().enumerate() {
        let before = match tokio::fs::read(&path).await {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        if before.as_ref().is_some_and(|s| s.len() > 1024 * 1024) {
            anyhow::bail!(
                "Editor guard cannot verify files larger than 1 MiB: {}",
                path.display()
            );
        }
        let request = format!("{}:{index}", ctx.tool_call_id);
        let key = (ctx.session_id.clone(), request.clone());
        let (tx, rx) = oneshot::channel();
        state()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending
            .insert(key.clone(), tx);
        let sent = sender.send(ServerEvent::AcpReadFile {
            request_id: request,
            path: path.to_string_lossy().into(),
        });
        let response = if sent.is_ok() {
            tokio::time::timeout(std::time::Duration::from_secs(15), rx)
                .await
                .ok()
                .and_then(Result::ok)
        } else {
            None
        };
        state()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending
            .remove(&key);
        let content = response
            .context("Editor did not respond; edit stopped before execution")?
            .map_err(anyhow::Error::msg)?;
        if !editor_matches_disk(content.as_deref(), before.as_deref()) {
            anyhow::bail!(
                "Unsaved editor changes or stale editor content in {}. The editor and disk differ. Reconcile the buffer before retrying; do not repeat the unchanged edit or bypass this check with shell writes.",
                path.display()
            );
        }
        // Detect disk changes during the client roundtrip too.
        let after = match tokio::fs::read(&path).await {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        if after != before {
            anyhow::bail!(
                "File changed while checking editor state; retry the edit: {}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn ctx(id: &str, dir: PathBuf) -> ToolContext {
        ToolContext {
            session_id: id.into(),
            message_id: "m".into(),
            tool_call_id: "t".into(),
            working_dir: Some(dir),
            stdin_request_tx: None,
            graceful_shutdown_signal: None,
            execution_mode: super::super::ToolExecutionMode::Direct,
        }
    }
    #[tokio::test]
    async fn dirty_editor_blocks_and_matching_editor_passes_without_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "disk").unwrap();
        for (session, content, passes) in [
            ("editor-dirty", "unsaved", false),
            ("editor-clean", "disk", true),
        ] {
            let context = ctx(session, dir.path().into());
            let (tx, mut rx) = mpsc::unbounded_channel();
            install(session, Some(tx));
            let task = tokio::spawn(async move {
                if let Some(ServerEvent::AcpReadFile { request_id, .. }) = rx.recv().await {
                    reply(session, &request_id, Some(content.into()), None);
                }
            });
            assert_eq!(
                check("edit", &json!({"file_path":"a.txt"}), &context)
                    .await
                    .is_ok(),
                passes
            );
            task.await.unwrap();
            install(session, None);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "disk");
        }
    }
    #[tokio::test]
    async fn acp_editor_new_file_and_normalized_text_preserve_dirty_buffers() {
        let dir = tempfile::tempdir().unwrap();
        for (index, disk, editor, passes) in [
            (0, None, "", true),
            (1, None, "unsaved new text", false),
            (2, Some("line\r\n"), "line\n", true),
            (3, Some("\u{feff}line\n"), "line\n", true),
            (4, Some("line\n"), "line", false),
            (5, Some("saved"), "", false),
        ] {
            let file = format!("case-{index}");
            let path = dir.path().join(&file);
            if let Some(disk) = disk {
                std::fs::write(&path, disk).unwrap();
            }
            let id = format!("acp-editor-case-{index}");
            let context = ctx(&id, dir.path().into());
            let (tx, mut rx) = mpsc::unbounded_channel();
            install(&id, Some(tx));
            let worker_id = id.clone();
            let task = tokio::spawn(async move {
                if let Some(ServerEvent::AcpReadFile { request_id, .. }) = rx.recv().await {
                    reply(&worker_id, &request_id, Some(editor.into()), None);
                }
            });
            assert_eq!(
                check("write", &json!({"file_path":file}), &context)
                    .await
                    .is_ok(),
                passes,
                "case {index}"
            );
            task.await.unwrap();
            install(&id, None);
            assert_eq!(
                std::fs::read(&path).ok(),
                disk.map(|d| d.as_bytes().to_vec())
            );
        }
    }

    #[test]
    fn patch_checks_source_and_move_destination_once() {
        let context = ctx("patch", PathBuf::from("/project"));
        let found = paths(
            "apply_patch",
            &json!({"patch_text":"*** Update File: a\n*** Move to: b\n*** Update File: a\n"}),
            &context,
        );
        assert_eq!(
            found,
            vec![PathBuf::from("/project/a"), PathBuf::from("/project/b")]
        );
    }
    #[tokio::test]
    async fn disconnected_editor_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let context = ctx("disconnected", dir.path().into());
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        install("disconnected", Some(tx));
        assert!(
            check("write", &json!({"file_path":"new.txt"}), &context)
                .await
                .is_err()
        );
        install("disconnected", None);
    }
}
