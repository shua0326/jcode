use super::dispatch;
use super::provider_init::ProviderChoice;
use crate::protocol::{Request, ServerEvent};
use crate::transport::{ReadHalf, WriteHalf};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

const ACP_PROTOCOL_VERSION: u64 = 1;

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_INVALID_REQUEST: i64 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INVALID_PARAMS: i64 = -32602;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;
// NOTE: `-32000` is reserved by ACP for `AuthRequired`, whose client-facing
// string is "Authentication required". Never return it for ordinary turn or
// provider failures: clients (Zed) render it as an authentication prompt, so a
// crashed model turn looked like an expired login and sent users chasing auth.
// Every failure below is an internal error, not an authentication request.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AcpProfile {
    Standard,
    Extended,
    Full,
}

impl AcpProfile {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "extended" => Self::Extended,
            "full" => Self::Full,
            _ => Self::Standard,
        }
    }

    fn is_extended(self) -> bool {
        matches!(self, Self::Extended | Self::Full)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Extended => "extended",
            Self::Full => "full",
        }
    }
}

#[derive(Debug)]
struct JsonRpcMessage {
    id: Option<Value>,
    method: Option<String>,
    params: Value,
}

impl JsonRpcMessage {
    fn parse(line: &str) -> std::result::Result<Self, (i64, String)> {
        let value: Value =
            serde_json::from_str(line).map_err(|err| (JSONRPC_PARSE_ERROR, err.to_string()))?;
        let object = value.as_object().ok_or_else(|| {
            (
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must be an object".to_string(),
            )
        })?;
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err((
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must include jsonrpc=\"2.0\"".to_string(),
            ));
        }
        Ok(Self {
            id: object.get("id").cloned(),
            method: object
                .get("method")
                .and_then(Value::as_str)
                .map(str::to_string),
            params: object.get("params").cloned().unwrap_or(Value::Null),
        })
    }
}

struct DaemonSession {
    session_id: String,
    reader: Mutex<BufReader<ReadHalf>>,
    writer: Mutex<WriteHalf>,
    next_request_id: AtomicU64,
    active_prompt_id: Mutex<Option<u64>>,
    prompt_running: AtomicBool,
    ui_state: Mutex<SessionUiState>,
    working_dir: Option<PathBuf>,
    event_queue: Mutex<Option<tokio::sync::mpsc::Receiver<Result<ServerEvent>>>>,
    event_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Events a control request consumed but did not need. They still belong to
    /// the prompt loop, so dropping them would lose streamed output; the prompt
    /// loop drains this before the live queue.
    deferred_events: Mutex<std::collections::VecDeque<ServerEvent>>,
}

/// Session-scoped provider/model state used to surface ACP `configOptions`
/// (model selector, reasoning effort) and `usage_update` notifications.
#[derive(Clone, Debug, Default)]
struct SessionUiState {
    provider_name: Option<String>,
    model: Option<String>,
    available_models: Vec<String>,
    reasoning_effort: Option<String>,
    model_routes: Vec<crate::provider::ModelRoute>,
    last_context_tokens: Option<u64>,
    last_turn_usage: Option<Value>,
    selected_model_value: Option<String>,
    connection_usage: TurnUsage,
    usage_reports: u64,
    service_tier: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct TurnUsage {
    reported: bool,
    input_tokens: u64,
    output_tokens: u64,
    cached_read_tokens: Option<u64>,
    cached_write_tokens: Option<u64>,
}

impl TurnUsage {
    fn add(
        &mut self,
        input_tokens: u64,
        output_tokens: u64,
        cached_read_tokens: Option<u64>,
        cached_write_tokens: Option<u64>,
    ) {
        self.reported = true;
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        add_optional_tokens(&mut self.cached_read_tokens, cached_read_tokens);
        add_optional_tokens(&mut self.cached_write_tokens, cached_write_tokens);
    }

    fn to_acp(&self) -> Option<Value> {
        if !self.reported {
            return None;
        }

        let total_tokens = self
            .input_tokens
            .saturating_add(self.output_tokens)
            .saturating_add(self.cached_read_tokens.unwrap_or(0))
            .saturating_add(self.cached_write_tokens.unwrap_or(0));
        let mut usage = json!({
            "totalTokens": total_tokens,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
        });
        let object = usage.as_object_mut().expect("usage is an object");
        if let Some(tokens) = self.cached_read_tokens {
            object.insert("cachedReadTokens".to_string(), json!(tokens));
        }
        if let Some(tokens) = self.cached_write_tokens {
            object.insert("cachedWriteTokens".to_string(), json!(tokens));
        }
        Some(usage)
    }
}

fn add_optional_tokens(total: &mut Option<u64>, tokens: Option<u64>) {
    if let Some(tokens) = tokens {
        *total = Some(total.unwrap_or(0).saturating_add(tokens));
    }
}

fn prompt_response(stop_reason: &str, usage: &TurnUsage) -> Value {
    let mut response = json!({ "stopReason": stop_reason });
    if let Some(usage) = usage.to_acp() {
        response
            .as_object_mut()
            .expect("prompt response is an object")
            .insert("usage".to_string(), usage);
    }
    response
}

impl SessionUiState {
    fn from_history_fields(
        provider_name: Option<String>,
        provider_model: Option<String>,
        available_models: Vec<String>,
        reasoning_effort: Option<String>,
    ) -> Self {
        Self {
            provider_name,
            model: provider_model,
            available_models,
            reasoning_effort,
            ..Self::default()
        }
    }

    fn with_model_routes(mut self, routes: Vec<crate::provider::ModelRoute>) -> Self {
        self.model_routes = routes;
        self
    }

    fn with_service_tier(mut self, tier: Option<String>) -> Self {
        self.service_tier = tier;
        self
    }

    fn context_limit(&self) -> u64 {
        self.model
            .as_deref()
            .and_then(|model| {
                crate::provider::context_limit_for_model_with_provider(
                    model,
                    self.provider_name.as_deref(),
                )
            })
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_LIMIT) as u64
    }

    /// Context meter payload for the active conversation: the last observed
    /// prompt+output token count paired with the *current* model's window.
    ///
    /// `None` until a turn has reported usage. A conversation that has never
    /// reported tokens has nothing to show, and reporting `0` would claim the
    /// context is empty.
    fn context_usage(&self) -> Option<(u64, u64)> {
        self.last_context_tokens
            .map(|used| (used, self.context_limit()))
    }
}

impl DaemonSession {
    fn new(session_id: String, reader: ReadHalf, writer: WriteHalf, next_request_id: u64) -> Self {
        Self {
            session_id,
            reader: Mutex::new(BufReader::new(reader)),
            writer: Mutex::new(writer),
            next_request_id: AtomicU64::new(next_request_id),
            active_prompt_id: Mutex::new(None),
            prompt_running: AtomicBool::new(false),
            ui_state: Mutex::new(SessionUiState::default()),
            working_dir: None,
            event_queue: Mutex::new(None),
            event_task: Mutex::new(None),
            deferred_events: Mutex::new(std::collections::VecDeque::new()),
        }
    }

    fn with_ui_state(self, state: SessionUiState) -> Self {
        Self {
            ui_state: Mutex::new(state),
            ..self
        }
    }

    fn with_working_dir(mut self, cwd: PathBuf) -> Self {
        self.working_dir = Some(cwd);
        self
    }

    fn next_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn send(&self, request: &Request) -> Result<()> {
        let mut json = serde_json::to_string(request)?;
        json.push('\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(json.as_bytes()).await?;
        writer.flush().await?;
        Ok(())
    }

    async fn read_event(&self) -> Result<ServerEvent> {
        if let Some(event) = self.deferred_events.lock().await.pop_front() {
            return Ok(event);
        }
        self.read_live_event().await
    }

    /// Read the next event *without* draining events deferred by control
    /// requests, for use by those same control requests.
    ///
    /// A control request must not use [`Self::read_event`]: it would immediately
    /// re-read the event it just deferred, defer it again, and spin on it
    /// forever instead of ever seeing its own reply.
    async fn read_event_for_control(&self) -> Result<ServerEvent> {
        self.read_live_event().await
    }

    async fn read_live_event(&self) -> Result<ServerEvent> {
        let mut queue = self.event_queue.lock().await;
        if let Some(rx) = queue.as_mut() {
            return rx
                .recv()
                .await
                .context("JCode event reader stopped; reload the session")?;
        }
        drop(queue);
        self.read_wire_event().await
    }

    /// Hand an event this request does not need back to whoever owns the stream.
    /// Bounded: a long control wait during a streaming turn must not grow without
    /// limit, so the oldest deferred event is dropped past the cap.
    async fn defer_event(&self, event: ServerEvent) {
        const MAX_DEFERRED_EVENTS: usize = 512;
        let mut deferred = self.deferred_events.lock().await;
        // Live output snapshots replace earlier ones for the same tool call, so a
        // long control wait cannot stack stale copies of the same stream.
        if let ServerEvent::ToolOutput { id, .. } = &event
            && let Some(slot) = deferred.iter_mut().find(|queued| {
                matches!(queued, ServerEvent::ToolOutput { id: queued_id, .. } if queued_id == id)
            })
        {
            *slot = event;
            return;
        }
        if deferred.len() >= MAX_DEFERRED_EVENTS {
            deferred.pop_front();
        }
        deferred.push_back(event);
    }

    /// Drop deferred events that predate a new turn so a stale delta cannot be
    /// replayed as if it arrived now.
    async fn clear_deferred_events(&self) {
        self.deferred_events.lock().await.clear();
    }

    async fn read_wire_event(&self) -> Result<ServerEvent> {
        let mut line = String::new();
        let mut reader = self.reader.lock().await;
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("Jcode daemon disconnected");
        }
        let event = serde_json::from_str(&line)
            .with_context(|| format!("failed to decode Jcode daemon event: {}", line.trim_end()))?;
        Ok(event)
    }
}

#[derive(Clone)]
struct AcpRuntime {
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    sessions: Arc<Mutex<HashMap<String, Arc<DaemonSession>>>>,
    profile: AcpProfile,
    provider_choice: ProviderChoice,
    model: Option<String>,
    provider_profile: Option<String>,
    editor_read: Arc<AtomicBool>,
    client_requests: Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<Value>>>>,
    client_request_id: Arc<AtomicU64>,
}

impl AcpRuntime {
    fn new(
        profile: AcpProfile,
        provider_choice: ProviderChoice,
        model: Option<String>,
        provider_profile: Option<String>,
    ) -> Self {
        Self {
            stdout: Arc::new(Mutex::new(tokio::io::stdout())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            profile,
            provider_choice,
            model,
            provider_profile,
            editor_read: Arc::new(AtomicBool::new(false)),
            client_requests: Arc::new(Mutex::new(HashMap::new())),
            client_request_id: Arc::new(AtomicU64::new(1)),
        }
    }

    async fn run(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                return Ok(());
            }
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                if value.get("method").is_none() {
                    if let Some(id) = value["id"].as_str() {
                        if let Some(reply) = self.client_requests.lock().await.remove(id) {
                            let _ = reply.send(value);
                            continue;
                        }
                    }
                }
            }
            let message = match JsonRpcMessage::parse(&line) {
                Ok(message) => message,
                Err((code, message)) => {
                    self.write_error_value(
                        Value::Null,
                        code,
                        format!("Invalid JSON-RPC request: {message}"),
                    )
                    .await?;
                    continue;
                }
            };

            self.handle_message(message).await?;
        }
    }

    async fn handle_message(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(method) = message.method.as_deref() else {
            if let Some(id) = message.id {
                self.write_error_value(
                    id,
                    JSONRPC_INVALID_REQUEST,
                    "JSON-RPC request missing method".to_string(),
                )
                .await?;
            }
            return Ok(());
        };

        match method {
            "initialize" => {
                self.editor_read.store(
                    message.params["clientCapabilities"]["fs"]["readTextFile"] == true,
                    Ordering::SeqCst,
                );
                if let Some(id) = message.id {
                    self.write_result(id, initialize_result(&message.params, self.profile))
                        .await?;
                }
            }
            "session/new" => self.handle_session_new(message).await?,
            "session/list" => self.handle_session_list(message).await?,
            "session/load" => self.handle_session_load(message, true).await?,
            "session/resume" => self.handle_session_load(message, false).await?,
            "session/prompt" => self.handle_session_prompt(message).await?,
            "session/cancel" => self.handle_session_cancel(message).await?,
            "session/close" => self.handle_session_close(message).await?,
            "session/set_config_option" => self.handle_set_config_option(message).await?,
            "session/set_model" => {
                self.handle_compat_config_option(
                    message,
                    CONFIG_ID_MODEL,
                    &["modelId", "model"],
                    "session/set_model",
                )
                .await?
            }
            "session/set_reasoning_effort" => {
                self.handle_compat_config_option(
                    message,
                    CONFIG_ID_EFFORT,
                    &["effort", "reasoningEffort"],
                    "session/set_reasoning_effort",
                )
                .await?
            }
            _ if method.starts_with('_') => {
                if let Some(id) = message.id {
                    self.write_error_value(
                        id,
                        JSONRPC_METHOD_NOT_FOUND,
                        format!("Unsupported Jcode ACP extension method: {method}"),
                    )
                    .await?;
                }
            }
            _ => {
                if let Some(id) = message.id {
                    self.write_error_value(
                        id,
                        JSONRPC_METHOD_NOT_FOUND,
                        format!("Unsupported ACP method: {method}"),
                    )
                    .await?;
                }
            }
        }

        Ok(())
    }

    async fn handle_session_new(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let cwd = match cwd_from_params(&message.params) {
            Ok(cwd) => cwd,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Err(err) = validate_acp_mcp_servers(&message.params) {
            self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                .await?;
            return Ok(());
        }

        match self
            .create_new_session(
                cwd,
                message
                    .params
                    .get("mcpServers")
                    .cloned()
                    .unwrap_or(json!([])),
            )
            .await
        {
            Ok(session) => {
                let session_id = session.session_id.clone();
                let state = session.ui_state.lock().await.clone();
                if let Some(previous) = self
                    .sessions
                    .lock()
                    .await
                    .insert(session_id.clone(), Arc::new(session))
                {
                    if let Some(task) = previous.event_task.lock().await.take() {
                        task.abort();
                    }
                }
                let mut result = json!({ "sessionId": session_id });
                insert_session_configuration(&mut result, &state);
                self.write_result(id, result).await?;
                self.write_available_commands(&session_id).await?;
                self.start_event_reader(&session_id).await?;
            }
            Err(err) => {
                self.write_error_value(
                    id,
                    JSONRPC_INTERNAL_ERROR,
                    format!("Failed to create Jcode session: {err:#}"),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn handle_session_list(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let params = message.params;
        let live: Vec<Value> = self.sessions.lock().await.values().filter_map(|session| {
            let cwd = session.working_dir.as_ref()?;
            Some(json!({"sessionId":session.session_id,"cwd":cwd.to_string_lossy(),"title":session.session_id,"updatedAt":chrono::Utc::now().to_rfc3339()}))
        }).collect();
        let result = tokio::task::spawn_blocking(move || acp_session_list(&params, live)).await?;
        match result {
            Ok(list) => self.write_result(id, list).await?,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err.to_string())
                    .await?
            }
        }
        Ok(())
    }

    async fn handle_session_load(
        &self,
        message: JsonRpcMessage,
        replay_history: bool,
    ) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let cwd = match cwd_from_params(&message.params) {
            Ok(cwd) => cwd,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Err(err) = validate_acp_mcp_servers(&message.params) {
            self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                .await?;
            return Ok(());
        }

        match self
            .attach_existing_session(
                session_id.clone(),
                cwd,
                replay_history,
                message
                    .params
                    .get("mcpServers")
                    .cloned()
                    .unwrap_or(json!([])),
            )
            .await
        {
            Ok(session) => {
                let state = session.ui_state.lock().await.clone();
                if let Some(previous) = self
                    .sessions
                    .lock()
                    .await
                    .insert(session.session_id.clone(), Arc::new(session))
                {
                    if let Some(task) = previous.event_task.lock().await.take() {
                        task.abort();
                    }
                }
                let mut result = json!({});
                insert_session_configuration(&mut result, &state);
                self.write_result(id, result).await?;
                self.write_available_commands(&session_id).await?;
                self.start_event_reader(&session_id).await?;
            }
            Err(err) => {
                self.write_error_value(
                    id,
                    JSONRPC_INTERNAL_ERROR,
                    format!("Failed to attach Jcode session '{session_id}': {err:#}"),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn handle_session_prompt(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let (text, images) = match prompt_from_params(&message.params) {
            Ok(prompt) => prompt,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let session = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).cloned()
        };
        let Some(session) = session else {
            self.write_error_value(
                id,
                JSONRPC_INVALID_PARAMS,
                format!("Unknown ACP session id: {session_id}"),
            )
            .await?;
            return Ok(());
        };

        if session
            .prompt_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            self.write_error_value(
                id,
                JSONRPC_INTERNAL_ERROR,
                format!("Session {session_id} is already processing a prompt"),
            )
            .await?;
            return Ok(());
        }

        let runtime = self.clone();
        tokio::spawn(async move {
            let result = runtime
                .run_prompt(id.clone(), session.clone(), text, images)
                .await;
            if let Err(err) = result {
                cleanup_prompt_state(&session).await;
                let _ = runtime
                    .write_error_value(
                        id,
                        JSONRPC_INTERNAL_ERROR,
                        format!("Prompt failed: {err:#}"),
                    )
                    .await;
            }
        });
        Ok(())
    }

    async fn handle_session_cancel(&self, message: JsonRpcMessage) -> Result<()> {
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                if let Some(id) = message.id {
                    self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                        .await?;
                }
                return Ok(());
            }
        };
        let session = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).cloned()
        };
        if let Some(session) = session {
            let cancel_id = session.next_id();
            let _ = session.send(&Request::Cancel { id: cancel_id }).await;
        }
        if let Some(id) = message.id {
            self.write_result(id, json!({})).await?;
        }
        Ok(())
    }

    async fn handle_session_close(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        if let Some(session) = self.sessions.lock().await.remove(&session_id) {
            if let Some(task) = session.event_task.lock().await.take() {
                task.abort();
            }
            let cancel_id = session.next_id();
            let _ = session.send(&Request::Cancel { id: cancel_id }).await;
        }
        self.write_result(id, json!({})).await?;
        Ok(())
    }

    async fn handle_set_config_option(&self, message: JsonRpcMessage) -> Result<()> {
        let Some(id) = message.id else {
            return Ok(());
        };
        let session_id = match required_session_id(&message.params) {
            Ok(session_id) => session_id,
            Err(err) => {
                self.write_error_value(id, JSONRPC_INVALID_PARAMS, err)
                    .await?;
                return Ok(());
            }
        };
        let config_id = message
            .params
            .get("configId")
            .and_then(Value::as_str)
            .map(str::to_string);
        let value = message
            .params
            .get("value")
            .and_then(Value::as_str)
            .map(str::to_string);
        let (Some(config_id), Some(value)) = (config_id, value) else {
            self.write_error_value(
                id,
                JSONRPC_INVALID_PARAMS,
                "session/set_config_option requires string configId and value".to_string(),
            )
            .await?;
            return Ok(());
        };

        let session = {
            let sessions = self.sessions.lock().await;
            sessions.get(&session_id).cloned()
        };
        let Some(session) = session else {
            self.write_error_value(
                id,
                JSONRPC_INVALID_PARAMS,
                format!("Unknown ACP session id: {session_id}"),
            )
            .await?;
            return Ok(());
        };
        if session.prompt_running.load(Ordering::SeqCst) {
            self.write_error_value(
                id,
                JSONRPC_INTERNAL_ERROR,
                format!("Session {session_id} is processing a prompt; retry when it finishes"),
            )
            .await?;
            return Ok(());
        }

        let request_id = session.next_id();
        let apply_result = match config_id.as_str() {
            CONFIG_ID_MODEL => set_acp_model(&session, request_id, &value).await,
            CONFIG_ID_EFFORT => {
                session
                    .send(&Request::SetReasoningEffort {
                        id: request_id,
                        effort: value.clone(),
                        target_session_id: None,
                    })
                    .await?;
                wait_for_effort_changed(&session, request_id).await
            }
            other => Err(anyhow::anyhow!("Unknown config option id: {other}")),
        };

        match apply_result {
            Ok(()) => {
                let config_options = session_config_options(&*session.ui_state.lock().await);
                // The spec requires the full option set in the response itself.
                self.write_result(id, json!({ "configOptions": config_options }))
                    .await?;
                self.write_notification(
                    "session/update",
                    json!({
                        "sessionId": session.session_id,
                        "update": {
                            "sessionUpdate": "config_option_update",
                            "configOptions": config_options,
                        }
                    }),
                )
                .await?;
                // `set_acp_model` consumes the daemon's ModelChanged event while
                // waiting for the switch, so this is the only place that can tell
                // the editor its context meter just changed models.
                if config_id == CONFIG_ID_MODEL {
                    let _ = self.write_usage_update(&session).await;
                }
            }
            Err(err) => {
                self.write_error_value(
                    id,
                    JSONRPC_INVALID_PARAMS,
                    format!("Failed to set {config_id}: {err:#}"),
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Compatibility entry points used by ACP hosts that implemented the
    /// pre-configOptions model and reasoning controls. Normalize them through
    /// the standard config option path so both interfaces stay in sync.
    async fn handle_compat_config_option(
        &self,
        mut message: JsonRpcMessage,
        config_id: &str,
        value_fields: &[&str],
        method: &str,
    ) -> Result<()> {
        let value = match compatibility_option_value(&message.params, value_fields, method) {
            Ok(value) => value,
            Err(error) => {
                if let Some(id) = message.id {
                    self.write_error_value(id, JSONRPC_INVALID_PARAMS, error)
                        .await?;
                }
                return Ok(());
            }
        };
        let Some(params) = message.params.as_object_mut() else {
            if let Some(id) = message.id {
                self.write_error_value(
                    id,
                    JSONRPC_INVALID_PARAMS,
                    format!("{method} params must be an object"),
                )
                .await?;
            }
            return Ok(());
        };
        params.insert("configId".to_string(), Value::String(config_id.to_string()));
        params.insert("value".to_string(), Value::String(value));
        self.handle_set_config_option(message).await
    }

    async fn write_available_commands(&self, session_id: &str) -> Result<()> {
        let session = self
            .sessions
            .lock()
            .await
            .get(session_id)
            .cloned()
            .context("Session missing while advertising commands")?;
        let skills =
            crate::skill::SkillRegistry::load_for_working_dir(session.working_dir.as_deref())?;
        self.write_notification(
            "session/update",
            json!({
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "available_commands_update",
                    "availableCommands": acp_commands_with_skills(&skills),
                }
            }),
        )
        .await
    }

    async fn read_editor_file(&self, session_id: &str, path: &str) -> Result<Option<String>> {
        if !self.editor_read.load(Ordering::SeqCst) {
            anyhow::bail!("Client does not support editor reads");
        }
        let id = format!(
            "jcode-editor-{}",
            self.client_request_id.fetch_add(1, Ordering::Relaxed)
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.client_requests.lock().await.insert(id.clone(), tx);
        let sent = self.write_value(json!({"jsonrpc":"2.0","id":id,"method":"fs/read_text_file","params":{"sessionId":session_id,"path":path}})).await;
        let response = if sent.is_ok() {
            tokio::time::timeout(std::time::Duration::from_secs(10), rx)
                .await
                .ok()
                .and_then(Result::ok)
        } else {
            None
        };
        self.client_requests.lock().await.remove(&id);
        let response = response.context("Editor read timed out or disconnected")?;
        if response.get("error").is_some() {
            // ACP resource-not-found for a new file is the only acceptable read error.
            if response["error"]["code"] == -32002 && !std::path::Path::new(path).exists() {
                return Ok(None);
            }
            anyhow::bail!("Editor could not verify this file; edit stopped");
        }
        let text = response["result"]["content"]
            .as_str()
            .context("Editor returned invalid file content")?;
        if text.len() > 1024 * 1024 {
            anyhow::bail!("Editor buffer exceeds the 1 MiB verification limit");
        }
        Ok(Some(text.into()))
    }

    async fn start_event_reader(&self, session_id: &str) -> Result<()> {
        let session = self
            .sessions
            .lock()
            .await
            .get(session_id)
            .cloned()
            .context("Missing session")?;
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        *session.event_queue.lock().await = Some(rx);
        let runtime = self.clone();
        let active = session.clone();
        let task = tokio::spawn(async move {
            let mut mapper = EventMapper::new(active.session_id.clone(), runtime.profile);
            mapper.working_dir = active.working_dir.clone();
            loop {
                // Session load/configuration can defer unsolicited events while
                // waiting for its control replies. Drain those first, but read
                // live events directly: `read_event` would wait on this task's
                // own event queue.
                let event = match if let Some(event) =
                    active.deferred_events.lock().await.pop_front()
                {
                    Ok(event)
                } else {
                    active.read_wire_event().await
                } {
                    Ok(event) => event,
                    Err(err) => {
                        let _ = runtime.write_notification("session/update", json!({"sessionId":active.session_id,"update":agent_message_chunk("JCode disconnected. Reload this session to reconnect; do not resend a command until its state is checked.".into())})).await;
                        let _ = tx.send(Err(err)).await;
                        break;
                    }
                };
                match event {
                    ServerEvent::AcpReadFile { request_id, path } => {
                        let runtime = runtime.clone();
                        let active = active.clone();
                        tokio::spawn(async move {
                            let result = runtime.read_editor_file(&active.session_id, &path).await;
                            let (content, error) = match result {
                                Ok(content) => (content, None),
                                Err(err) => (None, Some(err.to_string())),
                            };
                            let _ = active
                                .send(&Request::AcpFileContent {
                                    id: active.next_id(),
                                    request_id,
                                    content,
                                    error,
                                })
                                .await;
                        });
                    }
                    ServerEvent::TextDelta { .. }
                    | ServerEvent::TextReplace { .. }
                    | ServerEvent::ReasoningDelta { .. }
                    | ServerEvent::ReasoningDone { .. }
                    | ServerEvent::GeneratedImage { .. }
                    | ServerEvent::ToolStart { .. }
                    | ServerEvent::ToolInput { .. }
                    | ServerEvent::ToolExec { .. }
                    | ServerEvent::ToolOutput { .. }
                    | ServerEvent::ToolDone { .. }
                    | ServerEvent::SwarmStatus { .. }
                    | ServerEvent::SwarmPlan { .. } => {
                        for update in mapper.map_event(event) {
                            if runtime
                                .write_notification(
                                    "session/update",
                                    json!({"sessionId":active.session_id,"update":update}),
                                )
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                    event @ ServerEvent::TokenUsage {
                        input,
                        output,
                        cache_read_input,
                        cache_creation_input,
                    } => {
                        {
                            let mut state = active.ui_state.lock().await;
                            let full = acp_full_input_tokens(
                                state.provider_name.as_deref().unwrap_or_default(),
                                input,
                                cache_read_input,
                                cache_creation_input,
                            );
                            let cache = cache_read_input
                                .unwrap_or(0)
                                .saturating_add(cache_creation_input.unwrap_or(0));
                            state.connection_usage.add(
                                full.saturating_sub(cache),
                                output,
                                cache_read_input,
                                cache_creation_input,
                            );
                            state.usage_reports += 1;
                            state.last_context_tokens = Some(full.saturating_add(output));
                        }
                        let _ = runtime.write_usage_update(&active).await;
                        if active.prompt_running.load(Ordering::SeqCst) {
                            let _ = tx.send(Ok(event)).await;
                        }
                    }
                    event @ ServerEvent::ServiceTierChanged { .. } => {
                        if let ServerEvent::ServiceTierChanged {
                            service_tier,
                            error: None,
                            ..
                        } = &event
                        {
                            active.ui_state.lock().await.service_tier = service_tier.clone();
                        }
                        if active.prompt_running.load(Ordering::SeqCst) {
                            let _ = tx.send(Ok(event)).await;
                        }
                    }
                    event @ ServerEvent::ModelChanged { .. } => {
                        if let ServerEvent::ModelChanged {
                            model,
                            provider_name,
                            error,
                            ..
                        } = &event
                        {
                            if let Some(error) = error {
                                let _ = runtime.write_notification("session/update", json!({"sessionId":active.session_id,"update":agent_message_chunk(format!("Model switch failed: {error}"))})).await;
                            } else {
                                {
                                    let mut state = active.ui_state.lock().await;
                                    state.model = Some(model.clone());
                                    state.selected_model_value = None;
                                    if provider_name.is_some() {
                                        state.provider_name = provider_name.clone();
                                    }
                                }
                                let _ = runtime.write_config_option_update(&active).await;
                                // The new model has a different window, so the
                                // meter would keep the old denominator otherwise.
                                let _ = runtime.write_usage_update(&active).await;
                            }
                        }
                        if tx.send(Ok(event)).await.is_err() {
                            break;
                        }
                    }
                    ServerEvent::AvailableModelsUpdated {
                        provider_name,
                        provider_model,
                        available_models,
                        available_model_routes,
                    } => {
                        {
                            let mut state = active.ui_state.lock().await;
                            if provider_name.is_some() {
                                state.provider_name = provider_name;
                            }
                            if provider_model.is_some() {
                                state.model = provider_model;
                            }
                            state.available_models = available_models;
                            // Oversized upstream updates omit routes; preserve the full catalogue.
                            if !available_model_routes.is_empty() {
                                state.model_routes = available_model_routes;
                            }
                        }
                        let _ = runtime.write_config_option_update(&active).await;
                        // This event can also resolve a different provider/model
                        // for the session, which moves the context window.
                        let _ = runtime.write_usage_update(&active).await;
                    }
                    ServerEvent::Notification {
                        from_session,
                        from_name,
                        notification_type,
                        message,
                    } => {
                        // Rendered whether or not a turn is running: an idle
                        // session has no other channel for a coordinator's
                        // assignment, report or file conflict.
                        let text = notification_chunk_text(
                            &from_session,
                            from_name.as_deref(),
                            &notification_type,
                            &message,
                        );
                        if runtime
                            .write_notification(
                                "session/update",
                                json!({"sessionId":active.session_id,"update":agent_message_chunk(text)}),
                            )
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    other => {
                        // Unsolicited idle events must not fill a queue nobody is consuming.
                        if active.prompt_running.load(Ordering::SeqCst)
                            || matches!(
                                other,
                                ServerEvent::History { .. }
                                    | ServerEvent::ModelChanged { .. }
                                    | ServerEvent::ReasoningEffortChanged { .. }
                                    | ServerEvent::Error { .. }
                                    | ServerEvent::Done { .. }
                                    | ServerEvent::Ack { .. }
                            )
                        {
                            if tx.send(Ok(other)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        });
        *session.event_task.lock().await = Some(task);
        Ok(())
    }

    async fn ensure_daemon(&self) -> Result<()> {
        if dispatch::server_is_running().await {
            return Ok(());
        }
        // ACP and daemon extensions must come from the same build. A globally
        // installed stable/shared daemon may not understand the local protocol.
        dispatch::spawn_server_with_executable(
            &self.provider_choice,
            self.model.as_deref(),
            self.provider_profile.as_deref(),
            Some(std::env::current_exe().context("Cannot locate ACP executable")?),
        )
        .await
    }

    async fn connect_daemon(&self) -> Result<(ReadHalf, WriteHalf)> {
        self.ensure_daemon().await?;
        let stream = crate::server::connect_socket(&crate::server::socket_path()).await?;
        Ok(stream.into_split())
    }

    async fn create_new_session(&self, cwd: PathBuf, mcp_servers: Value) -> Result<DaemonSession> {
        let (reader, writer) = self.connect_daemon().await?;
        let session = DaemonSession::new(String::new(), reader, writer, 2);
        let subscribe_id = 1;
        session
            .send(&Request::Subscribe {
                crash_on_disconnect: false,
                continue_on_disconnect: true,
                id: subscribe_id,
                working_dir: Some(cwd.display().to_string()),
                selfdev: None,
                target_session_id: None,
                client_instance_id: Some("acp".to_string()),
                client_has_local_history: false,
                allow_session_takeover: false,
                terminal_env: crate::terminal_launch::snapshot_client_terminal_env(),
            })
            .await?;
        wait_for_done(&session, subscribe_id).await?;
        let history = request_history(&session).await?;
        let (session_id, ui_state) = match history {
            ServerEvent::History {
                session_id,
                provider_name,
                provider_model,
                available_models,
                available_model_routes,
                reasoning_effort,
                service_tier,
                ..
            } => (
                session_id,
                SessionUiState::from_history_fields(
                    provider_name,
                    provider_model,
                    available_models,
                    reasoning_effort,
                )
                .with_model_routes(available_model_routes)
                .with_service_tier(service_tier),
            ),
            other => anyhow::bail!("expected history after session creation, got {other:?}"),
        };
        let session = DaemonSession {
            session_id,
            ..session
        }
        .with_ui_state(ui_state)
        .with_working_dir(cwd);
        refresh_model_catalog(&session).await?;
        configure_acp_mcp(
            &session,
            &mcp_servers,
            self.editor_read.load(Ordering::SeqCst),
        )
        .await?;
        Ok(session)
    }

    async fn attach_existing_session(
        &self,
        target_session_id: String,
        cwd: PathBuf,
        replay_history: bool,
        mcp_servers: Value,
    ) -> Result<DaemonSession> {
        let (reader, writer) = self.connect_daemon().await?;
        let session = DaemonSession::new(String::new(), reader, writer, 2);
        let resume_id = 1;
        session
            .send(&Request::Subscribe {
                crash_on_disconnect: false,
                continue_on_disconnect: true,
                id: resume_id,
                working_dir: Some(cwd.display().to_string()),
                selfdev: None,
                target_session_id: Some(target_session_id.clone()),
                client_instance_id: Some("acp".to_string()),
                client_has_local_history: false,
                allow_session_takeover: false,
                terminal_env: crate::terminal_launch::snapshot_client_terminal_env(),
            })
            .await?;

        let mut attached_id = target_session_id;
        let mut ui_state = SessionUiState::default();
        loop {
            let event = session.read_event().await?;
            match event {
                ServerEvent::Ack { .. } => {}
                ServerEvent::History {
                    session_id,
                    messages,
                    provider_name,
                    provider_model,
                    available_models,
                    available_model_routes,
                    reasoning_effort,
                    service_tier,
                    ..
                } => {
                    attached_id = session_id.clone();
                    ui_state = SessionUiState::from_history_fields(
                        provider_name,
                        provider_model,
                        available_models,
                        reasoning_effort,
                    )
                    .with_model_routes(available_model_routes)
                    .with_service_tier(service_tier);
                    // The normal History payload is formatted for the terminal UI.
                    let _ = messages;
                }
                ServerEvent::Done { id } if id == resume_id => break,
                ServerEvent::Error { id, message, .. } if id == resume_id => {
                    anyhow::bail!(message);
                }
                other => {
                    if self.profile.is_extended() {
                        self.write_jcode_extension_event(&attached_id, &other)
                            .await?;
                    }
                }
            }
        }

        let session = DaemonSession {
            session_id: attached_id,
            ..session
        }
        .with_ui_state(ui_state)
        .with_working_dir(cwd);
        if replay_history {
            let id = session.next_id();
            session.send(&Request::GetAcpHistory { id }).await?;
            let messages = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                loop {
                    match session.read_event().await? {
                        ServerEvent::AcpHistory {
                            id: event_id,
                            messages,
                        } if event_id == id => return Ok::<_, anyhow::Error>(messages),
                        ServerEvent::Error {
                            id: event_id,
                            message,
                            ..
                        } if event_id == id => anyhow::bail!(message),
                        _ => {}
                    }
                }
            })
            .await
            .context("Structured history lookup timed out")??;
            self.replay_history(&session.session_id, messages).await?;
        }
        refresh_model_catalog(&session).await?;
        configure_acp_mcp(
            &session,
            &mcp_servers,
            self.editor_read.load(Ordering::SeqCst),
        )
        .await?;
        Ok(session)
    }

    async fn replay_history(
        &self,
        session_id: &str,
        messages: Vec<crate::protocol::HistoryMessage>,
    ) -> Result<()> {
        for update in acp_history_updates(messages) {
            self.write_notification(
                "session/update",
                json!({"sessionId":session_id,"update":update}),
            )
            .await?;
        }
        Ok(())
    }

    async fn run_prompt(
        &self,
        rpc_id: Value,
        session: Arc<DaemonSession>,
        text: String,
        images: Vec<(String, String)>,
    ) -> Result<()> {
        if let Some(command) = parse_acp_slash_command(&text) {
            let response = match command {
                Ok(command) => self.run_session_command(&session, command).await,
                Err(err) => Err(err),
            };
            cleanup_prompt_state(&session).await;
            let response = response?;
            self.write_notification(
                "session/update",
                json!({
                    "sessionId": session.session_id,
                    "update": agent_message_chunk(response),
                }),
            )
            .await?;
            self.write_result(rpc_id, prompt_response("end_turn", &TurnUsage::default()))
                .await?;
            return Ok(());
        }

        // Skill metadata is only needed for slash invocations. A normal prompt
        // must not pay a filesystem scan of every skill directory.
        let trimmed = text.trim_end();
        if matches!(trimmed, "/skills" | "/usage" | "/limits") {
            let listing = if trimmed == "/limits" {
                acp_account_limits().await
            } else if trimmed == "/usage" {
                acp_usage_listing(&*session.ui_state.lock().await)
            } else {
                let skills = crate::skill::SkillRegistry::load_for_working_dir(
                    session.working_dir.as_deref(),
                )?;
                acp_skill_listing(&skills)
            };
            self.write_available_commands(&session.session_id).await?;
            self.write_notification(
                "session/update",
                json!({
                    "sessionId": session.session_id,
                    "update": agent_message_chunk(listing),
                }),
            )
            .await?;
            cleanup_prompt_state(&session).await;
            self.write_result(rpc_id, prompt_response("end_turn", &TurnUsage::default()))
                .await?;
            return Ok(());
        }
        let (text, active_skill) = if text.starts_with('/') {
            let skills =
                crate::skill::SkillRegistry::load_for_working_dir(session.working_dir.as_deref())?;
            acp_skill_prompt(&skills, &text)
        } else {
            (text, None)
        };
        let prompt_id = session.next_id();
        {
            let mut active = session.active_prompt_id.lock().await;
            *active = Some(prompt_id);
        }
        // Events deferred by an earlier control request belong to that turn, not
        // this one; replaying them would duplicate streamed output.
        session.clear_deferred_events().await;

        let send_result = session
            .send(&Request::Message {
                id: prompt_id,
                content: text,
                images,
                system_reminder: None,
                active_skill,
                no_reply: false,
            })
            .await;
        if let Err(err) = send_result {
            cleanup_prompt_state(&session).await;
            return Err(err);
        }

        let mut mapper = EventMapper::new(session.session_id.clone(), self.profile);
        mapper.working_dir = session.working_dir.clone();
        let mut stop_reason = "end_turn".to_string();
        let mut turn_usage = TurnUsage::default();
        loop {
            let event = match session.read_event().await {
                Ok(event) => event,
                Err(err) => {
                    cleanup_prompt_state(&session).await;
                    return Err(err);
                }
            };
            if self.profile.is_extended() {
                self.write_jcode_extension_event(&session.session_id, &event)
                    .await?;
            }
            match event {
                ServerEvent::Ack { .. } => {}
                ServerEvent::Done { id } if id == prompt_id => break,
                ServerEvent::Interrupted => {
                    stop_reason = "cancelled".to_string();
                }
                ServerEvent::Error { id, message, .. } if id == prompt_id => {
                    cleanup_prompt_state(&session).await;
                    self.write_error_value(rpc_id, JSONRPC_INTERNAL_ERROR, message)
                        .await?;
                    return Ok(());
                }
                ServerEvent::TokenUsage {
                    input,
                    output,
                    cache_read_input,
                    cache_creation_input,
                } => {
                    let provider_name = session
                        .ui_state
                        .lock()
                        .await
                        .provider_name
                        .clone()
                        .unwrap_or_default();
                    let full_input = acp_full_input_tokens(
                        &provider_name,
                        input,
                        cache_read_input,
                        cache_creation_input,
                    );
                    let cache = cache_read_input
                        .unwrap_or(0)
                        .saturating_add(cache_creation_input.unwrap_or(0));
                    turn_usage.add(
                        full_input.saturating_sub(cache),
                        output,
                        cache_read_input,
                        cache_creation_input,
                    );
                }
                ServerEvent::AvailableModelsUpdated {
                    provider_name,
                    provider_model,
                    available_models,
                    available_model_routes,
                } => {
                    {
                        let mut state = session.ui_state.lock().await;
                        if provider_name.is_some() {
                            state.provider_name = provider_name;
                        }
                        if provider_model.is_some() {
                            state.model = provider_model;
                        }
                        state.available_models = available_models;
                        state.model_routes = available_model_routes;
                    }
                    self.write_config_option_update(&session).await?;
                }
                ServerEvent::ModelChanged {
                    model,
                    provider_name,
                    error,
                    ..
                } => {
                    // Mid-prompt model changes happen on provider failover;
                    // keep the selector in sync.
                    if error.is_none() {
                        let config_options = {
                            let mut state = session.ui_state.lock().await;
                            state.model = Some(model);
                            state.selected_model_value = None;
                            if provider_name.is_some() {
                                state.provider_name = provider_name;
                            }
                            session_config_options(&state)
                        };
                        if !config_options.is_empty() {
                            self.write_notification(
                                "session/update",
                                json!({
                                    "sessionId": session.session_id,
                                    "update": {
                                        "sessionUpdate": "config_option_update",
                                        "configOptions": config_options,
                                    }
                                }),
                            )
                            .await?;
                        }
                    }
                }
                other => {
                    for update in mapper.map_event(other) {
                        self.write_notification(
                            "session/update",
                            json!({
                                "sessionId": session.session_id,
                                "update": update,
                            }),
                        )
                        .await?;
                    }
                }
            }
        }

        session.ui_state.lock().await.last_turn_usage = turn_usage.to_acp();
        cleanup_prompt_state(&session).await;
        self.write_result(rpc_id, prompt_response(&stop_reason, &turn_usage))
            .await?;
        Ok(())
    }

    async fn run_session_command(
        &self,
        session: &DaemonSession,
        command: AcpSlashCommand,
    ) -> Result<String> {
        match command {
            AcpSlashCommand::Model(None) => {
                let state = session.ui_state.lock().await;
                Ok(match state.model.as_deref() {
                    Some(model) => format!("Current model: `{model}`"),
                    None => "The daemon did not report a current model.".to_string(),
                })
            }
            AcpSlashCommand::Model(Some(model)) => {
                let id = session.next_id();
                set_acp_model(session, id, &model).await?;
                self.write_config_option_update(session).await?;
                let selected = session.ui_state.lock().await.model.clone().unwrap_or(model);
                Ok(format!("Switched model to `{selected}`."))
            }
            AcpSlashCommand::Models => {
                let event = request_model_catalog(session).await?;
                let ServerEvent::History {
                    provider_name,
                    provider_model,
                    available_models,
                    available_model_routes,
                    ..
                } = event
                else {
                    unreachable!("request_model_catalog only returns history")
                };
                let (current, models) = {
                    let mut state = session.ui_state.lock().await;
                    if provider_name.is_some() {
                        state.provider_name = provider_name;
                    }
                    if provider_model.is_some() {
                        state.model = provider_model;
                    }
                    state.available_models = available_models;
                    state.model_routes = available_model_routes;
                    (state.model.clone(), state.available_models.clone())
                };
                self.write_config_option_update(session).await?;
                Ok(format_model_catalog(current.as_deref(), &models))
            }
            AcpSlashCommand::Effort(None) => {
                let state = session.ui_state.lock().await;
                let current = state
                    .reasoning_effort
                    .as_deref()
                    .unwrap_or("provider default");
                let available = available_efforts(&state);
                if available.is_empty() {
                    Ok(format!("Current reasoning effort: `{current}`."))
                } else {
                    Ok(format!(
                        "Current reasoning effort: `{current}`. Available: {}.",
                        available
                            .iter()
                            .map(|effort| format!("`{effort}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            }
            AcpSlashCommand::Effort(Some(effort)) => {
                let id = session.next_id();
                session
                    .send(&Request::SetReasoningEffort {
                        id,
                        effort: effort.clone(),
                        target_session_id: None,
                    })
                    .await?;
                wait_for_effort_changed(session, id).await?;
                self.write_config_option_update(session).await?;
                let selected = session
                    .ui_state
                    .lock()
                    .await
                    .reasoning_effort
                    .clone()
                    .unwrap_or(effort);
                Ok(format!("Set reasoning effort to `{selected}`."))
            }
            AcpSlashCommand::ZedUpdate(source) => run_zed_update(source).await,
            AcpSlashCommand::ZedUpdateStatus => run_zed_update_status().await,
        }
    }

    async fn write_config_option_update(&self, session: &DaemonSession) -> Result<()> {
        let config_options = session_config_options(&*session.ui_state.lock().await);
        self.write_notification(
            "session/update",
            json!({
                "sessionId": session.session_id,
                "update": {
                    "sessionUpdate": "config_option_update",
                    "configOptions": config_options,
                }
            }),
        )
        .await
    }

    /// Report the active context meter to the editor.
    ///
    /// The meter's denominator is the current model's window, so anything that
    /// can change the active model or provider has to re-report it. Only the
    /// turn-level token event used to, which left the meter showing the previous
    /// model's window until the next turn finished.
    async fn write_usage_update(&self, session: &DaemonSession) -> Result<()> {
        let Some((used, size)) = session.ui_state.lock().await.context_usage() else {
            return Ok(());
        };
        self.write_notification(
            "session/update",
            json!({
                "sessionId": session.session_id,
                "update": {
                    "sessionUpdate": "usage_update",
                    "used": used,
                    "size": size,
                }
            }),
        )
        .await
    }

    async fn write_result(&self, id: Value, result: Value) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }))
        .await
    }

    async fn write_error_value(&self, id: Value, code: i64, message: String) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message,
            }
        }))
        .await
    }

    async fn write_notification(&self, method: &str, params: Value) -> Result<()> {
        self.write_value(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
        .await
    }

    async fn write_jcode_extension_event(
        &self,
        session_id: &str,
        event: &ServerEvent,
    ) -> Result<()> {
        self.write_notification(
            "_jcode/server_event",
            json!({
                "sessionId": session_id,
                "event": serde_json::to_value(event).unwrap_or(Value::Null),
            }),
        )
        .await
    }

    async fn write_value(&self, value: Value) -> Result<()> {
        let mut stdout = self.stdout.lock().await;
        let mut line = serde_json::to_string(&value)?;
        line.push('\n');
        stdout.write_all(line.as_bytes()).await?;
        stdout.flush().await?;
        Ok(())
    }
}

async fn cleanup_prompt_state(session: &DaemonSession) {
    {
        let mut active = session.active_prompt_id.lock().await;
        *active = None;
    }
    session.prompt_running.store(false, Ordering::SeqCst);
}

async fn wait_for_done(session: &DaemonSession, request_id: u64) -> Result<()> {
    loop {
        match session.read_event_for_control().await? {
            ServerEvent::Ack { .. } => {}
            ServerEvent::Done { id } if id == request_id => return Ok(()),
            ServerEvent::Error { id, message, .. } if id == request_id => anyhow::bail!(message),
            other => session.defer_event(other).await,
        }
    }
}

async fn request_history(session: &DaemonSession) -> Result<ServerEvent> {
    let id = session.next_id();
    session.send(&Request::GetHistory { id }).await?;
    loop {
        match session.read_event_for_control().await? {
            ServerEvent::Ack { .. } => {}
            event @ ServerEvent::History { id: event_id, .. } if event_id == id => {
                return Ok(event);
            }
            ServerEvent::Error {
                id: event_id,
                message,
                ..
            } if event_id == id => anyhow::bail!(message),
            other => session.defer_event(other).await,
        }
    }
}

async fn request_model_catalog(session: &DaemonSession) -> Result<ServerEvent> {
    tokio::time::timeout(
        std::time::Duration::from_secs(30),
        request_model_catalog_inner(session),
    )
    .await
    .context("Model catalogue lookup timed out")?
}
async fn request_model_catalog_inner(session: &DaemonSession) -> Result<ServerEvent> {
    let id = session.next_id();
    session
        .send(&Request::GetModelCatalog {
            id,
            subscribe_usage_updates: false,
        })
        .await?;
    loop {
        match session.read_event_for_control().await? {
            ServerEvent::Ack { .. } => {}
            event @ ServerEvent::History { id: event_id, .. } if event_id == id => {
                return Ok(event);
            }
            ServerEvent::Error {
                id: event_id,
                message,
                ..
            } if event_id == id => anyhow::bail!(message),
            other => session.defer_event(other).await,
        }
    }
}

const CONFIG_ID_MODEL: &str = "model";
const CONFIG_ID_EFFORT: &str = "reasoning_effort";

fn acp_available_commands() -> Vec<Value> {
    vec![
        json!({"name":"limits", "description":"Fetch provider-reported account limits (separate from context and token usage)"}),
        json!({
            "name": "model",
            "description": "Switch the model for this session, or show the current model",
            "input": { "hint": "model id (optional)" },
        }),
        json!({
            "name": "models",
            "description": "Refresh models and provider routes available to JCode",
        }),
        json!({
            "name": "effort",
            "description": "Set reasoning effort, or show the current effort",
            "input": { "hint": "none|minimal|low|medium|high|xhigh|max (optional)" },
        }),
        json!({
            "name": "zed-update",
            "description": "Build, verify and install this patched JCode from a clean fork checkout",
            "input": { "hint": "checkout path (optional when configured)" },
        }),
        json!({
            "name": "zed-update-status",
            "description": "Show progress and recent output from the JCode adapter update",
        }),
    ]
}

// Preserve daemon-native activation; do not inline skill files into user messages.
fn acp_skill_prompt(skills: &crate::skill::SkillRegistry, text: &str) -> (String, Option<String>) {
    // Leading whitespace is ACP's escape for literal slash-command text.
    if text.starts_with('/') {
        if let Some(invocation) = skills.resolve_invocation(text)
            && skills.contains(invocation.name)
        {
            return (
                invocation.prompt.map(str::to_string).unwrap_or_else(|| {
                    format!(
                        "Apply the {} skill. If required task details are missing, ask for them.",
                        invocation.name
                    )
                }),
                Some(invocation.name.to_string()),
            );
        }
    }
    (text.to_string(), None)
}

fn acp_commands_with_skills(skills: &crate::skill::SkillRegistry) -> Vec<Value> {
    let mut commands = acp_available_commands();
    commands.push(json!({"name": "skills", "description": "List installed skills and refresh skill commands"}));
    commands.push(json!({"name": "usage", "description": "Show observed context and last-turn tokens (not account quota)"}));
    for skill in skills.list() {
        if commands
            .iter()
            .any(|command| command["name"].as_str() == Some(&skill.name))
        {
            continue;
        }
        commands.push(json!({
            "name": skill.name,
            "description": skill.description,
            "input": {"hint": "task or instructions (optional)"},
        }));
    }
    commands
}

fn acp_full_input_tokens(provider: &str, input: u64, read: Option<u64>, write: Option<u64>) -> u64 {
    let read = read.unwrap_or(0);
    let write = write.unwrap_or(0);
    let provider = provider.to_ascii_lowercase();
    if provider.contains("anthropic") || provider.contains("claude") || write > 0 || read > input {
        input.saturating_add(read).saturating_add(write)
    } else {
        input
    }
}

fn compact_swarm_label(text: &str) -> String {
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim();
    let line = line.strip_prefix("PURPOSE:").unwrap_or(line);
    // Zed places its pending-task badge over the right edge of this one-line
    // preview. Keep the display-only label short and free of inline Markdown.
    let before_code = match line.find('`') {
        Some(index) if index > 0 => line[..index].trim().to_string(),
        _ => line.replace('`', ""),
    };
    let collapsed = before_code.split_whitespace().collect::<Vec<_>>().join(" ");
    let sentence_end = collapsed
        .find(". ")
        .map(|index| index + 1)
        .unwrap_or(collapsed.len());
    let collapsed = collapsed[..sentence_end].trim();
    const MAX_LABEL_CHARS: usize = 64;
    let mut label: String = collapsed.chars().take(MAX_LABEL_CHARS).collect();
    if collapsed.chars().count() > MAX_LABEL_CHARS {
        label.push('…');
    }
    label
}

fn acp_usage_listing(state: &SessionUiState) -> String {
    let tier = state
        .service_tier
        .as_deref()
        .unwrap_or("provider default / unreported");
    let context = state
        .last_context_tokens
        .map(|used| {
            format!(
                "Last observed context: {used} / {} tokens.",
                state.context_limit()
            )
        })
        .unwrap_or_else(|| "Context usage has not been reported in this connection yet.".into());
    let turn = state.last_turn_usage.as_ref().map(|usage|
        format!("Last completed turn: {} total tokens; {} uncached input, {} output, {} cache reads, {} cache writes.",
            usage["totalTokens"], usage["inputTokens"], usage["outputTokens"],
            usage.get("cachedReadTokens").unwrap_or(&Value::Null), usage.get("cachedWriteTokens").unwrap_or(&Value::Null)))
        .unwrap_or_else(|| "No completed-turn usage has been reported yet.".into());
    format!(
        "Requested service tier: {tier}. Actual billed tier is not confirmed by this display.\n\n{context}\n\n{turn}\n\nSince this ACP connection opened: {} usage reports; totals {}. Includes background activity in this session; excludes separate workers and earlier connections.\n\nThese are token counts, not subscription quota or billing estimates. Null cache values mean unreported.",
        state.usage_reports,
        state
            .connection_usage
            .to_acp()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "not reported".into())
    )
}

fn acp_model_label(state: &SessionUiState, model: &str) -> String {
    let mut providers: Vec<&str> = state
        .model_routes
        .iter()
        .filter(|route| route.model == model && route.available)
        .map(|route| route.provider.as_str())
        .collect();
    providers.sort_unstable();
    providers.dedup();
    if providers.is_empty() && state.model.as_deref() == Some(model) {
        if let Some(provider) = state.provider_name.as_deref() {
            providers.push(provider);
        }
    }
    if providers.is_empty() {
        model.to_string()
    } else {
        format!("{} · {model}", providers.join(" / "))
    }
}

fn acp_skill_listing(skills: &crate::skill::SkillRegistry) -> String {
    let skills = skills.list();
    if skills.is_empty() {
        return "No installed skills found for this project.".to_string();
    }
    let mut output =
        String::from("Installed skills (invoke with /name followed by your task):\n\n");
    for skill in skills {
        output.push_str(&format!("- /{} — {}\n", skill.name, skill.description));
    }
    output
}

fn insert_session_configuration(result: &mut Value, state: &SessionUiState) {
    let Some(object) = result.as_object_mut() else {
        return;
    };
    let config_options = session_config_options(state);
    if !config_options.is_empty() {
        object.insert("configOptions".to_string(), Value::Array(config_options));
    }
    if let Some(models) = session_models(state) {
        object.insert("models".to_string(), models);
    }
}

fn model_route_value(route: &crate::provider::ModelRoute) -> String {
    // ACP values are opaque IDs. Preserve routes even when native model specs collide.
    format!(
        "jcode-route:{}",
        json!([route.provider, route.api_method, route.model])
    )
}

fn same_provider(a: &str, b: &str) -> bool {
    let normalize = |s: &str| {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    normalize(a) == normalize(b)
}

fn acp_model_options(state: &SessionUiState) -> (String, Vec<Value>) {
    let current = state.model.as_deref().unwrap_or_default();
    let mut routes: Vec<_> = state.model_routes.iter().filter(|r| r.available).collect();
    routes.sort_by_key(|r| (&r.provider, &r.model, &r.api_method));
    let matching: Vec<_> = routes
        .iter()
        .filter(|r| {
            r.model == current
                && state
                    .provider_name
                    .as_deref()
                    .is_some_and(|p| same_provider(p, &r.provider))
        })
        .collect();
    let selected = state
        .selected_model_value
        .as_ref()
        .filter(|value| {
            routes
                .iter()
                .any(|r| r.model == current && model_route_value(r) == **value)
        })
        .cloned()
        .or_else(|| {
            if matching.len() == 1 {
                Some(model_route_value(matching[0]))
            } else {
                None
            }
        })
        .unwrap_or_else(|| current.to_string());
    let mut options = Vec::new();
    for route in routes {
        let value = model_route_value(route);
        if options
            .iter()
            .any(|v: &Value| v["value"].as_str() == Some(value.as_str()))
        {
            continue;
        }
        options.push(
            json!({"value":value, "name":format!("{} · {}",route.provider,route.model),
            "description":format!("{} · {}", route.api_method, route.detail)}),
        );
    }
    if !options
        .iter()
        .any(|v| v["value"].as_str() == Some(selected.as_str()))
    {
        options.insert(0, json!({"value":selected,"name":acp_model_label(state,current),"description":"Current model; exact route was not reported"}));
    }
    (selected, options)
}

fn session_models(state: &SessionUiState) -> Option<Value> {
    state.model.as_ref()?;
    let (current, options) = acp_model_options(state);
    Some(
        json!({"currentModelId":current,"availableModels":options.iter().map(|v|
        json!({"modelId":v["value"],"name":v["name"],"description":v["description"]})).collect::<Vec<_>>()}),
    )
}

async fn refresh_model_catalog(session: &DaemonSession) -> Result<()> {
    let event = request_model_catalog(session).await?;
    if let ServerEvent::History {
        provider_name,
        provider_model,
        available_models,
        available_model_routes,
        reasoning_effort,
        ..
    } = event
    {
        let mut state = session.ui_state.lock().await;
        if provider_name.is_some() {
            state.provider_name = provider_name;
        }
        if provider_model.is_some() {
            state.model = provider_model;
        }
        state.available_models = available_models;
        state.model_routes = available_model_routes;
        state.reasoning_effort = reasoning_effort;
    }
    Ok(())
}

async fn set_acp_model(session: &DaemonSession, id: u64, value: &str) -> Result<()> {
    let route = {
        let state = session.ui_state.lock().await;
        let exact = state
            .model_routes
            .iter()
            .find(|r| r.available && model_route_value(r) == value)
            .cloned();
        let named: Vec<_> = state
            .model_routes
            .iter()
            .filter(|r| r.available && r.model == value)
            .collect();
        // Old saved defaults use bare names. Resolve a unique route explicitly too.
        exact.or_else(|| (named.len() == 1).then(|| named[0].clone()))
    };
    if let Some(route) = route {
        session
            .send(&Request::SetRoute {
                id,
                selection: crate::provider::RouteSelection::from_model_route(&route),
            })
            .await?;
        wait_for_model_changed(session, id).await?;
        session.ui_state.lock().await.selected_model_value = Some(model_route_value(&route));
    } else if value.starts_with("jcode-route:") {
        anyhow::bail!("This model route is no longer available; refresh /models and select again");
    } else {
        // Preserve explicit /model and the client's saved raw model defaults.
        session
            .send(&Request::SetModel {
                id,
                model: value.to_string(),
            })
            .await?;
        wait_for_model_changed(session, id).await?;
    }
    refresh_model_catalog(session).await?;
    Ok(())
}

fn available_efforts(state: &SessionUiState) -> Vec<&'static str> {
    let provider_name = state.provider_name.as_deref();
    let model = state.model.as_deref();
    let inferred = crate::provider::inferred_reasoning_efforts(provider_name, model).into_iter();
    // Name heuristics first so existing providers' ladders do not change, then
    // the ladder the model catalog publishes. The catalog is what gives
    // gateway-only models (OpenCode Go `union-alpha`) selectable levels.
    let efforts: Vec<&'static str> = inferred
        // `swarm`/`swarm-deep` are TUI sentinels, not provider effort levels.
        .filter(|effort| !effort.starts_with("swarm"))
        .collect();
    if !efforts.is_empty() {
        return efforts;
    }
    match (provider_name, model) {
        (Some(provider), Some(model)) => crate::model_pricing::discovered_efforts(provider, model),
        _ => Vec::new(),
    }
}

/// Build the ACP `configOptions` array (model selector plus reasoning effort)
/// from the current session provider state. Empty when the daemon reported no
/// usable model state.
fn session_config_options(state: &SessionUiState) -> Vec<Value> {
    let mut options = Vec::new();

    if state.model.is_some() {
        let (selected, select_options) = acp_model_options(state);
        options.push(
            json!({"type":"select","id":CONFIG_ID_MODEL,"name":"Model","category":"model",
            "currentValue":selected,"options":select_options}),
        );
    }

    let efforts = available_efforts(state);
    if !efforts.is_empty() {
        let current = state
            .reasoning_effort
            .as_deref()
            .filter(|effort| efforts.contains(effort))
            .unwrap_or_else(|| {
                if efforts.contains(&"medium") {
                    "medium"
                } else {
                    efforts[0]
                }
            });
        let select_options: Vec<Value> = efforts
            .iter()
            .map(|name| json!({ "value": name, "name": name }))
            .collect();
        options.push(json!({
            "type": "select",
            "id": CONFIG_ID_EFFORT,
            "name": "Reasoning effort",
            "category": "thought_level",
            "currentValue": current,
            "options": select_options,
        }));
    }

    options
}

async fn wait_for_model_changed(session: &DaemonSession, request_id: u64) -> Result<()> {
    tokio::time::timeout(std::time::Duration::from_secs(30), wait_for_model_changed_inner(session,request_id)).await.context("Model switch timed out; JCode may still complete it, so check the current selection before retrying")?
}
async fn wait_for_model_changed_inner(session: &DaemonSession, request_id: u64) -> Result<()> {
    loop {
        match session.read_event_for_control().await? {
            ServerEvent::Ack { .. } => {}
            ServerEvent::ModelChanged {
                id,
                model,
                provider_name,
                error,
            } if id == request_id => {
                if let Some(error) = error {
                    anyhow::bail!(error);
                }
                let mut state = session.ui_state.lock().await;
                state.model = Some(model);
                state.selected_model_value = None;
                if provider_name.is_some() {
                    state.provider_name = provider_name;
                }
                return Ok(());
            }
            ServerEvent::Error { id, message, .. } if id == request_id => {
                anyhow::bail!(message)
            }
            other => session.defer_event(other).await,
        }
    }
}

async fn wait_for_effort_changed(session: &DaemonSession, request_id: u64) -> Result<()> {
    loop {
        match session.read_event_for_control().await? {
            ServerEvent::Ack { .. } => {}
            ServerEvent::ReasoningEffortChanged { id, effort, error } if id == request_id => {
                if let Some(error) = error {
                    anyhow::bail!(error);
                }
                let mut state = session.ui_state.lock().await;
                state.reasoning_effort = effort;
                return Ok(());
            }
            ServerEvent::Error { id, message, .. } if id == request_id => {
                anyhow::bail!(message)
            }
            other => session.defer_event(other).await,
        }
    }
}

fn acp_history_updates(messages: Vec<crate::protocol::HistoryMessage>) -> Vec<Value> {
    let mut updates = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, message) in messages.into_iter().enumerate() {
        match message.role.as_str() {
            "tool_call" | "tool" | "tool_error" => {
                let tool = message.tool_data.unwrap_or_else(|| crate::message::ToolCall {id:format!("replay-tool-{index}"),name:"tool".into(),..Default::default()});
                if seen.insert(tool.id.clone()) {
                    updates.push(json!({"sessionUpdate":"tool_call","toolCallId":tool.id,"title":detailed_tool_title(&tool.name,&tool.input),"kind":tool_kind(&tool.name),"status":"pending","rawInput":tool.input}));
                }
                if message.role != "tool_call" {
                    updates.push(json!({"sessionUpdate":"tool_call_update","toolCallId":tool.id,"status":if message.role=="tool_error" {"failed"} else {"completed"},"content":[{"type":"content","content":{"type":"text","text":message.content}}],"rawOutput":{"output":message.content}}));
                }
            }
            "thought" | "reasoning" => updates.push(json!({"sessionUpdate":"agent_thought_chunk","messageId":format!("replay-thought-{index}"),"content":{"type":"text","text":message.content}})),
            "user" | "assistant" => {
                if !message.content.is_empty() { updates.push(json!({"sessionUpdate":if message.role=="user" {"user_message_chunk"} else {"agent_message_chunk"},"messageId":format!("replay-message-{index}"),"content":{"type":"text","text":message.content}})); }
            }
            _ => {} // Unknown/internal records must not become ordinary assistant prose.
        }
    }
    updates
}

struct EventMapper {
    session_id: String,
    profile: AcpProfile,
    current_tool_id: Option<String>,
    tool_inputs: HashMap<String, String>,
    working_dir: Option<PathBuf>,
    tool_names: HashMap<String, String>,
    file_before: HashMap<String, Vec<(PathBuf, Option<String>)>>,
    swarm_status: HashMap<String, String>,
}

impl EventMapper {
    fn new(session_id: String, profile: AcpProfile) -> Self {
        Self {
            session_id,
            profile,
            current_tool_id: None,
            tool_inputs: HashMap::new(),
            working_dir: None,
            tool_names: HashMap::new(),
            file_before: HashMap::new(),
            swarm_status: HashMap::new(),
        }
    }

    fn map_event(&mut self, event: ServerEvent) -> Vec<Value> {
        match event {
            ServerEvent::ReasoningDelta { text } => vec![
                json!({"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":text}}),
            ],
            ServerEvent::ReasoningDone { .. } => Vec::new(),
            ServerEvent::TextDelta { text } => vec![agent_message_chunk(text)],
            ServerEvent::TextReplace { text } => vec![agent_message_chunk(text)],
            // The turn loop explains a turn that ended with no visible output
            // (a dropped upstream stream, or a provider guardrail refusal).
            // Editors have no other channel for it, so an empty turn would
            // otherwise look like the agent silently did nothing.
            ServerEvent::ProviderGuardrail { message, .. } => vec![agent_message_chunk(format!(
                "\n[provider] {message}\n"
            ))],
            ServerEvent::ToolStart { id, name } => {
                self.current_tool_id = Some(id.clone());
                self.tool_inputs.entry(id.clone()).or_default();
                self.tool_names.insert(id.clone(), name.clone());
                vec![json!({
                    "sessionUpdate": "tool_call",
                    "toolCallId": id,
                    "title": tool_title(&name),
                    "kind": tool_kind(&name),
                    "status": "pending",
                })]
            }
            ServerEvent::ToolInput { delta } => {
                let Some(tool_id) = self.current_tool_id.clone() else {
                    return Vec::new();
                };
                let buffer = self.tool_inputs.entry(tool_id.clone()).or_default();
                buffer.push_str(&delta);
                let mut update = json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_id,
                });
                if let Some(raw_input) = parse_json_object(buffer)
                    && let Some(object) = update.as_object_mut()
                {
                    if let Some(name) = self.tool_names.get(&tool_id) {
                        object.insert(
                            "title".to_string(),
                            json!(detailed_tool_title(name, &raw_input)),
                        );
                    }
                    object.insert("rawInput".to_string(), raw_input);
                }
                vec![update]
            }
            ServerEvent::ToolExec { id, name } => {
                self.current_tool_id = Some(id.clone());
                self.tool_names.insert(id.clone(), name.clone());
                let input = self
                    .tool_inputs
                    .get(&id)
                    .and_then(|raw| parse_json_object(raw));
                if matches!(
                    name.as_str(),
                    "write" | "edit" | "multiedit" | "apply_patch" | "patch"
                ) {
                    if let Some(input) = input.as_ref() {
                        let snapshots = tool_affected_paths(input, self.working_dir.as_deref())
                            .into_iter()
                            .filter_map(|path| {
                                if !path.exists() {
                                    Some((path, None))
                                } else {
                                    bounded_file_text(&path).map(|text| (path, Some(text)))
                                }
                            })
                            .collect();
                        self.file_before.insert(id.clone(), snapshots);
                    }
                }
                let mut update = json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": id,
                    "title": tool_title(&name),
                    "kind": tool_kind(&name),
                    "status": "in_progress",
                });
                if let Some(input) = self
                    .tool_inputs
                    .get(update["toolCallId"].as_str().unwrap_or_default())
                    && let Some(raw_input) = parse_json_object(input)
                    && let Some(object) = update.as_object_mut()
                {
                    object.insert("rawInput".to_string(), raw_input);
                }
                if let Some(input) = input.as_ref() {
                    update["title"] = json!(detailed_tool_title(&name, input));
                    let paths = tool_affected_paths(input, self.working_dir.as_deref());
                    if !paths.is_empty() {
                        update["locations"] = json!(
                            paths
                                .iter()
                                .map(|path| json!({"path": path.to_string_lossy()}))
                                .collect::<Vec<_>>()
                        );
                    }
                }
                vec![update]
            }
            ServerEvent::ToolOutput { id, output } => {
                if !self.tool_names.contains_key(&id) {
                    return Vec::new();
                }
                vec![
                    json!({"sessionUpdate":"tool_call_update","toolCallId":id,"status":"in_progress", "content":[{"type":"content","content":{"type":"text","text":output}}]}),
                ]
            }
            ServerEvent::ToolDone {
                id,
                name,
                output,
                error,
            } => {
                let input = self
                    .tool_inputs
                    .remove(&id)
                    .and_then(|raw| parse_json_object(&raw));
                let title = input
                    .as_ref()
                    .map(|v| detailed_tool_title(&name, v))
                    .unwrap_or_else(|| tool_title(&name));
                let mut content =
                    vec![json!({"type": "content", "content": {"type": "text", "text": output}})];
                if let Some(error) = error.as_ref() {
                    content.push(json!({"type": "content", "content": {"type": "text", "text": format!("Error: {error}")}}));
                }
                if let Some(snapshots) = self.file_before.remove(&id) {
                    for (path, before) in snapshots {
                        if error.is_none() {
                            if let Some(after) = bounded_file_text(&path).or_else(|| {
                                if !path.exists() {
                                    Some(String::new())
                                } else {
                                    None
                                }
                            }) {
                                if before.as_deref() != Some(after.as_str()) {
                                    content.push(json!({"type": "diff", "path": path.to_string_lossy(), "oldText": before, "newText": after}));
                                }
                            }
                        }
                    }
                }
                self.tool_names.remove(&id);
                vec![json!({
                    "sessionUpdate": "tool_call_update", "toolCallId": id,
                    "title": title, "kind": tool_kind(&name),
                    "status": if error.is_some() { "failed" } else { "completed" },
                    "content": content, "rawOutput": {"output": output, "error": error},
                })]
            }
            ServerEvent::GeneratedImage {
                id,
                path,
                output_format,
                revised_prompt,
                ..
            } => vec![json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": id,
                "status": "completed",
                "content": [{
                    "type": "content",
                    "content": {
                        "type": "text",
                        "text": format!("Generated image: {path} ({output_format}){}", revised_prompt.map(|prompt| format!("\nRevised prompt: {prompt}")).unwrap_or_default()),
                    }
                }]
            })],
            ServerEvent::SwarmStatus { members } => {
                let mut updates = Vec::new();
                for member in members.iter().filter(|m| {
                    m.report_back_to_session_id.as_deref() == Some(self.session_id.as_str())
                }) {
                    let id = format!("swarm-member-{}", member.session_id);
                    let name = member
                        .friendly_name
                        .as_deref()
                        .unwrap_or(&member.session_id);
                    let mut title = format!("🐝 {} · {}", compact_swarm_label(name), member.status);
                    if let Some((done, total)) = member.todo_progress {
                        title.push_str(&format!(" · {done}/{total}"));
                    }
                    if let Some(secs) = member.runtime.elapsed_secs.filter(|secs| *secs >= 15) {
                        title.push_str(&format!(" · {}s", secs / 15 * 15));
                    }
                    let detail = member
                        .task_label
                        .as_deref()
                        .or(member.detail.as_deref())
                        .map(compact_swarm_label)
                        .unwrap_or_default();
                    let snapshot = title.clone();
                    if self.swarm_status.get(&id) == Some(&snapshot) {
                        continue;
                    }
                    let first = self.swarm_status.insert(id.clone(), snapshot).is_none();
                    let status = match member.status.as_str() {
                        "completed" | "done" => "completed",
                        "failed" | "stopped" | "cancelled" => "failed",
                        "running" | "working" | "busy" | "thinking" | "streaming" => "in_progress",
                        _ => "pending",
                    };
                    updates.push(json!({
                        "sessionUpdate": if first { "tool_call" } else { "tool_call_update" },
                        "toolCallId": id, "title": title,
                        "kind": "other", "status": status,
                        "content": [{"type":"content", "content":{"type":"text", "text":detail}}],
                    }));
                }
                updates
            }
            ServerEvent::SwarmPlan {
                items,
                participants,
                ..
            } => {
                if !participants.is_empty() && !participants.contains(&self.session_id) {
                    return Vec::new();
                }
                let entries: Vec<Value> = items
                    .iter()
                    .map(|item| {
                        let status = match item.status.as_str() {
                            "completed" | "done" => "completed",
                            "in_progress" | "running" => "in_progress",
                            _ => "pending",
                        };
                        let content =
                            if matches!(item.status.as_str(), "failed" | "blocked" | "cancelled") {
                                format!("[{}] {}", item.status, compact_swarm_label(&item.content))
                            } else {
                                compact_swarm_label(&item.content)
                            };
                        let priority = match item.priority.as_str() {
                            "high" => "high",
                            "low" => "low",
                            _ => "medium",
                        };
                        json!({"content": content, "status": status, "priority": priority})
                    })
                    .collect();
                vec![json!({"sessionUpdate": "plan", "entries": entries})]
            }
            ServerEvent::Compaction { trigger, .. } if self.profile.is_extended() => vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": format!("\n[Jcode compacted context: {trigger}]\n"),
                }
            })],
            ServerEvent::SessionRenamed { display_title, .. } => vec![json!({
                "sessionUpdate": "session_info_update",
                "title": display_title,
            })],
            ServerEvent::McpStatus { servers } if self.profile.is_extended() => vec![json!({
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": format!("\n[Jcode MCP status: {}]\n", servers.join(", ")),
                }
            })],
            _ => {
                let _ = &self.session_id;
                Vec::new()
            }
        }
    }
}

fn parse_json_object(input: &str) -> Option<Value> {
    let value: Value = serde_json::from_str(input).ok()?;
    value.as_object()?;
    Some(value)
}

fn compatibility_option_value(
    params: &Value,
    value_fields: &[&str],
    method: &str,
) -> std::result::Result<String, String> {
    if !params.is_object() {
        return Err(format!("{method} params must be an object"));
    }
    value_fields
        .iter()
        .find_map(|field| {
            params
                .get(*field)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            format!(
                "{method} requires a non-empty string {}",
                value_fields.join(" or ")
            )
        })
}

fn initialize_result(params: &Value, profile: AcpProfile) -> Value {
    // We only speak exactly ACP_PROTOCOL_VERSION; the response pins to our
    // version regardless of the `protocolVersion` the client requested.
    let _ = params;
    let protocol_version = ACP_PROTOCOL_VERSION;
    let mut agent_capabilities = json!({
        "loadSession": true,
        "promptCapabilities": {
            "image": true,
            "audio": false,
            "embeddedContext": true,
        },
        "mcpCapabilities": {
            "http": false,
            "sse": false,
        },
        "sessionCapabilities": {
            "close": {},
            "resume": {},
            "list": {},
        }
    });

    if profile.is_extended()
        && let Some(object) = agent_capabilities.as_object_mut()
    {
        object.insert(
            "_meta".to_string(),
            json!({
                "jcode": {
                    "profile": profile.as_str(),
                    "extensions": ["raw_server_event"]
                }
            }),
        );
    }

    json!({
        "protocolVersion": protocol_version,
        "agentCapabilities": agent_capabilities,
        "agentInfo": {
            "name": "jcode",
            "title": "Jcode",
            "version": format!("{}+zed-acp.8", jcode_build_meta::pkg_version()),
        },
        "authMethods": [],
    })
}

fn cwd_from_params(params: &Value) -> std::result::Result<PathBuf, String> {
    let cwd = match params.get("cwd").and_then(Value::as_str) {
        Some(cwd) if !cwd.trim().is_empty() => PathBuf::from(cwd),
        _ => std::env::current_dir().map_err(|err| err.to_string())?,
    };
    if !cwd.is_absolute() {
        return Err(format!("ACP cwd must be absolute: {}", cwd.display()));
    }
    Ok(cwd)
}

fn required_session_id(params: &Value) -> std::result::Result<String, String> {
    params
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "Missing required sessionId".to_string())
}

fn acp_session_list(params: &Value, live: Vec<Value>) -> Result<Value> {
    let cwd = params.get("cwd").and_then(Value::as_str);
    if cwd.is_some_and(|p| !std::path::Path::new(p).is_absolute()) {
        anyhow::bail!("cwd must be absolute");
    }
    let offset = match params.get("cursor") {
        None | Some(Value::Null) => 0,
        Some(Value::String(s)) => s
            .strip_prefix("offset:")
            .context("Invalid session cursor")?
            .parse::<usize>()
            .context("Invalid session cursor")?,
        _ => anyhow::bail!("Invalid session cursor"),
    };
    let dir = crate::storage::jcode_dir()?.join("sessions");
    let mut rows: Vec<Value> = live
        .into_iter()
        .filter(|row| cwd.is_none_or(|filter| row["cwd"].as_str() == Some(filter)))
        .collect();
    let entries = if dir.exists() {
        Some(std::fs::read_dir(dir)?)
    } else {
        None
    };
    for entry in entries.into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json")
            || !entry.file_type().is_ok_and(|t| t.is_file())
        {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(session) = crate::session::Session::load_startup_stub(id) else {
            continue;
        };
        // Keep internal/swarm children out of the user-facing history importer.
        if session.parent_id.is_some() || crate::storage::session_is_internal(id) {
            continue;
        }
        let Some(dir) = session
            .working_dir
            .filter(|d| std::path::Path::new(d).is_absolute())
        else {
            continue;
        };
        if cwd.is_some_and(|filter| filter != dir) {
            continue;
        }
        if rows
            .iter()
            .any(|row| row["sessionId"].as_str() == Some(session.id.as_str()))
        {
            continue;
        }
        rows.push(json!({"sessionId":session.id,"cwd":dir,"title":session.custom_title.or(session.title).unwrap_or_else(|| id.into()),"updatedAt":session.updated_at.to_rfc3339()}));
    }
    rows.sort_by(|a, b| {
        b["updatedAt"]
            .as_str()
            .cmp(&a["updatedAt"].as_str())
            .then(a["sessionId"].as_str().cmp(&b["sessionId"].as_str()))
    });
    if offset > rows.len() {
        anyhow::bail!("Session cursor expired; restart listing");
    }
    let end = offset.saturating_add(50).min(rows.len());
    let mut result = json!({"sessions": &rows[offset..end]});
    if end < rows.len() {
        result["nextCursor"] = json!(format!("offset:{end}"));
    }
    Ok(result)
}

async fn acp_account_limits() -> String {
    let reports = match tokio::time::timeout(std::time::Duration::from_secs(20), crate::usage::fetch_all_provider_usage()).await {
        Ok(reports) => reports,
        Err(_) => return "Provider limit lookup timed out. Token/context information remains available through /usage.".into(),
    };
    let mut lines = vec![
        "Provider-reported account limits (JCode may cache reports for two minutes):".to_string(),
    ];
    for report in reports {
        lines.push(format!("\n{}", report.provider_name));
        if report.error.is_some() {
            lines.push(
                "Limits unavailable: the provider lookup failed. Check JCode authentication."
                    .into(),
            );
            continue;
        }
        if report.limits.is_empty() {
            lines.push("No numeric limits reported.".into());
        }
        for limit in report.limits {
            if limit.usage_percent.is_finite() {
                lines.push(format!(
                    "{}: {:.1}% used{}",
                    limit.name,
                    limit.usage_percent,
                    limit
                        .resets_at
                        .map(|r| format!("; resets {r}"))
                        .unwrap_or_default()
                ));
            }
        }
    }
    if lines.len() == 1 {
        lines.push(
            "No supported account-limit reports are available for the configured providers.".into(),
        );
    }
    lines.push("\nThese are account limits, not this turn's tokens or monetary cost.".into());
    lines.join("\n")
}

fn acp_mcp_config(servers: &Value) -> Result<Value> {
    let entries = servers
        .as_array()
        .context("ACP mcpServers must be an array")?;
    if entries.len() > 32 {
        anyhow::bail!("At most 32 Zed MCP servers are supported per session");
    }
    let mut mapped = serde_json::Map::new();
    for server in entries {
        let name = server["name"]
            .as_str()
            .filter(|n| !n.is_empty())
            .context("MCP server requires name")?;
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            anyhow::bail!("MCP server name must contain letters, digits, underscores or hyphens");
        }
        if server
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t != "stdio")
        {
            anyhow::bail!("JCode currently supports only stdio MCP servers");
        }
        let command = server["command"]
            .as_str()
            .filter(|s| !s.is_empty())
            .context("MCP stdio server requires command")?;
        let args = server.get("args").cloned().unwrap_or(json!([]));
        if !args
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_string))
        {
            anyhow::bail!("MCP args must be strings");
        }
        let mut env = serde_json::Map::new();
        if let Some(values) = server.get("env") {
            for entry in values.as_array().context("MCP env must be an array")? {
                env.insert(
                    entry["name"]
                        .as_str()
                        .context("MCP env requires name")?
                        .into(),
                    json!(entry["value"].as_str().context("MCP env requires value")?),
                );
            }
        }
        let key = format!("acp_zed_{}", name.replace('-', "_"));
        if mapped.contains_key(&key) {
            anyhow::bail!("Duplicate MCP server name after normalization");
        }
        mapped.insert(key, json!({"command":command,"args":args,"env":env,"shared":false,"type":"stdio","timeout_secs":30}));
    }
    Ok(json!({"servers":mapped}))
}

async fn configure_acp_mcp(
    session: &DaemonSession,
    servers: &Value,
    editor_read: bool,
) -> Result<()> {
    let config = acp_mcp_config(servers)?;
    let id = session.next_id();
    session
        .send(&Request::ConfigureAcpMcp {
            id,
            servers: config,
            editor_read,
        })
        .await?;
    tokio::time::timeout(
        std::time::Duration::from_secs(35),
        wait_for_done(session, id),
    )
    .await
    .context("Timed out configuring Zed MCP servers")??;
    Ok(())
}

fn validate_acp_mcp_servers(params: &Value) -> std::result::Result<(), String> {
    acp_mcp_config(params.get("mcpServers").unwrap_or(&json!([])))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[derive(Debug, PartialEq, Eq)]
enum AcpSlashCommand {
    Model(Option<String>),
    Models,
    Effort(Option<String>),
    ZedUpdate(Option<String>),
    ZedUpdateStatus,
}

fn parse_acp_slash_command(text: &str) -> Option<Result<AcpSlashCommand>> {
    // A leading space is the ACP client convention for escaping slash command
    // interpretation and sending the text to the model literally.
    let trimmed = text.trim_end();
    let body = trimmed.strip_prefix('/')?;
    let mut parts = body.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|argument| !argument.is_empty())
        .map(str::to_string);
    match name {
        "model" => Some(Ok(AcpSlashCommand::Model(argument))),
        "models" if argument.is_none() => Some(Ok(AcpSlashCommand::Models)),
        "models" => Some(Err(anyhow::anyhow!("/models does not accept an argument"))),
        "effort" => Some(Ok(AcpSlashCommand::Effort(argument))),
        "zed-update" => Some(Ok(AcpSlashCommand::ZedUpdate(argument))),
        "zed-update-status" if argument.is_none() => Some(Ok(AcpSlashCommand::ZedUpdateStatus)),
        "zed-update-status" => Some(Err(anyhow::anyhow!(
            "/zed-update-status does not accept an argument"
        ))),
        _ => None,
    }
}

async fn run_zed_update(source: Option<String>) -> Result<String> {
    let source = source
        .or_else(|| std::env::var("JCODE_ZED_SOURCE_DIR").ok())
        .filter(|value| !value.trim().is_empty())
        .context(
            "No checkout was supplied. Configure JCODE_ZED_SOURCE_DIR or run /zed-update /absolute/path/to/jcode",
        )?;
    run_zed_update_helper("start", Some(source)).await
}

async fn run_zed_update_status() -> Result<String> {
    run_zed_update_helper("status", None).await
}

async fn run_zed_update_helper(action: &str, source: Option<String>) -> Result<String> {
    let helper = std::env::var("JCODE_ZED_UPDATE_HELPER")
        .context("JCODE_ZED_UPDATE_HELPER is not configured for this ACP server")?;
    let helper_path = std::path::Path::new(&helper);
    if !helper_path.is_file() {
        anyhow::bail!(
            "JCode update helper was not found at {}",
            helper_path.display()
        );
    }
    let mut command = tokio::process::Command::new(helper_path);
    command.arg(action).kill_on_drop(true);
    if let Some(source) = source {
        command.arg("--source").arg(source);
        if std::env::var("JCODE_ZED_UPDATE_OFFLINE")
            .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        {
            command.arg("--offline");
        }
    }
    if let Ok(state_dir) = std::env::var("JCODE_ZED_UPDATE_STATE_DIR")
        && !state_dir.trim().is_empty()
    {
        command.arg("--state-dir").arg(state_dir);
    }
    let output = tokio::time::timeout(std::time::Duration::from_secs(15), command.output())
        .await
        .context("JCode update helper did not respond within 15 seconds")??;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        anyhow::bail!(
            "{}",
            if stderr.is_empty() {
                format!("JCode update helper exited with {}", output.status)
            } else {
                stderr
            }
        );
    }
    Ok(if stdout.is_empty() {
        "JCode update helper completed without a status message.".to_string()
    } else {
        stdout
    })
}

fn format_model_catalog(current: Option<&str>, models: &[String]) -> String {
    if models.is_empty() {
        return match current {
            Some(current) => format!("Current model: `{current}`. No model catalog was reported."),
            None => "The active provider did not report a model catalog.".to_string(),
        };
    }
    let mut output = String::from("Available models:\n");
    for model in models {
        let selected = if Some(model.as_str()) == current {
            " (current)"
        } else {
            ""
        };
        output.push_str(&format!("- `{model}`{selected}\n"));
    }
    output.pop();
    output
}

fn prompt_from_params(
    params: &Value,
) -> std::result::Result<(String, Vec<(String, String)>), String> {
    let prompt = params
        .get("prompt")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing required prompt array".to_string())?;
    let mut text_parts = Vec::new();
    let mut images = Vec::new();

    for block in prompt {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                }
            }
            Some("image") => {
                let mime_type = block
                    .get("mimeType")
                    .or_else(|| block.get("mime_type"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Image content block missing mimeType".to_string())?;
                let data = block
                    .get("data")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Image content block missing data".to_string())?;
                images.push((mime_type.to_string(), data.to_string()));
            }
            Some("resource") => {
                if let Some(resource) = block.get("resource") {
                    text_parts.push(format_resource_block(resource));
                }
            }
            Some("resource_link") => {
                let uri = block.get("uri").and_then(Value::as_str).unwrap_or("");
                let name = block.get("name").and_then(Value::as_str).unwrap_or(uri);
                text_parts.push(format!("[Resource link: {name} <{uri}>]"));
            }
            Some(other) => {
                return Err(format!(
                    "Unsupported ACP prompt content block type: {other}"
                ));
            }
            None => return Err("Prompt content block missing type".to_string()),
        }
    }

    Ok((text_parts.join("\n\n"), images))
}

fn format_resource_block(resource: &Value) -> String {
    let uri = resource
        .get("uri")
        .and_then(Value::as_str)
        .unwrap_or("resource");
    if let Some(text) = resource.get("text").and_then(Value::as_str) {
        format!("[Embedded resource: {uri}]\n{text}")
    } else if let Some(blob) = resource.get("blob").and_then(Value::as_str) {
        let mime = resource
            .get("mimeType")
            .or_else(|| resource.get("mime_type"))
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        format!(
            "[Embedded binary resource: {uri} ({mime}, {} base64 bytes)]",
            blob.len()
        )
    } else {
        format!("[Embedded resource: {uri}]")
    }
}

fn agent_message_chunk(text: String) -> Value {
    json!({
        "sessionUpdate": "agent_message_chunk",
        "content": {
            "type": "text",
            "text": text,
        }
    })
}

/// One display line for inter-agent traffic. Assignments, reports, channel
/// posts and file conflicts are the only way a coordinator reaches an idle
/// session, so they belong in the transcript rather than being dropped.
fn notification_chunk_text(
    from_session: &str,
    from_name: Option<&str>,
    notification_type: &crate::protocol::NotificationType,
    message: &str,
) -> String {
    let sender = from_name.unwrap_or(from_session);
    let label = match notification_type {
        crate::protocol::NotificationType::Message { scope, channel, .. } => {
            match (scope.as_deref(), channel.as_deref()) {
                (Some("channel"), Some(channel)) => format!("#{channel} from {sender}"),
                (Some("broadcast"), _) => format!("Broadcast from {sender}"),
                _ => format!("Message from {sender}"),
            }
        }
        crate::protocol::NotificationType::FileConflict { path, .. } => {
            format!("File conflict on {path} from {sender}")
        }
        crate::protocol::NotificationType::SharedContext { key, .. } => {
            format!("Shared context `{key}` from {sender}")
        }
    };
    format!("{label}: {message}")
}

/// Upper bound for a file preview shipped to an ACP client. The client renders
/// `oldText`/`newText` as the whole file, so larger paths are skipped instead of
/// sending megabytes of JSON per edit.
const ACP_DIFF_MAX_BYTES: u64 = 256 * 1024;

fn bounded_file_text(path: &std::path::Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > ACP_DIFF_MAX_BYTES {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

fn tool_file_path(input: &Value, cwd: Option<&std::path::Path>) -> Option<PathBuf> {
    let raw = input
        .get("file_path")
        .or_else(|| input.get("path"))?
        .as_str()?;
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        Some(path)
    } else {
        cwd.map(|base| base.join(path))
    }
}

fn tool_affected_paths(input: &Value, cwd: Option<&std::path::Path>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = tool_file_path(input, cwd) {
        paths.push(path);
    }
    if let Some(patch) = input.get("patch_text").and_then(Value::as_str) {
        for line in patch.lines() {
            let raw = [
                "*** Update File: ",
                "*** Add File: ",
                "*** Delete File: ",
                "*** Move to: ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix));
            if let Some(raw) = raw {
                if let Some(path) = tool_file_path(&json!({"file_path":raw}), cwd) {
                    paths.push(path);
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn detailed_tool_title(name: &str, input: &Value) -> String {
    if name == "bg" {
        return match input.get("action").and_then(Value::as_str) {
            Some("wait") => "Waiting for background task".to_string(),
            Some("status" | "list") => "Checking background tasks".to_string(),
            Some("output" | "tail") => "Reading background task output".to_string(),
            Some("watch" | "delivery" | "subscribe") => "Watching background task".to_string(),
            Some("cancel") => "Stopping background task".to_string(),
            Some("cleanup") => "Cleaning up background tasks".to_string(),
            _ => "Background task".to_string(),
        };
    }
    let detail = match name {
        "bash" => input
            .get("command")
            .or_else(|| input.get("cmd"))
            .and_then(Value::as_str),
        "read" | "view_file" | "read_file" | "write" | "edit" | "multiedit" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .or_else(|| input.get("AbsolutePath"))
            .and_then(Value::as_str),
        "swarm" | "skill_manage" => input.get("action").and_then(Value::as_str),
        _ => None,
    };
    match detail {
        Some(detail) => {
            let single_line = detail.split_whitespace().collect::<Vec<_>>().join(" ");
            let label: String = single_line.chars().take(180).collect();
            if name == "bash" {
                label
            } else {
                format!("{}: {label}", tool_title(name))
            }
        }
        None => tool_title(name),
    }
}

fn tool_title(name: &str) -> String {
    match name {
        "bash" => "Running shell command".to_string(),
        "read" | "view_file" | "read_file" => "Reading file".to_string(),
        "write" => "Writing file".to_string(),
        "edit" | "multiedit" | "patch" | "apply_patch" => "Editing files".to_string(),
        "agentgrep" | "grep" | "glob" | "ls" | "list_dir" | "find_by_name" => {
            "Searching workspace".to_string()
        }
        "webfetch" | "websearch" | "search_web" => "Fetching web content".to_string(),
        "bg" => "Background task".to_string(),
        other => other.replace('_', " "),
    }
}

pub(crate) fn tool_kind(name: &str) -> &'static str {
    match name {
        "read" | "view_file" | "read_file" => "read",
        "write" | "edit" | "multiedit" | "patch" | "apply_patch" => "edit",
        "bash" | "selfdev" => "execute",
        "bg" => "other",
        "agentgrep"
        | "grep"
        | "glob"
        | "ls"
        | "list_dir"
        | "find_by_name"
        | "session_search"
        | "conversation_search" => "search",
        "webfetch" | "websearch" | "search_web" | "codesearch" => "fetch",
        _ => "other",
    }
}

pub(crate) async fn run_acp_command(
    provider_choice: ProviderChoice,
    model: Option<String>,
    provider_profile: Option<String>,
    explicit_tool_profile: bool,
) -> Result<()> {
    crate::env::set_var("JCODE_NON_INTERACTIVE", "1");
    let acp_config = crate::config::config().acp.clone();
    if !explicit_tool_profile {
        crate::env::set_var("JCODE_TOOL_PROFILE", acp_config.tool_profile.trim());
        crate::config::invalidate_config_cache();
    }
    let profile = AcpProfile::parse(&acp_config.profile);
    AcpRuntime::new(profile, provider_choice, model, provider_profile)
        .run()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn compact_swarm_display_does_not_embed_assignment_packet() {
        assert_eq!(
            compact_swarm_label("\nPURPOSE: Draft section 3.4\nPRIVATE INPUT: long packet"),
            "Draft section 3.4"
        );
        assert_eq!(compact_swarm_label(&"文".repeat(200)).chars().count(), 65);
        assert_eq!(
            compact_swarm_label(
                "Swarm smoke test A (rerun). Your label is smoke_a. Run `echo swarm-ok-A2`"
            ),
            "Swarm smoke test A (rerun)."
        );
        assert_eq!(
            compact_swarm_label("`cargo test` validates the adapter"),
            "cargo test validates the adapter"
        );
    }

    #[test]
    fn acp_tool_kind_maps_core_tools() {
        assert_eq!(tool_kind("read"), "read");
        assert_eq!(tool_kind("apply_patch"), "edit");
        assert_eq!(tool_kind("bash"), "execute");
        assert_eq!(tool_kind("bg"), "other");
        assert_eq!(tool_kind("agentgrep"), "search");
        assert_eq!(tool_kind("webfetch"), "fetch");
        assert_eq!(tool_kind("swarm"), "other");
    }

    #[test]
    fn acp_background_titles_describe_the_action() {
        assert_eq!(
            detailed_tool_title("bg", &json!({"action":"wait"})),
            "Waiting for background task"
        );
        assert_eq!(
            detailed_tool_title("bg", &json!({"action":"tail"})),
            "Reading background task output"
        );
    }

    #[test]
    fn json_rpc_parse_errors_use_standard_codes() {
        let (code, _) = JsonRpcMessage::parse("not json").unwrap_err();
        assert_eq!(code, JSONRPC_PARSE_ERROR);

        let (code, message) = JsonRpcMessage::parse(r#"{"method":"initialize"}"#).unwrap_err();
        assert_eq!(code, JSONRPC_INVALID_REQUEST);
        assert!(message.contains("jsonrpc"));
    }

    #[test]
    fn prompt_from_params_accepts_text_images_and_resources() {
        let params = json!({
            "sessionId": "s1",
            "prompt": [
                {"type": "text", "text": "hello"},
                {"type": "image", "mimeType": "image/png", "data": "abc"},
                {"type": "resource", "resource": {"uri": "file:///tmp/a.rs", "text": "fn main(){}"}},
                {"type": "resource_link", "uri": "file:///tmp/b.rs", "name": "b.rs"}
            ]
        });
        let (text, images) = prompt_from_params(&params).unwrap();
        assert!(text.contains("hello"));
        assert!(text.contains("Embedded resource: file:///tmp/a.rs"));
        assert!(text.contains("Resource link: b.rs"));
        assert_eq!(images, vec![("image/png".to_string(), "abc".to_string())]);
    }

    #[test]
    fn prompt_response_reports_usage_accumulated_across_the_turn() {
        let mut usage = TurnUsage::default();
        usage.add(10, 2, Some(4), Some(5));
        usage.add(20, 3, Some(6), Some(7));

        assert_eq!(
            prompt_response("end_turn", &usage),
            json!({
                "stopReason": "end_turn",
                "usage": {
                    "totalTokens": 57,
                    "inputTokens": 30,
                    "outputTokens": 5,
                    "cachedReadTokens": 10,
                    "cachedWriteTokens": 12,
                }
            })
        );
    }

    #[test]
    fn prompt_response_omits_unreported_usage_and_cache_fields() {
        assert_eq!(
            prompt_response("end_turn", &TurnUsage::default()),
            json!({ "stopReason": "end_turn" })
        );

        let mut usage = TurnUsage::default();
        usage.add(10, 2, None, None);
        assert_eq!(
            prompt_response("cancelled", &usage),
            json!({
                "stopReason": "cancelled",
                "usage": {
                    "totalTokens": 12,
                    "inputTokens": 10,
                    "outputTokens": 2,
                }
            })
        );
    }

    #[test]
    fn initialize_standard_omits_jcode_meta() {
        let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Standard);
        assert_eq!(result["protocolVersion"], 1);
        assert!(result["agentCapabilities"].get("_meta").is_none());
        assert_eq!(result["agentCapabilities"]["loadSession"], true);
    }

    #[test]
    fn initialize_full_advertises_jcode_extension_meta() {
        let result = initialize_result(&json!({"protocolVersion": 1}), AcpProfile::Full);
        assert_eq!(
            result["agentCapabilities"]["_meta"]["jcode"]["profile"],
            "full"
        );
    }

    #[test]
    fn event_mapper_maps_tool_lifecycle() {
        let mut mapper = EventMapper::new("session1".to_string(), AcpProfile::Standard);
        let start = mapper.map_event(ServerEvent::ToolStart {
            id: "tool1".to_string(),
            name: "bash".to_string(),
        });
        assert_eq!(start[0]["sessionUpdate"], "tool_call");
        assert_eq!(start[0]["kind"], "execute");

        let input = mapper.map_event(ServerEvent::ToolInput {
            delta: "{\"command\":\"true\"}".to_string(),
        });
        assert_eq!(input[0]["rawInput"]["command"], "true");

        let done = mapper.map_event(ServerEvent::ToolDone {
            id: "tool1".to_string(),
            name: "bash".to_string(),
            output: "ok".to_string(),
            error: None,
        });
        assert_eq!(done[0]["status"], "completed");
        assert_eq!(done[0]["content"][0]["content"]["text"], "ok");
    }

    #[test]
    fn non_empty_mcp_servers_are_tolerated_until_session_scoped_mcp_is_supported() {
        let params =
            json!({"mcpServers": [{"name": "fs", "command":"example", "args":[], "env":[]}]});
        assert!(validate_acp_mcp_servers(&params).is_ok());

        let params = json!({"mcpServers": []});
        assert!(validate_acp_mcp_servers(&params).is_ok());
    }

    #[test]
    fn cache_usage_does_not_double_count_inclusive_provider_inputs() {
        assert_eq!(acp_full_input_tokens("openai", 100, Some(40), None), 100);
        assert_eq!(
            acp_full_input_tokens("anthropic", 100, Some(40), Some(10)),
            150
        );
        assert_eq!(acp_full_input_tokens("anthropic", 0, Some(40), None), 40);
        let mut usage = TurnUsage::default();
        let full = acp_full_input_tokens("openai", 100, Some(40), None);
        usage.add(full - 40, 20, Some(40), None);
        assert_eq!(usage.to_acp().unwrap()["totalTokens"], 120);
    }

    #[test]
    fn replay_does_not_flatten_tools_or_thoughts_into_answer_text() {
        let messages=serde_json::from_value(json!([
            {"role":"thought","content":"Fixture thought","tool_calls":null,"tool_data":null},
            {"role":"tool_call","content":"","tool_calls":null,"tool_data":{"id":"t","name":"view_file","input":{"path":"/fixture"}}},
            {"role":"tool_error","content":"fixture failure","tool_calls":null,"tool_data":{"id":"t","name":"view_file","input":{"path":"/fixture"}}},
            {"role":"assistant","content":"Answer","tool_calls":null,"tool_data":null}
        ])).unwrap();
        let updates = acp_history_updates(messages);
        assert_eq!(updates[0]["sessionUpdate"], "agent_thought_chunk");
        assert_eq!(updates[1]["kind"], "read");
        assert_eq!(updates[2]["status"], "failed");
        assert_eq!(
            updates
                .iter()
                .filter(|v| v["sessionUpdate"] == "agent_message_chunk")
                .count(),
            1
        );
        let mut mapper = EventMapper::new("live".into(), AcpProfile::Standard);
        assert_eq!(
            mapper.map_event(ServerEvent::ReasoningDelta {
                text: "Fixture thought".into()
            })[0]["sessionUpdate"],
            "agent_thought_chunk"
        );
    }

    #[test]
    fn forwarded_mcp_is_private_and_rejects_colliding_or_unsupported_servers() {
        let config = acp_mcp_config(&json!([{"name":"fixture","command":"/test/server","env":[{"name":"TOKEN","value":"fixture-only"}]}])).unwrap();
        assert_eq!(config["servers"]["acp_zed_fixture"]["shared"], false);
        assert_eq!(
            config["servers"]["acp_zed_fixture"]["env"]["TOKEN"],
            "fixture-only"
        );
        assert!(
            acp_mcp_config(&json!([{"name":"x","type":"http","url":"https://example.com"}]))
                .is_err()
        );
        assert!(
            acp_mcp_config(&json!([{"name":"a-b","command":"x"},{"name":"a_b","command":"y"}]))
                .is_err()
        );
    }
    #[test]
    fn late_stream_output_cannot_revive_completed_tools() {
        let mut mapper = EventMapper::new("test".into(), AcpProfile::Standard);
        mapper.map_event(ServerEvent::ToolStart {
            id: "tool".into(),
            name: "bash".into(),
        });
        let progress = mapper.map_event(ServerEvent::ToolOutput {
            id: "tool".into(),
            output: "working".into(),
        });
        assert_eq!(progress[0]["status"], "in_progress");
        mapper.map_event(ServerEvent::ToolDone {
            id: "tool".into(),
            name: "bash".into(),
            output: "done".into(),
            error: None,
        });
        assert!(
            mapper
                .map_event(ServerEvent::ToolOutput {
                    id: "tool".into(),
                    output: "late".into()
                })
                .is_empty()
        );
    }
    #[test]
    fn session_listing_rejects_invalid_filters_and_cursor() {
        assert!(acp_session_list(&json!({"cwd":"relative"}), Vec::new()).is_err());
        assert!(acp_session_list(&json!({"cursor":"bad"}), Vec::new()).is_err());
    }

    #[test]
    fn empty_turn_notice_reaches_the_editor_as_agent_text() {
        // A turn that ends with no visible output (dropped upstream stream,
        // guardrail refusal) has no other channel to the editor; without this
        // mapping the user sees the turn end silently.
        let mut mapper = EventMapper::new("test".into(), AcpProfile::Standard);
        let updates = mapper.map_event(ServerEvent::ProviderGuardrail {
            stop_reason: None,
            message: "The provider returned an empty response.".into(),
        });
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0]["sessionUpdate"], "agent_message_chunk");
        let text = updates[0]["content"]["text"].as_str().unwrap();
        assert!(
            text.contains("The provider returned an empty response."),
            "the notice text must survive, got {text:?}"
        );
    }

    #[test]
    fn distinct_provider_routes_keep_distinct_selector_values() {
        let a = crate::provider::ModelRoute {
            model: "shared-model".into(),
            provider: "Gateway A".into(),
            api_method: "unknown".into(),
            available: true,
            detail: String::new(),
            cheapness: None,
            usage: None,
        };
        let mut b = a.clone();
        b.provider = "Gateway B".into();
        assert_ne!(model_route_value(&a), model_route_value(&b));
    }

    #[test]
    fn route_catalog_labels_every_provider_before_selection_and_excludes_unavailable() {
        let route =
            |model: &str, provider: &str, api: &str, available| crate::provider::ModelRoute {
                model: model.into(),
                provider: provider.into(),
                api_method: api.into(),
                available,
                detail: String::new(),
                cheapness: None,
                usage: None,
            };
        let state = SessionUiState {
            model: Some("deepseek-flash".into()),
            provider_name: Some("OpenCode Go".into()),
            model_routes: vec![
                route(
                    "deepseek-flash",
                    "OpenCode Go",
                    "openai-compatible:opencode-go",
                    true,
                ),
                route("claude-opus-4-6", "Anthropic", "claude-oauth", true),
                route("gpt-6-astra", "OpenAI", "openai-oauth", true),
                route(
                    "claude-opus-4-6",
                    "Anthropic API",
                    "anthropic-api-key",
                    false,
                ),
            ],
            ..Default::default()
        };
        let (_current, options) = acp_model_options(&state);
        assert_eq!(options.len(), 3);
        assert!(
            options
                .iter()
                .any(|v| v["name"] == "Anthropic · claude-opus-4-6"
                    && v["value"] == model_route_value(&state.model_routes[1]))
        );
        assert!(options.iter().any(|v| v["name"] == "OpenAI · gpt-6-astra"
            && v["value"] == model_route_value(&state.model_routes[2])));
        assert!(
            options
                .iter()
                .all(|v| !v["name"].as_str().unwrap().contains("Anthropic API"))
        );
        let legacy = session_models(&state).unwrap();
        assert_eq!(legacy["availableModels"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn provider_labels_preserve_model_selection_ids() {
        let state = SessionUiState {
            provider_name: Some("claude".into()),
            model: Some("claude-opus-4-6".into()),
            available_models: vec!["claude-opus-4-6".into()],
            ..Default::default()
        };
        let options = session_config_options(&state);
        assert_eq!(options[0]["options"][0]["value"], "claude-opus-4-6");
        assert_eq!(options[0]["options"][0]["name"], "claude · claude-opus-4-6");
        assert!(acp_usage_listing(&state).contains("not been reported"));
    }

    #[test]
    fn command_titles_and_errors_are_visible() {
        let mut mapper = EventMapper::new("test".into(), AcpProfile::Standard);
        mapper.map_event(ServerEvent::ToolStart {
            id: "cmd".into(),
            name: "bash".into(),
        });
        let input = mapper.map_event(ServerEvent::ToolInput {
            delta: r#"{"command":"printf hello"}"#.into(),
        });
        assert_eq!(input[0]["title"], "printf hello");
        mapper.map_event(ServerEvent::ToolExec {
            id: "cmd".into(),
            name: "bash".into(),
        });
        let done = mapper.map_event(ServerEvent::ToolDone {
            id: "cmd".into(),
            name: "bash".into(),
            output: String::new(),
            error: Some("test failure".into()),
        });
        assert_eq!(done[0]["title"], "printf hello");
        assert_eq!(done[0]["status"], "failed");
        assert!(
            done[0]["content"][1]["content"]["text"]
                .as_str()
                .unwrap()
                .contains("test failure")
        );
    }

    #[test]
    fn edit_diff_uses_complete_before_and_after_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("example.txt");
        std::fs::write(&path, "before\nunchanged\n").unwrap();
        let mut mapper = EventMapper::new("test".into(), AcpProfile::Standard);
        mapper.working_dir = Some(dir.path().into());
        mapper.map_event(ServerEvent::ToolStart {
            id: "edit".into(),
            name: "edit".into(),
        });
        mapper.map_event(ServerEvent::ToolInput {
            delta: r#"{"file_path":"example.txt","old_string":"before","new_string":"after"}"#
                .into(),
        });
        let exec = mapper.map_event(ServerEvent::ToolExec {
            id: "edit".into(),
            name: "edit".into(),
        });
        assert_eq!(
            exec[0]["locations"][0]["path"],
            path.to_string_lossy().as_ref()
        );
        std::fs::write(&path, "after\nunchanged\n").unwrap();
        let done = mapper.map_event(ServerEvent::ToolDone {
            id: "edit".into(),
            name: "edit".into(),
            output: "saved".into(),
            error: None,
        });
        let diff = &done[0]["content"][1];
        assert_eq!(diff["type"], "diff");
        assert_eq!(diff["oldText"], "before\nunchanged\n");
        assert_eq!(diff["newText"], "after\nunchanged\n");
    }

    #[test]
    fn patch_previews_find_each_changed_file_once() {
        let dir = tempfile::tempdir().unwrap();
        let paths = tool_affected_paths(
            &json!({"patch_text": "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n*** Update File: a.txt\n*** Add File: b.txt\n+hello\n*** End Patch"}),
            Some(dir.path()),
        );
        assert_eq!(
            paths,
            vec![dir.path().join("a.txt"), dir.path().join("b.txt")]
        );
    }

    #[test]
    fn swarm_status_is_scoped_and_repeated_snapshots_are_suppressed() {
        let mut mapper = EventMapper::new("parent".into(), AcpProfile::Standard);
        let member: crate::protocol::SwarmMemberStatus = serde_json::from_value(json!({
            "session_id": "child", "status": "running", "friendly_name": "Researcher",
            "todo_progress": [1, 3], "report_back_to_session_id": "parent",
            "runtime": {"model":"deepseek-v4.1-flash", "elapsed_secs":31}
        }))
        .unwrap();
        let event = mapper.map_event(ServerEvent::SwarmStatus {
            members: vec![member.clone()],
        });
        assert_eq!(event[0]["sessionUpdate"], "tool_call");
        assert_eq!(event[0]["status"], "in_progress");
        assert_eq!(event[0]["title"], "🐝 Researcher · running · 1/3 · 30s");
        assert!(
            mapper
                .map_event(ServerEvent::SwarmStatus {
                    members: vec![member.clone()]
                })
                .is_empty()
        );
        let mut advanced = member.clone();
        advanced.todo_progress = Some((2, 3));
        advanced.runtime.elapsed_secs = Some(46);
        let update = mapper.map_event(ServerEvent::SwarmStatus {
            members: vec![advanced],
        });
        assert_eq!(update[0]["sessionUpdate"], "tool_call_update");
        assert_eq!(update[0]["title"], "🐝 Researcher · running · 2/3 · 45s");
        let mut unrelated = member;
        unrelated.report_back_to_session_id = Some("someone-else".into());
        assert!(
            mapper
                .map_event(ServerEvent::SwarmStatus {
                    members: vec![unrelated]
                })
                .is_empty()
        );
    }

    fn acp_test_skills() -> (tempfile::TempDir, crate::skill::SkillRegistry) {
        let dir = tempfile::tempdir().unwrap();
        for name in ["review-example", "model"] {
            let folder = dir.path().join(".jcode/skills").join(name);
            std::fs::create_dir_all(&folder).unwrap();
            std::fs::write(
                folder.join("SKILL.md"),
                format!(
                    "---\nname: {name}\ndescription: Test skill\n---\nReview the supplied task.\n"
                ),
            )
            .unwrap();
        }
        let skills = crate::skill::SkillRegistry::load_project_overlay(Some(dir.path())).unwrap();
        (dir, skills)
    }

    #[test]
    fn skill_commands_include_discovery_and_do_not_shadow_model_control() {
        let (_dir, skills) = acp_test_skills();
        let commands = acp_commands_with_skills(&skills);
        assert!(commands.iter().any(|v| v["name"] == "skills"));
        assert!(commands.iter().any(|v| v["name"] == "review-example"));
        assert_eq!(commands.iter().filter(|v| v["name"] == "model").count(), 1);
        assert!(acp_skill_listing(&skills).contains("/review-example"));
    }

    #[test]
    fn skill_invocation_activates_native_skill_and_preserves_literal_inputs() {
        let (_dir, skills) = acp_test_skills();
        assert_eq!(
            acp_skill_prompt(&skills, "/review-example inspect this"),
            ("inspect this".into(), Some("review-example".into()))
        );
        assert_eq!(
            acp_skill_prompt(&skills, " /review-example inspect this"),
            (" /review-example inspect this".into(), None)
        );
        assert_eq!(
            acp_skill_prompt(&skills, "/unknown task"),
            ("/unknown task".into(), None)
        );
        assert_eq!(
            acp_skill_prompt(&skills, "/tmp/example.png"),
            ("/tmp/example.png".into(), None)
        );
        let (prompt, active) = acp_skill_prompt(&skills, "/review-example");
        assert_eq!(active.as_deref(), Some("review-example"));
        assert!(!prompt.is_empty());
    }

    #[test]
    fn advertised_commands_cover_all_acp_daemon_model_controls() {
        let commands = acp_available_commands();
        let names: Vec<&str> = commands
            .iter()
            .map(|command| command["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "limits",
                "model",
                "models",
                "effort",
                "zed-update",
                "zed-update-status"
            ]
        );
        assert_eq!(commands[1]["input"]["hint"], "model id (optional)");
        assert!(commands[2].get("input").is_none());
        assert!(
            commands[3]["input"]["hint"]
                .as_str()
                .unwrap()
                .contains("high")
        );
        assert!(
            commands[4]["input"]["hint"]
                .as_str()
                .unwrap()
                .contains("checkout")
        );
        assert!(commands[5].get("input").is_none());
    }

    #[test]
    fn advertised_commands_parse_to_real_dispatch_variants() {
        assert_eq!(
            parse_acp_slash_command("/model claude-sonnet-4-5")
                .unwrap()
                .unwrap(),
            AcpSlashCommand::Model(Some("claude-sonnet-4-5".to_string()))
        );
        assert_eq!(
            parse_acp_slash_command("/model ").unwrap().unwrap(),
            AcpSlashCommand::Model(None)
        );
        assert_eq!(
            parse_acp_slash_command("/models").unwrap().unwrap(),
            AcpSlashCommand::Models
        );
        assert_eq!(
            parse_acp_slash_command("/effort xhigh").unwrap().unwrap(),
            AcpSlashCommand::Effort(Some("xhigh".to_string()))
        );
        assert_eq!(
            parse_acp_slash_command("/zed-update /tmp/jcode fork")
                .unwrap()
                .unwrap(),
            AcpSlashCommand::ZedUpdate(Some("/tmp/jcode fork".to_string()))
        );
        assert_eq!(
            parse_acp_slash_command("/zed-update-status")
                .unwrap()
                .unwrap(),
            AcpSlashCommand::ZedUpdateStatus
        );
        assert!(parse_acp_slash_command("/models now").unwrap().is_err());
        assert!(
            parse_acp_slash_command("/zed-update-status now")
                .unwrap()
                .is_err()
        );
        assert!(parse_acp_slash_command("/not-advertised").is_none());
        assert!(parse_acp_slash_command(" /model literal").is_none());
        assert!(parse_acp_slash_command("ordinary prompt").is_none());
    }

    #[test]
    fn compatibility_methods_accept_host_field_names_and_aliases() {
        assert_eq!(
            compatibility_option_value(
                &json!({"modelId": "deepseek-v4-flash"}),
                &["modelId", "model"],
                "session/set_model"
            )
            .unwrap(),
            "deepseek-v4-flash"
        );
        assert_eq!(
            compatibility_option_value(
                &json!({"reasoningEffort": "high"}),
                &["effort", "reasoningEffort"],
                "session/set_reasoning_effort"
            )
            .unwrap(),
            "high"
        );
        assert!(
            compatibility_option_value(
                &json!({"effort": ""}),
                &["effort", "reasoningEffort"],
                "session/set_reasoning_effort"
            )
            .unwrap_err()
            .contains("non-empty")
        );
    }

    #[test]
    fn cwd_must_be_absolute() {
        let params = json!({"cwd": "relative"});
        assert!(cwd_from_params(&params).is_err());
        let params = json!({"cwd": "/tmp"});
        assert_eq!(cwd_from_params(&params).unwrap(), Path::new("/tmp"));
    }

    #[test]
    fn config_options_include_model_selector_and_effort_ladder() {
        let state = SessionUiState {
            provider_name: Some("openai".to_string()),
            model: Some("gpt-5.2".to_string()),
            available_models: vec!["gpt-5.2".to_string(), "gpt-5.2-codex".to_string()],
            reasoning_effort: Some("high".to_string()),
            ..SessionUiState::default()
        };
        let options = session_config_options(&state);
        assert_eq!(options.len(), 2);

        let model = &options[0];
        assert_eq!(model["id"], CONFIG_ID_MODEL);
        assert_eq!(model["category"], "model");
        assert_eq!(model["type"], "select");
        assert_eq!(model["currentValue"], "gpt-5.2");
        assert_eq!(model["options"].as_array().unwrap().len(), 1); // No unverified routes advertised.

        let effort = &options[1];
        assert_eq!(effort["id"], CONFIG_ID_EFFORT);
        assert_eq!(effort["category"], "thought_level");
        assert_eq!(effort["currentValue"], "high");
        let effort_values: Vec<&str> = effort["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|option| option["value"].as_str().unwrap())
            .collect();
        assert!(effort_values.contains(&"medium"));
        assert!(
            !effort_values.iter().any(|value| value.starts_with("swarm")),
            "swarm sentinels are TUI-only and must not leak over ACP: {effort_values:?}"
        );
    }

    #[test]
    fn config_options_include_opencode_go_muse_effort_ladder() {
        let state = SessionUiState {
            provider_name: Some("OpenCode Go".to_string()),
            model: Some("muse-spark-1.3-contributor".to_string()),
            reasoning_effort: Some("medium".to_string()),
            ..SessionUiState::default()
        };
        let options = session_config_options(&state);
        let effort = options
            .iter()
            .find(|option| option["id"] == CONFIG_ID_EFFORT)
            .expect("Muse should expose an ACP reasoning control");
        let values = effort["options"]
            .as_array()
            .expect("effort options")
            .iter()
            .filter_map(|option| option["value"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec!["none", "minimal", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(effort["currentValue"], "medium");
    }

    #[test]
    fn config_options_current_model_prepended_when_not_listed() {
        let state = SessionUiState {
            provider_name: Some("anthropic".to_string()),
            model: Some("claude-opus-4-6".to_string()),
            available_models: vec!["claude-sonnet-4-5".to_string()],
            reasoning_effort: None,
            ..SessionUiState::default()
        };
        let options = session_config_options(&state);
        let model_values: Vec<&str> = options[0]["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|option| option["value"].as_str().unwrap())
            .collect();
        assert_eq!(model_values[0], "claude-opus-4-6");
        assert!(!model_values.contains(&"claude-sonnet-4-5")); // A name alone is not a selectable route.
    }

    #[test]
    fn legacy_models_catalog_is_emitted_alongside_config_options() {
        let state = SessionUiState {
            provider_name: Some("deepseek".to_string()),
            model: Some("deepseek-v4-flash".to_string()),
            available_models: vec!["deepseek-v4-pro".to_string()],
            reasoning_effort: Some("high".to_string()),
            ..SessionUiState::default()
        };
        let mut result = json!({"sessionId": "s1"});
        insert_session_configuration(&mut result, &state);

        assert!(result["configOptions"].is_array());
        assert_eq!(result["models"]["currentModelId"], "deepseek-v4-flash");
        let ids: Vec<&str> = result["models"]["availableModels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["modelId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["deepseek-v4-flash"]);
    }

    #[test]
    fn config_options_empty_without_model_state() {
        let options = session_config_options(&SessionUiState::default());
        assert!(options.is_empty());
    }

    #[test]
    fn context_limit_falls_back_to_default_for_unknown_models() {
        let state = SessionUiState {
            provider_name: Some("mystery".to_string()),
            model: Some("mystery-model-9000".to_string()),
            available_models: Vec::new(),
            reasoning_effort: None,
            ..SessionUiState::default()
        };
        assert_eq!(
            state.context_limit(),
            crate::provider::DEFAULT_CONTEXT_LIMIT as u64
        );
    }

    #[test]
    fn context_usage_denominator_follows_the_active_model() {
        let mut state = SessionUiState {
            provider_name: Some("OpenAI".to_string()),
            model: Some("gpt-5.6-sol".to_string()),
            last_context_tokens: Some(167_000),
            ..SessionUiState::default()
        };
        assert_eq!(state.context_usage(), Some((167_000, 272_000)));

        // The same conversation on a wider model must report the wider window,
        // which is what an editor's context meter divides the usage by.
        state.provider_name = Some("OpenCode Go".to_string());
        state.model = Some("muse-spark-1.3-contributor".to_string());
        assert_eq!(state.context_usage(), Some((167_000, 1_048_576)));
    }

    #[test]
    fn context_usage_is_absent_until_a_turn_reports_tokens() {
        // Reporting `0` would claim an empty context, so a session that has
        // never seen usage stays silent no matter how the model changes.
        let mut state = SessionUiState {
            provider_name: Some("OpenAI".to_string()),
            model: Some("gpt-5.6-sol".to_string()),
            ..SessionUiState::default()
        };
        assert_eq!(state.context_usage(), None);
        state.model = Some("muse-spark-1.3-contributor".to_string());
        assert_eq!(state.context_usage(), None);
    }

    #[cfg(unix)]
    fn test_daemon_session() -> DaemonSession {
        let (client, _server) = tokio::net::UnixStream::pair().expect("unix socket pair");
        let (reader, writer) = client.into_split();
        DaemonSession::new("deferred-test".to_string(), reader, writer, 1)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn deferred_control_events_replay_in_order_and_stop_at_the_cap() {
        let session = test_daemon_session();
        // A control request (model switch, catalog refresh) must not swallow
        // streamed output belonging to the running turn.
        session
            .defer_event(ServerEvent::TextDelta {
                text: "kept".into(),
            })
            .await;
        session
            .defer_event(ServerEvent::ToolOutput {
                id: "tool-1".into(),
                output: "streamed".into(),
            })
            .await;

        match session.read_event().await.expect("first deferred event") {
            ServerEvent::TextDelta { text } => assert_eq!(text, "kept"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
        assert!(matches!(
            session.read_event().await.expect("second deferred event"),
            ServerEvent::ToolOutput { .. }
        ));

        // A new turn starts clean: the previous turn's deferred events are gone.
        session.clear_deferred_events().await;
        assert!(session.deferred_events.lock().await.is_empty());

        // Repeated live snapshots for one tool call collapse to the newest.
        session
            .defer_event(ServerEvent::ToolOutput {
                id: "tool-1".into(),
                output: "first".into(),
            })
            .await;
        session
            .defer_event(ServerEvent::ToolOutput {
                id: "tool-1".into(),
                output: "second".into(),
            })
            .await;
        {
            let deferred = session.deferred_events.lock().await;
            assert_eq!(deferred.len(), 1, "one stream must not stack snapshots");
            match deferred.front() {
                Some(ServerEvent::ToolOutput { output, .. }) => assert_eq!(output, "second"),
                other => panic!("expected the newest snapshot, got {other:?}"),
            }
        }
        session.clear_deferred_events().await;

        // The queue is bounded, so a long control wait cannot grow without limit.
        for index in 0..1100 {
            session
                .defer_event(ServerEvent::TextDelta {
                    text: format!("event-{index}"),
                })
                .await;
        }
        let deferred = session.deferred_events.lock().await;
        assert_eq!(deferred.len(), 512);
        match deferred.front() {
            Some(ServerEvent::TextDelta { text }) => assert_eq!(text, "event-588"),
            other => panic!("expected the oldest surviving TextDelta, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn control_reads_do_not_reconsume_deferred_events() {
        use tokio::io::AsyncWriteExt;
        let (client, mut server) = tokio::net::UnixStream::pair().expect("unix socket pair");
        let (reader, writer) = client.into_split();
        let session = DaemonSession::new("deferred-control".to_string(), reader, writer, 1);
        let wire_event = serde_json::to_string(&ServerEvent::TextDelta {
            text: "from-wire".into(),
        })
        .expect("serialize wire event");
        server
            .write_all(format!("{wire_event}\n").as_bytes())
            .await
            .expect("write wire event");

        // A control request defers an unrelated event and must still reach its own
        // reply: draining the deferred queue here would spin on that one event
        // forever, which is exactly how session/new used to hang.
        session
            .defer_event(ServerEvent::ToolOutput {
                id: "tool-1".into(),
                output: "deferred".into(),
            })
            .await;
        match session
            .read_event_for_control()
            .await
            .expect("control read")
        {
            ServerEvent::TextDelta { text } => assert_eq!(text, "from-wire"),
            other => panic!("control read stole a deferred event: {other:?}"),
        }

        // The stream owner still receives the deferred event before new ones.
        assert!(matches!(
            session.read_event().await.expect("stream read"),
            ServerEvent::ToolOutput { .. }
        ));
    }

    #[test]
    fn inter_agent_notifications_render_with_their_sender_and_scope() {
        use crate::protocol::NotificationType;
        assert_eq!(
            notification_chunk_text(
                "session-abc",
                Some("Researcher"),
                &NotificationType::Message {
                    scope: Some("dm".into()),
                    channel: None,
                    tldr: None,
                },
                "please verify section 3"
            ),
            "Message from Researcher: please verify section 3"
        );
        // Falls back to the session id when no friendly name was reported.
        assert_eq!(
            notification_chunk_text(
                "session-abc",
                None,
                &NotificationType::Message {
                    scope: Some("channel".into()),
                    channel: Some("parser".into()),
                    tldr: None,
                },
                "landed the fix"
            ),
            "#parser from session-abc: landed the fix"
        );
        assert_eq!(
            notification_chunk_text(
                "session-abc",
                Some("Writer"),
                &NotificationType::FileConflict {
                    path: "src/lib.rs".into(),
                    operation: "wrote".into(),
                    intent: None,
                    summary: None,
                    detail: None,
                },
                "another agent edited this file"
            ),
            "File conflict on src/lib.rs from Writer: another agent edited this file"
        );
    }
}
