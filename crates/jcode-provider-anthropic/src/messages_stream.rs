//! Anthropic Messages SSE translation shared by every Messages-wire caller.
//!
//! The direct Anthropic runtime (`jcode-provider-anthropic-runtime`) and
//! OpenAI-compatible gateways that serve individual models through the Messages
//! API (e.g. OpenCode Go's `union-alpha`, `minimax-m3`, `qwen3.8-flash`) must
//! translate the same SSE vocabulary into jcode stream events. Keeping one
//! translator means thinking deltas, tool-call accumulation, usage bookkeeping,
//! and served-model substitution warnings behave identically on both paths.
//!
//! The translator is pure: callers own their HTTP transport, idle timeouts,
//! retries, and channel plumbing and simply feed response chunks in.

use jcode_message_types::{ConnectionPhase, StreamEvent};
use jcode_provider_core::anthropic::{
    anthropic_map_tool_name_from_oauth, anthropic_strip_1m_suffix,
};
use serde::Deserialize;

/// Per-stream settings for [`MessagesSseTranslator`].
#[derive(Debug, Clone, Default)]
pub struct MessagesStreamOptions {
    /// Base id of the model the client asked for. Empty disables the
    /// served-model substitution warning (unit tests, unknown model).
    pub requested_model: String,
    /// Human label for the upstream used in log lines and warnings.
    pub provider_label: String,
    /// Apply Claude Code OAuth tool-name mapping. Only the direct OAuth
    /// transport needs this; API-key gateways use plain tool names.
    pub oauth_tool_name_mapping: bool,
}

/// Token usage accumulated across a Messages stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessagesUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}

impl MessagesUsage {
    /// True when the stream reported at least one usage counter.
    pub fn has_any(&self) -> bool {
        self.input_tokens.is_some() || self.output_tokens.is_some()
    }
}

/// Incremental Messages SSE translator.
///
/// Feed response chunks with [`Self::push_chunk`]; each call returns the stream
/// events completed by those bytes (possibly none when a chunk splits an event).
pub struct MessagesSseTranslator {
    buffer: String,
    state: SseStreamState,
    options: MessagesStreamOptions,
}

impl MessagesSseTranslator {
    pub fn new(options: MessagesStreamOptions) -> Self {
        let mut state = SseStreamState {
            requested_model_base: anthropic_strip_1m_suffix(options.requested_model.trim())
                .to_ascii_lowercase(),
            ..SseStreamState::default()
        };
        state.provider_label = if options.provider_label.trim().is_empty() {
            "Anthropic".to_string()
        } else {
            options.provider_label.clone()
        };
        Self {
            buffer: String::new(),
            state,
            options,
        }
    }

    /// Translate one response chunk into the stream events it completed.
    pub fn push_chunk(&mut self, chunk: &[u8]) -> Vec<StreamEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut events = Vec::new();
        while let Some(event) = parse_sse_event(&mut self.buffer) {
            events.extend(process_sse_event(
                &event,
                &mut self.state,
                self.options.oauth_tool_name_mapping,
            ));
        }
        events
    }

    /// Usage counters seen so far.
    pub fn usage(&self) -> MessagesUsage {
        MessagesUsage {
            input_tokens: self.state.input_tokens,
            output_tokens: self.state.output_tokens,
            cache_read_input_tokens: self.state.cache_read_input_tokens,
            cache_creation_input_tokens: self.state.cache_creation_input_tokens,
        }
    }
}

/// SSE event from the stream
struct SseEvent {
    event_type: String,
    data: String,
}

struct ToolUseAccumulator {
    #[expect(
        dead_code,
        reason = "accumulated for debugging parity with the direct runtime's stream state"
    )]
    input_json: String,
}

#[derive(Default)]
struct SseStreamState {
    current_tool_use: Option<ToolUseAccumulator>,
    current_thinking_block: bool,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    /// Lowercased base id of the model we asked for, so `message_start` can flag
    /// a silent server-side substitution (e.g. an unavailable id aliased to a
    /// different model). Empty when unknown (e.g. in unit tests).
    requested_model_base: String,
    /// Set once we have warned about a substitution, so we only warn per stream.
    warned_model_substitution: bool,
    provider_label: String,
}

/// Parse a single SSE event from the buffer
fn parse_sse_event(buffer: &mut String) -> Option<SseEvent> {
    // Look for complete event (ends with double newline)
    let event_end = buffer.find("\n\n")?;
    let event_str = buffer[..event_end].to_string();
    buffer.drain(..event_end + 2);

    let mut event_type = String::new();
    let mut data = String::new();

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event_type = rest.to_string();
        } else if let Some(rest) = data_line(line) {
            data = rest.to_string();
        }
    }

    if event_type.is_empty() && data.is_empty() {
        return None;
    }

    Some(SseEvent { event_type, data })
}

/// Extract the payload of an SSE `data:` line (with or without a space).
fn data_line(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

/// Process an SSE event and return StreamEvents if applicable
fn process_sse_event(
    event: &SseEvent,
    state: &mut SseStreamState,
    is_oauth: bool,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    match event.event_type.as_str() {
        "message_start" => {
            // Extract usage from message_start (includes cache info)
            if let Ok(parsed) = serde_json::from_str::<MessageStartEvent>(&event.data) {
                // The server echoes the model that actually served the request.
                // Log it so we can confirm there was no silent server-side
                // substitution (and surface it under JCODE_LOG_SERVED_MODEL).
                if let Some(served) = parsed.message.model.as_deref() {
                    jcode_logging::info(&format!(
                        "{} served model={}",
                        state.provider_label, served
                    ));
                    if std::env::var("JCODE_LOG_SERVED_MODEL").is_ok() {
                        eprintln!("[messages] served model={served}");
                    }
                    // Anthropic can silently alias an unavailable/retired model
                    // id to a different model (observed: claude-fable-5 ->
                    // claude-haiku-4-5). That is a correctness hazard: the user
                    // believes they are on the requested flagship. Warn loudly
                    // once per stream when the served base id differs.
                    let served_base = anthropic_strip_1m_suffix(served).to_ascii_lowercase();
                    if !state.requested_model_base.is_empty()
                        && !state.warned_model_substitution
                        && served_base != state.requested_model_base
                    {
                        state.warned_model_substitution = true;
                        jcode_logging::warn(&format!(
                            "{} served a DIFFERENT model than requested: requested '{}', served '{}'. The requested model is likely unavailable and is being substituted server-side.",
                            state.provider_label, state.requested_model_base, served_base
                        ));
                        events.push(StreamEvent::StatusDetail {
                            detail: format!(
                                "⚠ {} served '{}' instead of requested '{}' (requested model unavailable)",
                                state.provider_label, served_base, state.requested_model_base
                            ),
                        });
                    }
                }
                if let Some(usage) = parsed.message.usage {
                    state.input_tokens = usage.input_tokens.map(|t| t as u64);
                    state.cache_read_input_tokens = usage.cache_read_input_tokens.map(|t| t as u64);
                    state.cache_creation_input_tokens =
                        usage.cache_creation_input_tokens.map(|t| t as u64);
                    if let Some(tier) = usage.service_tier.as_deref() {
                        jcode_logging::info(&format!(
                            "{} granted service_tier={}",
                            state.provider_label, tier
                        ));
                        if std::env::var("JCODE_LOG_SERVICE_TIER").is_ok() {
                            eprintln!("[messages] granted service_tier={tier}");
                        }
                    }
                }
            }
        }
        "content_block_start" => {
            if let Ok(parsed) = serde_json::from_str::<ContentBlockStartEvent>(&event.data) {
                match parsed.content_block {
                    ApiContentBlockStart::Text { .. } => {
                        // Text block starting - nothing to emit yet
                    }
                    ApiContentBlockStart::Thinking { thinking, .. } => {
                        state.current_thinking_block = true;
                        events.push(StreamEvent::ThinkingStart);
                        if !thinking.is_empty() {
                            events.push(StreamEvent::ThinkingDelta(thinking));
                        }
                    }
                    ApiContentBlockStart::RedactedThinking { .. } => {
                        state.current_thinking_block = true;
                        events.push(StreamEvent::ThinkingStart);
                    }
                    ApiContentBlockStart::ToolUse { id, name } => {
                        let mapped_name = if is_oauth {
                            anthropic_map_tool_name_from_oauth(&name)
                        } else {
                            name.clone()
                        };
                        // Start accumulating tool use
                        state.current_tool_use = Some(ToolUseAccumulator {
                            input_json: String::new(),
                        });
                        events.push(StreamEvent::ToolUseStart {
                            id,
                            name: mapped_name,
                        });
                    }
                    ApiContentBlockStart::Unknown => {
                        // Newer/unsupported block type. Parsing succeeded, so
                        // the rest of the stream stays intact; there is simply
                        // nothing for this build to surface.
                        jcode_logging::warn(
                            "Messages stream sent an unrecognized content_block_start type; ignoring the block",
                        );
                    }
                }
            }
        }
        "content_block_delta" => {
            if let Ok(parsed) = serde_json::from_str::<ContentBlockDeltaEvent>(&event.data) {
                match parsed.delta {
                    ApiDelta::Text { text } => {
                        events.push(StreamEvent::TextDelta(text));
                    }
                    ApiDelta::InputJson { partial_json } => {
                        if let Some(tool) = state.current_tool_use.as_mut() {
                            tool.input_json.push_str(&partial_json);
                        }
                        events.push(StreamEvent::ToolInputDelta(partial_json));
                    }
                    ApiDelta::Thinking { thinking } => {
                        events.push(StreamEvent::ThinkingDelta(thinking));
                    }
                    ApiDelta::Signature { signature } => {
                        events.push(StreamEvent::ThinkingSignatureDelta(signature));
                    }
                }
            }
        }
        "content_block_stop" => {
            // If we were accumulating a tool_use, it's complete now
            if state.current_tool_use.take().is_some() {
                events.push(StreamEvent::ToolUseEnd);
            } else if state.current_thinking_block {
                state.current_thinking_block = false;
                events.push(StreamEvent::ThinkingEnd);
            }
        }
        "message_delta" => {
            if let Ok(parsed) = serde_json::from_str::<MessageDeltaEvent>(&event.data) {
                if let Some(usage) = parsed.usage {
                    state.output_tokens = usage.output_tokens.map(|t| t as u64);
                }
                if let Some(stop_reason) = parsed.delta.stop_reason {
                    events.push(StreamEvent::MessageEnd {
                        stop_reason: Some(stop_reason),
                    });
                }
            }
        }
        "message_stop" => {
            // Final message stop - we may have already sent MessageEnd via message_delta
        }
        "ping" => {
            // Keepalive. Surface it as a phase event instead of swallowing it:
            // during silent reasoning phases (adaptive thinking with hidden or
            // summarized display) pings can be the only upstream traffic, and
            // downstream consumers (the TUI stall guard) need to see *some*
            // event to know the stream is alive (issue #451).
            events.push(StreamEvent::ConnectionPhase {
                phase: ConnectionPhase::Streaming,
            });
        }
        "error" => {
            jcode_logging::error(&format!(
                "{} stream error: {}",
                state.provider_label, event.data
            ));
            events.push(StreamEvent::Error {
                message: event.data.clone(),
                retry_after_secs: None,
            });
        }
        _ => {
            // Unknown event type, ignore
        }
    }

    events
}

// ============================================================================
// Wire types
// ============================================================================

#[derive(Deserialize)]
struct MessageStartEvent {
    message: MessageStartMessage,
}

#[derive(Deserialize)]
struct MessageStartMessage {
    #[serde(default)]
    model: Option<String>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct ContentBlockStartEvent {
    #[serde(rename = "index")]
    _index: u32,
    content_block: ApiContentBlockStart,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiContentBlockStart {
    #[serde(rename = "text")]
    Text {
        #[serde(rename = "text")]
        _text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default, rename = "thinking")]
        thinking: String,
        #[serde(default, rename = "signature")]
        _signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(default, rename = "data")]
        _data: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    /// A block type this build does not recognize (for example a newer
    /// server-side tool block). Kept as an explicit catch-all so the
    /// surrounding `content_block_start` event still deserializes instead of
    /// being dropped whole.
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct ContentBlockDeltaEvent {
    #[serde(rename = "index")]
    _index: u32,
    delta: ApiDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
}

#[derive(Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDeltaDelta,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct MessageDeltaDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct UsageInfo {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    service_tier: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translator() -> MessagesSseTranslator {
        MessagesSseTranslator::new(MessagesStreamOptions {
            requested_model: "union-alpha".to_string(),
            provider_label: "Messages".to_string(),
            oauth_tool_name_mapping: false,
        })
    }

    fn sse(event: &str, data: &str) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn text_deltas(events: &[StreamEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::TextDelta(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    fn thinking_deltas(events: &[StreamEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ThinkingDelta(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn text_and_usage_events_translate() {
        let mut translator = translator();
        let mut events = translator.push_chunk(
            sse(
                "message_start",
                r#"{"message":{"model":"union-alpha","usage":{"input_tokens":12,"output_tokens":0,"cache_read_input_tokens":3}}}"#,
            )
            .as_bytes(),
        );
        events.extend(
            translator.push_chunk(
                sse(
                    "content_block_start",
                    r#"{"index":0,"content_block":{"type":"text","text":""}}"#,
                )
                .as_bytes(),
            ),
        );
        events.extend(
            translator.push_chunk(
                sse(
                    "content_block_delta",
                    r#"{"index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
                )
                .as_bytes(),
            ),
        );
        events.extend(
            translator.push_chunk(
                sse(
                    "message_delta",
                    r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}"#,
                )
                .as_bytes(),
            ),
        );

        assert_eq!(text_deltas(&events), vec!["ok".to_string()]);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd { stop_reason } if stop_reason.as_deref() == Some("end_turn")
        )));
        assert_eq!(
            translator.usage(),
            MessagesUsage {
                input_tokens: Some(12),
                output_tokens: Some(4),
                cache_read_input_tokens: Some(3),
                cache_creation_input_tokens: None,
            }
        );
        assert!(translator.usage().has_any());
    }

    #[test]
    fn tool_use_blocks_emit_start_deltas_and_end() {
        let mut translator = translator();
        let mut events = translator.push_chunk(
            sse(
                "content_block_start",
                r#"{"index":0,"content_block":{"type":"tool_use","id":"chatcmpl-tool-1","name":"bash","input":{}}}"#,
            )
            .as_bytes(),
        );
        events.extend(translator.push_chunk(
            sse(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"hi\"}"}}"#,
            )
            .as_bytes(),
        ));
        events
            .extend(translator.push_chunk(sse("content_block_stop", r#"{"index":0}"#).as_bytes()));

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolUseStart { name, .. } if name == "bash"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolInputDelta(json) if json == r#"{"command":"hi"}"#
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::ToolUseEnd))
        );
    }

    #[test]
    fn thinking_deltas_and_signatures_translate() {
        let mut translator = translator();
        let mut events = translator.push_chunk(
            sse(
                "content_block_start",
                r#"{"index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            )
            .as_bytes(),
        );
        events.extend(
            translator.push_chunk(
                sse(
                    "content_block_delta",
                    r#"{"index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
                )
                .as_bytes(),
            ),
        );
        events.extend(
            translator.push_chunk(
                sse(
                    "content_block_delta",
                    r#"{"index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
                )
                .as_bytes(),
            ),
        );
        events
            .extend(translator.push_chunk(sse("content_block_stop", r#"{"index":0}"#).as_bytes()));

        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::ThinkingStart))
        );
        assert_eq!(thinking_deltas(&events), vec!["hmm".to_string()]);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ThinkingSignatureDelta(signature) if signature == "sig"
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::ThinkingEnd))
        );
    }

    #[test]
    fn split_chunks_buffer_until_event_completes() {
        let mut translator = translator();
        let payload = sse(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
        );
        let (head, tail) = payload.split_at(24);
        assert!(translator.push_chunk(head.as_bytes()).is_empty());
        let events = translator.push_chunk(tail.as_bytes());
        assert_eq!(text_deltas(&events), vec!["partial".to_string()]);
    }

    #[test]
    fn served_model_substitution_warns_once() {
        let mut translator = translator();
        let events = translator.push_chunk(
            sse(
                "message_start",
                r#"{"message":{"model":"some-other-model","usage":{"input_tokens":1}}}"#,
            )
            .as_bytes(),
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, StreamEvent::StatusDetail { .. }))
        );
        let second = translator.push_chunk(
            sse(
                "message_start",
                r#"{"message":{"model":"some-other-model","usage":{"input_tokens":1}}}"#,
            )
            .as_bytes(),
        );
        assert!(
            !second
                .iter()
                .any(|event| matches!(event, StreamEvent::StatusDetail { .. }))
        );
    }

    #[test]
    fn matching_served_model_does_not_warn() {
        let mut translator = translator();
        let events = translator.push_chunk(
            sse(
                "message_start",
                r#"{"message":{"model":"union-alpha","usage":{"input_tokens":1}}}"#,
            )
            .as_bytes(),
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, StreamEvent::StatusDetail { .. }))
        );
    }

    /// A `content_block_start` carrying an unrecognized block type must still
    /// deserialize. Before the `Unknown` catch-all the whole event failed to
    /// parse and was dropped, so an unknown *tool* block produced a turn that
    /// reported `stop_reason: tool_use` with no tool call for the agent to run.
    #[test]
    fn unknown_content_block_start_is_ignored_without_dropping_the_event() {
        for block_type in [
            "server_tool_use",
            "web_search_tool_result",
            "some_future_block",
        ] {
            let mut translator = translator();
            let events = translator.push_chunk(
                sse(
                    "content_block_start",
                    &format!(
                        r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"{block_type}","id":"srvtoolu_1","name":"web_search"}}}}"#
                    ),
                )
                .as_bytes(),
            );
            assert!(
                events.is_empty(),
                "{block_type}: unknown block must not synthesize stream events"
            );
        }
    }

    #[test]
    fn pings_surface_as_streaming_phase_and_errors_pass_through() {
        let mut translator = translator();
        let ping = translator.push_chunk(sse("ping", r#"{"type":"ping"}"#).as_bytes());
        assert!(ping.iter().any(|event| matches!(
            event,
            StreamEvent::ConnectionPhase {
                phase: ConnectionPhase::Streaming
            }
        )));

        let error = translator.push_chunk(
            sse(
                "error",
                r#"{"type":"error","error":{"type":"api_error","message":"Endpoint is unavailable."}}"#,
            )
            .as_bytes(),
        );
        assert!(error.iter().any(|event| matches!(
            event,
            StreamEvent::Error { message, .. } if message.contains("Endpoint is unavailable")
        )));
    }
}
