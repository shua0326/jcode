//! Wire-API and effort-ladder discovery from published model metadata.
//!
//! Aggregator gateways do not serve every model through one request shape.
//! OpenCode Go/Zen, for example, serves `deepseek-*` through
//! `POST /chat/completions`, `muse-spark-*` and `gpt-5.6-luna` through
//! `POST /responses`, and `union-alpha`, `minimax-m3` or `qwen3.8-flash`
//! through the Anthropic Messages API (`POST /messages`). Sending the wrong
//! shape is not a useful error: the gateway answers a bare
//! `500 {"type":"error","error":{"message":"Internal server error"}}`.
//!
//! models.dev (which jcode already ingests for pricing) publishes the mapping
//! per model as the AI-SDK package the gateway uses, plus an optional reasoning
//! effort ladder. These helpers turn that metadata into jcode's request shape
//! and effort ladder without hardcoding model families.

/// Request shape a gateway serves a model with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireApi {
    /// OpenAI-compatible `POST /chat/completions`.
    ChatCompletions,
    /// OpenAI `POST /responses`.
    Responses,
    /// Anthropic `POST /messages`.
    Messages,
}

impl WireApi {
    /// Stable identifier persisted in caches and used in logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "openai-chat",
            Self::Responses => "openai-responses",
            Self::Messages => "anthropic-messages",
        }
    }

    /// Parse a persisted identifier. Unknown values are ignored so a catalog
    /// written by a newer jcode cannot break an older one.
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai-chat" | "chat" | "chat-completions" => Some(Self::ChatCompletions),
            "openai-responses" | "responses" => Some(Self::Responses),
            "anthropic-messages" | "messages" | "anthropic" => Some(Self::Messages),
            _ => None,
        }
    }
}

/// Map a models.dev AI-SDK package (`provider.npm`) to the request shape the
/// gateway serves that model with.
///
/// Returns `None` for packages whose shape jcode cannot derive from the
/// package name alone (e.g. `@ai-sdk/google` served through a
/// chat/completions-compatible facade), so callers keep their own fallback.
pub fn wire_api_for_npm(npm: &str) -> Option<WireApi> {
    let npm = npm.trim().to_ascii_lowercase();
    if npm.is_empty() {
        return None;
    }
    match npm.as_str() {
        "@ai-sdk/anthropic" => Some(WireApi::Messages),
        "@ai-sdk/openai" => Some(WireApi::Responses),
        "@ai-sdk/openai-compatible" => Some(WireApi::ChatCompletions),
        _ => None,
    }
}

/// Normalize a published effort value to jcode's vocabulary.
///
/// models.dev publishes per-provider ladders (`none|minimal|low|medium|high|
/// xhigh|max`). Values jcode does not recognize are dropped so the picker never
/// offers an effort the request builder cannot send.
pub fn normalize_published_effort(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Some("none"),
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" => Some("max"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npm_packages_map_to_gateway_request_shapes() {
        assert_eq!(
            wire_api_for_npm("@ai-sdk/anthropic"),
            Some(WireApi::Messages)
        );
        assert_eq!(wire_api_for_npm("@ai-sdk/openai"), Some(WireApi::Responses));
        assert_eq!(
            wire_api_for_npm("@ai-sdk/openai-compatible"),
            Some(WireApi::ChatCompletions)
        );
        assert_eq!(
            wire_api_for_npm(" @AI-SDK/Anthropic "),
            Some(WireApi::Messages)
        );
        assert_eq!(wire_api_for_npm("@ai-sdk/google"), None);
        assert_eq!(wire_api_for_npm(""), None);
    }

    #[test]
    fn wire_api_identifiers_round_trip() {
        for api in [
            WireApi::ChatCompletions,
            WireApi::Responses,
            WireApi::Messages,
        ] {
            assert_eq!(WireApi::from_str(api.as_str()), Some(api));
        }
        assert_eq!(WireApi::from_str("something-new"), None);
    }

    #[test]
    fn published_efforts_normalize_and_reject_unknown() {
        assert_eq!(normalize_published_effort("XHigh"), Some("xhigh"));
        assert_eq!(normalize_published_effort(" max "), Some("max"));
        assert_eq!(normalize_published_effort("turbo"), None);
        assert_eq!(normalize_published_effort(""), None);
    }
}
