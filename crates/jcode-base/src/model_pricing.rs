//! Live model pricing catalog backed by <https://models.dev>.
//!
//! jcode's static pricing tables (`jcode_provider_core::pricing`) only cover
//! first-party Anthropic/OpenAI models and go stale whenever a provider ships
//! new models or changes prices. models.dev publishes a free, no-auth JSON
//! catalog (`https://models.dev/api.json`) with per-model `input`/`output`/
//! `cache_read`/`cache_write` USD prices per million tokens across 140+
//! providers, including every OpenAI-compatible profile jcode ships.
//!
//! This module mirrors the OpenRouter catalog pattern:
//!   - a 24h disk cache under `~/.jcode/cache/models_dev_pricing.json`,
//!   - synchronous lookups that never block on the network,
//!   - a background refresh scheduled on cache miss/staleness.
//!
//! Lookup order for callers is curated static table first (exact, reviewed),
//! then this catalog, then provider-specific sources (OpenRouter endpoints),
//! and only then a generic fallback.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const API_URL: &str = "https://models.dev/api.json";
const CACHE_FILE: &str = "models_dev_pricing.json";
const CACHE_TTL_SECS: u64 = 24 * 60 * 60;
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
/// On-disk cache layout version. Bumped whenever the parsed fields change so an
/// older cache is refreshed instead of silently serving missing capabilities.
const CACHE_SCHEMA_VERSION: u32 = 2;

/// Per-model USD prices per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelCost {
    pub input_usd_per_mtok: f64,
    pub output_usd_per_mtok: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_usd_per_mtok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_usd_per_mtok: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PricingCache {
    #[serde(default)]
    schema_version: u32,
    cached_at_unix_secs: u64,
    /// provider id -> model id -> cost. Provider ids are models.dev ids
    /// (e.g. `anthropic`, `openai`, `deepseek`, `moonshotai`).
    providers: HashMap<String, HashMap<String, ModelCost>>,
    /// provider id -> model id -> request-shape/effort metadata. Kept beside
    /// pricing because both come from the same catalog fetch: gateways such as
    /// OpenCode Go serve different models through different wire APIs, and
    /// jcode must know which one before spending a model turn (issue: the free
    /// `union-alpha` model 500s on `chat/completions` and only answers on the
    /// Anthropic Messages shape).
    #[serde(default)]
    capabilities: HashMap<String, HashMap<String, ModelCapability>>,
}

/// Per-model request-shape and reasoning metadata published by models.dev.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCapability {
    /// Wire API the gateway serves this model with, derived from the AI-SDK
    /// package (`provider.npm`). `None` when the catalog does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire_api: Option<String>,
    /// Published reasoning effort ladder, normalized to jcode's vocabulary.
    /// Empty when the catalog lists no explicit effort enum for the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
    /// Whether the catalog marks the model as reasoning-capable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
}

/// In-memory pricing cache keyed by the on-disk cache path it was loaded
/// from, so changing `JCODE_HOME` (tests, multi-home setups) never serves
/// pricing that belongs to a different home directory.
///
/// Held behind an `Arc` so per-route pricing lookups share one parsed catalog
/// instead of deep-cloning the multi-thousand-model map on every call (that
/// clone dominated server CPU during client connect bursts).
static MEMORY_CACHE: Mutex<Option<(PathBuf, Arc<PricingCache>)>> = Mutex::new(None);
static REFRESH_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

fn cache_path() -> PathBuf {
    crate::storage::jcode_dir()
        .unwrap_or_else(|_| PathBuf::from(".").join(".jcode"))
        .join("cache")
        .join(CACHE_FILE)
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_cache() -> Option<Arc<PricingCache>> {
    let path = cache_path();
    {
        let memory = MEMORY_CACHE.lock().ok()?;
        if let Some((cached_path, cache)) = memory.as_ref()
            && cached_path == &path
        {
            return Some(Arc::clone(cache));
        }
    }
    let cache: PricingCache = crate::storage::read_json(&path).ok()?;
    let cache = Arc::new(cache);
    if let Ok(mut memory) = MEMORY_CACHE.lock() {
        *memory = Some((path, Arc::clone(&cache)));
    }
    Some(cache)
}

fn save_cache(cache: &PricingCache) {
    let path = cache_path();
    if let Ok(mut memory) = MEMORY_CACHE.lock() {
        *memory = Some((path.clone(), Arc::new(cache.clone())));
    }
    let _ = crate::storage::write_json(&path, cache);
}

/// Translate a jcode provider key (runtime key, activity source key, or
/// compatible-profile id) to the models.dev provider id.
pub fn models_dev_provider_id(jcode_provider: &str) -> Option<&'static str> {
    let key = jcode_provider
        .trim()
        .strip_prefix("openai-compatible:")
        .unwrap_or_else(|| jcode_provider.trim())
        .trim();
    // Callers pass runtime keys (`opencode-go`), route api methods
    // (`openai-compatible:opencode-go`), and display names (`OpenCode Go`).
    // Normalizing here keeps every caller from re-implementing the mapping.
    let key = key.to_ascii_lowercase().replace([' ', '_'], "-");
    Some(match key.as_str() {
        "anthropic" | "claude" | "claude:api-key" | "anthropic-api" => "anthropic",
        "openai" | "openai:api-key" | "openai-api" => "openai",
        "openrouter" => "openrouter",
        "opencode" => "opencode",
        "opencode-go" => "opencode-go",
        "deepseek" => "deepseek",
        "moonshotai" => "moonshotai",
        "kimi" => "kimi-for-coding",
        "zai" => "zai",
        "cerebras" => "cerebras",
        "groq" => "groq",
        "mistral" => "mistral",
        "xai" => "xai",
        "minimax" => "minimax",
        "togetherai" => "togetherai",
        "fireworks" => "fireworks-ai",
        "deepinfra" => "deepinfra",
        "perplexity" => "perplexity",
        "nebius" => "nebius",
        "scaleway" => "scaleway",
        "stackit" => "stackit",
        "huggingface" => "huggingface",
        "baseten" => "baseten",
        "chutes" => "chutes",
        "nvidia-nim" => "nvidia",
        "302ai" => "302ai",
        "cortecs" => "cortecs",
        "alibaba-coding-plan" => "alibaba",
        "bedrock" => "amazon-bedrock",
        "azure-openai" | "azure" => "azure",
        "gemini" | "gemini-api" => "google",
        _ => return None,
    })
}

/// Strip jcode-local suffixes/prefixes a model id may carry before catalog
/// lookup (`[1m]` long-context alias, `provider/` prefixes for OpenRouter ids).
fn normalize_model_id(model: &str) -> &str {
    let model = jcode_provider_core::model_id::strip_long_context_suffix(model).trim();
    model
        .rsplit_once('@')
        .map_or(model, |(bare, _)| bare.trim())
}

/// Look up live pricing for `model` under a jcode provider key. Returns `None`
/// when the catalog has no entry; never blocks on the network. Schedules a
/// background refresh when the disk cache is missing or stale.
pub fn lookup(jcode_provider: &str, model: &str) -> Option<ModelCost> {
    let provider_id = models_dev_provider_id(jcode_provider)?;
    let cache = ensure_cache_fresh()?;
    let models = cache.providers.get(provider_id)?;
    let model = normalize_model_id(model);
    if let Some(cost) = models.get(model) {
        return Some(*cost);
    }
    // OpenRouter-style ids (`anthropic/claude-...`) may reach here with the
    // provider prefix still attached; retry on the bare model name.
    if let Some((_, bare)) = model.rsplit_once('/') {
        return models.get(bare).copied();
    }
    None
}

/// Look up published request-shape/reasoning metadata for `model` under a jcode
/// provider key. Returns `None` when the catalog has no entry for the model.
///
/// Callers use this to pick the wire API a gateway actually serves a model with
/// (e.g. OpenCode Go answers `union-alpha` only on the Anthropic Messages
/// shape) and to expose the effort ladder the provider documents.
pub fn lookup_capability(jcode_provider: &str, model: &str) -> Option<ModelCapability> {
    let provider_id = models_dev_provider_id(jcode_provider)?;
    let cache = ensure_cache_fresh()?;
    let models = cache.capabilities.get(provider_id)?;
    let model = normalize_model_id(model);
    if let Some(capability) = models.get(model) {
        return Some(capability.clone());
    }
    // OpenRouter-style ids (`anthropic/claude-...`) may reach here with the
    // provider prefix still attached; retry on the bare model name.
    if let Some((_, bare)) = model.rsplit_once('/') {
        return models.get(bare).cloned();
    }
    None
}

/// jcode's default thinking ladder for a Messages-wire reasoning model whose
/// gateway publishes no explicit effort enum (e.g. OpenCode Go's `union-alpha`).
/// The Messages API takes a token budget, so these names map to budgets in the
/// request builder.
const MESSAGES_DEFAULT_EFFORTS: [&str; 3] = ["low", "medium", "high"];

/// Effort ladder to offer for a provider+model, in provider order.
///
/// 1. the ladder the catalog publishes for the model, when it has one;
/// 2. for Messages-wire reasoning models, jcode's default budget ladder;
/// 3. otherwise empty, so callers keep their own name-based heuristics.
pub fn discovered_efforts(provider: &str, model: &str) -> Vec<&'static str> {
    let Some(capability) = lookup_capability(provider, model) else {
        return Vec::new();
    };
    if !capability.reasoning_efforts.is_empty() {
        return capability
            .reasoning_efforts
            .iter()
            .filter_map(|effort| jcode_provider_core::normalize_published_effort(effort))
            .collect();
    }
    let messages_wire = capability.wire_api.as_deref() == Some("anthropic-messages");
    if messages_wire && capability.reasoning != Some(false) {
        return MESSAGES_DEFAULT_EFFORTS.to_vec();
    }
    Vec::new()
}

/// Return the freshest cache available, scheduling a refresh if needed.
fn ensure_cache_fresh() -> Option<Arc<PricingCache>> {
    let cache = load_cache();
    let stale = cache
        .as_ref()
        .map(|c| {
            c.schema_version < CACHE_SCHEMA_VERSION
                || now_unix_secs().saturating_sub(c.cached_at_unix_secs) >= CACHE_TTL_SECS
        })
        .unwrap_or(true);
    if stale {
        schedule_refresh();
    }
    cache
}

/// Spawn one background refresh at a time. Safe to call from sync contexts;
/// uses a thread + ad-hoc runtime when no Tokio runtime is active.
pub fn schedule_refresh() {
    // Keep tests hermetic: never hit the network from test builds (the
    // `test-support` feature also covers downstream crates' test targets via
    // feature unification), and let users opt out entirely.
    // JCODE_FORCE_PRICING_REFRESH=1 re-enables the fetch for manual e2e checks
    // (e.g. `cargo run --example pricing_e2e_check`, which builds with
    // test-support unified in).
    let forced = std::env::var_os("JCODE_FORCE_PRICING_REFRESH").is_some();
    if !forced
        && (cfg!(any(test, feature = "test-support"))
            || std::env::var_os("JCODE_DISABLE_PRICING_REFRESH").is_some())
    {
        return;
    }
    if REFRESH_IN_FLIGHT.swap(true, Ordering::SeqCst) {
        return;
    }
    let work = || async {
        if let Err(e) = refresh_now().await {
            crate::logging::warn(&format!("models.dev pricing refresh failed: {e:#}"));
        }
        REFRESH_IN_FLIGHT.store(false, Ordering::SeqCst);
    };
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(work());
    } else {
        std::thread::spawn(move || {
            if let Ok(runtime) = tokio::runtime::Runtime::new() {
                runtime.block_on(work());
            } else {
                REFRESH_IN_FLIGHT.store(false, Ordering::SeqCst);
            }
        });
    }
}

/// Fetch the catalog and persist the parsed cache.
async fn refresh_now() -> anyhow::Result<()> {
    let client = crate::provider::shared_http_client();
    let response = client
        .get(API_URL)
        .header("Accept", "application/json")
        .timeout(HTTP_TIMEOUT)
        .send()
        .await?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {}", response.status());
    }
    let body = response.text().await?;
    let cache = parse_api_response(&body)?;
    save_cache(&cache);
    crate::logging::info(&format!(
        "models.dev pricing refreshed: {} providers, {} priced models",
        cache.providers.len(),
        cache.providers.values().map(HashMap::len).sum::<usize>()
    ));
    Ok(())
}

fn parse_api_response(body: &str) -> anyhow::Result<PricingCache> {
    let json: serde_json::Value = serde_json::from_str(body)?;
    let top = json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected top-level provider object"))?;

    let mut providers: HashMap<String, HashMap<String, ModelCost>> = HashMap::new();
    let mut capabilities: HashMap<String, HashMap<String, ModelCapability>> = HashMap::new();
    for (provider_id, provider) in top {
        let Some(models) = provider.get("models").and_then(|m| m.as_object()) else {
            continue;
        };
        // The gateway-level AI-SDK package applies to every model unless a
        // model overrides it (models.dev sets `provider.npm` per model for the
        // models a gateway serves through a different API).
        let provider_npm = provider
            .get("npm")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let mut parsed_models = HashMap::new();
        let mut parsed_capabilities = HashMap::new();
        for (model_id, model) in models {
            let capability = parse_model_capability(model, provider_npm);
            if capability != ModelCapability::default() {
                parsed_capabilities.insert(model_id.clone(), capability);
            }
            let Some(cost) = model.get("cost") else {
                continue;
            };
            let (Some(input), Some(output)) = (
                cost.get("input").and_then(|v| v.as_f64()),
                cost.get("output").and_then(|v| v.as_f64()),
            ) else {
                continue;
            };
            parsed_models.insert(
                model_id.clone(),
                ModelCost {
                    input_usd_per_mtok: input,
                    output_usd_per_mtok: output,
                    cache_read_usd_per_mtok: cost.get("cache_read").and_then(|v| v.as_f64()),
                    cache_write_usd_per_mtok: cost.get("cache_write").and_then(|v| v.as_f64()),
                },
            );
        }
        if !parsed_models.is_empty() {
            providers.insert(provider_id.clone(), parsed_models);
        }
        if !parsed_capabilities.is_empty() {
            capabilities.insert(provider_id.clone(), parsed_capabilities);
        }
    }

    if providers.is_empty() {
        anyhow::bail!("no priced models in models.dev response");
    }
    Ok(PricingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        cached_at_unix_secs: now_unix_secs(),
        providers,
        capabilities,
    })
}

/// Read the request-shape/reasoning metadata models.dev publishes for one model.
fn parse_model_capability(model: &serde_json::Value, provider_npm: &str) -> ModelCapability {
    let npm = model
        .get("provider")
        .and_then(|provider| provider.get("npm"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|npm| !npm.is_empty())
        .unwrap_or(provider_npm);

    let reasoning_efforts = model
        .get("reasoning_options")
        .and_then(|options| options.as_array())
        .map(|options| {
            let mut efforts = Vec::new();
            for option in options {
                if option.get("type").and_then(|v| v.as_str()) != Some("effort") {
                    continue;
                }
                let Some(values) = option.get("values").and_then(|v| v.as_array()) else {
                    continue;
                };
                for value in values {
                    let Some(normalized) = value
                        .as_str()
                        .and_then(jcode_provider_core::normalize_published_effort)
                    else {
                        continue;
                    };
                    if !efforts.iter().any(|existing| existing == normalized) {
                        efforts.push(normalized.to_string());
                    }
                }
            }
            efforts
        })
        .unwrap_or_default();

    ModelCapability {
        wire_api: jcode_provider_core::wire_api_for_npm(npm).map(|api| api.as_str().to_string()),
        reasoning_efforts,
        reasoning: model.get("reasoning").and_then(|v| v.as_bool()),
    }
}

#[cfg(test)]
pub(crate) fn save_test_cache(entries: &[(&str, &str, ModelCost)]) {
    let mut providers: HashMap<String, HashMap<String, ModelCost>> = HashMap::new();
    for (provider, model, cost) in entries {
        providers
            .entry((*provider).to_string())
            .or_default()
            .insert((*model).to_string(), *cost);
    }
    save_cache(&PricingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        cached_at_unix_secs: now_unix_secs(),
        providers,
        capabilities: HashMap::new(),
    });
}

/// Test helper: persist capability metadata the way a real catalog fetch would.
#[cfg(test)]
pub(crate) fn save_test_capability(provider: &str, model: &str, capability: ModelCapability) {
    let mut cache = load_cache()
        .map(|cache| (*cache).clone())
        .unwrap_or_default();
    cache.schema_version = CACHE_SCHEMA_VERSION;
    cache.cached_at_unix_secs = now_unix_secs();
    cache
        .capabilities
        .entry(provider.to_string())
        .or_default()
        .insert(model.to_string(), capability);
    save_cache(&cache);
}

#[cfg(test)]
pub(crate) fn clear_memory_cache_for_tests() {
    if let Ok(mut memory) = MEMORY_CACHE.lock() {
        *memory = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_models_dev_shape() {
        let body = r#"{
            "deepseek": {
                "id": "deepseek",
                "models": {
                    "deepseek-v4-flash": {
                        "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.0028}
                    },
                    "free-model": {"cost": {"input": 0, "output": 0}},
                    "no-cost-model": {}
                }
            },
            "anthropic": {
                "models": {
                    "claude-fable-5": {
                        "cost": {"input": 10, "output": 50, "cache_read": 1, "cache_write": 12.5}
                    }
                }
            }
        }"#;
        let cache = parse_api_response(body).expect("parsed");
        let deepseek = cache.providers.get("deepseek").expect("deepseek");
        assert_eq!(deepseek.len(), 2, "model without cost is skipped");
        let flash = deepseek.get("deepseek-v4-flash").expect("flash");
        assert!((flash.input_usd_per_mtok - 0.14).abs() < 1e-9);
        assert!((flash.output_usd_per_mtok - 0.28).abs() < 1e-9);
        assert_eq!(flash.cache_read_usd_per_mtok, Some(0.0028));
        assert_eq!(flash.cache_write_usd_per_mtok, None);

        let fable = cache
            .providers
            .get("anthropic")
            .and_then(|m| m.get("claude-fable-5"))
            .expect("fable");
        assert!((fable.input_usd_per_mtok - 10.0).abs() < 1e-9);
        assert_eq!(fable.cache_write_usd_per_mtok, Some(12.5));
    }

    #[test]
    fn rejects_empty_response() {
        assert!(parse_api_response("{}").is_err());
        assert!(parse_api_response("[]").is_err());
    }

    #[test]
    fn parses_gateway_wire_api_and_effort_metadata() {
        // Trimmed models.dev shape: the opencode-go gateway serves models
        // through different AI-SDK packages, which is what decides the wire API.
        let body = r#"{
            "opencode-go": {
                "npm": "@ai-sdk/openai-compatible",
                "models": {
                    "deepseek-v4.1-flash": {
                        "reasoning": true,
                        "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
                        "cost": {"input": 0.15, "output": 0.6}
                    },
                    "union-alpha": {
                        "reasoning": true,
                        "reasoning_options": [],
                        "provider": {"npm": "@ai-sdk/anthropic"},
                        "cost": {"input": 0, "output": 0}
                    },
                    "muse-spark-1.3-contributor": {
                        "provider": {"npm": "@ai-sdk/openai"},
                        "reasoning": true,
                        "reasoning_options": [
                            {"type": "effort", "values": ["minimal", "high", "xhigh"]},
                            {"type": "toggle"}
                        ],
                        "cost": {"input": 0.1, "output": 0.2}
                    },
                    "gpt-5.6-luna": {
                        "reasoning_options": [
                            {"type": "effort", "values": ["low", "turbo", "xhigh"]}
                        ],
                        "cost": {"input": 0.2, "output": 0.8}
                    }
                }
            }
        }"#;
        let cache = parse_api_response(body).expect("parsed");
        let capabilities = cache.capabilities.get("opencode-go").expect("capabilities");

        let union = capabilities.get("union-alpha").expect("union-alpha");
        assert_eq!(union.wire_api.as_deref(), Some("anthropic-messages"));
        assert!(union.reasoning_efforts.is_empty());
        assert_eq!(union.reasoning, Some(true));

        let muse = capabilities
            .get("muse-spark-1.3-contributor")
            .expect("muse");
        assert_eq!(muse.wire_api.as_deref(), Some("openai-responses"));
        assert_eq!(muse.reasoning_efforts, vec!["minimal", "high", "xhigh"]);

        let deepseek = capabilities.get("deepseek-v4.1-flash").expect("deepseek");
        assert_eq!(deepseek.wire_api.as_deref(), Some("openai-chat"));
        assert_eq!(deepseek.reasoning_efforts, vec!["low", "high", "max"]);

        // Unknown published values are dropped rather than offered in the picker.
        let luna = capabilities.get("gpt-5.6-luna").expect("luna");
        assert_eq!(luna.wire_api.as_deref(), Some("openai-chat"));
        assert_eq!(luna.reasoning_efforts, vec!["low", "xhigh"]);
    }

    #[test]
    fn capability_lookup_reads_the_saved_catalog() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", temp.path());
        clear_memory_cache_for_tests();

        save_test_capability(
            "opencode-go",
            "union-alpha",
            ModelCapability {
                wire_api: Some("anthropic-messages".to_string()),
                reasoning_efforts: Vec::new(),
                reasoning: Some(true),
            },
        );

        let capability =
            lookup_capability("openai-compatible:opencode-go", "union-alpha").expect("capability");
        assert_eq!(capability.wire_api.as_deref(), Some("anthropic-messages"));
        // Provider-prefixed ids fall back to the bare model name.
        assert!(lookup_capability("opencode-go", "opencode-go/union-alpha").is_some());
        assert!(lookup_capability("opencode-go", "not-a-model").is_none());

        clear_memory_cache_for_tests();
        if let Some(prev) = prev_home {
            crate::env::set_var("JCODE_HOME", prev);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }

    #[test]
    fn provider_key_mapping_covers_jcode_providers() {
        assert_eq!(models_dev_provider_id("claude:api-key"), Some("anthropic"));
        assert_eq!(models_dev_provider_id("openai:api-key"), Some("openai"));
        assert_eq!(
            models_dev_provider_id("openai-compatible:deepseek"),
            Some("deepseek")
        );
        assert_eq!(
            models_dev_provider_id("openai-compatible:nvidia-nim"),
            Some("nvidia")
        );
        assert_eq!(models_dev_provider_id("bedrock"), Some("amazon-bedrock"));
        assert_eq!(models_dev_provider_id("unknown-thing"), None);
    }

    #[test]
    fn lookup_normalizes_model_ids() {
        let _guard = crate::storage::lock_test_env();
        let temp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("JCODE_HOME");
        crate::env::set_var("JCODE_HOME", temp.path());
        clear_memory_cache_for_tests();

        save_test_cache(&[
            (
                "anthropic",
                "claude-opus-4-6",
                ModelCost {
                    input_usd_per_mtok: 5.0,
                    output_usd_per_mtok: 25.0,
                    cache_read_usd_per_mtok: Some(0.5),
                    cache_write_usd_per_mtok: Some(6.25),
                },
            ),
            (
                "openrouter",
                "kimi-k2",
                ModelCost {
                    input_usd_per_mtok: 0.5,
                    output_usd_per_mtok: 2.0,
                    cache_read_usd_per_mtok: None,
                    cache_write_usd_per_mtok: None,
                },
            ),
        ]);

        // [1m] suffix strips before lookup.
        let opus = lookup("claude:api-key", "claude-opus-4-6[1m]").expect("priced");
        assert!((opus.input_usd_per_mtok - 5.0).abs() < 1e-9);

        // provider/model ids fall back to the bare model name.
        let kimi = lookup("openrouter", "moonshotai/kimi-k2").expect("priced");
        assert!((kimi.output_usd_per_mtok - 2.0).abs() < 1e-9);
        let pinned = lookup("openrouter", "moonshotai/kimi-k2@Sail Research").expect("priced");
        assert!((pinned.output_usd_per_mtok - 2.0).abs() < 1e-9);

        assert!(lookup("claude:api-key", "claude-unknown").is_none());

        clear_memory_cache_for_tests();
        if let Some(prev) = prev_home {
            crate::env::set_var("JCODE_HOME", prev);
        } else {
            crate::env::remove_var("JCODE_HOME");
        }
    }
}
