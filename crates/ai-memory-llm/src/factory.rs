//! Provider factory.
//!
//! Maps the user-visible `ProviderChoice` + env config into a
//! concrete `Arc<dyn LlmProvider>`.

use std::sync::Arc;

use secrecy::SecretString;

use crate::AnthropicProvider;
use crate::OpenAiCompatProvider;
use crate::OpenAiProvider;
use crate::embedding::{Embedder, OpenAiEmbedder, VoyageEmbedder};
use crate::error::{LlmError, LlmResult};
use crate::provider::LlmProvider;

/// Three providers ship in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderChoice {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Chat Completions.
    OpenAi,
    /// OpenAI-compatible (Ollama / vLLM / LM Studio).
    OpenAiCompat,
}

/// All settings needed to construct one of the three providers.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider selection.
    pub provider: ProviderChoice,
    /// Model id (`claude-opus-4-7`, `gpt-4o-mini`, `llama3.1:8b`, …).
    pub model: String,
    /// API key. Required for Anthropic + OpenAI; optional for compat.
    pub api_key: Option<SecretString>,
    /// Base URL override (required for OpenAI-compat).
    pub base_url: Option<String>,
}

/// Build a [`ProviderConfig`] from the environment.
///
/// Reads `AI_MEMORY_LLM_PROVIDER`, `AI_MEMORY_LLM_MODEL`,
/// `AI_MEMORY_LLM_BASE_URL`, and the appropriate API key
/// (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `LLM_API_KEY`).
/// Returns `Ok(None)` when `AI_MEMORY_LLM_PROVIDER` is unset — that
/// is the canonical "no LLM features" path.
///
/// # Errors
/// Returns [`LlmError::NotConfigured`] when the provider env var is
/// set to an unknown value or when the model env var is missing.
pub fn provider_from_env() -> LlmResult<Option<ProviderConfig>> {
    let provider = match std::env::var("AI_MEMORY_LLM_PROVIDER") {
        Ok(s) => match s.as_str() {
            "anthropic" => ProviderChoice::Anthropic,
            "openai" => ProviderChoice::OpenAi,
            "openai-compat" | "openai_compat" => ProviderChoice::OpenAiCompat,
            other => {
                return Err(LlmError::NotConfigured(format!(
                    "AI_MEMORY_LLM_PROVIDER={other} is not one of anthropic|openai|openai-compat"
                )));
            }
        },
        Err(_) => return Ok(None),
    };
    let model = std::env::var("AI_MEMORY_LLM_MODEL")
        .map_err(|_| LlmError::NotConfigured("AI_MEMORY_LLM_MODEL".into()))?;
    let base_url = std::env::var("AI_MEMORY_LLM_BASE_URL").ok();
    let api_key = match provider {
        ProviderChoice::Anthropic => std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .map(secrecy::SecretString::from),
        ProviderChoice::OpenAi => std::env::var("OPENAI_API_KEY")
            .ok()
            .map(secrecy::SecretString::from),
        ProviderChoice::OpenAiCompat => std::env::var("LLM_API_KEY")
            .ok()
            .map(secrecy::SecretString::from),
    };
    Ok(Some(ProviderConfig {
        provider,
        model,
        api_key,
        base_url,
    }))
}

/// Three embedders ship in v0.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderChoice {
    /// OpenAI Embeddings API.
    OpenAi,
    /// Voyage Embeddings API.
    Voyage,
}

impl EmbedderChoice {
    /// Wire-format provider name; matches what the `Embedder::provider`
    /// implementations return so the refuse-on-mismatch query lines up.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Voyage => "voyage",
        }
    }
}

/// Settings to build an embedder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Provider selection.
    pub provider: EmbedderChoice,
    /// Model id (e.g. `text-embedding-3-small`).
    pub model: String,
    /// Vector dimensionality. Refused on mismatch with the stored
    /// pages' dim.
    pub dim: u32,
    /// API key.
    pub api_key: SecretString,
    /// Optional base URL override.
    pub base_url: Option<String>,
}

/// Construct an `Arc<dyn Embedder>` from the config.
///
/// # Errors
/// Propagates HTTP-client construction errors.
pub fn build_embedder(config: EmbedderConfig) -> LlmResult<Arc<dyn Embedder>> {
    let arc: Arc<dyn Embedder> = match config.provider {
        EmbedderChoice::OpenAi => {
            let mut e = OpenAiEmbedder::new(config.api_key, config.model, config.dim)?;
            if let Some(url) = config.base_url {
                e = e.with_base_url(url);
            }
            Arc::new(e)
        }
        EmbedderChoice::Voyage => {
            let mut e = VoyageEmbedder::new(config.api_key, config.model, config.dim)?;
            if let Some(url) = config.base_url {
                e = e.with_base_url(url);
            }
            Arc::new(e)
        }
    };
    Ok(arc)
}

/// Read an `EmbedderConfig` from the environment.
///
/// Honours `AI_MEMORY_EMBEDDING_PROVIDER` (`openai` / `voyage`),
/// `AI_MEMORY_EMBEDDING_MODEL`, `AI_MEMORY_EMBEDDING_DIM`,
/// `AI_MEMORY_EMBEDDING_BASE_URL` (optional), and the appropriate
/// API key env (`OPENAI_API_KEY` / `VOYAGE_API_KEY`).
///
/// Returns `Ok(None)` when `AI_MEMORY_EMBEDDING_PROVIDER` is unset —
/// the canonical "no embeddings, FTS5 only" path.
///
/// # Errors
/// Returns [`LlmError::NotConfigured`] when the provider value is
/// unrecognised or when required env vars are missing.
pub fn embedder_from_env() -> LlmResult<Option<EmbedderConfig>> {
    let provider = match std::env::var("AI_MEMORY_EMBEDDING_PROVIDER") {
        Ok(s) => match s.as_str() {
            "openai" => EmbedderChoice::OpenAi,
            "voyage" => EmbedderChoice::Voyage,
            other => {
                return Err(LlmError::NotConfigured(format!(
                    "AI_MEMORY_EMBEDDING_PROVIDER={other} not one of openai|voyage"
                )));
            }
        },
        Err(_) => return Ok(None),
    };
    let model = std::env::var("AI_MEMORY_EMBEDDING_MODEL")
        .map_err(|_| LlmError::NotConfigured("AI_MEMORY_EMBEDDING_MODEL".into()))?;
    let dim: u32 = std::env::var("AI_MEMORY_EMBEDDING_DIM")
        .map_err(|_| LlmError::NotConfigured("AI_MEMORY_EMBEDDING_DIM".into()))?
        .parse()
        .map_err(|e| LlmError::NotConfigured(format!("AI_MEMORY_EMBEDDING_DIM: {e}")))?;
    let base_url = std::env::var("AI_MEMORY_EMBEDDING_BASE_URL").ok();
    let api_key = match provider {
        EmbedderChoice::OpenAi => std::env::var("OPENAI_API_KEY")
            .map_err(|_| LlmError::NotConfigured("OPENAI_API_KEY".into()))?,
        EmbedderChoice::Voyage => std::env::var("VOYAGE_API_KEY")
            .map_err(|_| LlmError::NotConfigured("VOYAGE_API_KEY".into()))?,
    };
    Ok(Some(EmbedderConfig {
        provider,
        model,
        dim,
        api_key: SecretString::from(api_key),
        base_url,
    }))
}

/// Construct an `Arc<dyn LlmProvider>` matching the config.
///
/// # Errors
/// Returns [`LlmError::NotConfigured`] if a required env value (API
/// key, base URL) is missing.
pub fn build_provider(config: ProviderConfig) -> LlmResult<Arc<dyn LlmProvider>> {
    match config.provider {
        ProviderChoice::Anthropic => {
            let key = config
                .api_key
                .ok_or_else(|| LlmError::NotConfigured("ANTHROPIC_API_KEY".into()))?;
            Ok(Arc::new(AnthropicProvider::new(key, config.model)?))
        }
        ProviderChoice::OpenAi => {
            let key = config
                .api_key
                .ok_or_else(|| LlmError::NotConfigured("OPENAI_API_KEY".into()))?;
            Ok(Arc::new(OpenAiProvider::new(key, config.model)?))
        }
        ProviderChoice::OpenAiCompat => {
            let base = config
                .base_url
                .ok_or_else(|| LlmError::NotConfigured("LLM_BASE_URL".into()))?;
            Ok(Arc::new(OpenAiCompatProvider::new(
                base,
                config.api_key,
                config.model,
            )?))
        }
    }
}
