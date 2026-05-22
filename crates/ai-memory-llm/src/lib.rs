//! LLM provider abstraction for ai-memory.
//!
//! Three providers ship in v1, each with a *native, typed*
//! `reqwest`-based client — never a generic gateway. The cognee
//! issue tracker showed that LiteLLM + Instructor silently drop
//! unknown kwargs, which makes the wrapper layer drift away from
//! the provider's wire protocol over time (#2840, #2608, #2782).
//! Our clients deserialise into named structs that `serde` rejects
//! on unknown fields, surfacing breakage immediately.
//!
//! Three structured-output strategies:
//!
//! * **Anthropic**: `tools[0]` is set to a single tool whose input
//!   schema we want filled, with `tool_choice = "tool"`. The
//!   model's `tool_use` content block is the structured payload.
//! * **OpenAI**: `response_format = { type: "json_schema", strict: true }`.
//! * **OpenAI-compat** (Ollama, vLLM, LM Studio): we ask for
//!   `response_format: { type: "json_object" }` when supported,
//!   otherwise parse the first balanced `{…}` from the text body.
//!   No tenacity-style 8-128s backoff (cognee #2840 lesson).

pub mod anthropic;
pub mod embedding;
pub mod error;
pub mod factory;
pub mod openai;
pub mod openai_compat;
pub mod provider;
pub mod types;

pub use anthropic::AnthropicProvider;
pub use embedding::{Embedder, OpenAiEmbedder, SyntheticEmbedder, VoyageEmbedder, cosine};
pub use error::{LlmError, LlmResult};
pub use factory::{
    EmbedderChoice, EmbedderConfig, ProviderChoice, ProviderConfig, build_embedder, build_provider,
    embedder_from_env, provider_from_env,
};
pub use openai::OpenAiProvider;
pub use openai_compat::OpenAiCompatProvider;
pub use provider::{LlmProvider, complete_structured};
pub use types::{ChatMessage, ChatRequest, ChatResponse, Role, Usage};
