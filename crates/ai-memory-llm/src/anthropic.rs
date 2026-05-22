//! Anthropic Messages API client.

use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{LlmError, LlmResult};
use crate::provider::LlmProvider;
use crate::types::{ChatRequest, ChatResponse, Role, Usage};

/// Default Anthropic API base.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Pinned Anthropic API version header.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages-API-backed provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: SecretString,
    base_url: String,
    model: String,
}

impl AnthropicProvider {
    /// Construct a provider given an API key and model id.
    ///
    /// # Errors
    /// Returns a `reqwest::Error` if the underlying HTTP client cannot
    /// be built.
    pub fn new(api_key: SecretString, model: impl Into<String>) -> LlmResult<Self> {
        // 300s matches the OpenAI/openai-compat client — same reason:
        // first request after a model swap on a local inference server
        // (Ollama, llama-swap, vLLM) can take 30-90s of cold-load.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            client,
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
        })
    }

    /// Override the API base URL (mostly for tests against wiremock).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMsg<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Serialize)]
struct AnthropicMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolChoice {
    Tool { name: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    ToolUse {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: ChatRequest) -> LlmResult<ChatResponse> {
        let messages: Vec<AnthropicMsg<'_>> = request
            .messages
            .iter()
            .map(|m| AnthropicMsg {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: request.system.as_deref(),
            messages,
            temperature: request.temperature,
            tools: None,
            tool_choice: None,
        };
        let response: AnthropicResponse = self.post(&body).await?;
        let text = response
            .content
            .iter()
            .filter_map(|c| match c {
                AnthropicContent::Text { text } => Some(text.as_str()),
                AnthropicContent::ToolUse { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ChatResponse {
            text,
            usage: response.usage.map(|u| Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
            }),
            model: response.model,
        })
    }

    async fn complete_structured_raw(
        &self,
        request: ChatRequest,
        schema: serde_json::Value,
    ) -> LlmResult<serde_json::Value> {
        let tool = AnthropicTool {
            name: "result".into(),
            description: "Emit the structured result.".into(),
            input_schema: schema,
        };
        let messages: Vec<AnthropicMsg<'_>> = request
            .messages
            .iter()
            .map(|m| AnthropicMsg {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: request.system.as_deref(),
            messages,
            temperature: request.temperature,
            tools: Some(vec![tool]),
            tool_choice: Some(AnthropicToolChoice::Tool {
                name: "result".into(),
            }),
        };
        let response: AnthropicResponse = self.post(&body).await?;
        for c in response.content {
            if let AnthropicContent::ToolUse { input, .. } = c {
                return Ok(input);
            }
        }
        Err(LlmError::UnexpectedShape(
            "anthropic response had no tool_use block".into(),
        ))
    }
}

impl AnthropicProvider {
    async fn post<B: Serialize, R: DeserializeOwned>(&self, body: &B) -> LlmResult<R> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        debug!(url, "POST anthropic");
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Provider {
                status: status.as_u16(),
                body: truncate(&body, 1024),
            });
        }
        resp.json::<R>().await.map_err(LlmError::from)
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}
