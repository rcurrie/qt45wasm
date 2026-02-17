use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::runtime::WasmRuntime;

const MAX_RETRIES: usize = 3;
const PROMPTS_DIR: &str = "prompts";

/// Client for an OpenAI-compatible LLM API (Ollama, OpenAI, etc.)
pub struct LlmClient {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

impl LlmClient {
    /// Create a client pointing at an Ollama instance.
    /// Default: http://localhost:11434/v1/chat/completions
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    /// Generate a WAT module for the given function spec.
    /// Retries up to MAX_RETRIES times if compilation fails, feeding
    /// the compiler error back to the LLM.
    pub fn generate_wat(
        &self,
        name: &str,
        description: &str,
        signature: &str,
        runtime: &WasmRuntime,
    ) -> Result<String> {
        let system_prompt = load_prompt("system.txt")?
            .replace("{name}", name)
            .replace("{signature}", signature);

        let user_prompt = load_prompt("generate.txt")?
            .replace("{name}", name)
            .replace("{description}", description)
            .replace("{signature}", signature);

        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt,
            },
            Message {
                role: "user".to_string(),
                content: user_prompt,
            },
        ];

        for attempt in 1..=MAX_RETRIES {
            println!("  [llm] attempt {attempt}/{MAX_RETRIES}...");

            let response_text = self.chat(&messages)?;
            let wat = extract_wat(&response_text);

            match runtime.compile_wat(&wat) {
                Ok(_) => {
                    println!("  [llm] compilation successful");
                    return Ok(wat);
                }
                Err(e) if attempt < MAX_RETRIES => {
                    let error_msg = format!("{e:#}");
                    println!("  [llm] compilation failed: {error_msg}");

                    // Feed the error back for self-correction
                    messages.push(Message {
                        role: "assistant".to_string(),
                        content: response_text,
                    });
                    let retry_prompt = load_prompt("retry.txt")?
                        .replace("{error}", &error_msg);
                    messages.push(Message {
                        role: "user".to_string(),
                        content: retry_prompt,
                    });
                }
                Err(e) => {
                    return Err(e).context(format!(
                        "LLM failed to generate valid WAT for '{name}' after {MAX_RETRIES} attempts"
                    ));
                }
            }
        }

        unreachable!()
    }

    fn chat(&self, messages: &[Message]) -> Result<String> {
        let req = ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            temperature: 0.2,
        };

        let resp = self
            .client
            .post(&self.base_url)
            .json(&req)
            .send()
            .context("Failed to reach LLM API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            anyhow::bail!("LLM API error (HTTP {status}): {body}");
        }

        let chat_resp: ChatResponse = resp.json().context("Failed to parse LLM response")?;
        let content = chat_resp
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(content)
    }
}

/// Load a prompt template from the prompts directory.
fn load_prompt(filename: &str) -> Result<String> {
    let path = Path::new(PROMPTS_DIR).join(filename);
    std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to load prompt: {}", path.display()))
}

/// Extract WAT code from an LLM response.
/// Handles cases where the LLM wraps code in markdown fences.
fn extract_wat(response: &str) -> String {
    let text = response.trim();

    // Strip markdown fences if present
    if let Some(rest) = text.strip_prefix("```wat") {
        if let Some(code) = rest.strip_suffix("```") {
            return code.trim().to_string();
        }
    }
    if let Some(rest) = text.strip_prefix("```wasm") {
        if let Some(code) = rest.strip_suffix("```") {
            return code.trim().to_string();
        }
    }
    if let Some(rest) = text.strip_prefix("```") {
        if let Some(code) = rest.strip_suffix("```") {
            return code.trim().to_string();
        }
    }

    // Try to extract just the (module ...) portion
    if let Some(start) = text.find("(module") {
        // Find the matching closing paren
        let bytes = text.as_bytes();
        let mut depth = 0;
        let mut end = start;
        for (i, &b) in bytes[start..].iter().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end > start {
            return text[start..end].to_string();
        }
    }

    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_wat_plain() {
        let input = "(module (func (export \"add\") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))";
        assert_eq!(extract_wat(input), input);
    }

    #[test]
    fn test_extract_wat_markdown_fences() {
        let input = "```wat\n(module\n  (func (export \"add\") (param i32 i32) (result i32)\n    local.get 0\n    local.get 1\n    i32.add))\n```";
        let result = extract_wat(input);
        assert!(result.starts_with("(module"));
        assert!(result.ends_with("))"));
    }

    #[test]
    fn test_extract_wat_with_explanation() {
        let input = "Here is the WAT code:\n\n(module\n  (func (export \"add\") (param i32 i32) (result i32)\n    local.get 0\n    local.get 1\n    i32.add))\n\nThis adds two integers.";
        let result = extract_wat(input);
        assert!(result.starts_with("(module"));
        assert!(result.ends_with("))"));
        assert!(!result.contains("Here is"));
    }
}
