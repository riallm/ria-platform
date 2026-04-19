//! HTTP request handlers

use crate::types::*;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use ria_core::{GenerationConfig, Generator, RIAModel, RIATokenizer};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

/// Application state
pub struct AppState {
    pub model: RIAModel,
    pub tokenizer: Option<RIATokenizer>,
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    // If streaming requested, use SSE
    if req.stream {
        return completions_stream(State(state), req).await;
    }

    let state = state.lock().await;

    // Encode prompt
    let prompt_tokens = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .encode(&req.prompt, true)
            .map_err(|e| internal_error(format!("Tokenization failed: {}", e)))?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Tokenizer not loaded"})),
        ));
    };

    let prompt_len = prompt_tokens.len();

    // Build generation config
    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p,
        top_k: req.top_k,
        repeat_penalty: req.repeat_penalty.unwrap_or(1.1),
        repeat_last_n: 64,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        stop_sequences: req.stop.clone().unwrap_or_default(),
        logprobs: false,
        seed: req.seed,
    };

    // Generate
    let mut generator = Generator::new(gen_config);
    let output = generator
        .generate(&state.model, &prompt_tokens)
        .map_err(|e| internal_error(format!("Generation failed: {}", e)))?;

    // Decode output
    let text = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .decode(&output.tokens, true)
            .map_err(|e| internal_error(format!("Decoding failed: {}", e)))?
    } else {
        format!("{:?}", output.tokens)
    };

    let completion_tokens = output.tokens.len();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: match output.finish_reason {
                ria_core::generation::FinishReason::StopToken => "stop".to_string(),
                ria_core::generation::FinishReason::MaxTokens => "length".to_string(),
                ria_core::generation::FinishReason::StopSequence(_) => "stop".to_string(),
            },
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    }))
}

/// Streaming completions with SSE
async fn completions_stream(
    State(state): State<Arc<Mutex<AppState>>>,
    req: CompletionRequest,
) -> Result<Json<CompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    // For now, fall back to non-streaming with a note
    // Full SSE implementation would use tokio::spawn and channels
    let state = state.lock().await;

    let prompt_tokens = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .encode(&req.prompt, true)
            .map_err(|e| internal_error(format!("Tokenization failed: {}", e)))?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Tokenizer not loaded"})),
        ));
    };

    let prompt_len = prompt_tokens.len();

    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p,
        top_k: req.top_k,
        repeat_penalty: req.repeat_penalty.unwrap_or(1.1),
        repeat_last_n: 64,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        stop_sequences: req.stop.clone().unwrap_or_default(),
        logprobs: false,
        seed: req.seed,
    };

    let mut generator = Generator::new(gen_config);
    let output = generator
        .generate(&state.model, &prompt_tokens)
        .map_err(|e| internal_error(format!("Generation failed: {}", e)))?;

    let text = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .decode(&output.tokens, true)
            .map_err(|e| internal_error(format!("Decoding failed: {}", e)))?
    } else {
        format!("{:?}", output.tokens)
    };

    let completion_tokens = output.tokens.len();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid_simple()),
        object: "text_completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    }))
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    let state = state.lock().await;

    // Format chat messages into prompt
    let prompt = format_chat_messages(&req.messages);

    let prompt_tokens = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .encode(&prompt, true)
            .map_err(|e| internal_error(format!("Tokenization failed: {}", e)))?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Tokenizer not loaded"})),
        ));
    };

    let prompt_len = prompt_tokens.len();

    // Generate
    let gen_config = GenerationConfig {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p,
        top_k: None,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        stop_sequences: req.stop.clone().unwrap_or_default(),
        logprobs: false,
        seed: None,
    };

    let mut generator = Generator::new(gen_config);
    let output = generator
        .generate(&state.model, &prompt_tokens)
        .map_err(|e| internal_error(format!("Generation failed: {}", e)))?;

    let text = if let Some(ref tokenizer) = state.tokenizer {
        tokenizer
            .decode(&output.tokens, true)
            .map_err(|e| internal_error(format!("Decoding failed: {}", e)))?
    } else {
        format!("{:?}", output.tokens)
    };

    let completion_tokens = output.tokens.len();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid_simple()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    }))
}

/// GET /v1/models
pub async fn list_models() -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "ria-model".to_string(),
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            owned_by: "riallm".to_string(),
        }],
    })
}

/// GET /health
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

/// Format chat messages into a prompt string
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Generate simple UUID-like string
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", timestamp)
}

/// Create internal error response
fn internal_error(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": msg})),
    )
}
