//! Model and generation configuration

use serde::{Deserialize, Serialize};

/// Model configuration loaded from GGUF metadata
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub name: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub vocab_size: u32,
}

impl ModelConfig {
    /// Get number of KV heads (for GQA)
    pub fn kv_heads(&self) -> u32 {
        self.attention_head_count_kv
    }

    /// Get head dimension
    pub fn head_dim(&self) -> u32 {
        self.embedding_length / self.attention_head_count
    }

    /// Check if model uses grouped query attention
    pub fn uses_gqa(&self) -> bool {
        self.attention_head_count_kv < self.attention_head_count
    }
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Sampling temperature (0.0-2.0)
    pub temperature: f64,

    /// Top-p nucleus sampling (0.0-1.0)
    pub top_p: Option<f64>,

    /// Top-k sampling
    pub top_k: Option<usize>,

    /// Repeat penalty
    pub repeat_penalty: f32,

    /// Repeat last N tokens
    pub repeat_last_n: usize,

    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: f32,

    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: f32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// Return token probabilities
    pub logprobs: bool,

    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: Some(0.95),
            top_k: Some(50),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
            logprobs: false,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a new config with temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Create a new config with top_p
    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Create a new config with max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_new_tokens = max;
        self
    }
}
