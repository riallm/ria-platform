//! Server configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Model file path
    pub model_path: String,

    /// Tokenizer file path
    pub tokenizer_path: Option<String>,

    /// Device (cpu/cuda/metal)
    #[serde(default = "default_device")]
    pub device: String,

    /// Maximum sequence length
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Enable profiling
    #[serde(default)]
    pub profile: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            model_path: String::new(),
            tokenizer_path: None,
            device: default_device(),
            max_seq_len: default_max_seq_len(),
            profile: false,
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_max_seq_len() -> usize {
    4096
}
