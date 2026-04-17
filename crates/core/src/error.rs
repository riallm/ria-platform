//! Error types for RIA core

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RIAError>;

#[derive(Error, Debug)]
pub enum RIAError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF error: {0}")]
    GGUF(#[from] ria_gguf::GGUFError),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Out of memory: needed {needed} bytes, available {available} bytes")]
    OutOfMemory { needed: u64, available: u64 },
}
