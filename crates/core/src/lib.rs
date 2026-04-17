//! RIA Core - Memory-optimized LLM inference engine
//!
//! This crate implements the RIA model architecture with:
//! - GGUF model loading
//! - Layer-by-layer inference for minimal memory usage
//! - RoPE position embeddings
//! - KV cache for efficient generation
//! - Multiple sampling strategies

pub mod config;
pub mod model;
pub mod generation;
pub mod cache;
pub mod error;
pub mod tokenizer;

pub use config::{ModelConfig, GenerationConfig};
pub use model::RIAModel;
pub use generation::Generator;
pub use cache::KVCache;
pub use error::{Result, RIAError};
pub use tokenizer::RIATokenizer;
