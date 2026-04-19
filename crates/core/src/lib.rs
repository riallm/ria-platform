//! RIA Core - Memory-optimized LLM inference engine
//!
//! This crate implements the RIA model architecture with:
//! - GGUF model loading
//! - Layer-by-layer inference for minimal memory usage
//! - RoPE position embeddings
//! - KV cache for efficient generation
//! - Multiple sampling strategies

pub mod cache;
pub mod config;
pub mod error;
pub mod generation;
pub mod model;
pub mod tokenizer;

pub use cache::KVCache;
pub use config::{GenerationConfig, ModelConfig};
pub use error::{RIAError, Result};
pub use generation::Generator;
pub use model::RIAModel;
pub use tokenizer::RIATokenizer;
