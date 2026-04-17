//! Error types for GGUF parsing

use thiserror::Error;

pub type Result<T> = std::result::Result<T, GGUFError>;

#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF magic number: 0x{0:08X}, expected 0x46554747")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata value type: {0}")]
    InvalidMetadataType(u32),

    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),

    #[error("Tensor data overflow")]
    TensorDataOverflow,

    #[error("Tensor shape error: {0}")]
    TensorShapeError(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Metadata key not found: {0}")]
    MetadataNotFound(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Unexpected end of file")]
    UnexpectedEof,
}
