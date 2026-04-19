//! Tokenizer integration for RIA models

use crate::error::{RIAError, Result};
use tokenizers::{Encoding, Tokenizer};

/// RIA Tokenizer wrapper
pub struct RIATokenizer {
    tokenizer: Tokenizer,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl RIATokenizer {
    /// Load tokenizer from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| RIAError::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // Get special tokens
        let bos_token = None;
        let eos_token = None;

        Ok(Self {
            tokenizer,
            bos_token,
            eos_token,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| RIAError::Tokenizer(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| RIAError::Tokenizer(format!("Decoding failed: {}", e)))?;

        Ok(text)
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        None // TODO: Implement when tokenizers crate API stabilizes
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        None // TODO: Implement when tokenizers crate API stabilizes
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Encode multiple texts
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| RIAError::Tokenizer(format!("Batch encoding failed: {}", e)))?;

        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }
}
