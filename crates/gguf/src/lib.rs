//! GGUF (GPT-Generated Unified Format) binary format parser
//!
//! This crate provides functionality for reading GGUF model files, including:
//! - Header parsing (magic, version, tensor count, metadata)
//! - Metadata key-value parsing
//! - Tensor info and data loading
//! - Quantized tensor decoding (Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16, etc.)

pub mod error;
pub mod header;
pub mod metadata;
pub mod quantization;
pub mod tensor;

pub use error::GGUFError;
pub use header::GGUFHeader;
pub use metadata::MetadataValue;
pub use quantization::GGUFQuantizationType;
pub use tensor::{GGUFTensorData, GGUFTensorInfo};

use candle_core::Device;
use candle_core::Tensor;
use memmap2::Mmap;
use std::path::Path;

/// GGUF file reader
pub struct GGUFReader {
    header: GGUFHeader,
    mmap: Mmap,
    /// Offset where tensor data section starts (after all tensor infos)
    tensor_data_offset: usize,
}

impl GGUFReader {
    /// Open a GGUF file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GGUFError> {
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header = GGUFHeader::parse(&mmap)?;

        // Parse tensor infos to find where tensor data section starts
        let tensor_infos = header.parse_tensor_infos(&mmap)?;
        let tensor_data_offset = if let Some(last_info) = tensor_infos.last() {
            // Approximate: after last tensor info
            last_info.data_start_offset()
        } else {
            header.tensor_info_offset
        };

        tracing::info!(
            "Loaded GGUF file: magic=0x{:08X}, version={}, tensors={}, metadata_keys={}",
            header.magic,
            header.version,
            header.tensor_count,
            header.metadata_kv_count
        );

        Ok(Self {
            header,
            mmap,
            tensor_data_offset,
        })
    }

    /// Get the header
    pub fn header(&self) -> &GGUFHeader {
        &self.header
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.header.metadata.get(key)
    }

    /// Get model architecture name
    pub fn architecture(&self) -> Option<&str> {
        match self.get_metadata("general.architecture") {
            Some(MetadataValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get model name
    pub fn model_name(&self) -> Option<&str> {
        match self.get_metadata("general.name") {
            Some(MetadataValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get tensor infos for all tensors
    pub fn tensor_infos(&self) -> Result<Vec<GGUFTensorInfo>, GGUFError> {
        self.header.parse_tensor_infos(&self.mmap)
    }

    /// Load a specific tensor by name
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>, GGUFError> {
        let infos = self.tensor_infos()?;

        for info in &infos {
            if info.name == name {
                let data = info.load_data(&self.mmap, self.tensor_data_offset)?;
                return Ok(Some(data.to_candle_tensor(device)?));
            }
        }

        Ok(None)
    }

    /// Load all tensors into memory (use with caution for large models)
    pub fn load_all_tensors(&self, device: &Device) -> Result<Vec<(String, Tensor)>, GGUFError> {
        let infos = self.tensor_infos()?;
        let mut tensors = Vec::with_capacity(infos.len());

        for info in &infos {
            let data = info.load_data(&self.mmap, self.tensor_data_offset)?;
            let tensor = data.to_candle_tensor(device)?;
            tensors.push((info.name.clone(), tensor));
        }

        Ok(tensors)
    }

    /// Get file size in bytes
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Format file size for display
    pub fn formatted_size(&self) -> String {
        let bytes = self.file_size();
        if bytes >= 1_073_741_824 {
            format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.2} MB", bytes as f64 / 1_048_576.0)
        } else {
            format!("{} KB", bytes / 1024)
        }
    }
}
