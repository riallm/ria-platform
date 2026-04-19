//! GGUF header parsing

use super::error::{GGUFError, Result};
use super::metadata::{parse_metadata_value, MetadataValue};
use super::tensor::GGUFTensorInfo;
use memmap2::Mmap;
use std::collections::HashMap;

/// GGUF magic number: "GGUF" = 0x46554747
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Current GGUF version
pub const GGUF_VERSION: u32 = 3;

/// GGUF file header
#[derive(Debug)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    /// Offset where tensor info section starts (after metadata)
    pub tensor_info_offset: usize,
}

impl GGUFHeader {
    /// Parse GGUF header from memory-mapped file
    pub fn parse(mmap: &Mmap) -> super::error::Result<Self> {
        let mut offset = 0usize;

        // Magic number (4 bytes)
        let magic = read_u32(mmap, &mut offset)?;
        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic(magic));
        }

        // Version (4 bytes)
        let version = read_u32(mmap, &mut offset)?;
        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        // Tensor count (8 bytes)
        let tensor_count = read_u64(mmap, &mut offset)?;

        // Metadata KV count (8 bytes)
        let metadata_kv_count = read_u64(mmap, &mut offset)?;

        // Parse metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_string(mmap, &mut offset)?;
            let value = parse_metadata_value(mmap, &mut offset)?;
            metadata.insert(key, value);
        }

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensor_info_offset: offset,
        })
    }

    /// Parse tensor info section
    pub fn parse_tensor_infos(&self, mmap: &Mmap) -> super::error::Result<Vec<GGUFTensorInfo>> {
        // Offset starts after metadata section (tracked by tensor_info_offset)
        let mut offset = self.tensor_info_offset;
        let mut infos = Vec::with_capacity(self.tensor_count as usize);

        for _ in 0..self.tensor_count {
            let info = GGUFTensorInfo::parse(mmap, &mut offset)?;
            infos.push(info);
        }

        Ok(infos)
    }
}

/// Read a little-endian u16
pub fn read_u8(mmap: &Mmap, offset: &mut usize) -> super::error::Result<u8> {
    if *offset >= mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let val = mmap[*offset];
    *offset += 1;
    Ok(val)
}

pub fn read_u16(mmap: &Mmap, offset: &mut usize) -> super::error::Result<u16> {
    if *offset + 2 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 2];
    *offset += 2;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Read a little-endian u32
pub fn read_u32(mmap: &Mmap, offset: &mut usize) -> super::error::Result<u32> {
    if *offset + 4 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 4];
    *offset += 4;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a little-endian u64
pub fn read_u64(mmap: &Mmap, offset: &mut usize) -> super::error::Result<u64> {
    if *offset + 8 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 8];
    *offset += 8;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read a little-endian i32
pub fn read_i32(mmap: &Mmap, offset: &mut usize) -> super::error::Result<i32> {
    if *offset + 4 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 4];
    *offset += 4;
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a little-endian i64
pub fn read_i64(mmap: &Mmap, offset: &mut usize) -> super::error::Result<i64> {
    if *offset + 8 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 8];
    *offset += 8;
    Ok(i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read a little-endian f32
pub fn read_f32(mmap: &Mmap, offset: &mut usize) -> super::error::Result<f32> {
    if *offset + 4 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 4];
    *offset += 4;
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a little-endian f64
pub fn read_f64(mmap: &Mmap, offset: &mut usize) -> super::error::Result<f64> {
    if *offset + 8 > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }
    let bytes = &mmap[*offset..*offset + 8];
    *offset += 8;
    Ok(f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read a GGUF string (length-prefixed UTF-8)
pub fn read_string(mmap: &Mmap, offset: &mut usize) -> super::error::Result<String> {
    let len = read_u64(mmap, offset)? as usize;

    if *offset + len > mmap.len() {
        return Err(GGUFError::UnexpectedEof);
    }

    let bytes = &mmap[*offset..*offset + len];
    *offset += len;

    String::from_utf8(bytes.to_vec())
        .map_err(|e| GGUFError::Quantization(format!("Invalid UTF-8: {}", e)))
}
