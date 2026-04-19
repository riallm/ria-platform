//! GGUF metadata parsing

use super::error::{GGUFError, Result};
use super::header::{
    read_f32, read_f64, read_i32, read_i64, read_string, read_u16, read_u32, read_u64, read_u8,
};
use memmap2::Mmap;

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    /// Get size in bytes of this value (excluding type tag)
    pub fn size_bytes(&self) -> usize {
        match self {
            MetadataValue::U8(_) | MetadataValue::I8(_) | MetadataValue::Bool(_) => 1,
            MetadataValue::U16(_) | MetadataValue::I16(_) => 2,
            MetadataValue::U32(_) | MetadataValue::I32(_) | MetadataValue::F32(_) => 4,
            MetadataValue::U64(_) | MetadataValue::I64(_) | MetadataValue::F64(_) => 8,
            MetadataValue::String(s) => 8 + s.len(), // length prefix + string
            MetadataValue::Array(arr) => {
                // Each element: type (4) + value
                8 + arr.iter().map(|v| 4 + v.size_bytes()).sum::<usize>()
            }
        }
    }
}

/// Metadata value type tags
pub const METADATA_TYPE_U8: u32 = 0;
pub const METADATA_TYPE_I8: u32 = 1;
pub const METADATA_TYPE_U16: u32 = 2;
pub const METADATA_TYPE_I16: u32 = 3;
pub const METADATA_TYPE_U32: u32 = 4;
pub const METADATA_TYPE_I32: u32 = 5;
pub const METADATA_TYPE_F32: u32 = 6;
pub const METADATA_TYPE_BOOL: u32 = 7;
pub const METADATA_TYPE_STRING: u32 = 8;
pub const METADATA_TYPE_ARRAY: u32 = 9;
pub const METADATA_TYPE_U64: u32 = 10;
pub const METADATA_TYPE_I64: u32 = 11;
pub const METADATA_TYPE_F64: u32 = 12;

/// Parse a metadata value based on its type
pub fn parse_metadata_value(
    mmap: &Mmap,
    offset: &mut usize,
) -> super::error::Result<MetadataValue> {
    let value_type = read_u32(mmap, offset)?;

    match value_type {
        METADATA_TYPE_U8 => {
            let val = read_u8(mmap, offset)?;
            Ok(MetadataValue::U8(val))
        }
        METADATA_TYPE_I8 => {
            let val = mmap[*offset];
            *offset += 1;
            Ok(MetadataValue::I8(val as i8))
        }
        METADATA_TYPE_U16 => {
            let val = read_u16(mmap, offset)?;
            Ok(MetadataValue::U16(val))
        }
        METADATA_TYPE_I16 => {
            let val = read_u16(mmap, offset)?;
            Ok(MetadataValue::I16(val as i16))
        }
        METADATA_TYPE_U32 => {
            let val = read_u32(mmap, offset)?;
            Ok(MetadataValue::U32(val))
        }
        METADATA_TYPE_I32 => {
            let val = read_i32(mmap, offset)?;
            Ok(MetadataValue::I32(val))
        }
        METADATA_TYPE_F32 => {
            let val = read_f32(mmap, offset)?;
            Ok(MetadataValue::F32(val))
        }
        METADATA_TYPE_BOOL => {
            let val = read_u8(mmap, offset)?;
            Ok(MetadataValue::Bool(val != 0))
        }
        METADATA_TYPE_STRING => {
            let val = read_string(mmap, offset)?;
            Ok(MetadataValue::String(val))
        }
        METADATA_TYPE_ARRAY => {
            let len = read_u64(mmap, offset)?;
            let mut arr = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let elem = parse_metadata_value(mmap, offset)?;
                arr.push(elem);
            }
            Ok(MetadataValue::Array(arr))
        }
        METADATA_TYPE_U64 => {
            let val = read_u64(mmap, offset)?;
            Ok(MetadataValue::U64(val))
        }
        METADATA_TYPE_I64 => {
            let val = read_i64(mmap, offset)?;
            Ok(MetadataValue::I64(val))
        }
        METADATA_TYPE_F64 => {
            let val = read_f64(mmap, offset)?;
            Ok(MetadataValue::F64(val))
        }
        _ => Err(GGUFError::InvalidMetadataType(value_type)),
    }
}
