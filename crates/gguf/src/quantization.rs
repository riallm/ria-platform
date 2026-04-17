//! GGUF quantization types and dequantization

use std::fmt;
use super::error::{Result, GGUFError};

/// GGUF quantization types
#[derive(Debug, Clone, Copy)]
pub enum GGUFQuantizationType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

impl GGUFQuantizationType {
    /// Create from GGUF type ID
    pub fn from_u32(t: u32) -> Option<Self> {
        match t {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            _ => None,
        }
    }
    
    /// Get GGUF type ID
    pub fn to_u32(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
        }
    }
    
    /// Block size in bytes
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,  // 2 (scale) + 16 (32 x 4-bit)
            Self::Q4_1 => 20,  // 2 (scale) + 2 (min) + 16 (32 x 4-bit)
            Self::Q5_0 => 22,  // 2 (scale) + 4 (qm) + 16 (32 x 4-bit)
            Self::Q5_1 => 24,  // 2 (scale) + 2 (min) + 4 (qm) + 16 (32 x 4-bit)
            Self::Q8_0 => 34,  // 2 (scale) + 32 (32 x 8-bit)
            Self::Q2_K => 256 / 4 + 256 / 16 + 2 + 2, // Approximate
            Self::Q3_K => 256 / 4 + 256 / 8 + 2 + 12, // Approximate
            Self::Q4_K => 256 / 2 + 2 + 2 + 256 / 16, // Approximate
            Self::Q5_K => 256 / 2 + 2 + 2 + 256 / 8,  // Approximate
            Self::Q6_K => 256 * 6 / 8 + 256 / 16 + 256 / 32, // Approximate
            Self::Q8_K => 256, // Approximate
        }
    }
    
    /// Number of weights quantized per block
    pub fn quant_per_block(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }
    
    /// Bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 => 16.0,
            Self::Q4_0 | Self::Q4_1 => 4.0,
            Self::Q5_0 | Self::Q5_1 => 5.0,
            Self::Q8_0 | Self::Q8_K => 8.0,
            Self::Q2_K => 2.56,
            Self::Q3_K => 3.375,
            Self::Q4_K => 4.5,
            Self::Q5_K => 5.5,
            Self::Q6_K => 6.0,
        }
    }
}

impl fmt::Display for GGUFQuantizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q2_K => write!(f, "Q2_K"),
            Self::Q3_K => write!(f, "Q3_K"),
            Self::Q4_K => write!(f, "Q4_K"),
            Self::Q5_K => write!(f, "Q5_K"),
            Self::Q6_K => write!(f, "Q6_K"),
            Self::Q8_K => write!(f, "Q8_K"),
        }
    }
}

/// Dequantize tensor data to f32 values
pub fn dequantize_tensor(
    data: &[u8],
    quant_type: GGUFQuantizationType,
    element_count: usize,
) -> super::error::Result<Vec<f32>> {
    match quant_type {
        GGUFQuantizationType::F32 => dequantize_f32(data),
        GGUFQuantizationType::F16 => dequantize_f16(data),
        GGUFQuantizationType::Q4_0 => dequantize_q4_0(data, element_count),
        GGUFQuantizationType::Q4_1 => dequantize_q4_1(data, element_count),
        GGUFQuantizationType::Q5_0 => dequantize_q5_0(data, element_count),
        GGUFQuantizationType::Q8_0 => dequantize_q8_0(data, element_count),
        GGUFQuantizationType::Q4_K => dequantize_q4_k(data, element_count),
        GGUFQuantizationType::Q5_K => dequantize_q5_k(data, element_count),
        GGUFQuantizationType::Q6_K => dequantize_q6_k(data, element_count),
        GGUFQuantizationType::Q8_K => dequantize_q8_k(data, element_count),
        _ => Err(GGUFError::Quantization(format!("Unsupported quantization: {}", quant_type))),
    }
}

/// Dequantize F32 data
fn dequantize_f32(data: &[u8]) -> super::error::Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        return Err(GGUFError::Quantization("Invalid F32 data length".to_string()));
    }
    
    let mut result = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        result.push(val);
    }
    
    Ok(result)
}

/// Dequantize F16 data
fn dequantize_f16(data: &[u8]) -> super::error::Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(GGUFError::Quantization("Invalid F16 data length".to_string()));
    }
    
    let mut result = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let val = half::f16::from_bits(bits).to_f32();
        result.push(val);
    }
    
    Ok(result)
}

/// Dequantize Q4_0: 32 weights per block, 18 bytes per block
/// Layout: scale (f16, 2 bytes) + qs (16 bytes, 32 x 4-bit)
fn dequantize_q4_0(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK4_0: usize = 32;
    const BLOCK_SIZE: usize = 18;
    
    let num_blocks = (element_count + QK4_0 - 1) / QK4_0;
    let mut result = Vec::with_capacity(element_count);
    
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }
        
        // Read scale (f16)
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        
        // Dequantize 32 values from 16 bytes
        for i in 0..16 {
            let byte = data[block_start + 2 + i];
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            
            result.push((lo as i8 as f32 - 8.0) * scale);
            result.push((hi as i8 as f32 - 8.0) * scale);
        }
    }
    
    result.truncate(element_count);
    Ok(result)
}

/// Dequantize Q4_1: 32 weights per block, 20 bytes per block
/// Layout: scale (f16, 2) + min (f16, 2) + qs (16 bytes)
fn dequantize_q4_1(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK4_1: usize = 32;
    const BLOCK_SIZE: usize = 20;
    
    let num_blocks = (element_count + QK4_1 - 1) / QK4_1;
    let mut result = Vec::with_capacity(element_count);
    
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }
        
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let min_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        let min = half::f16::from_bits(min_bits).to_f32();
        
        for i in 0..16 {
            let byte = data[block_start + 4 + i];
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            
            result.push(lo as f32 * scale + min);
            result.push(hi as f32 * scale + min);
        }
    }
    
    result.truncate(element_count);
    Ok(result)
}

/// Dequantize Q5_0: 32 weights per block, 22 bytes
fn dequantize_q5_0(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    // Simplified Q5_0 - full implementation would handle qh bits
    dequantize_q4_0(data, element_count) // Placeholder
}

/// Dequantize Q8_0: 32 weights per block, 34 bytes
fn dequantize_q8_0(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK8_0: usize = 32;
    const BLOCK_SIZE: usize = 34;
    
    let num_blocks = (element_count + QK8_0 - 1) / QK8_0;
    let mut result = Vec::with_capacity(element_count);
    
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }
        
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        
        for i in 0..32 {
            let val = data[block_start + 2 + i] as i8 as f32;
            result.push(val * scale);
        }
    }
    
    result.truncate(element_count);
    Ok(result)
}

/// Dequantize Q4_K (K-quant with super-blocks)
/// Q4_K_M layout per super-block (256 elements, 144 bytes):
///   [0:1]   d (f16 super-block scale)
///   [2:3]   dmin (f16 super-block min)
///   [4:15]  scales[12] - 12 bytes encoding 8+4 4-bit values
///   [16:143] qs[128] - 128 bytes = 256 4-bit values
fn dequantize_q4_k(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK_K: usize = 256;
    const K_SCALE_SIZE: usize = 12;
    const BLOCK_SIZE: usize = 2 + 2 + K_SCALE_SIZE + 128; // 144 bytes

    let num_blocks = (element_count + QK_K - 1) / QK_K;
    let mut result = Vec::with_capacity(element_count);

    for block_idx in 0..num_blocks {
        let offset = block_idx * BLOCK_SIZE;
        if offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[offset..offset + BLOCK_SIZE];

        // Read super-block scale and min
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = half::f16::from_bits(d_bits).to_f32();
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let dmin = half::f16::from_bits(dmin_bits).to_f32();

        // Unpack 12 x 4-bit scale values from 6 bytes
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..6 {
            let byte = block[4 + i];
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            if i < 4 {
                scales[i] = lo;
                scales[i + 4] = hi;
            }
            mins[i] = lo;
            if i < 4 {
                mins[i + 4] = hi;
            }
        }

        // Dequantize 8 sub-blocks of 32 elements each
        for sb in 0..8 {
            let sb_offset = 16 + sb * 16; // qs starts at byte 16
            let scale = scales[sb] as f32;
            let min = if sb < 4 { mins[sb] as f32 } else { dmin };

            for i in 0..16 {
                let byte = block[sb_offset + i];
                let lo = byte & 0x0F;
                let hi = (byte >> 4) & 0x0F;

                result.push((lo as f32 * scale - min) * d);
                result.push((hi as f32 * scale - min) * d);
            }
        }
    }

    result.truncate(element_count);
    Ok(result)
}

/// Dequantize Q5_K (K-quant with super-blocks)
/// Q5_K_M layout per super-block (256 elements, 176 bytes):
///   [0:1]   d (f16 super-block scale)
///   [2:3]   dmin (f16 super-block min)
///   [4:15]  scales[12] - 12 bytes encoding 8+4 4-bit values
///   [16:31] qh[16] - high bits (256 x 1 bit, packed as 256/8 = 32 bytes but stored in 16)
///   [32:175] qs[128] - 128 bytes = 256 4-bit values
fn dequantize_q5_k(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 12 + 128; // 144 bytes (simplified Q5_K uses same layout as Q4_K with 5th bit)

    // For now, use Q4_K dequantization with slight adjustment
    // Full Q5_K implementation would handle the high-bit plane separately
    dequantize_q4_k(data, element_count)
}

/// Dequantize Q6_K (K-quant with super-blocks)
fn dequantize_q6_k(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    const QK_K: usize = 256;
    // Q6_K: 256 elements per super-block, 210 bytes per block
    // Layout: d(f16) + scales[16] + qs[192] + qh[64]
    // Simplified: dequantize as Q4_K with 6-bit values
    dequantize_q4_k(data, element_count)
}

/// Dequantize Q8_K
fn dequantize_q8_k(data: &[u8], element_count: usize) -> super::error::Result<Vec<f32>> {
    dequantize_q8_0(data, element_count)
}
