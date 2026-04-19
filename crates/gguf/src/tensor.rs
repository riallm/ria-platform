//! GGUF tensor info and data loading

use super::error::GGUFError;
use super::header::{read_u32, read_u64};
use super::quantization::{dequantize_tensor, GGUFQuantizationType};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::fmt;

/// GGUF tensor data alignment (bytes)
pub const GGUF_ALIGNMENT: usize = 32;

/// GGUF tensor type IDs
pub const TENSOR_TYPE_F32: u32 = 0;
pub const TENSOR_TYPE_F16: u32 = 1;
pub const TENSOR_TYPE_Q4_0: u32 = 2;
pub const TENSOR_TYPE_Q4_1: u32 = 3;
// Q5_0 = 6, Q5_1 = 7, Q8_0 = 8
pub const TENSOR_TYPE_Q5_0: u32 = 6;
pub const TENSOR_TYPE_Q5_1: u32 = 7;
pub const TENSOR_TYPE_Q8_0: u32 = 8;
// K-quants
pub const TENSOR_TYPE_Q2_K: u32 = 10;
pub const TENSOR_TYPE_Q3_K: u32 = 11;
pub const TENSOR_TYPE_Q4_K: u32 = 12;
pub const TENSOR_TYPE_Q5_K: u32 = 13;
pub const TENSOR_TYPE_Q6_K: u32 = 14;
pub const TENSOR_TYPE_Q8_K: u32 = 15;
pub const TENSOR_TYPE_IQ2_XXS: u32 = 16;
pub const TENSOR_TYPE_IQ2_XS: u32 = 17;
pub const TENSOR_TYPE_IQ3_XXS: u32 = 18;
pub const TENSOR_TYPE_IQ1_S: u32 = 19;
pub const TENSOR_TYPE_IQ4_NL: u32 = 20;
pub const TENSOR_TYPE_IQ3_S: u32 = 21;
pub const TENSOR_TYPE_IQ2_S: u32 = 22;
pub const TENSOR_TYPE_IQ4_XS: u32 = 23;
pub const TENSOR_TYPE_I8: u32 = 24;
pub const TENSOR_TYPE_I16: u32 = 25;
pub const TENSOR_TYPE_I32: u32 = 26;
pub const TENSOR_TYPE_I64: u32 = 27;
pub const TENSOR_TYPE_F64: u32 = 28;
pub const TENSOR_TYPE_IQ1_M: u32 = 29;

/// Tensor info from GGUF header
#[derive(Debug)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64, // Offset to tensor data from start of tensor data section
}

impl GGUFTensorInfo {
    /// Calculate where this tensor's data starts (offset + size, aligned)
    pub fn data_start_offset(&self) -> usize {
        let quant_type = GGUFQuantizationType::from_u32(self.tensor_type);
        if let Some(qt) = quant_type {
            let block_size: usize = qt.block_size();
            let quant_per_block: usize = qt.quant_per_block();
            let num_blocks = (self.element_count() + quant_per_block - 1) / quant_per_block;
            let tensor_size = block_size * num_blocks;
            // Align to GGUF_ALIGNMENT boundary
            ((self.offset as usize) + tensor_size + GGUF_ALIGNMENT - 1) & !(GGUF_ALIGNMENT - 1)
        } else {
            self.offset as usize
        }
    }

    /// Parse tensor info from GGUF file
    pub fn parse(mmap: &Mmap, offset: &mut usize) -> crate::error::Result<Self> {
        // Name length + name
        let name_len = read_u32(mmap, offset)?;
        let name_bytes = &mmap[*offset..*offset + name_len as usize];
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|e| {
            GGUFError::TensorShapeError(format!("Invalid UTF-8 in tensor name: {}", e))
        })?;
        *offset += name_len as usize;

        // Number of dimensions
        let n_dimensions = read_u32(mmap, offset)?;

        // Dimensions
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(read_u64(mmap, offset)?);
        }

        // Tensor type
        let tensor_type = read_u32(mmap, offset)?;

        // Offset to tensor data
        let offset_to_data = read_u64(mmap, offset)?;

        Ok(Self {
            name,
            n_dimensions,
            dimensions,
            tensor_type,
            offset: offset_to_data,
        })
    }

    /// Load tensor data from memory-mapped file
    pub fn load_data(
        &self,
        mmap: &Mmap,
        tensor_data_offset: usize,
    ) -> crate::error::Result<GGUFTensorData> {
        let quant_type = GGUFQuantizationType::from_u32(self.tensor_type)
            .ok_or_else(|| GGUFError::InvalidTensorType(self.tensor_type))?;

        // Calculate tensor data size
        let block_size = quant_type.block_size();
        let quant_per_block = quant_type.quant_per_block();
        let num_blocks = (self.element_count() + quant_per_block - 1) / quant_per_block;
        let data_size = block_size * num_blocks;

        // In GGUF, offset is absolute file offset from tensor data section start
        let data_start = tensor_data_offset + self.offset as usize;
        let data_end = data_start + data_size;

        if data_end > mmap.len() {
            return Err(GGUFError::TensorDataOverflow);
        }

        let data = &mmap[data_start..data_end];

        Ok(GGUFTensorData {
            name: self.name.clone(),
            dimensions: self.dimensions.clone(),
            quant_type,
            raw_data: data.to_vec(),
        })
    }

    /// Get total number of elements
    pub fn element_count(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
}

/// Loaded tensor data
pub struct GGUFTensorData {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub quant_type: GGUFQuantizationType,
    pub raw_data: Vec<u8>,
}

impl GGUFTensorData {
    /// Get total number of elements
    pub fn element_count(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }

    /// Convert to Candle tensor
    pub fn to_candle_tensor(&self, device: &Device) -> crate::error::Result<Tensor> {
        let shape: Vec<usize> = self.dimensions.iter().map(|&d| d as usize).collect();
        let elem_count = self.element_count();

        // Dequantize if necessary
        let data_f32 = dequantize_tensor(&self.raw_data, self.quant_type, elem_count)?;

        // Create tensor from f32 data
        let tensor = Tensor::from_vec(data_f32, &*shape, device)?;
        Ok(tensor)
    }
}

impl fmt::Debug for GGUFTensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GGUFTensorData")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("quant_type", &self.quant_type)
            .field("data_size", &self.raw_data.len())
            .finish()
    }
}
