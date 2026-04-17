//! Layer-by-layer loading engine for memory-optimized inference
//!
//! This module implements the core AirLLM-style optimization:
//! loading one transformer layer at a time instead of the entire model,
//! enabling running large models on limited GPU memory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use candle_core::{Tensor, Device};
use ria_gguf::GGUFReader;
use crate::error::{Result, RIAError};
use crate::config::ModelConfig;
use crate::cache::KVCache;

/// Configuration for layer-by-layer loading
#[derive(Debug, Clone)]
pub struct LayerLoadingConfig {
    /// Device for active layer inference
    pub device: Device,
    /// CPU device for storing unloaded layers
    pub cpu_device: Device,
    /// Number of layers to prefetch ahead
    pub prefetch_count: usize,
    /// Enable memory-mapped loading
    pub use_mmap: bool,
}

impl Default for LayerLoadingConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            cpu_device: Device::Cpu,
            prefetch_count: 2,
            use_mmap: true,
        }
    }
}

/// Manages layer loading and unloading for memory-optimized inference
pub struct LayerEngine {
    /// Path to GGUF model file
    model_path: PathBuf,
    /// Model configuration
    config: ModelConfig,
    /// Layer loading configuration
    config_loading: LayerLoadingConfig,
    /// Currently loaded layer weights (by tensor name)
    current_layer: Option<usize>,
    /// Cached layer weights on CPU (loaded from GGUF on demand)
    cpu_cache: HashMap<usize, LayerWeights>,
}

/// Weights for a single transformer layer
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm_weight: Tensor,
    pub wq_weight: Tensor,
    pub wk_weight: Tensor,
    pub wv_weight: Tensor,
    pub wo_weight: Tensor,
    pub ffn_norm_weight: Tensor,
    pub w1_weight: Tensor,
    pub w2_weight: Tensor,
    pub w3_weight: Tensor,
}

impl LayerWeights {
    /// Move all weights to specified device
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            attn_norm_weight: self.attn_norm_weight.to_device(device)?,
            wq_weight: self.wq_weight.to_device(device)?,
            wk_weight: self.wk_weight.to_device(device)?,
            wv_weight: self.wv_weight.to_device(device)?,
            wo_weight: self.wo_weight.to_device(device)?,
            ffn_norm_weight: self.ffn_norm_weight.to_device(device)?,
            w1_weight: self.w1_weight.to_device(device)?,
            w2_weight: self.w2_weight.to_device(device)?,
            w3_weight: self.w3_weight.to_device(device)?,
        })
    }

    /// Clear weights (drop from memory)
    pub fn clear(&mut self) {
        // Drop tensors by replacing with CPU dummy tensors
        let cpu = Device::Cpu;
        self.attn_norm_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.wq_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.wk_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.wv_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.wo_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.ffn_norm_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.w1_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.w2_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
        self.w3_weight = Tensor::zeros((0,), candle_core::DType::F32, &cpu).unwrap();
    }
}

impl LayerEngine {
    /// Create a new layer engine
    pub fn new(
        model_path: impl AsRef<Path>,
        config: ModelConfig,
        config_loading: LayerLoadingConfig,
    ) -> Result<Self> {
        if !model_path.as_ref().exists() {
            return Err(RIAError::ModelLoading(
                format!("Model file not found: {:?}", model_path.as_ref())
            ));
        }

        Ok(Self {
            model_path: model_path.as_ref().to_path_buf(),
            config,
            config_loading,
            current_layer: None,
            cpu_cache: HashMap::new(),
        })
    }

    /// Load a specific layer from GGUF file into CPU memory
    pub fn load_layer_to_cpu(&mut self, layer_idx: usize) -> Result<()> {
        if self.cpu_cache.contains_key(&layer_idx) {
            return Ok(()); // Already loaded
        }

        let reader = GGUFReader::open(&self.model_path)
            .map_err(RIAError::GGUF)?;

        let prefix = format!("blk.{}.", layer_idx);
        let get_tensor = |name: &str| -> Result<Tensor> {
            let full_name = format!("{}{}", prefix, name);
            reader.load_tensor(&full_name, &self.config_loading.cpu_device)
                .map_err(RIAError::GGUF)?
                .ok_or_else(|| RIAError::ModelLoading(
                    format!("Missing tensor: {}", full_name)
                ))
        };

        let weights = LayerWeights {
            attn_norm_weight: get_tensor("attn_norm.weight")?,
            wq_weight: get_tensor("attn_q.weight")?,
            wk_weight: get_tensor("attn_k.weight")?,
            wv_weight: get_tensor("attn_v.weight")?,
            wo_weight: get_tensor("attn_output.weight")?,
            ffn_norm_weight: get_tensor("ffn_norm.weight")?,
            w1_weight: get_tensor("ffn_gate.weight")?,
            w2_weight: get_tensor("ffn_down.weight")?,
            w3_weight: get_tensor("ffn_up.weight")?,
        };

        self.cpu_cache.insert(layer_idx, weights);
        Ok(())
    }

    /// Load layer to inference device (CPU → GPU)
    pub fn load_layer_to_device(&mut self, layer_idx: usize) -> Result<LayerWeights> {
        // Ensure layer is in CPU cache
        self.load_layer_to_cpu(layer_idx)?;

        // Move to device
        let weights = self.cpu_cache.get(&layer_idx)
            .ok_or_else(|| RIAError::ModelLoading(
                format!("Layer {} not in CPU cache", layer_idx)
            ))?;

        let device_weights = weights.to_device(&self.config_loading.device)?;
        self.current_layer = Some(layer_idx);

        Ok(device_weights)
    }

    /// Unload a layer from device memory
    pub fn unload_layer(&mut self, layer_idx: usize) -> Result<()> {
        if self.current_layer == Some(layer_idx) {
            self.current_layer = None;
        }
        // Keep in CPU cache for potential reuse
        Ok(())
    }

    /// Prefetch next N layers into CPU cache
    pub fn prefetch_layers(&mut self, current_layer: usize) -> Result<()> {
        for i in 1..=self.config_loading.prefetch_count {
            let next_layer = current_layer + i;
            if next_layer < self.config.block_count as usize {
                self.load_layer_to_cpu(next_layer)?;
            }
        }
        Ok(())
    }

    /// Get embedding weights
    pub fn load_embeddings(&self) -> Result<Tensor> {
        let reader = GGUFReader::open(&self.model_path)
            .map_err(RIAError::GGUF)?;

        reader.load_tensor("token_embd.weight", &self.config_loading.device)
            .map_err(RIAError::GGUF)?
            .ok_or_else(|| RIAError::ModelLoading(
                "Missing tensor: token_embd.weight".to_string()
            ))
    }

    /// Get output norm weights
    pub fn load_output_norm(&self) -> Result<(Tensor, f64)> {
        let reader = GGUFReader::open(&self.model_path)
            .map_err(RIAError::GGUF)?;

        let weight = reader.load_tensor("output_norm.weight", &self.config_loading.device)
            .map_err(RIAError::GGUF)?
            .ok_or_else(|| RIAError::ModelLoading(
                "Missing tensor: output_norm.weight".to_string()
            ))?;

        Ok((weight, self.config.layer_norm_rms_epsilon as f64))
    }

    /// Get output (LM head) weights
    pub fn load_output(&self) -> Result<Tensor> {
        let reader = GGUFReader::open(&self.model_path)
            .map_err(RIAError::GGUF)?;

        reader.load_tensor("output.weight", &self.config_loading.device)
            .map_err(RIAError::GGUF)?
            .ok_or_else(|| RIAError::ModelLoading(
                "Missing tensor: output.weight".to_string()
            ))
    }

    /// Clear CPU cache to free system memory
    pub fn clear_cpu_cache(&mut self) {
        self.cpu_cache.clear();
    }

    /// Get memory usage estimate
    pub fn memory_usage_bytes(&self) -> (u64, u64) {
        let mut cpu_bytes = 0u64;
        let mut device_bytes = 0u64;

        for weights in self.cpu_cache.values() {
            // Rough estimate: count tensor elements * dtype size
            cpu_bytes += estimate_tensor_bytes(&weights.attn_norm_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.wq_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.wk_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.wv_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.wo_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.ffn_norm_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.w1_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.w2_weight);
            cpu_bytes += estimate_tensor_bytes(&weights.w3_weight);
        }

        (cpu_bytes, device_bytes)
    }
}

/// Estimate tensor memory usage in bytes
fn estimate_tensor_bytes(tensor: &Tensor) -> u64 {
    let elem_count = tensor.shape().elem_count();
    let dtype_size = match tensor.dtype() {
        candle_core::DType::F32 => 4,
        candle_core::DType::F16 | candle_core::DType::BF16 => 2,
        candle_core::DType::U8 => 1,
        candle_core::DType::U32 => 4,
        candle_core::DType::I64 => 8,
        _ => 4,
    };
    (elem_count * dtype_size) as u64
}

/// Forward pass through a single transformer layer
pub fn forward_layer(
    hidden: &Tensor,
    weights: &LayerWeights,
    cache: &mut KVCache,
    layer_idx: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<Tensor> {
    use candle_core::D;

    let (_batch_size, seq_len, _) = hidden.dims3()?;
    let residual = hidden.clone();

    // Attention pre-norm
    let normed = candle_nn::RmsNorm::new(
        weights.attn_norm_weight.clone(),
        1e-5,
    ).forward(hidden)?;

    // Q, K, V projections
    let q = candle_nn::Linear::new(weights.wq_weight.clone(), None).forward(&normed)?;
    let k = candle_nn::Linear::new(weights.wk_weight.clone(), None).forward(&normed)?;
    let v = candle_nn::Linear::new(weights.wv_weight.clone(), None).forward(&normed)?;

    // Reshape for attention: (batch, seq, heads, head_dim)
    let q = q.reshape(((), seq_len, num_heads, head_dim))?
        .transpose(1, 2)?.contiguous()?;
    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?.contiguous()?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?.contiguous()?;

    // Apply RoPE (would need rope_apply_interleaved function)
    // For now, skip RoPE since it's defined in model.rs
    // let (q, k) = rope_apply_interleaved(&q, &k, ...)?;

    // Update KV cache
    let (k_cached, v_cached) = cache.update(layer_idx, k.clone(), v.clone())?;

    // GQA: repeat KV heads if needed
    let gqa_group_size = num_heads / num_kv_heads;
    let k_repeated = if gqa_group_size > 1 {
        repeat_kv_tensor(&k_cached, gqa_group_size)?
    } else {
        k_cached
    };
    let v_repeated = if gqa_group_size > 1 {
        repeat_kv_tensor(&v_cached, gqa_group_size)?
    } else {
        v_cached
    };

    // Scaled dot-product attention
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut attn_weights = q.matmul(&k_repeated.transpose(2, 3)?)?;
    attn_weights = (attn_weights * scale)?;

    let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
    let attn_output = attn_weights.matmul(&v_repeated)?;

    // Reshape back
    let attn_output = attn_output.transpose(1, 2)?
        .reshape(((), seq_len, num_heads * head_dim))?;

    // Output projection
    let attn_output = candle_nn::Linear::new(weights.wo_weight.clone(), None)
        .forward(&attn_output)?;

    let mut hidden = (residual + attn_output)?;

    // FFN pre-norm
    let residual_ffn = hidden.clone();
    let normed_ffn = candle_nn::RmsNorm::new(
        weights.ffn_norm_weight.clone(),
        1e-5,
    ).forward(&hidden)?;

    // SwiGLU FFN
    let gate = candle_nn::Linear::new(weights.w1_weight.clone(), None)
        .forward(&normed_ffn)?;
    let gate = candle_nn::ops::silu(&gate)?;
    let up = candle_nn::Linear::new(weights.w3_weight.clone(), None)
        .forward(&normed_ffn)?;
    let hidden_ffn = (gate * up)?;
    let ffn_output = candle_nn::Linear::new(weights.w2_weight.clone(), None)
        .forward(&hidden_ffn)?;

    hidden = (residual_ffn + ffn_output)?;

    Ok(hidden)
}

/// Repeat KV heads for GQA
fn repeat_kv_tensor(x: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }

    let dims = x.shape().dims().to_vec();
    let batch_size = dims[0];
    let num_kv_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    // Reshape → Expand → Reshape
    let x = x.reshape((batch_size, num_kv_heads, 1, seq_len, head_dim))?;
    let x = x.expand((batch_size, num_kv_heads, group_size, seq_len, head_dim))?;
    x.reshape((batch_size, num_kv_heads * group_size, seq_len, head_dim))
}
