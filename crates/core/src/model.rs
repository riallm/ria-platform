//! RIA Model - Complete transformer implementation with GGUF loading

use crate::cache::KVCache;
use crate::config::ModelConfig;
use crate::error::{RIAError, Result};
use candle_core::Module;
use candle_core::{Device, Tensor, D};
use candle_nn::{Embedding, Linear, RmsNorm};
use ria_gguf::{GGUFQuantizationType, GGUFReader};
use std::collections::HashMap;

/// RIA Transformer Model
pub struct RIAModel {
    pub config: ModelConfig,
    pub device: Device,

    // Model weights
    pub tok_embeddings: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub output: Linear,

    // For layer-by-layer loading
    pub gguf_path: Option<String>,
    pub loaded_weights: HashMap<String, Tensor>,
}

/// Single transformer block
pub struct TransformerBlock {
    pub attention_norm: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: RmsNorm,
    pub w1: Linear, // Gate (SwiGLU)
    pub w2: Linear, // Down
    pub w3: Linear, // Up
    pub layer_idx: usize,
}

impl TransformerBlock {
    /// Build transformer block from tensors
    pub fn from_tensors(
        prefix: &str,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        device: Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let get = |name: &str| -> Result<Tensor> {
            let full_name = format!("{}{}", prefix, name);
            tensors
                .get(&full_name)
                .cloned()
                .ok_or_else(|| RIAError::ModelLoading(format!("Missing tensor: {}", full_name)))
        };

        let hidden = config.embedding_length as usize;
        let ffn = config.feed_forward_length as usize;

        // Attention norm
        let attn_norm_weight = get("attn_norm.weight")?;
        let attention_norm = RmsNorm::new(attn_norm_weight, config.layer_norm_rms_epsilon as f64);

        // Q, K, V, O projections
        let wq_weight = get("attn_q.weight")?;
        let wq = Linear::new(wq_weight, None);

        let wk_weight = get("attn_k.weight")?;
        let wk = Linear::new(wk_weight, None);

        let wv_weight = get("attn_v.weight")?;
        let wv = Linear::new(wv_weight, None);

        let wo_weight = get("attn_output.weight")?;
        let wo = Linear::new(wo_weight, None);

        // FFN norm
        let ffn_norm_weight = get("ffn_norm.weight")?;
        let ffn_norm = RmsNorm::new(ffn_norm_weight, config.layer_norm_rms_epsilon as f64);

        // SwiGLU FFN: w1 (gate), w2 (down), w3 (up)
        let w1_weight = get("ffn_gate.weight")?;
        let w1 = Linear::new(w1_weight, None);

        let w2_weight = get("ffn_down.weight")?;
        let w2 = Linear::new(w2_weight, None);

        let w3_weight = get("ffn_up.weight")?;
        let w3 = Linear::new(w3_weight, None);

        Ok(Self {
            attention_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w1,
            w2,
            w3,
            layer_idx,
        })
    }
}

impl RIAModel {
    /// Load model from GGUF file
    pub fn from_gguf(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self> {
        let reader = GGUFReader::open(&path).map_err(RIAError::GGUF)?;

        // Parse config from metadata
        let config = Self::parse_config(&reader)?;

        tracing::info!(
            "Loading {} model: {} layers, {} dim, {} vocab",
            config.architecture,
            config.block_count,
            config.embedding_length,
            config.vocab_size
        );

        // Load all tensors from GGUF
        let tensors = reader.load_all_tensors(&device).map_err(RIAError::GGUF)?;

        let tensor_map: HashMap<String, Tensor> = tensors.into_iter().collect();

        // Build model
        let model = Self::build_from_tensors(&config, &tensor_map, device.clone())?;

        Ok(model)
    }

    /// Parse ModelConfig from GGUF metadata
    fn parse_config(reader: &GGUFReader) -> Result<ModelConfig> {
        let get_str = |key: &str| -> Result<String> {
            match reader.get_metadata(key) {
                Some(ria_gguf::MetadataValue::String(s)) => Ok(s.clone()),
                _ => Err(RIAError::ModelLoading(format!("Missing metadata: {}", key))),
            }
        };

        let get_u32 = |key: &str| -> Result<u32> {
            match reader.get_metadata(key) {
                Some(ria_gguf::MetadataValue::U32(v)) => Ok(*v),
                Some(ria_gguf::MetadataValue::U64(v)) => Ok(*v as u32),
                _ => Err(RIAError::ModelLoading(format!(
                    "Missing/invalid metadata: {}",
                    key
                ))),
            }
        };

        let get_f32 = |key: &str| -> Result<f32> {
            match reader.get_metadata(key) {
                Some(ria_gguf::MetadataValue::F32(v)) => Ok(*v),
                _ => Err(RIAError::ModelLoading(format!(
                    "Missing/invalid metadata: {}",
                    key
                ))),
            }
        };

        let architecture = get_str("general.architecture")?;
        let name = get_str("general.name").unwrap_or_else(|_| "ria-model".to_string());

        let context_length =
            get_u32("llama.context_length").or_else(|_| get_u32("ria.context_length"))?;
        let embedding_length =
            get_u32("llama.embedding_length").or_else(|_| get_u32("ria.embedding_length"))?;
        let block_count = get_u32("llama.block_count").or_else(|_| get_u32("ria.block_count"))?;
        let feed_forward_length =
            get_u32("llama.feed_forward_length").or_else(|_| get_u32("ria.feed_forward_length"))?;
        let attention_head_count = get_u32("llama.attention.head_count")
            .or_else(|_| get_u32("ria.attention.head_count"))?;
        let attention_head_count_kv = get_u32("llama.attention.head_count_kv")
            .or_else(|_| get_u32("ria.attention.head_count_kv"))
            .unwrap_or(attention_head_count);
        let layer_norm_rms_epsilon = get_f32("llama.attention.layer_norm_rms_epsilon")
            .or_else(|_| get_f32("ria.attention.layer_norm_rms_epsilon"))?;
        let rope_freq_base = get_f32("llama.rope.freq_base")
            .or_else(|_| get_f32("ria.rope.freq_base"))
            .unwrap_or(10000.0);
        let vocab_size = get_u32("llama.vocab_size").or_else(|_| get_u32("ria.vocab_size"))?;

        Ok(ModelConfig {
            architecture,
            name,
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            layer_norm_rms_epsilon,
            rope_freq_base,
            vocab_size,
        })
    }

    /// Build model from tensor map
    fn build_from_tensors(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let get_tensor = |name: &str| -> Result<Tensor> {
            tensors
                .get(name)
                .cloned()
                .ok_or_else(|| RIAError::ModelLoading(format!("Missing tensor: {}", name)))
        };

        // Token embeddings - create directly from tensor
        let tok_embed_tensor = get_tensor("token_embd.weight")?;
        let (_vocab_size, embed_dim) = tok_embed_tensor.dims2()?;
        let tok_embeddings = Embedding::new(tok_embed_tensor, embed_dim);

        // Build transformer layers
        let mut layers = Vec::with_capacity(config.block_count as usize);
        for i in 0..config.block_count {
            let prefix = format!("blk.{}.", i);
            let layer = TransformerBlock::from_tensors(
                &prefix,
                config,
                tensors,
                device.clone(),
                i as usize,
            )?;
            layers.push(layer);
        }

        // Final norm
        let norm_weight = get_tensor("output_norm.weight")?;
        let norm = RmsNorm::new(norm_weight, config.layer_norm_rms_epsilon as f64);

        // Output projection
        let output_weight = get_tensor("output.weight")?;
        let output = Linear::new(output_weight, None);

        Ok(Self {
            config: config.clone(),
            device: device.clone(),
            tok_embeddings,
            layers,
            norm,
            output,
            gguf_path: None,
            loaded_weights: HashMap::new(),
        })
    }

    /// Forward pass with KV cache
    pub fn forward(&self, input_ids: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;
        let head_dim = self.config.head_dim() as usize;
        let num_heads = self.config.attention_head_count as usize;
        let num_kv_heads = self.config.kv_heads() as usize;
        let gqa_group_size = num_heads / num_kv_heads;

        // Token embeddings: lookup embeddings for each token ID
        let mut hidden = self
            .tok_embeddings
            .embeddings()
            .index_select(input_ids, 0)?;

        // Process each layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();

            // Attention pre-norm
            let normed = layer.attention_norm.forward(&hidden)?;

            // Q, K, V projections
            let q = layer.wq.forward(&normed)?;
            let k = layer.wk.forward(&normed)?;
            let v = layer.wv.forward(&normed)?;

            // Reshape for attention: (batch, seq, heads, head_dim)
            let q = q
                .reshape(((), seq_len, num_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape(((), seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape(((), seq_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            // Apply RoPE
            let rope_theta = self.config.rope_freq_base;
            let (q, k) = apply_rope(&q, &k, cache.seq_len(), rope_theta)?;

            // Update KV cache
            let (k_cached, v_cached) = cache.update(layer_idx, k.clone(), v.clone())?;

            // GQA: repeat KV heads to match query heads
            // k_cached: (batch, num_kv_heads, seq, head_dim)
            // After repeat_kv: (batch, num_heads, seq, head_dim)
            let k_repeated = repeat_kv(&k_cached, gqa_group_size)?;
            let v_repeated = repeat_kv(&v_cached, gqa_group_size)?;

            // Scaled dot-product attention with causal mask
            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut attn_weights = q.matmul(&k_repeated.transpose(2, 3)?)?;
            attn_weights = (attn_weights * scale)?;

            // Apply causal mask
            let total_seq = k_repeated.dim(2)?;
            if seq_len < total_seq {
                // Generation mode: mask future tokens
                let mask = create_causal_mask(seq_len, total_seq, &self.device)?;
                attn_weights = attn_weights.add(&mask)?;
            }

            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            let attn_output = attn_weights.matmul(&v_repeated)?;

            // Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
            let attn_output = attn_output.transpose(1, 2)?.reshape((
                (),
                seq_len,
                self.config.embedding_length as usize,
            ))?;

            // Output projection
            let attn_output = layer.wo.forward(&attn_output)?;
            hidden = (residual + attn_output)?;

            // FFN pre-norm
            let residual_ffn = hidden.clone();
            let normed_ffn = layer.ffn_norm.forward(&hidden)?;

            // SwiGLU FFN: w2(silu(w1(x)) * w3(x))
            let gate = layer.w1.forward(&normed_ffn)?;
            let gate = candle_nn::ops::silu(&gate)?;
            let up = layer.w3.forward(&normed_ffn)?;
            let hidden_ffn = (gate * up)?;
            let ffn_output = layer.w2.forward(&hidden_ffn)?;

            hidden = (residual_ffn + ffn_output)?;
        }

        // Final norm
        let output = self.norm.forward(&hidden)?;

        // LM head
        let logits = self.output.forward(&output)?;

        cache.increment_seq_len();

        Ok(logits)
    }

    /// Get model size in parameters
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;

        // Embedding
        count += self.config.vocab_size as usize * self.config.embedding_length as usize;

        // Per layer
        let hidden = self.config.embedding_length as usize;
        let ffn = self.config.feed_forward_length as usize;
        let per_layer = hidden * hidden * 4 +  // Q, K, V, O
            hidden * ffn * 3; // w1, w2, w3

        count += per_layer * self.config.block_count as usize;

        // Norms
        count += hidden * self.config.block_count as usize * 2;

        // Output
        count += hidden * self.config.vocab_size as usize;

        count
    }
}

/// Repeat KV heads for Grouped Query Attention
/// Input: (batch, num_kv_heads, seq_len, head_dim)
/// Output: (batch, num_kv_heads * group_size, seq_len, head_dim)
fn repeat_kv(x: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(x.clone());
    }

    let dims = x.shape().dims();
    let batch_size = dims[0];
    let num_kv_heads = dims[1];
    let seq_len = dims[2];
    let head_dim = dims[3];

    // Reshape: (batch, num_kv_heads, 1, seq_len, head_dim)
    let x = x.reshape((batch_size, num_kv_heads, 1, seq_len, head_dim))?;

    // Expand: (batch, num_kv_heads, group_size, seq_len, head_dim)
    let x = x.expand((batch_size, num_kv_heads, group_size, seq_len, head_dim))?;

    // Reshape: (batch, num_kv_heads * group_size, seq_len, head_dim)
    Ok(x.reshape((batch_size, num_kv_heads * group_size, seq_len, head_dim))?)
}

/// Apply RoPE (Rotary Position Embedding)
fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    seq_offset: usize,
    rope_theta: f32,
) -> Result<(Tensor, Tensor)> {
    let dims = q.shape().dims();
    let seq_len = dims[2];
    let head_dim = dims[3];
    let device = q.device();

    // RoPE operates on pairs of dimensions
    let half_dim = head_dim / 2;

    // Create position IDs: [seq_len]
    let positions: Vec<u32> = (seq_offset..(seq_offset + seq_len))
        .map(|i| i as u32)
        .collect();
    let positions = Tensor::from_slice(&positions, seq_len, device)?;

    // Inverse frequency: 1 / (theta^(i/dim)) for i in 0, 2, 4, ..., head_dim-2
    // Shape: [half_dim]
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

    // Compute freq: positions @ inv_freq.T -> [seq_len, half_dim]
    let positions = positions.reshape((seq_len, 1))?;
    let freqs = positions.broadcast_matmul(&inv_freq)?;

    // cos and sin: [1, 1, seq_len, half_dim]
    let cos = freqs.cos()?.reshape((1, 1, seq_len, half_dim))?;
    let sin = freqs.sin()?.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotary embedding using standard interleaved approach
    let q_rope = rope_apply_interleaved(q, &cos, &sin)?;
    let k_rope = rope_apply_interleaved(k, &cos, &sin)?;

    Ok((q_rope, k_rope))
}

/// Apply RoPE with interleaved cos/sin pairs
/// This is the standard LLaMA-style RoPE implementation
fn rope_apply_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(candle_core::D::Minus1)?;
    let half_dim = head_dim / 2;

    // Split x into two halves along the last dimension
    // x = [x[..., :half_dim], x[..., half_dim:]]
    let x_first_half = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let x_second_half = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;

    // Get the shape for broadcasting
    let half_shape = x_first_half.shape().clone();

    // Broadcast cos/sin to match half dimension shape
    let cos_bc = cos.broadcast_as(&half_shape)?;
    let sin_bc = sin.broadcast_as(&half_shape)?;

    // RoPE formula:
    // out_first = x_first * cos - x_second * sin
    // out_second = x_second * cos + x_first * sin
    let out_first =
        ((x_first_half.clone() * cos_bc.clone())? - (x_second_half.clone() * sin_bc.clone())?)?;
    let out_second = ((x_second_half * cos_bc)? + (x_first_half * sin_bc)?)?;

    // Concatenate: [out_first, out_second] -> [batch, heads, seq, head_dim]
    Ok(Tensor::cat(
        &[&out_first, &out_second],
        candle_core::D::Minus1,
    )?)
}

/// Create causal attention mask
fn create_causal_mask(seq_len: usize, total_seq: usize, device: &Device) -> Result<Tensor> {
    // Create mask where future positions are -inf
    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_seq)
                .map(|j| {
                    if j > i + (total_seq - seq_len) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, total_seq), device)?;
    Ok(mask)
}
