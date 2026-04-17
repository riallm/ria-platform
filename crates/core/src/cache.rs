//! KV Cache for efficient autoregressive generation

use candle_core::{Tensor, Device, D};
use crate::error::Result;

/// KV Cache stores key/value tensors for efficient generation
pub struct KVCache {
    /// Key cache per layer: Vec<(layer_idx, Tensor)>
    key_cache: Vec<Option<Tensor>>,
    
    /// Value cache per layer
    value_cache: Vec<Option<Tensor>>,
    
    /// Current sequence length (tokens generated so far)
    seq_len: usize,
    
    /// Number of layers
    num_layers: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(num_layers: usize) -> Self {
        Self {
            key_cache: vec![None; num_layers],
            value_cache: vec![None; num_layers],
            seq_len: 0,
            num_layers,
        }
    }
    
    /// Get cached keys for a layer
    pub fn get_keys(&self, layer_idx: usize) -> Option<&Tensor> {
        self.key_cache.get(layer_idx).and_then(|opt| opt.as_ref())
    }
    
    /// Get cached values for a layer
    pub fn get_values(&self, layer_idx: usize) -> Option<&Tensor> {
        self.value_cache.get(layer_idx).and_then(|opt| opt.as_ref())
    }
    
    /// Update cache with new key/value tensors
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if let (Some(existing_keys), Some(existing_values)) = 
            (&self.key_cache[layer_idx], &self.value_cache[layer_idx]) 
        {
            // Concatenate with existing cache along seq dimension (dim 2)
            let updated_keys = candle_core::Tensor::cat(&[existing_keys, &new_keys], 2)?;
            let updated_values = candle_core::Tensor::cat(&[existing_values, &new_values], 2)?;
            
            self.key_cache[layer_idx] = Some(updated_keys.clone());
            self.value_cache[layer_idx] = Some(updated_values.clone());
            
            Ok((updated_keys, updated_values))
        } else {
            // First tokens - no concatenation needed
            self.key_cache[layer_idx] = Some(new_keys.clone());
            self.value_cache[layer_idx] = Some(new_values.clone());
            
            Ok((new_keys, new_values))
        }
    }
    
    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    
    /// Update sequence length
    pub fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }
    
    /// Increment sequence length
    pub fn increment_seq_len(&mut self) {
        self.seq_len += 1;
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        self.key_cache = vec![None; self.num_layers];
        self.value_cache = vec![None; self.num_layers];
        self.seq_len = 0;
    }
    
    /// Get memory usage estimate (bytes)
    pub fn memory_usage_bytes(&self) -> u64 {
        let mut total = 0u64;
        
        for cache in [&self.key_cache, &self.value_cache] {
            for tensor_opt in cache.iter().flatten() {
                let shape = tensor_opt.shape().dims().to_vec();
                let elem_count: usize = shape.iter().product();
                total += (elem_count * 2) as u64; // Assuming F16
            }
        }
        
        total
    }
}

/// Cache layer for a single transformer layer
pub struct CacheLayer {
    pub key: Tensor,
    pub value: Tensor,
}

impl CacheLayer {
    pub fn new(key: Tensor, value: Tensor) -> Self {
        Self { key, value }
    }
}
