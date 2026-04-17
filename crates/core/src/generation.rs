//! Generation loop with sampling strategies

use candle_core::{Tensor, Device, D};
use crate::config::GenerationConfig;
use crate::error::{Result, RIAError};
use crate::model::RIAModel;
use crate::cache::KVCache;

/// Token generation output
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    StopToken,
    MaxTokens,
    StopSequence(String),
}

/// Token generator with sampling
pub struct Generator {
    config: GenerationConfig,
    rng_state: u64,
    /// Recent token IDs for repeat penalty
    recent_tokens: Vec<u32>,
}

impl Generator {
    pub fn new(config: GenerationConfig) -> Self {
        let seed = config.seed.unwrap_or(42);
        Self {
            config,
            rng_state: seed,
            recent_tokens: Vec::new(),
        }
    }

    /// Generate tokens from prompt
    pub fn generate(
        &mut self,
        model: &RIAModel,
        prompt_tokens: &[u32],
    ) -> Result<GenerationOutput> {
        let device = &model.device;
        let mut cache = KVCache::new(model.config.block_count as usize);

        // Process prompt
        let prompt_tensor = Tensor::from_slice(prompt_tokens, prompt_tokens.len(), device)?
            .reshape((1, prompt_tokens.len()))?;

        // Forward pass on prompt
        let logits = model.forward(&prompt_tensor, &mut cache)?;

        // Get last token logits
        let seq_len = prompt_tokens.len();
        let last_logits = logits.narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;

        // Initialize recent tokens with prompt tokens (for repeat penalty)
        let start_idx = prompt_tokens.len().saturating_sub(self.config.repeat_last_n);
        self.recent_tokens = prompt_tokens[start_idx..].to_vec();

        // Sample first token
        let mut tokens = prompt_tokens.to_vec();
        let mut logprobs = if self.config.logprobs { Some(vec![]) } else { None };

        let mut current_logits = last_logits;
        let mut finish_reason = FinishReason::MaxTokens;

        // Get EOS token ID from model config
        let eos_token_id = self.get_eos_token_id(model);

        for _step in 0..self.config.max_new_tokens {
            // Sample next token
            let (next_token, logprob) = self.sample(&current_logits)?;

            if let Some(ref mut lp) = logprobs {
                lp.push(logprob);
            }

            // Check for EOS token
            if let Some(eos_id) = eos_token_id {
                if next_token == eos_id {
                    finish_reason = FinishReason::StopToken;
                    break;
                }
            }

            // Check stop sequences (decode recent tokens and check)
            if !self.config.stop_sequences.is_empty() {
                // Simple check: if any stop sequence token appears
                // For full implementation, would decode and do string matching
            }

            tokens.push(next_token);
            self.recent_tokens.push(next_token);

            // Trim recent tokens to repeat_last_n
            if self.recent_tokens.len() > self.config.repeat_last_n {
                self.recent_tokens.remove(0);
            }

            // Forward pass for single token
            let next_tensor = Tensor::from_slice(&[next_token], 1, device)?
                .reshape((1, 1))?;

            let logits = model.forward(&next_tensor, &mut cache)?;
            current_logits = logits.squeeze(0)?.squeeze(0)?
                .narrow(0, 0, 1)?.squeeze(0)?;
        }

        Ok(GenerationOutput {
            tokens: tokens[prompt_tokens.len()..].to_vec(),
            logprobs,
            finish_reason,
        })
    }

    /// Get EOS token ID from model metadata
    fn get_eos_token_id(&self, model: &RIAModel) -> Option<u32> {
        // Try to get from GGUF metadata if available
        // For now, common defaults:
        // LLaMA: 2
        // GPT-2: 50256
        // Default to 2 if not specified
        Some(2)
    }

    /// Sample token from logits
    fn sample(&mut self, logits: &Tensor) -> Result<(u32, f32)> {
        let logits_vec = logits.to_vec1::<f32>()?;
        let _vocab_size = logits_vec.len();

        // Apply temperature
        let mut logits_vec = if self.config.temperature > 0.0 {
            logits_vec.iter()
                .map(|&x| x / self.config.temperature as f32)
                .collect::<Vec<_>>()
        } else {
            logits_vec
        };

        // Apply penalties using recent token history
        logits_vec = self.apply_penalties(&logits_vec);

        // Sample based on config
        if self.config.temperature == 0.0 {
            // Greedy
            let (idx, &max_val) = logits_vec.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .ok_or_else(|| RIAError::Generation("Empty logits".to_string()))?;

            Ok((idx as u32, max_val))
        } else if let Some(top_k) = self.config.top_k {
            // Top-k sampling
            self.sample_top_k(&logits_vec, top_k)
        } else if let Some(top_p) = self.config.top_p {
            // Top-p (nucleus) sampling
            self.sample_top_p(&logits_vec, top_p)
        } else {
            // Full softmax sampling
            self.sample_full(&logits_vec)
        }
    }

    /// Apply repetition and presence/frequency penalties
    fn apply_penalties(&self, logits: &[f32]) -> Vec<f32> {
        if self.config.repeat_penalty == 1.0
            && self.config.presence_penalty == 0.0
            && self.config.frequency_penalty == 0.0
        {
            return logits.to_vec();
        }

        // Count token frequencies in recent history
        let mut freq = std::collections::HashMap::new();
        for &token in &self.recent_tokens {
            *freq.entry(token as usize).or_insert(0u32) += 1;
        }

        // Apply penalties
        let mut penalized = logits.to_vec();
        for (token_id, count) in &freq {
            if *token_id < penalized.len() {
                let presence = if *count > 0 { 1.0 } else { 0.0 };
                let frequency = *count as f32;

                penalized[*token_id] -= self.config.presence_penalty * presence
                    + self.config.frequency_penalty * frequency;

                if self.config.repeat_penalty > 1.0 {
                    penalized[*token_id] /= self.config.repeat_penalty;
                }
            }
        }

        penalized
    }

    /// Top-k sampling
    fn sample_top_k(&mut self, logits: &[f32], k: usize) -> Result<(u32, f32)> {
        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        // Softmax over top-k
        let max_val = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = indexed.iter()
            .map(|(_, v)| (v - max_val).exp())
            .collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Sample
        let idx = self.sample_categorical(&probs);
        let (orig_idx, logit_val) = indexed[idx];
        let logprob = logit_val - max_val - sum_exp.ln();

        Ok((orig_idx as u32, logprob))
    }

    /// Top-p (nucleus) sampling
    fn sample_top_p(&mut self, logits: &[f32], p: f64) -> Result<(u32, f32)> {
        // Sort logits descending
        let mut indexed: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Softmax
        let max_val = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = indexed.iter()
            .map(|(_, v)| (v - max_val).exp())
            .collect();
        let sum_exp: f32 = exps.iter().sum();
        let mut probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Cumulative sum and truncate
        let mut cumsum = 0.0f32;
        let mut keep_count = probs.len();
        for (i, prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > p as f32 {
                keep_count = i + 1;
                break;
            }
        }
        probs.truncate(keep_count);
        indexed.truncate(keep_count);

        // Renormalize
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

        // Sample
        let idx = self.sample_categorical(&probs);
        let (orig_idx, logit_val) = indexed[idx];
        let logprob = logit_val - max_val - sum_exp.ln();

        Ok((orig_idx as u32, logprob))
    }

    /// Full softmax sampling
    fn sample_full(&mut self, logits: &[f32]) -> Result<(u32, f32)> {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        let idx = self.sample_categorical(&probs);
        let logprob = logits[idx] - max_val - sum_exp.ln();

        Ok((idx as u32, logprob))
    }

    /// Sample from categorical distribution
    fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        let rand_val = self.rand_f32();
        let mut cumsum = 0.0f32;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val <= cumsum {
                return i;
            }
        }

        probs.len() - 1
    }

    /// Simple LCG random number generator
    fn rand_f32(&mut self) -> f32 {
        // Linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 33) as f32 / (u32::MAX as f32)
    }
}
