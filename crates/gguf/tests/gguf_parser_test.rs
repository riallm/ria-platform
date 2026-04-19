//! Unit tests for GGUF parser

#[cfg(test)]
mod tests {
    use ria_gguf::{dequantize_tensor, GGUFQuantizationType};

    #[test]
    fn test_q4_0_dequantization() {
        // Create a simple Q4_0 block: 32 values, 18 bytes
        // Scale = 1.0 (f16), qs = [0x00, 0x11, 0x22, ...]
        let mut data = vec![0u8; 18];

        // Scale = 1.0 as f16
        let scale_f16 = half::f16::from_f32(1.0).to_bits();
        data[0] = (scale_f16 & 0xFF) as u8;
        data[1] = ((scale_f16 >> 8) & 0xFF) as u8;

        // Fill qs with known pattern: all 8s (value 0 after -8 offset)
        for i in 2..18 {
            data[i] = 0x88; // Two 8s per byte
        }

        let result = dequantize_tensor(&data, GGUFQuantizationType::Q4_0, 32).unwrap();

        assert_eq!(result.len(), 32);
        // All values should be (8 - 8) * 1.0 = 0.0
        for val in &result {
            assert!((val - 0.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_dequantization() {
        // Create F16 data: 4 values = 8 bytes
        let mut data = vec![0u8; 8];

        // Values: 1.0, 2.0, 3.0, 4.0
        let values = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &v) in values.iter().enumerate() {
            let bits = half::f16::from_f32(v).to_bits();
            data[i * 2] = (bits & 0xFF) as u8;
            data[i * 2 + 1] = ((bits >> 8) & 0xFF) as u8;
        }

        let result = dequantize_tensor(&data, GGUFQuantizationType::F16, 4).unwrap();

        assert_eq!(result.len(), 4);
        for (i, &expected) in values.iter().enumerate() {
            assert!((result[i] - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_f32_dequantization() {
        // Create F32 data: 4 values = 16 bytes
        let mut data = vec![0u8; 16];

        let values = [1.5f32, -2.5, 0.0, 100.0];
        for (i, &v) in values.iter().enumerate() {
            let bytes = v.to_le_bytes();
            data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        let result = dequantize_tensor(&data, GGUFQuantizationType::F32, 4).unwrap();

        assert_eq!(result.len(), 4);
        for (i, &expected) in values.iter().enumerate() {
            assert!((result[i] - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_q8_0_dequantization() {
        // Create Q8_0 block: 32 values, 34 bytes
        // Scale = 1.0, qs = signed bytes
        let mut data = vec![0u8; 34];

        // Scale = 1.0 as f16
        let scale_f16 = half::f16::from_f32(1.0).to_bits();
        data[0] = (scale_f16 & 0xFF) as u8;
        data[1] = ((scale_f16 >> 8) & 0xFF) as u8;

        // Fill qs with pattern: 0, 1, 2, ..., 31
        for i in 0..32 {
            data[2 + i] = i as u8;
        }

        let result = dequantize_tensor(&data, GGUFQuantizationType::Q8_0, 32).unwrap();

        assert_eq!(result.len(), 32);
        // Values should be i * 1.0 for i in 0..32
        for i in 0..32 {
            assert!((result[i] - i as f32).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantization_type_from_u32() {
        assert_eq!(
            GGUFQuantizationType::from_u32(0),
            Some(GGUFQuantizationType::F32)
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(1),
            Some(GGUFQuantizationType::F16)
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(2),
            Some(GGUFQuantizationType::Q4_0)
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(3),
            Some(GGUFQuantizationType::Q4_1)
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(8),
            Some(GGUFQuantizationType::Q8_0)
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(12),
            Some(GGUFQuantizationType::Q4_K)
        );
        assert_eq!(GGUFQuantizationType::from_u32(99), None);
    }

    #[test]
    fn test_block_size_calculations() {
        assert_eq!(GGUFQuantizationType::F32.block_size(), 4);
        assert_eq!(GGUFQuantizationType::F16.block_size(), 2);
        assert_eq!(GGUFQuantizationType::Q4_0.block_size(), 18);
        assert_eq!(GGUFQuantizationType::Q8_0.block_size(), 34);
        assert_eq!(GGUFQuantizationType::Q4_K.block_size(), 144);
    }

    #[test]
    fn test_quant_per_block() {
        assert_eq!(GGUFQuantizationType::F32.quant_per_block(), 1);
        assert_eq!(GGUFQuantizationType::F16.quant_per_block(), 1);
        assert_eq!(GGUFQuantizationType::Q4_0.quant_per_block(), 32);
        assert_eq!(GGUFQuantizationType::Q8_0.quant_per_block(), 32);
        assert_eq!(GGUFQuantizationType::Q4_K.quant_per_block(), 256);
    }
}
