// vecdb-core/src/quantize.rs

/// Scalar Quantization to 8-bit integers.
/// Store min/scale per-dimension for exact dequantization.
pub struct SQ8Config {
    pub dim: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
}

impl SQ8Config {
    /// Compute quantization parameters from a training set of vectors.
    pub fn fit(vectors: &[Vec<f32>], dim: usize) -> Self {
        let mut mins = vec![f32::MAX; dim];
        let mut maxes = vec![f32::MIN; dim];

        for v in vectors {
            for (d, &x) in v.iter().enumerate() {
                if x < mins[d] { mins[d] = x; }
                if x > maxes[d] { maxes[d] = x; }
            }
        }

        let scales: Vec<f32> = mins.iter().zip(maxes.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range == 0.0 { 1.0 } else { range / 255.0 }
            }).collect();

        SQ8Config { dim, mins, scales }
    }

    /// Quantize an f32 vector to u8.
    pub fn quantize(&self, v: &[f32]) -> Vec<u8> {
        v.iter().enumerate().map(|(d, &x)| {
            let q = (x - self.mins[d]) / self.scales[d];
            q.round().clamp(0.0, 255.0) as u8
        }).collect()
    }

    /// Dequantize u8 back to approximate f32.
    pub fn dequantize(&self, v: &[u8]) -> Vec<f32> {
        v.iter().enumerate().map(|(d, &q)| {
            (q as f32) * self.scales[d] + self.mins[d]
        }).collect()
    }

    /// Approximate cosine similarity directly on quantized vectors.
    pub fn cosine_sq8(a: &[u8], b: &[u8]) -> f32 {
        let (mut dot, mut na, mut nb) = (0u64, 0u64, 0u64);
        
        for (&x, &y) in a.iter().zip(b.iter()) {
            let x_u64 = x as u64;
            let y_u64 = y as u64;
            
            dot += x_u64 * y_u64;
            na += x_u64 * x_u64;
            nb += y_u64 * y_u64;
        }
        
        if na == 0 || nb == 0 { return 0.0; }
        (dot as f32) / ((na as f32).sqrt() * (nb as f32).sqrt())
    }
}