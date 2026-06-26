// vecdb-core/src/distance.rs
// All distance/similarity metrics

/// Cosine similarity: 1.0 = identical, -1.0 = opposite.
/// Assumes vectors are NOT pre-normalised.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// L2 squared distance (no sqrt, monotone for NN search).
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Inner (dot) product used when vectors are pre-normalised.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_zero_distance() {
        let v = vec![3.0_f32, 4.0];
        assert!(l2_sq(&v, &v) < 1e-6);
    }
}
#[cfg(target_arch = "x86_64")]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let n = a.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();
        
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        
        // Horizontal sum of 8 lanes
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let sum = _mm_add_ps(lo, hi);
        let s2 = _mm_movehl_ps(sum, sum);
        let s3 = _mm_add_ps(sum, s2);
        let s4 = _mm_shuffle_ps(s3, s3, 0x55);
        let s5 = _mm_add_ss(s3, s4);
        let mut result = _mm_cvtss_f32(s5);
        
        // Handle remainder for dimensions not perfectly divisible by 8
        for i in (chunks * 8)..n {
            result += a[i] * b[i];
        }
        
        result
    }

    // Scalar fallback for older CPUs or when AVX2 is not enabled at compile time
    #[cfg(not(target_feature = "avx2"))]
    {
        dot_product(a, b)
    }
}
#[cfg(target_arch = "x86_64")]
pub fn cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = dot_product_simd(a, a).sqrt();
    let norm_b = dot_product_simd(b, b).sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 { 
        0.0 
    } else { 
        dot / (norm_a * norm_b) 
    }
}