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