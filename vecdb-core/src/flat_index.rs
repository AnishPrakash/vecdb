// vecdb-core/src/flat_index.rs
use crate::distance::cosine_similarity;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// Min-heap wrapper (BinaryHeap is max-heap by default in Rust)
// We reverse the Ord implementation to keep the Top-K highest similarity scores.
#[derive(PartialEq)]
struct Ranked(f32, u64); // (score, id)

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the comparison to create a min-heap
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

pub struct FlatIndex {
    dim: usize,
    vectors: Vec<f32>, // flat array: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...]
    ids: Vec<u64>,
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self {
        FlatIndex { 
            dim, 
            vectors: Vec::new(), 
            ids: Vec::new() 
        }
    }

    pub fn insert(&mut self, id: u64, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim);
        self.ids.push(id);
        self.vectors.extend_from_slice(vec);
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut heap: BinaryHeap<Ranked> = BinaryHeap::new();
        let n = self.ids.len();
        
        for i in 0..n {
            let start = i * self.dim;
            let v = &self.vectors[start..start + self.dim];
            let score = cosine_similarity(query, v);
            
            heap.push(Ranked(score, self.ids[i]));
            if heap.len() > top_k { 
                heap.pop(); 
            }
        }
        
        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|r| (r.1, r.0)).collect();
        // Sort results descending (highest similarity first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}