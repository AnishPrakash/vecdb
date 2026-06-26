use ndarray::{Array2, ArrayView1};
use linfa_clustering::KMeans;
use linfa::prelude::*;

pub struct Quantizer {
    pub m: usize,         // Number of sub-vectors
    pub k: usize,         // Number of centroids per codebook
    pub codebooks: Vec<Array2<f32>>, // M codebooks of size (K x D*)
}

impl Quantizer {
    // Train the codebooks using your existing dataset
    pub fn train(data: &Array2<f32>, m: usize, k: usize) -> Self {
        let dim = data.ncols();
        let sub_dim = dim / m;
        let mut codebooks = Vec::with_capacity(m);

        for i in 0..m {
            // Slice the data into sub-vectors
            let sub_data = data.slice(ndarray::s![.., i*sub_dim..(i+1)*sub_dim]);
            
            // Train K-Means
            let dataset = Dataset::from(sub_data.to_owned());
            let model = KMeans::params(k)
                .fit(&dataset)
                .expect("KMeans training failed");
            
            codebooks.push(model.centroids().clone());
        }

        Quantizer { m, k, codebooks }
    }

    // Compress a vector into code indices (u8)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = vector.len() / self.m;
        let mut codes = Vec::with_capacity(self.m);

        for i in 0..self.m {
            let sub = &vector[i*sub_dim..(i+1)*sub_dim];
            // Find nearest centroid (simple Euclidean distance)
            let mut best_idx = 0;
            let mut min_dist = f32::MAX;
            for (idx, centroid) in self.codebooks[i].rows().into_iter().enumerate() {
                let dist = centroid.iter().zip(sub).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }
            codes.push(best_idx as u8);
        }
        codes
    }
    pub fn compute_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = query.len() / self.m;
        self.codebooks.iter().enumerate().map(|(i, cb)| {
            let sub_q = &query[i*sub_dim..(i+1)*sub_dim];
            cb.rows().into_iter().map(|centroid| {
                centroid.iter().zip(sub_q).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            }).collect()
        }).collect()
    }

    pub fn estimate_distance(&self, codes: &[u8], lut: &[Vec<f32>]) -> f32 {
        let mut total_dist = 0.0;
        for (m, &code) in codes.iter().enumerate() {
            total_dist += lut[m][code as usize];
        }
        total_dist
    }
}