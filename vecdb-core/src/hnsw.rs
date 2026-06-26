// vecdb-core/src/hnsw.rs
use crate::distance::cosine_similarity;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// Key HNSW Parameters
pub const M: usize = 16;
pub const M0: usize = 32;
pub const EF_CONSTRUCTION: usize = 200;

// Candidate wrapper for heap ordering (id, distance)
#[derive(Clone, PartialEq)]
pub struct Candidate(pub u64, pub f32); 

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(Ordering::Equal)
    }
}

// A node in the HNSW graph
pub struct Node {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: serde_json::Value, // arbitrary metadata
    pub neighbours: Vec<Vec<u64>>,  // neighbours[layer] -> list of neighbour IDs
    pub max_layer: usize,
}

// The HNSW index
pub struct HnswIndex {
    pub dim: usize,
    pub nodes: HashMap<u64, Node>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
    pub ef_construction: usize,
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        HnswIndex {
            dim,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            ef_construction: EF_CONSTRUCTION,
        }
    }

    pub fn len(&self) -> usize { self.nodes.len() }

    // Sample the max layer for a new node using exponential decay
    fn random_level(&self) -> usize {
        let ml = 1.0 / (M as f64).ln();
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * ml).floor() as usize
    }

    // Distance from query to a node with a given id
    fn dist(&self, query: &[f32], id: u64) -> f32 {
        let node = &self.nodes[&id];
        // Convert cosine similarity to distance (1 - sim)
        1.0 - cosine_similarity(query, &node.vector)
    }
    // Greedy search at a single layer
    fn search_layer(
        &self,
        query: &[f32],
        ep: u64,
        ef: usize,
        layer: usize,
    ) -> BinaryHeap<Candidate> {
        let mut visited = HashSet::new();
        let ep_dist = self.dist(query, ep);
        let mut candidates = BinaryHeap::new(); // min-heap by dist
        let mut results = BinaryHeap::new();    // max-heap by dist

        candidates.push(std::cmp::Reverse(Candidate(ep, ep_dist)));
        results.push(Candidate(ep, ep_dist));
        visited.insert(ep);

        while let Some(std::cmp::Reverse(curr)) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if curr.1 > worst.1 && results.len() >= ef { break; }
            }

            let nbrs = {
                let node = &self.nodes[&curr.0];
                if layer < node.neighbours.len() {
                    node.neighbours[layer].clone()
                } else {
                    vec![]
                }
            };

            for nbr_id in nbrs {
                if visited.insert(nbr_id) {
                    let d = self.dist(query, nbr_id);
                    let worst_dist = results.peek().map(|c| c.1).unwrap_or(f32::MAX);
                    
                    if d < worst_dist || results.len() < ef {
                        candidates.push(std::cmp::Reverse(Candidate(nbr_id, d)));
                        results.push(Candidate(nbr_id, d));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        results
    }

    // Insert a new vector into the index
    pub fn insert(
        &mut self,
        id: u64,
        vector: Vec<f32>,
        payload: serde_json::Value,
    ) {
        let new_level = self.random_level();
        let max_m = |layer: usize| if layer == 0 { M0 } else { M };

        let mut ep = match self.entry_point {
            Some(ep) => ep,
            None => {
                // First node
                let node = Node {
                    id, vector, payload,
                    neighbours: vec![vec![]; new_level + 1],
                    max_layer: new_level,
                };
                self.nodes.insert(id, node);
                self.entry_point = Some(id);
                self.max_layer = new_level;
                return;
            }
        };

        let q = vector.clone();

        // Phase 1: Descend from top layer to new_level + 1
        for layer in (new_level + 1..=self.max_layer).rev() {
            let candidates = self.search_layer(&q, ep, 1, layer);
            ep = candidates.into_iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0).unwrap_or(ep);
        }

        // Phase 2: from min(new_level, max_layer) down to 0
        let mut new_nbrs: Vec<Vec<u64>> = vec![vec![]; new_level + 1];
        
        for layer in (0..=new_level.min(self.max_layer)).rev() {
            let ef = self.ef_construction;
            let mut candidates = self.search_layer(&q, ep, ef, layer);
            
            let limit = max_m(layer);
            let mut sorted: Vec<Candidate> = candidates.drain().collect();
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let selected: Vec<Candidate> = sorted.into_iter().take(limit).collect();
            new_nbrs[layer] = selected.iter().map(|c| c.0).collect();
            
            ep = new_nbrs[layer].first().copied().unwrap_or(ep);

            // Add back-links
            for c in &selected {
                if let Some(nbr_node) = self.nodes.get_mut(&c.0) {
                    if layer < nbr_node.neighbours.len() {
                        nbr_node.neighbours[layer].push(id);
                        let nbr_limit = max_m(layer);
                        if nbr_node.neighbours[layer].len() > nbr_limit {
                            nbr_node.neighbours[layer].truncate(nbr_limit);
                        }
                    }
                }
            }
        }

        let node = Node {
            id, vector, payload,
            neighbours: new_nbrs,
            max_layer: new_level,
        };
        self.nodes.insert(id, node);

        if new_level > self.max_layer {
            self.max_layer = new_level;
            self.entry_point = Some(id);
        }
    }

    // Query the index for top-k nearest neighbours
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
    ) -> Vec<(u64, f32)> {
        let Some(ep) = self.entry_point else { return vec![]; };
        let mut curr_ep = ep;

        for layer in (1..=self.max_layer).rev() {
            let cands = self.search_layer(query, curr_ep, 1, layer);
            curr_ep = cands.into_iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0).unwrap_or(curr_ep);
        }

        let results = self.search_layer(query, curr_ep, ef.max(top_k), 0);
        let mut sorted: Vec<Candidate> = results.into_iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        sorted.into_iter()
            .take(top_k)
            .map(|c| (c.0, 1.0 - c.1)) // back to similarity
            .collect()
    }
}