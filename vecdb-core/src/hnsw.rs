use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

const EF_CONSTRUCTION: usize = 50;

#[derive(Clone)]
pub struct Node {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: serde_json::Value,
    pub neighbours: Vec<Vec<u64>>,
    pub max_layer: usize,
}

pub struct HnswIndex {
    pub dim: usize,
    pub nodes: HashMap<u64, Node>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
    pub ef_construction: usize,
    pub tombstones: HashSet<u64>,
}

#[derive(Clone, Copy, PartialEq)]
struct Candidate(pub u64, pub f32);

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(Ordering::Equal)
    }
}

impl HnswIndex {
    pub fn new(dim: usize) -> Self {
        HnswIndex {
            dim,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            ef_construction: EF_CONSTRUCTION,
            tombstones: HashSet::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len() - self.tombstones.len()
    }

    pub fn delete(&mut self, id: u64) {
        if self.nodes.contains_key(&id) {
            self.tombstones.insert(id);
        }
    }

    fn dist(&self, query: &[f32], id: u64) -> f32 {
        let node_vec = &self.nodes[&id].vector;
        crate::distance::cosine_simd(query, node_vec)
    }

    fn random_level(&self) -> usize {
        let mut level = 0;
        while rand::random::<f32>() < 0.5 && level < 16 {
            level += 1;
        }
        level
    }

    fn matches_filter(payload: &serde_json::Value, filter: &serde_json::Value) -> bool {
        // Force the server to print what it's comparing
        let result = if let (Some(f_obj), Some(p_obj)) = (filter.as_object(), payload.as_object()) {
            let mut matched = true;
            for (k, v) in f_obj {
                if p_obj.get(k) != Some(v) { 
                    matched = false; 
                    break; 
                }
            }
            matched
        } else {
            false
        };
        
        // This log will reveal exactly which node is lying to us
        if result == true {
             println!("DEBUG: MATCH FOUND! Payload: {:?} vs Filter: {:?}", payload, filter);
        }
        
        result
    }

    fn search_layer(
        &self,
        query: &[f32],
        ep: u64,
        ef: usize,
        layer: usize,
        filter: Option<&serde_json::Value>,
    ) -> BinaryHeap<Candidate> {
        if self.nodes.is_empty() || !self.nodes.contains_key(&ep) {
            return BinaryHeap::new();
        }

        let mut visited = HashSet::new();
        let ep_dist = self.dist(query, ep);
        let mut candidates = BinaryHeap::new(); 
        let mut results = BinaryHeap::new();    

        candidates.push(std::cmp::Reverse(Candidate(ep, ep_dist)));
        
        let ep_node = &self.nodes[&ep];
        let is_match = filter.map_or(true, |f| Self::matches_filter(&ep_node.payload, f));
        if is_match && !self.tombstones.contains(&ep) {
            results.push(Candidate(ep, ep_dist));
        }
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
                    candidates.push(std::cmp::Reverse(Candidate(nbr_id, d)));
                    
                    let nbr_node = &self.nodes[&nbr_id];
                    let matches = filter.map_or(true, |f| Self::matches_filter(&nbr_node.payload, f));
                    
                    if matches && !self.tombstones.contains(&nbr_id) {
                        results.push(Candidate(nbr_id, d));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        results
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>, payload: serde_json::Value) {
        let level = self.random_level();
        self.nodes.insert(id, Node {
            id,
            vector: vector.clone(),
            payload,
            neighbours: vec![vec![]; level + 1],
            max_layer: level,
        });

        let mut curr_ep = match self.entry_point {
            Some(ep) => ep,
            None => {
                self.entry_point = Some(id);
                self.max_layer = level;
                return;
            }
        };

        for layer in (level + 1..=self.max_layer).rev() {
            let cands = self.search_layer(&vector, curr_ep, 1, layer, None);
            curr_ep = cands.into_iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0).unwrap_or(curr_ep);
        }

        for layer in (0..=level.min(self.max_layer)).rev() {
            let mut candidates = self.search_layer(&vector, curr_ep, self.ef_construction, layer, None);
            let mut nbrs = Vec::new();
            while let Some(c) = candidates.pop() {
                nbrs.push(c.0);
                if nbrs.len() >= 16 { break; } 
            }
            
            // SAFE PUSH: use get_mut to avoid index out of bounds
            for &nbr_id in &nbrs {
                if let Some(node) = self.nodes.get_mut(&nbr_id) {
                    if let Some(layer_nbrs) = node.neighbours.get_mut(layer) {
                        layer_nbrs.push(id);
                    }
                }
            }
            
            // SAFE ASSIGN: use get_mut
            if let Some(node) = self.nodes.get_mut(&id) {
                if let Some(layer_nbrs) = node.neighbours.get_mut(layer) {
                    *layer_nbrs = nbrs;
                }
            }
            
            if layer == 0 { break; }
            
            // SAFE UPDATE
            if let Some(node) = self.nodes.get(&id) {
                if let Some(layer_0_nbrs) = node.neighbours.get(layer) {
                    if let Some(&first_nbr) = layer_0_nbrs.first() {
                        curr_ep = first_nbr;
                    }
                }
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }
    }
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        filter: Option<&serde_json::Value>,
    ) -> Vec<(u64, f32)> {
        let Some(ep) = self.entry_point else { return vec![]; };
        let mut curr_ep = ep;

        for layer in (1..=self.max_layer).rev() {
            let cands = self.search_layer(query, curr_ep, 1, layer, None);
            curr_ep = cands.into_iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|c| c.0).unwrap_or(curr_ep);
        }

        let results = self.search_layer(query, curr_ep, ef.max(top_k), 0, filter);
        
        let mut sorted: Vec<Candidate> = results.into_iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        sorted.into_iter()
            .take(top_k)
            .map(|c| {
                let score = (1.0-c.1).max(0.0);
                (c.0, score)
            })
            .collect()
    }
}