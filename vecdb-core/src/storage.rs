// vecdb-core/src/storage.rs
use memmap2::MmapMut;
use std::fs::OpenOptions;

/// Memory-mapped flat vector storage.
/// Layout: [n_vecs: u64][dim: u64][f32 ... f32]
pub struct MmapStorage {
    pub n_vecs: usize,
    pub dim: usize,
    mmap: MmapMut,
}

const HEADER_BYTES: usize = 16; // 2 x u64 (8 bytes each)

impl MmapStorage {
    pub fn create(path: &str, capacity: usize, dim: usize) -> std::io::Result<Self> {
        // total bytes = header + (capacity * dimensions * 4 bytes per f32)
        let total = HEADER_BYTES + capacity * dim * 4;
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
            
        file.set_len(total as u64)?;
        
        // unsafe is required here because another process *could* modify the file 
        // underneath us, but we assume exclusive access for our database engine.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Write initial header (0 vectors, and our dimension size)
        mmap[0..8].copy_from_slice(&(0u64).to_le_bytes());
        mmap[8..16].copy_from_slice(&(dim as u64).to_le_bytes());
        
        Ok(MmapStorage { n_vecs: 0, dim, mmap })
    }

    pub fn append_vector(&mut self, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim);
        
        // Calculate where this specific vector should start in the byte array
        let offset = HEADER_BYTES + self.n_vecs * self.dim * 4;
        
        for (i, &v) in vec.iter().enumerate() {
            let o = offset + (i * 4);
            self.mmap[o..o+4].copy_from_slice(&v.to_le_bytes());
        }
        
        // Update the global vector count in memory and on disk
        self.n_vecs += 1;
        self.mmap[0..8].copy_from_slice(&(self.n_vecs as u64).to_le_bytes());
        self.mmap.flush().ok();
    }

    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        let offset = HEADER_BYTES + idx * self.dim * 4;
        
        (0..self.dim).map(|i| {
            let o = offset + (i * 4);
            f32::from_le_bytes(self.mmap[o..o+4].try_into().unwrap())
        }).collect()
    }
}