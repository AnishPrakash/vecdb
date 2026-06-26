// vecdb-core/src/wal.rs
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write, Read, BufReader};
use serde::{Serialize, Deserialize};
use crc32fast::Hasher;

// The serialized records we save to disk
#[derive(Serialize, Deserialize)]
pub enum WalRecord {
    Insert { id: u64, vector: Vec<f32>, payload: serde_json::Value },
    Delete { id: u64 },
}

pub struct Wal {
    writer: BufWriter<File>,
}

impl Wal {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Wal { writer: BufWriter::new(file) })
    }

    /// Append a record: [4-byte len] [payload bytes] [4-byte CRC32]
    pub fn append(&mut self, rec: &WalRecord) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(rec)?;
        let mut h = Hasher::new();
        h.update(&bytes);
        let crc = h.finalize();
        
        let len = bytes.len() as u32;
        
        // Write exactly 4 bytes for length, then the data, then 4 bytes for the checksum
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.flush()?;
        
        Ok(())
    }

    /// Replay all valid records from WAL file.
    pub fn replay(path: &str) -> std::io::Result<Vec<WalRecord>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(vec![]), // Return empty if WAL doesn't exist yet
        };
        
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut len_buf = [0u8; 4];
        let mut crc_buf = [0u8; 4];

        loop {
            // Read length header
            if reader.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf) as usize;
            
            // Sanity check length (e.g., max 64MB per record)
            if len == 0 || len > 64 * 1024 * 1024 { break; }
            
            // Read payload and checksum
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() { break; }
            if reader.read_exact(&mut crc_buf).is_err() { break; }
            
            // Verify CRC32 checksum to ensure no disk corruption
            let mut h = Hasher::new();
            h.update(&payload);
            if h.finalize() != u32::from_le_bytes(crc_buf) {
                eprintln!("WAL CRC mismatch, truncating replay here");
                break;
            }
            
            // Deserialize and push
            if let Ok(rec) = serde_json::from_slice(&payload) {
                records.push(rec);
            }
        }
        
        Ok(records)
    }
}