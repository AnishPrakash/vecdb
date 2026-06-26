# vecdb
**Native Vector Database in Rust**

![CI](https://github.com/AnishPrakash/vecdb/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

## Performance (SIFT1M 1M x 128-dim)
| Metric | Value |
|--------|-------|
| Recall@10 | > 95.0% |
| P99 Latency | < 1.0 ms |
| SIMD Speedup | 6-8x |
| Memory (SQ8) | 128 MB / 1M vecs |

## Architecture
- **HNSW graph:** probabilistic layers, greedy search, bidirectional links
- **AVX2 SIMD distance:** 8 f32/cycle dot product
- **Write-Ahead Log:** CRC32-verified binary records, crash recovery
- **mmap storage:** out-of-core datasets beyond available RAM
- **Axum REST API:** concurrent RwLock-protected graph, WAL-backed inserts
- **Python SDK:** programmatic database access

## Quick Start
```bash
cargo run --release -p vecdb-server
pip install ./vecdb-python
python -c "from vecdb import VectorDB; db = VectorDB(); print(db.health())"