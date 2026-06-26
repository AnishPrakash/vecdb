// vecdb-core/benches/hnsw_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};
use vecdb_core::hnsw::HnswIndex;
use rand::{Rng, SeedableRng};

fn bench_insert(c: &mut Criterion) {
    // We use a deterministic seed so our benchmarks are consistent
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let dim = 128;
    let mut index = HnswIndex::new(dim);

    // Pre-insert 10K vectors to build a baseline graph
    for i in 0u64..10_000 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        index.insert(i, v, serde_json::Value::Null);
    }

    let mut id = 10_000u64;
    c.bench_function("hnsw_insert_10k", |b| {
        b.iter(|| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            index.insert(id, v, serde_json::Value::Null);
            id += 1;
        })
    });
}

fn bench_search(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let dim = 128;
    let mut index = HnswIndex::new(dim);

    // Fill the graph with 100K vectors
    for i in 0u64..100_000 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        index.insert(i, v, serde_json::Value::Null);
    }

    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    c.bench_function("hnsw_search_100k", |b| {
        b.iter(|| index.search(&query, 10, 50))
    });
}

criterion_group!(benches, bench_insert, bench_search);
criterion_main!(benches);