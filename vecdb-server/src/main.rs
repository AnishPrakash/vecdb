// vecdb-server/src/main.rs
use axum::{extract::State, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use vecdb_core::{
    hnsw::HnswIndex,
    wal::{Wal, WalRecord},
};

// Shared application state
struct AppState {
    index: RwLock<HnswIndex>,
    wal: Mutex<Wal>,
}

// Request / Response types
#[derive(Deserialize)]
struct InsertReq {
    id: u64,
    vector: Vec<f32>,
    payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    top_k: usize,
    ef: Option<usize>,
}

#[derive(Serialize)]
struct SearchResult {
    id: u64,
    score: f32,
}

// Handlers
async fn insert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertReq>,
) -> Json<serde_json::Value> {
    let payload = req.payload.unwrap_or(serde_json::Value::Null);
    
    // Write to WAL first (durability guarantee)
    {
        let rec = WalRecord::Insert {
            id: req.id,
            vector: req.vector.clone(),
            payload: payload.clone(),
        };
        let mut wal = state.wal.lock().await;
        wal.append(&rec).expect("WAL write failed");
    }

    // Then update in-memory index
    let mut index = state.index.write().await;
    index.insert(req.id, req.vector, payload);

    Json(serde_json::json!({ "status": "ok" }))
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchReq>,
) -> Json<Vec<SearchResult>> {
    let ef = req.ef.unwrap_or(50);
    let index = state.index.read().await;
    let results = index.search(&req.vector, req.top_k, ef);
    
    Json(
        results
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect(),
    )
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Replay WAL on startup to restore index
    // NOTE: Dimension is set to 3 here for easy testing. 
    // We will bump this to 128 in Phase 9 for the SIFT benchmark.
    let mut index = HnswIndex::new(3); 
    
    let records = Wal::replay("vecdb.wal").expect("WAL replay failed");
    let wal_obj = Wal::open("vecdb.wal").expect("WAL open failed");

    for rec in records {
        match rec {
            WalRecord::Insert { id, vector, payload } => {
                index.insert(id, vector, payload);
            }
            WalRecord::Delete { .. } => {
                // Delete handling can be added later
            }
        }
    }

    let state = Arc::new(AppState {
        index: RwLock::new(index),
        wal: Mutex::new(wal_obj),
    });

    let app = Router::new()
        .route("/insert", post(insert))
        .route("/search", post(search))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    tracing::info!("vecdb listening on port 8080");
    axum::serve(listener, app).await.unwrap();
}