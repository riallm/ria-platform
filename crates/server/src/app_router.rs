//! Router creation

use crate::handlers::{self, AppState};
use crate::server::ServerConfig;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Create the Axum router with all endpoints
pub fn create_router(state: Arc<Mutex<AppState>>, config: &ServerConfig) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        // Health and diagnostics
        .route("/health", get(handlers::health))
        // State and middleware
        .with_state(state)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}
