//! RIA HTTP API Server - OpenAI-compatible endpoints

pub mod app_router;
pub mod handlers;
pub mod server;
pub mod types;

pub use app_router::create_router;
pub use handlers::AppState;
pub use server::ServerConfig;
