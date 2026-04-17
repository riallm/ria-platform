//! RIA HTTP API Server - OpenAI-compatible endpoints

pub mod server;
pub mod handlers;
pub mod types;
pub mod app_router;

pub use app_router::create_router;
pub use server::ServerConfig;
pub use handlers::AppState;
