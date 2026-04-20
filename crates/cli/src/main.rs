//! RIA CLI - Command-line interface for RIA models

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser)]
#[command(name = "ria")]
#[command(about = "RIA - Memory-optimized LLM inference with agentic coding capabilities", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve a model via HTTP API
    Serve {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to tokenizer file
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Device (cpu/cuda/metal)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Maximum sequence length
        #[arg(long, default_value = "4096")]
        max_seq_len: usize,
    },

    /// Generate text from prompt
    Generate {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to tokenizer file
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f64,

        /// Top-p sampling
        #[arg(long, default_value = "0.95")]
        top_p: f64,

        /// Device (cpu/cuda/metal)
        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Inspect a GGUF model file
    Inspect {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model,
            tokenizer,
            host,
            port,
            device: device_str,
            max_seq_len,
        } => {
            cmd_serve(
                &model,
                tokenizer.as_ref(),
                &host,
                port,
                &device_str,
                max_seq_len,
            )
            .await?;
        }
        Commands::Generate {
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            top_p,
            device,
        } => {
            cmd_generate(
                &model,
                tokenizer.as_ref(),
                &prompt,
                max_tokens,
                temperature,
                top_p,
                &device,
            )
            .await?;
        }
        Commands::Inspect { model } => {
            cmd_inspect(&model).await?;
        }
    }

    Ok(())
}

/// Serve command
async fn cmd_serve(
    model_path: &PathBuf,
    tokenizer_path: Option<&PathBuf>,
    host: &str,
    port: u16,
    device_str: &str,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    use candle_core::Device;
    use ria_inference_core
::{RIAModel, RIATokenizer};
    use ria_server::{create_router, AppState, ServerConfig};

    println!("Loading model from {:?}", model_path);

    let device = parse_device(device_str)?;
    let model = RIAModel::from_gguf(model_path, device)?;

    let tokenizer = if let Some(path) = tokenizer_path {
        println!("Loading tokenizer from {:?}", path);
        Some(RIATokenizer::from_file(path)?)
    } else {
        println!("No tokenizer specified, using raw tokens");
        None
    };

    println!("Model loaded: {} parameters", model.parameter_count());
    println!("Server starting on http://{}:{}", host, port);
    println!("API endpoints:");
    println!("  POST /v1/completions");
    println!("  POST /v1/chat/completions");
    println!("  GET  /v1/models");
    println!("  GET  /health");

    let state = Arc::new(Mutex::new(AppState { model, tokenizer }));
    let config = ServerConfig {
        host: host.to_string(),
        port,
        model_path: model_path.to_string_lossy().to_string(),
        tokenizer_path: tokenizer_path.map(|p| p.to_string_lossy().to_string()),
        device: device_str.to_string(),
        max_seq_len,
        profile: false,
    };

    let app = create_router(state, &config);
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;

    axum::serve(listener, app).await?;

    Ok(())
}

/// Generate command
async fn cmd_generate(
    model_path: &PathBuf,
    tokenizer_path: Option<&PathBuf>,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    device: &str,
) -> anyhow::Result<()> {
    use candle_core::Device;
    use ria_inference_core
::{GenerationConfig, Generator, RIAModel, RIATokenizer};

    let device = parse_device(device)?;
    let model = RIAModel::from_gguf(model_path, device)?;

    let tokenizer = if let Some(path) = tokenizer_path {
        Some(RIATokenizer::from_file(path)?)
    } else {
        None
    };

    // Encode prompt
    let prompt_tokens = if let Some(ref tok) = tokenizer {
        tok.encode(prompt, true)?
    } else {
        eprintln!("Warning: No tokenizer, using raw token IDs");
        vec![]
    };

    println!("Prompt: {}", prompt);
    println!("Tokens: {}", prompt_tokens.len());
    println!("Generating...\n");

    // Generate
    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        temperature,
        top_p: Some(top_p),
        top_k: Some(50),
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        stop_sequences: vec![],
        logprobs: false,
        seed: None,
    };

    let mut generator = Generator::new(config);
    let output = generator.generate(&model, &prompt_tokens)?;

    // Decode output
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&output.tokens, true)?;
        print!("{}", text);
    } else {
        println!("Generated {} tokens", output.tokens.len());
    }

    Ok(())
}

/// Inspect command
async fn cmd_inspect(model_path: &PathBuf) -> anyhow::Result<()> {
    use ria_gguf::GGUFReader;

    let reader = GGUFReader::open(model_path)?;

    println!("GGUF File Inspection");
    println!("===================");
    println!("File size: {}", reader.formatted_size());
    println!("Magic: 0x{:08X}", reader.header().magic);
    println!("Version: {}", reader.header().version);
    println!("Tensor count: {}", reader.header().tensor_count);
    println!("Metadata keys: {}", reader.header().metadata_kv_count);

    if let Some(arch) = reader.architecture() {
        println!("Architecture: {}", arch);
    }

    if let Some(name) = reader.model_name() {
        println!("Model name: {}", name);
    }

    // Print key metadata
    println!("\nMetadata:");
    for (key, value) in reader.header().metadata.iter() {
        if key.starts_with("llama") || key.starts_with("ria") || key.starts_with("general") {
            println!("  {}: {:?}", key, value);
        }
    }

    Ok(())
}

/// Parse device string
fn parse_device(s: &str) -> anyhow::Result<candle_core::Device> {
    match s {
        "cpu" => Ok(candle_core::Device::Cpu),
        "cuda" => Ok(candle_core::Device::new_cuda(0)?),
        "metal" => Ok(candle_core::Device::new_metal(0)?),
        _ => anyhow::bail!("Unknown device: {}. Use cpu, cuda, or metal", s),
    }
}
