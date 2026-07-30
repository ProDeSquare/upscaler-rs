use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    Router,
    extract::{Multipart, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use image::ImageFormat;
use ndarray::Array4;
use ort::ep::CUDAExecutionProvider;
use ort::ep::ExecutionProvider;
use ort::session::{Session, builder::GraphOptimizationLevel};
use serde::Deserialize;
use tokio::sync::oneshot;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer};
use tracing_subscriber::EnvFilter;

const MODEL_PATH_ENV: &str = "MODEL_PATH";
const DEFAULT_MODEL_PATH: &str = "./models/RealESRGAN_x4plus_anime_6B.onnx";
const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

#[derive(Deserialize)]
struct UpscaleParams {
    task_id: String,
}

struct WorkItem {
    task_id: String,
    bytes: Vec<u8>,
    respond_to: oneshot::Sender<anyhow::Result<Vec<u8>>>,
}

struct AppState {
    workers: Vec<std::sync::mpsc::Sender<WorkItem>>,
    free_workers: Mutex<Vec<usize>>,
    free_notify: tokio::sync::Notify,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let env_path = "<ENV_PATH>";

    match dotenvy::from_path(env_path) {
        Ok(_) => println!("Successfully loaded app env"),
        Err(e) => eprintln!("Warning failed to load env: {}", e),
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let idle_shrink_secs: u64 = std::env::var("IDLE_SHRINK_AFTER")
        .unwrap_or_else(|_| "120".to_string())
        .parse()
        .expect("IDLE_SHRINK_AFTER must be a valid u64 int");

    let idle_shrink_after: Duration = Duration::from_secs(idle_shrink_secs);

    let num_workers: usize = std::env::var("NUM_WORKERS")
        .unwrap_or_else(|_| "3".to_string())
        .parse()
        .expect("NUM_WORKERS must be a valid unsinged int");

    let port: u16 = std::env::var("APP_PORT")
        .unwrap_or_else(|_| "8377".to_string())
        .parse()
        .expect("APP_PORT must be a valid u16 int");

    let model_path =
        std::env::var(MODEL_PATH_ENV).unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());

    let cuda = CUDAExecutionProvider::default();
    let cuda_available = cuda.is_available().unwrap_or(false);
    if !cuda_available {
        anyhow::bail!("CUDA execution provider is not available in this container.");
    }

    tracing::info!(
        model_path,
        workers = num_workers,
        "Spawning inference workers"
    );

    let mut workers = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        let (tx, rx) = std::sync::mpsc::channel::<WorkItem>();
        let model_path = model_path.clone();

        std::thread::Builder::new()
            .name(format!("ort-worker-{worker_id}"))
            .spawn(move || {
                let cuda_options = CUDAExecutionProvider::default().with_device_id(1);

                let mut session = Session::builder()
                    .expect("session builder")
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .expect("opt level")
                    .with_execution_providers([cuda_options.build().error_on_failure()])
                    .expect("execution providers")
                    .commit_from_file(&model_path)
                    .expect("load model");

                tracing::info!(worker_id, "working session loaded, entering serve loop");

                loop {
                    match rx.recv_timeout(idle_shrink_after) {
                        Ok(item) => {
                            tracing::info!(task_id = %item.task_id, worker_id, "initializing");

                            let res = run_upscale(&mut session, &item.bytes, &item.task_id);
                            let _ = item.respond_to.send(res);
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if let Err(e) = shrink_arena(&mut session) {
                                tracing::warn!(worker_id, "arena shrink failed: {e:#}");
                            }
                            // match shrink_arena(&mut session) {
                            //     Ok(()) => tracing::info!("{worker_id} arena freed"),
                            //     Err(e) => tracing::warn!("{worker_id} arena shrink failed: {e:#}"),
                            // }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                // while let Ok(item) = rx.recv() {
                //     let res = run_upscale(&mut session, &item.bytes);
                //     let _ = item.respond_to.send(res);
                // }
            })
            .expect("failed to spawn inference worker thread");

        workers.push(tx);
    }

    let state = std::sync::Arc::new(AppState {
        workers,
        free_workers: Mutex::new((0..num_workers).collect()),
        free_notify: tokio::sync::Notify::new(),
    });

    let app = Router::new()
        .route("/upscale", post(upscale))
        .layer(RequestBodyLimitLayer::new(MAX_UPLOAD_BYTES))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(%addr, "listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn upscale(
    State(state): State<Arc<AppState>>,
    Query(params): Query<UpscaleParams>,
    mut multipart: Multipart,
) -> Response {
    let field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        Ok(None) => {
            return (StatusCode::BAD_REQUEST, "expected a multipart file field").into_response();
        }
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("multipart error: {e}")).into_response();
        }
    };

    let bytes = match field.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("failed to read upload: {e}"),
            )
                .into_response();
        }
    };

    let worker_idx = loop {
        let maybe_idx = state.free_workers.lock().unwrap().pop();
        match maybe_idx {
            Some(idx) => break idx,
            None => state.free_notify.notified().await,
        }
    };

    let (resp_tx, resp_rx) = oneshot::channel();
    let item = WorkItem {
        task_id: params.task_id.clone(),
        bytes: bytes.to_vec(),
        respond_to: resp_tx,
    };

    if state.workers[worker_idx].send(item).is_err() {
        state.free_workers.lock().unwrap().push(worker_idx);
        state.free_notify.notify_one();

        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference worker unavailable",
        )
            .into_response();
    }

    let result = resp_rx.await;

    state.free_workers.lock().unwrap().push(worker_idx);
    state.free_notify.notify_one();

    match result {
        Ok(Ok(png_bytes)) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/png")],
            png_bytes,
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!(task_id = %params.task_id, "upscale failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("upscale failed: {e}"),
            )
                .into_response()
        }
        Err(join_err) => {
            tracing::error!(task_id = %params.task_id, "worker task panicked: {join_err:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

fn run_upscale(
    session: &mut Session,
    input_bytes: &[u8],
    task_id: &str,
) -> anyhow::Result<Vec<u8>> {
    let start = Instant::now();

    let img = image::load_from_memory(input_bytes)?.to_rgb8();
    let (w, h) = img.dimensions();

    let scale: u32 = 4;
    let tile_size: u32 = 256;
    let tile_pad: u32 = 32;

    let fixed_in_dim: usize = 320;

    let out_w = w * scale;
    let out_h = h * scale;
    let mut out_img = image::RgbImage::new(out_w, out_h);

    let input_name = session.inputs()[0].name().to_string();
    let output_name = session.outputs()[0].name().to_string();

    for tile_y in (0..h).step_by(tile_size as usize) {
        for tile_x in (0..w).step_by(tile_size as usize) {
            let in_x_min = tile_x.saturating_sub(tile_pad);
            let in_y_min = tile_y.saturating_sub(tile_pad);
            let in_x_max = (tile_x + tile_size + tile_pad).min(w);
            let in_y_max = (tile_y + tile_size + tile_pad).min(h);

            let current_in_w = in_x_max - in_x_min;
            let current_in_h = in_y_max - in_y_min;

            let mut input = Array4::<f32>::zeros((1, 3, fixed_in_dim, fixed_in_dim));
            for dy in 0..current_in_h {
                for dx in 0..current_in_w {
                    let px = img.get_pixel(in_x_min + dx, in_y_min + dy);
                    input[[0, 0, dy as usize, dx as usize]] = px[0] as f32 / 255.0;
                    input[[0, 1, dy as usize, dx as usize]] = px[1] as f32 / 255.0;
                    input[[0, 2, dy as usize, dx as usize]] = px[2] as f32 / 255.0;
                }
            }

            let input_tensor = ort::value::Tensor::from_array(input)?;
            let outputs = session.run(ort::inputs![input_name.as_str() => input_tensor])?;

            let (shape, out_slice) = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
            // let (out_tensor_h, out_tensor_w) = (shape[2] as usize, shape[3] as usize);
            let out_tensor_w = shape[3] as usize;
            let plane = (shape[2] as usize) * out_tensor_w;

            let out_x_min = tile_x * scale;
            let out_y_min = tile_y * scale;
            let out_x_max = (tile_x + tile_size).min(w) * scale;
            let out_y_max = (tile_y + tile_size).min(h) * scale;

            let offset_x = (tile_x - in_x_min) * scale;
            let offset_y = (tile_y - in_y_min) * scale;

            for dy in 0..(out_y_max - out_y_min) {
                for dx in 0..(out_x_max - out_x_min) {
                    let tensor_y = (offset_y + dy) as usize;
                    let tensor_x = (offset_x + dx) as usize;
                    let idx = tensor_y * out_tensor_w + tensor_x;

                    let r = (out_slice[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let g = (out_slice[plane + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let b = (out_slice[2 * plane + idx].clamp(0.0, 1.0) * 255.0).round() as u8;

                    out_img.put_pixel(out_x_min + dx, out_y_min + dy, image::Rgb([r, g, b]));
                }
            }
        }
    }

    let mut buf = Vec::new();
    out_img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)?;

    tracing::info!(
        task_id = %task_id,
        in_w = w,
        in_h = h,
        out_w = out_w,
        out_h = out_h,
        elapsed_ms = start.elapsed().as_millis(),
        "upscale done"
    );

    Ok(buf)
}

fn shrink_arena(session: &mut Session) -> anyhow::Result<()> {
    let mut run_options = ort::session::RunOptions::new()?;

    run_options.add_config_entry("memory.enable_memory_arena_shrinkage", "gpu:1")?;

    let dummy = ndarray::Array4::<f32>::zeros((1, 3, 320, 320));
    let input_name = session.inputs()[0].name().to_string();
    let tensor = ort::value::Tensor::from_array(dummy)?;

    session.run_with_options(ort::inputs![input_name.as_str() => tensor], &run_options)?;

    Ok(())
}
