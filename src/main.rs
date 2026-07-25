use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};

use axum::{
    Router,
    extract::{Multipart, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use image::ImageFormat;
use ndarray::Array4;
use ort::ep::CUDAExecutionProvider;
use ort::ep::ExecutionProvider;
use ort::session::{Session, builder::GraphOptimizationLevel};
use tower_http::limit::RequestBodyLimitLayer;
use tracing_subscriber::EnvFilter;

const MODEL_PATH_ENV: &str = "MODEL_PATH";
const DEFAULT_MODEL_PATH: &str = "/app/models/RealESRGAN_x4plus_anime_6B.onnx";
const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

struct AppState {
    session: Mutex<Session>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let model_path =
        std::env::var(MODEL_PATH_ENV).unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());

    let cuda = CUDAExecutionProvider::default();
    let cuda_available = cuda.is_available().unwrap_or(false);
    if !cuda_available {
        anyhow::bail!("CUDA execution provider is not available in this container.");
    }

    tracing::info!(model_path, "loading model");
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([cuda.build().error_on_failure()])?
        .commit_from_file(&model_path)?;

    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    tracing::info!(
        ?input_names,
        ?output_names,
        "model loaded, GPU execution provider active"
    );

    let state = Arc::new(AppState {
        session: Mutex::new(session),
    });

    let app = Router::new()
        .route("/upscale", post(upscale))
        .layer(RequestBodyLimitLayer::new(MAX_UPLOAD_BYTES))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!(%addr, "listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn upscale(State(state): State<Arc<AppState>>, mut multipart: Multipart) -> Response {
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

    let result = tokio::task::spawn_blocking(move || run_upscale(&state.session, &bytes)).await;

    match result {
        Ok(Ok(png_bytes)) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/png")],
            png_bytes,
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!("upscale failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("upscale failed: {e}"),
            )
                .into_response()
        }
        Err(join_err) => {
            tracing::error!("worker task panicked: {join_err:#}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        }
    }
}

fn run_upscale(session: &Mutex<Session>, input_bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    let start = Instant::now();

    let img = image::load_from_memory(input_bytes)?.to_rgb8();
    let (w, h) = img.dimensions();

    let scale: u32 = 4;
    let tile_size: u32 = 256;
    let tile_pad: u32 = 32;

    let out_w = w * scale;
    let out_h = h * scale;
    let mut out_img = image::RgbImage::new(out_w, out_h);

    let mut session_lock = session.lock().expect("session mutex poisoned");
    let input_name = session_lock.inputs()[0].name().to_string();
    let output_name = session_lock.outputs()[0].name().to_string();

    for tile_y in (0..h).step_by(tile_size as usize) {
        for tile_x in (0..w).step_by(tile_size as usize) {
            let in_x_min = tile_x.saturating_sub(tile_pad);
            let in_y_min = tile_y.saturating_sub(tile_pad);
            let in_x_max = (tile_x + tile_size + tile_pad).min(w);
            let in_y_max = (tile_y + tile_size + tile_pad).min(h);

            let current_in_w = in_x_max - in_x_min;
            let current_in_h = in_y_max - in_y_min;

            let mut input =
                Array4::<f32>::zeros((1, 3, current_in_h as usize, current_in_w as usize));
            for dy in 0..current_in_h {
                for dx in 0..current_in_w {
                    let px = img.get_pixel(in_x_min + dx, in_y_min + dy);
                    input[[0, 0, dy as usize, dx as usize]] = px[0] as f32 / 255.0;
                    input[[0, 1, dy as usize, dx as usize]] = px[1] as f32 / 255.0;
                    input[[0, 2, dy as usize, dx as usize]] = px[2] as f32 / 255.0;
                }
            }

            let input_tensor = ort::value::Tensor::from_array(input)?;
            let outputs = session_lock.run(ort::inputs![input_name.as_str() => input_tensor])?;

            let (shape, out_slice) = outputs[output_name.as_str()].try_extract_tensor::<f32>()?;
            let (out_tensor_h, out_tensor_w) = (shape[2] as usize, shape[3] as usize);
            let plane = out_tensor_h * out_tensor_w;

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
        in_w = w,
        in_h = h,
        out_w = out_w,
        out_h = out_h,
        elapsed_ms = start.elapsed().as_millis(),
        "upscale done"
    );

    Ok(buf)
}
