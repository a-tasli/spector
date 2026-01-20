use macroquad::prelude::*;
use ringbuf::HeapRb; 
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit}; 
use spectrum_analyzer::scaling::divide_by_N_sqrt;
use spectrum_analyzer::windows::hann_window; 
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use libpulse_binding::sample::{Spec, Format};
use libpulse_binding::stream::Direction;
use libpulse_binding::def::BufferAttr; 
use libpulse_simple_binding::Simple;

// --- PRO CONFIG ---
const FFT_SIZE: usize = 4096;      
const SAMPLE_RATE: u32 = 44100;
const HISTORY_WIDTH: usize = 1024; 
const HOP_SIZE: usize = 1024;      

// --- FIX: ADDED "Debug" HERE ---
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix }

struct AppSettings {
    mel_scale: bool,
    colormap: ColorMapType,
    redraw_flag: bool, 
}

#[macroquad::main("Rusty Spectrogram")]
async fn main() {
    // 1. SHARED MEMORY
    let tex_height = FFT_SIZE / 2;
    
    // A. The Visuals (Pixels)
    let shared_pixels = Arc::new(Mutex::new(vec![0u8; HISTORY_WIDTH * tex_height * 4]));
    
    // B. The Data (Floats) - The "Save File"
    let shared_history = Arc::new(Mutex::new(vec![0.0f32; HISTORY_WIDTH * tex_height]));

    let shared_settings = Arc::new(Mutex::new(AppSettings { 
        mel_scale: true, 
        colormap: ColorMapType::Magma,
        redraw_flag: false,
    }));

    // 2. AUDIO SETUP
    let rb = HeapRb::<f32>::new(65536); 
    let (mut producer, mut consumer) = rb.split();

    // --- THREAD A: RECORDER ---
    thread::spawn(move || {
        let output = std::process::Command::new("pactl")
            .arg("get-default-sink")
            .output()
            .expect("pactl failed");
        let sink_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let monitor = format!("{}.monitor", sink_name);

        let spec = Spec { format: Format::F32le, channels: 1, rate: SAMPLE_RATE };
        let frag_size = (SAMPLE_RATE as u32 * 4 * 20) / 1000; 
        let attr = BufferAttr {
            maxlength: u32::MAX, tlength: u32::MAX, prebuf: u32::MAX, minreq: u32::MAX,
            fragsize: frag_size, 
        };

        if let Ok(stream) = Simple::new(None, "RustySpec", Direction::Record, Some(&monitor), "Recorder", &spec, None, Some(&attr)) {
            let mut buf = [0u8; 4096]; 
            loop {
                if let Ok(_) = stream.read(&mut buf) {
                    let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) };
                    producer.push_slice(floats);
                }
            }
        }
    });

    // --- THREAD B: BRAIN (FFT + History Manager) ---
    let pixels_ref = shared_pixels.clone();
    let history_ref = shared_history.clone();
    let settings_ref = shared_settings.clone();

    thread::spawn(move || {
        let mut rolling_audio = vec![0.0; FFT_SIZE];
        
        loop {
            // A. Check for "Redraw All" Command
            let mut redraw_needed = false;
            {
                let mut s = settings_ref.lock().unwrap();
                if s.redraw_flag {
                    redraw_needed = true;
                    s.redraw_flag = false; 
                }
            }

            // B. Redraw Loop
            if redraw_needed {
                let mut pixels = pixels_ref.lock().unwrap();
                let history = history_ref.lock().unwrap();
                let settings = settings_ref.lock().unwrap();

                for x in 0..HISTORY_WIDTH {
                    let start = x * tex_height;
                    let end = start + tex_height;
                    let column = &history[start..end];
                    paint_column(&mut pixels, x, column, &settings, tex_height);
                }
                continue; 
            }

            // C. Process New Audio
            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                rolling_audio.extend(temp);
                if rolling_audio.len() > FFT_SIZE {
                    let remove = rolling_audio.len() - FFT_SIZE;
                    rolling_audio.drain(0..remove);
                }
            }

            if num_new < HOP_SIZE {
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            // FFT
            let windowed = hann_window(&rolling_audio);
            let spectrum = samples_fft_to_spectrum(
                &windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)
            ).unwrap();
            
            // Extract Magnitudes
            let raw_col: Vec<f32> = spectrum.data().iter().map(|(_, v)| v.val()).collect();
            
            // Write to Memory
            {
                let mut pixels = pixels_ref.lock().unwrap();
                let mut history = history_ref.lock().unwrap();
                let settings = settings_ref.lock().unwrap();

                // 1. Shift History
                let h_len = history.len();
                history.copy_within(tex_height..h_len, 0);
                
                let start_idx = (HISTORY_WIDTH - 1) * tex_height;
                let copy_len = raw_col.len().min(tex_height);
                history[start_idx..start_idx+copy_len].copy_from_slice(&raw_col[0..copy_len]);

                // 2. Shift Pixels
                let p_len = pixels.len();
                pixels.copy_within(4..p_len, 0);

                // 3. Paint New Column
                paint_column(&mut pixels, HISTORY_WIDTH - 1, &raw_col, &settings, tex_height);
            }
        }
    });

    // --- THREAD C: FACE (UI) ---
    let texture = Texture2D::from_image(&Image {
        width: HISTORY_WIDTH as u16, height: tex_height as u16,
        bytes: vec![0; HISTORY_WIDTH * tex_height * 4],
    });
    texture.set_filter(FilterMode::Linear);
    
    let mut local_mel = true;
    let mut local_cmap = ColorMapType::Magma;

    loop {
        let mut changed = false;
        if is_key_pressed(KeyCode::Space) { local_mel = !local_mel; changed = true; }
        if is_key_pressed(KeyCode::C) { 
            local_cmap = cycle_colormap(local_cmap); 
            changed = true; 
        }

        if changed {
            let mut s = shared_settings.lock().unwrap();
            s.mel_scale = local_mel;
            s.colormap = local_cmap;
            s.redraw_flag = true; 
        }

        {
            let pixels = shared_pixels.lock().unwrap();
            let img = Image {
                width: HISTORY_WIDTH as u16, height: tex_height as u16,
                bytes: pixels.clone(),
            };
            texture.update(&img);
        }

        clear_background(BLACK);
        draw_texture_ex(&texture, 0.0, 0.0, WHITE, DrawTextureParams {
            dest_size: Some(vec2(screen_width(), screen_height())), ..Default::default()
        });

        draw_ui_overlay(local_mel, local_cmap);

        next_frame().await
    }
}

// --- HELPERS ---

fn paint_column(pixels: &mut Vec<u8>, x: usize, column_data: &[f32], settings: &AppSettings, height: usize) {
    let gradient = get_gradient(settings.colormap);
    let max_freq_idx = column_data.len();

    let mel_min = 0.0;
    let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();

    for y in 0..height {
        let norm_y = 1.0 - (y as f32 / height as f32); 
        
        let target_idx = if settings.mel_scale {
            let mel = norm_y * (mel_max - mel_min) + mel_min;
            let freq = 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
            let idx = freq / (SAMPLE_RATE as f32 / FFT_SIZE as f32);
            idx as usize
        } else {
            (norm_y * max_freq_idx as f32) as usize
        };

        let safe_idx = target_idx.min(max_freq_idx - 1);
        let magnitude = column_data[safe_idx];
        let intensity = (magnitude * 2000.0).ln() / 8.0; 
        let c = gradient.eval_continuous(intensity.clamp(0.0, 1.0) as f64);

        let idx = (y * HISTORY_WIDTH + x) * 4;
        
        if idx + 3 < pixels.len() {
            pixels[idx] = c.r;
            pixels[idx+1] = c.g;
            pixels[idx+2] = c.b;
            pixels[idx+3] = 255;
        }
    }
}

fn get_gradient(t: ColorMapType) -> colorous::Gradient {
    match t {
        ColorMapType::Magma => colorous::MAGMA,
        ColorMapType::Inferno => colorous::INFERNO,
        ColorMapType::Viridis => colorous::VIRIDIS,
        ColorMapType::Plasma => colorous::PLASMA,
        ColorMapType::Turbo => colorous::TURBO,
        ColorMapType::Cubehelix => colorous::CUBEHELIX,
    }
}

fn cycle_colormap(c: ColorMapType) -> ColorMapType {
    match c {
        ColorMapType::Magma => ColorMapType::Inferno,
        ColorMapType::Inferno => ColorMapType::Viridis,
        ColorMapType::Viridis => ColorMapType::Plasma,
        ColorMapType::Plasma => ColorMapType::Turbo,
        ColorMapType::Turbo => ColorMapType::Cubehelix,
        ColorMapType::Cubehelix => ColorMapType::Magma,
    }
}

fn draw_ui_overlay(mel: bool, cmap: ColorMapType) {
    draw_rectangle(0.0, 0.0, 240.0, 100.0, Color::new(0.0, 0.0, 0.0, 0.8));
    draw_text(format!("FPS: {}", get_fps()).as_str(), 10.0, 20.0, 20.0, GREEN);
    
    let scale_text = if mel { "Scale: Mel (Human)" } else { "Scale: Linear (Data)" };
    draw_text(scale_text, 10.0, 45.0, 20.0, ORANGE);

    let pretty_name = format!("{:?}", cmap);
    draw_text(&format!("Color: {}", pretty_name), 10.0, 65.0, 20.0, YELLOW);

    draw_text("[Space] Scale  [C] Color", 10.0, 90.0, 20.0, LIGHTGRAY);
}