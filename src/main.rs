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
const HISTORY_LEN: usize = 1024; 
const HOP_SIZE: usize = 512;      

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix }

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScrollDirection { RTL, LTR, DownToUp, UpToDown }

struct AppSettings {
    mel_scale: bool,
    colormap: ColorMapType,
    direction: ScrollDirection,
    redraw_flag: bool, 
}

#[macroquad::main("Rusty Spectrogram")]
async fn main() {
    let freq_bins = FFT_SIZE / 2; // 2048
    
    let shared_pixels = Arc::new(Mutex::new(vec![0u8; HISTORY_LEN * freq_bins * 4]));
    let shared_history = Arc::new(Mutex::new(vec![0.0f32; HISTORY_LEN * freq_bins]));
    
    let shared_settings = Arc::new(Mutex::new(AppSettings { 
        mel_scale: true, 
        colormap: ColorMapType::Magma,
        direction: ScrollDirection::RTL,
        redraw_flag: false,
    }));

    // Audio Setup
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

    // --- THREAD B: BRAIN ---
    let pixels_ref = shared_pixels.clone();
    let history_ref = shared_history.clone();
    let settings_ref = shared_settings.clone();

    thread::spawn(move || {
        let mut rolling_audio = vec![0.0; FFT_SIZE];
        
        loop {
            // A. Handle Redraws
            let mut redraw_needed = false;
            {
                if let Ok(mut s) = settings_ref.lock() {
                    if s.redraw_flag {
                        redraw_needed = true;
                        s.redraw_flag = false; 
                    }
                }
            }

            if redraw_needed {
                if let (Ok(mut pixels), Ok(history), Ok(settings)) = (pixels_ref.lock(), history_ref.lock(), settings_ref.lock()) {
                    pixels.fill(0);
                    
                    // History is ALWAYS stored [Oldest -> Newest]
                    for t in 0..HISTORY_LEN {
                        let start = t * freq_bins;
                        let spectrum_slice = &history[start..start + freq_bins];
                        
                        // Calculate where this history slice belongs on screen
                        // For RTL: History[0] (Old) goes to Screen[0] (Left)
                        // For LTR: History[0] (Old) goes to Screen[Max] (Right)
                        let screen_pos = match settings.direction {
                            ScrollDirection::RTL | ScrollDirection::DownToUp => t,
                            ScrollDirection::LTR | ScrollDirection::UpToDown => (HISTORY_LEN - 1) - t,
                        };
                        
                        paint_slice(&mut pixels, screen_pos, spectrum_slice, &settings, freq_bins);
                    }
                }
                continue; 
            }

            // B. Process Audio
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

            // C. FFT
            let windowed = hann_window(&rolling_audio);
            if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                let raw_col: Vec<f32> = spectrum.data().iter().map(|(_, v)| v.val()).collect();
                
                if let (Ok(mut pixels), Ok(mut history), Ok(settings)) = (pixels_ref.lock(), history_ref.lock(), settings_ref.lock()) {
                    
                    // 1. Update Float History (Always Shift Left: Old -> New)
                    let h_len = history.len();
                    history.copy_within(freq_bins..h_len, 0);
                    let start_idx = (HISTORY_LEN - 1) * freq_bins;
                    let copy_len = raw_col.len().min(freq_bins);
                    history[start_idx..start_idx+copy_len].copy_from_slice(&raw_col[0..copy_len]);

                    // 2. Scroll Pixels (Direction Aware)
                    shift_pixels(&mut pixels, &settings.direction, freq_bins);

                    // 3. Paint New Slice (At the "New" Edge)
                    let paint_idx = match settings.direction {
                        ScrollDirection::RTL | ScrollDirection::DownToUp => HISTORY_LEN - 1,
                        ScrollDirection::LTR | ScrollDirection::UpToDown => 0,
                    };
                    
                    paint_slice(&mut pixels, paint_idx, &raw_col, &settings, freq_bins);
                }
            }
        }
    });

    // --- THREAD C: FACE ---
    let mut current_width = HISTORY_LEN;
    let mut current_height = freq_bins;

    let mut texture = Texture2D::from_image(&Image {
        width: current_width as u16, height: current_height as u16,
        bytes: vec![0; current_width * current_height * 4],
    });
    texture.set_filter(FilterMode::Linear);
    
    let mut local_mel = true;
    let mut local_cmap = ColorMapType::Magma;
    let mut local_dir = ScrollDirection::RTL;

    loop {
        let mut changed = false;
        if is_key_pressed(KeyCode::Space) { local_mel = !local_mel; changed = true; }
        if is_key_pressed(KeyCode::C) { local_cmap = cycle_colormap(local_cmap); changed = true; }
        if is_key_pressed(KeyCode::D) { local_dir = cycle_direction(local_dir); changed = true; }

        if changed {
            if let Ok(mut s) = shared_settings.lock() {
                s.mel_scale = local_mel;
                s.colormap = local_cmap;
                s.direction = local_dir;
                s.redraw_flag = true; 
            }
        }

        // Handle Dimension Changes
        let (req_w, req_h) = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (HISTORY_LEN, freq_bins),
            ScrollDirection::DownToUp | ScrollDirection::UpToDown => (freq_bins, HISTORY_LEN),
        };

        if req_w != current_width || req_h != current_height {
            current_width = req_w;
            current_height = req_h;
            texture = Texture2D::from_image(&Image {
                width: current_width as u16, height: current_height as u16,
                bytes: vec![0; current_width * current_height * 4],
            });
            texture.set_filter(FilterMode::Linear);
        }

        if let Ok(pixels) = shared_pixels.lock() {
            let img = Image {
                width: current_width as u16, height: current_height as u16,
                bytes: pixels.clone(),
            };
            texture.update(&img);
        }

        clear_background(BLACK);
        draw_texture_ex(&texture, 0.0, 0.0, WHITE, DrawTextureParams {
            dest_size: Some(vec2(screen_width(), screen_height())), ..Default::default()
        });

        draw_ui_overlay(local_mel, local_cmap, local_dir);
        next_frame().await
    }
}

// --- HELPER FUNCTIONS ---

fn shift_pixels(pixels: &mut Vec<u8>, dir: &ScrollDirection, freq_bins: usize) {
    let len = pixels.len();
    match dir {
        // Horizontal: Width is HISTORY_LEN (1024), Stride is 4
        ScrollDirection::RTL => {
            // New at Right. Old goes Left.
            if len > 4 { pixels.copy_within(4..len, 0); }
        },
        ScrollDirection::LTR => {
            // New at Left. Old goes Right.
            if len > 4 { pixels.copy_within(0..len-4, 4); }
        },
        // Vertical: Width is FREQ_BINS (2048), Stride is FREQ_BINS * 4
        ScrollDirection::DownToUp => {
            // New at Bottom. Old goes Up.
            let row_bytes = freq_bins * 4;
            if len > row_bytes { pixels.copy_within(row_bytes..len, 0); }
        },
        ScrollDirection::UpToDown => {
            // New at Top. Old goes Down.
            let row_bytes = freq_bins * 4;
            if len > row_bytes { pixels.copy_within(0..len-row_bytes, row_bytes); }
        }
    }
}

fn paint_slice(pixels: &mut Vec<u8>, pos_index: usize, data: &[f32], settings: &AppSettings, freq_bins: usize) {
    let gradient = get_gradient(settings.colormap);
    let max_freq_idx = data.len();
    let mel_min = 0.0;
    let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();

    for i in 0..freq_bins {
        let norm_i = i as f32 / freq_bins as f32;
        
        let target_idx = if settings.mel_scale {
            let mel = norm_i * (mel_max - mel_min) + mel_min;
            let freq = 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
            let idx = freq / (SAMPLE_RATE as f32 / FFT_SIZE as f32);
            idx as usize
        } else {
            (norm_i * max_freq_idx as f32) as usize
        };

        let safe_idx = target_idx.min(max_freq_idx - 1);
        let magnitude = data[safe_idx];
        let intensity = (magnitude * 2000.0).ln() / 8.0; 
        let c = gradient.eval_continuous(intensity.clamp(0.0, 1.0) as f64);

        let idx = match settings.direction {
            // Horizontal: Time is X, Freq is Y (inverted)
            ScrollDirection::RTL | ScrollDirection::LTR => {
                let x = pos_index; 
                let y = (freq_bins - 1) - i; 
                (y * HISTORY_LEN + x) * 4
            },
            // Vertical: Time is Y, Freq is X
            ScrollDirection::DownToUp | ScrollDirection::UpToDown => {
                let y = pos_index;
                let x = i; 
                (y * freq_bins + x) * 4
            }
        };

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

fn cycle_direction(d: ScrollDirection) -> ScrollDirection {
    match d {
        ScrollDirection::RTL => ScrollDirection::LTR,
        ScrollDirection::LTR => ScrollDirection::DownToUp,
        ScrollDirection::DownToUp => ScrollDirection::UpToDown,
        ScrollDirection::UpToDown => ScrollDirection::RTL,
    }
}

fn draw_ui_overlay(mel: bool, cmap: ColorMapType, dir: ScrollDirection) {
    draw_rectangle(0.0, 0.0, 260.0, 120.0, Color::new(0.0, 0.0, 0.0, 0.8));
    draw_text(format!("FPS: {}", get_fps()).as_str(), 10.0, 20.0, 20.0, GREEN);
    
    let scale_text = if mel { "Scale: Mel" } else { "Scale: Linear" };
    draw_text(scale_text, 10.0, 40.0, 20.0, ORANGE);

    let map_name = format!("{:?}", cmap);
    draw_text(&format!("Color: {}", map_name), 10.0, 60.0, 20.0, YELLOW);

    let dir_name = match dir {
        ScrollDirection::RTL => "Right -> Left",
        ScrollDirection::LTR => "Left -> Right",
        ScrollDirection::DownToUp => "Down -> Up",
        ScrollDirection::UpToDown => "Up -> Down",
    };
    draw_text(&format!("Flow: {}", dir_name), 10.0, 80.0, 20.0, SKYBLUE);
    draw_text("[Space] Scale  [C] Color  [D] Flow", 10.0, 105.0, 20.0, LIGHTGRAY);
}