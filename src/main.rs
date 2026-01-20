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

// --- CONFIG ---
const FFT_SIZE: usize = 4096;      
const SAMPLE_RATE: u32 = 44100;
const HISTORY_WIDTH: usize = 1024; 
const HOP_SIZE: usize = 1024;      

// Settings shared between UI and Background Processor
struct AppSettings {
    mel_scale: bool,
}

#[macroquad::main("Rusty Spectrogram")]
async fn main() {
    // 1. SHARED MEMORY
    // The "Brain" writes to this. The "Face" reads from this.
    // We protect it with a Mutex so they don't fight.
    let tex_height = FFT_SIZE / 2;
    let shared_pixels = Arc::new(Mutex::new(vec![0u8; HISTORY_WIDTH * tex_height * 4]));
    let shared_settings = Arc::new(Mutex::new(AppSettings { mel_scale: true }));

    // 2. SETUP AUDIO PIPELINE
    let rb = HeapRb::<f32>::new(65536); 
    let (mut producer, mut consumer) = rb.split();

    // --- THREAD A: RECORDER (Audio) ---
    thread::spawn(move || {
        let output = std::process::Command::new("pactl")
            .arg("get-default-sink")
            .output()
            .expect("Failed to run pactl");
        let sink_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let monitor = format!("{}.monitor", sink_name);

        let spec = Spec { format: Format::F32le, channels: 1, rate: SAMPLE_RATE };
        // 20ms latency buffer (balanced for background stability)
        let frag_size = (SAMPLE_RATE as u32 * 4 * 20) / 1000; 
        let attr = BufferAttr {
            maxlength: u32::MAX, tlength: u32::MAX, prebuf: u32::MAX, minreq: u32::MAX,
            fragsize: frag_size, 
        };

        let s = Simple::new(
            None, "RustySpec", Direction::Record, Some(&monitor), 
            "Recorder", &spec, None, Some(&attr) 
        );

        if let Ok(stream) = s {
            let mut buf = [0u8; 4096]; 
            loop {
                if let Ok(_) = stream.read(&mut buf) {
                    let floats: &[f32] = unsafe {
                        std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4)
                    };
                    producer.push_slice(floats);
                }
            }
        }
    });

    // --- THREAD B: BRAIN (FFT + Drawing) ---
    // This thread runs FOREVER, even if the window is minimized.
    let pixels_ref = shared_pixels.clone();
    let settings_ref = shared_settings.clone();

    thread::spawn(move || {
        let mut rolling_audio = vec![0.0; FFT_SIZE];
        
        loop {
            // A. Consume Audio
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

            // B. Wait for HOP_SIZE (Flow Control)
            if num_new < HOP_SIZE {
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            // C. The Math
            let windowed = hann_window(&rolling_audio);
            let spectrum = samples_fft_to_spectrum(
                &windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)
            ).unwrap();
            let raw_data = spectrum.data();
            let max_freq_idx = raw_data.len();

            // D. Update the Pixel Buffer (Lock -> Write -> Unlock)
            {
                let mut pixels = pixels_ref.lock().unwrap();
                let settings = settings_ref.lock().unwrap();

                // Shift Image Left
                let len = pixels.len();
                pixels.copy_within(4..len, 0);

                // Pre-calc Mel constants if needed
                let mel_min = 0.0;
                let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();
                
                // Draw new column
                for y in 0..tex_height {
                    let norm_y = 1.0 - (y as f32 / tex_height as f32); // 0.0 = bottom, 1.0 = top
                    
                    let target_freq_idx = if settings.mel_scale {
                        // Inverse Mel Scale Formula
                        let mel = norm_y * (mel_max - mel_min) + mel_min;
                        let freq = 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
                        let idx = freq / (SAMPLE_RATE as f32 / FFT_SIZE as f32);
                        idx as usize
                    } else {
                        // Linear
                        (norm_y * max_freq_idx as f32) as usize
                    };

                    let safe_idx = target_freq_idx.min(max_freq_idx - 1);
                    // Fast approximate lookup
                    let (_, val) = raw_data.iter().nth(safe_idx).unwrap();
                    let magnitude = val.val();

                    // Coloring (Magma-ish)
                    let intensity = (magnitude * 2000.0).ln() / 8.0; 
                    let c = colorous::MAGMA.eval_continuous(intensity.clamp(0.0, 1.0) as f64);

                    let idx = (y * HISTORY_WIDTH + (HISTORY_WIDTH - 1)) * 4;
                    pixels[idx] = c.r;
                    pixels[idx+1] = c.g;
                    pixels[idx+2] = c.b;
                    pixels[idx+3] = 255;
                }
            }
        }
    });

    // --- THREAD C: THE FACE (Rendering) ---
    // Init Texture
    let texture = Texture2D::from_image(&Image {
        width: HISTORY_WIDTH as u16,
        height: tex_height as u16,
        bytes: vec![0; HISTORY_WIDTH * tex_height * 4],
    });
    texture.set_filter(FilterMode::Linear);
    
    let mut local_mel = true;

    loop {
        // 1. Sync Settings (UI -> Thread B)
        {
            let mut s = shared_settings.lock().unwrap();
            s.mel_scale = local_mel;
        }

        // 2. Upload Pixels (Thread B -> GPU)
        {
            let pixels = shared_pixels.lock().unwrap();
            let img = Image {
                width: HISTORY_WIDTH as u16,
                height: tex_height as u16,
                bytes: pixels.clone(),
            };
            texture.update(&img);
        }

        // 3. Draw
        clear_background(BLACK);
        draw_texture_ex(
            &texture,
            0.0, 0.0, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        // UI Overlay
        draw_rectangle(0.0, 0.0, 220.0, 80.0, Color::new(0.0, 0.0, 0.0, 0.8));
        draw_text(format!("FPS: {}", get_fps()).as_str(), 10.0, 20.0, 20.0, GREEN);
        
        let scale_text = if local_mel { "Scale: Mel (Standard)" } else { "Scale: Linear" };
        draw_text(scale_text, 10.0, 45.0, 20.0, ORANGE);
        draw_text("[Space] Toggle Scale", 10.0, 65.0, 20.0, LIGHTGRAY);

        if is_key_pressed(KeyCode::Space) {
            local_mel = !local_mel;
        }

        next_frame().await
    }
}