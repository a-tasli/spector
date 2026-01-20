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
const SAMPLE_RATE: u32 = 44100;
const HISTORY_LEN: usize = 1024; 
const HOP_SIZE: usize = 512;       

const RESOLUTIONS: [usize; 3] = [2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix }

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScrollDirection { RTL, LTR, DownToUp, UpToDown }

#[derive(Clone)]
struct AppSettings {
    mel_scale: bool,
    colormap: ColorMapType,
    redraw_flag: bool, 
}

struct SpectrogramLayer {
    fft_size: usize,
    freq_bins: usize,
    pixels: Vec<u8>,
    history: Vec<f32>,
}

impl SpectrogramLayer {
    fn new(fft_size: usize) -> Self {
        let bins = fft_size / 2;
        Self {
            fft_size,
            freq_bins: bins,
            pixels: vec![0u8; HISTORY_LEN * bins * 4],
            history: vec![0.0f32; HISTORY_LEN * bins],
        }
    }
}

#[macroquad::main("Rusty Spectrogram")]
async fn main() {
    let mut layers = Vec::new();
    for &size in RESOLUTIONS.iter() {
        layers.push(SpectrogramLayer::new(size));
    }
    
    let shared_layers = Arc::new(Mutex::new(layers));
    let shared_settings = Arc::new(Mutex::new(AppSettings { 
        mel_scale: true, 
        colormap: ColorMapType::Magma,
        redraw_flag: false,
    }));

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
    let layers_ref = shared_layers.clone();
    let settings_ref = shared_settings.clone();

    thread::spawn(move || {
        let max_fft = *RESOLUTIONS.iter().max().unwrap();
        let mut rolling_audio = vec![0.0; max_fft];
        
        loop {
            // Check Redraw
            let mut redraw_needed = false;
            let mut current_settings = {
                let mut s = settings_ref.lock().unwrap();
                if s.redraw_flag {
                    redraw_needed = true;
                    s.redraw_flag = false; 
                    Some(s.clone())
                } else {
                    None
                }
            };

            if redraw_needed {
                if let Some(settings) = current_settings {
                    for i in 0..RESOLUTIONS.len() {
                        if let Ok(mut layers) = layers_ref.lock() {
                            let layer = &mut layers[i];
                            layer.pixels.fill(0);
                            for t in 0..HISTORY_LEN {
                                let start = t * layer.freq_bins;
                                let slice = &layer.history[start..start + layer.freq_bins];
                                paint_slice(&mut layer.pixels, t, slice, &settings, layer.freq_bins, layer.fft_size);
                            }
                        }
                        thread::yield_now();
                    }
                }
                continue; 
            }

            // Audio Ingest
            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                rolling_audio.extend(temp);
                if rolling_audio.len() > max_fft {
                    let remove = rolling_audio.len() - max_fft;
                    rolling_audio.drain(0..remove);
                }
            }

            if num_new < HOP_SIZE {
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            // Parallel FFT
            if let (Ok(mut layers), Ok(settings)) = (layers_ref.lock(), settings_ref.lock()) {
                for layer in layers.iter_mut() {
                    if rolling_audio.len() < layer.fft_size { continue; }

                    let start_sample = rolling_audio.len() - layer.fft_size;
                    let audio_slice = &rolling_audio[start_sample..];

                    let windowed = hann_window(audio_slice);
                    if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                        let raw_col: Vec<f32> = spectrum.data().iter().map(|(_, v)| v.val()).collect();
                        
                        let h_len = layer.history.len();
                        layer.history.copy_within(layer.freq_bins..h_len, 0);
                        let start_idx = (HISTORY_LEN - 1) * layer.freq_bins;
                        let copy_len = raw_col.len().min(layer.freq_bins);
                        layer.history[start_idx..start_idx+copy_len].copy_from_slice(&raw_col[0..copy_len]);

                        let p_len = layer.pixels.len();
                        if p_len > 4 { layer.pixels.copy_within(4..p_len, 0); }

                        paint_slice(&mut layer.pixels, HISTORY_LEN - 1, &raw_col, &settings, layer.freq_bins, layer.fft_size);
                    }
                }
            }
        }
    });

    // --- THREAD C: FACE ---
    let mut current_fft_idx = 1;
    let mut current_width = HISTORY_LEN;
    let mut current_height = RESOLUTIONS[current_fft_idx] / 2;

    let mut texture = Texture2D::from_image(&Image {
        width: current_width as u16, height: current_height as u16,
        bytes: vec![0; current_width * current_height * 4],
    });
    texture.set_filter(FilterMode::Linear);
    
    let mut local_mel = true;
    let mut local_cmap = ColorMapType::Magma;
    let mut local_dir = ScrollDirection::RTL;
    
    loop {
        let mut visual_changed = false;
        
        if is_key_pressed(KeyCode::S) { local_mel = !local_mel; visual_changed = true; }
        if is_key_pressed(KeyCode::C) { local_cmap = cycle_colormap(local_cmap); visual_changed = true; }
        if is_key_pressed(KeyCode::F) { local_dir = cycle_direction(local_dir); }
        if is_key_pressed(KeyCode::R) { current_fft_idx = (current_fft_idx + 1) % RESOLUTIONS.len(); }

        if visual_changed {
            if let Ok(mut s) = shared_settings.lock() {
                s.mel_scale = local_mel;
                s.colormap = local_cmap;
                s.redraw_flag = true; 
            }
        }

        let bins = RESOLUTIONS[current_fft_idx] / 2;
        if bins != current_height {
            current_height = bins;
            texture = Texture2D::from_image(&Image {
                width: current_width as u16, height: current_height as u16,
                bytes: vec![0; current_width * current_height * 4],
            });
            texture.set_filter(FilterMode::Linear);
        }

        if let Ok(layers) = shared_layers.lock() {
            let layer = &layers[current_fft_idx];
            if layer.pixels.len() == current_width * current_height * 4 {
                let img = Image {
                    width: current_width as u16, height: current_height as u16,
                    bytes: layer.pixels.clone(),
                };
                texture.update(&img);
            }
        }

        clear_background(BLACK);

        let sw = screen_width();
        let sh = screen_height();
        
        // --- VIEWPORT LOGIC ---
        // Determine how much history we can show based on window dimensions
        // If window is smaller than HISTORY_LEN, we crop.
        // If window is larger, we show all (and stretch).
        let (visible_bins, tex_w, tex_h, rotation, flip_x, flip_y) = match local_dir {
            ScrollDirection::RTL => {
                let vis = (sw as usize).min(HISTORY_LEN);
                (vis, vis, bins, 0.0, false, false)
            },
            ScrollDirection::LTR => {
                let vis = (sw as usize).min(HISTORY_LEN);
                (vis, vis, bins, 0.0, true, false)
            },
            ScrollDirection::DownToUp => {
                let vis = (sh as usize).min(HISTORY_LEN);
                (vis, vis, bins, std::f32::consts::FRAC_PI_2, false, false)
            },
            ScrollDirection::UpToDown => {
                let vis = (sh as usize).min(HISTORY_LEN);
                (vis, vis, bins, -std::f32::consts::FRAC_PI_2, false, true)
            },
        };

        // Determine SOURCE RECT (What part of texture to draw)
        // Texture is always [0..HISTORY_LEN].
        // Newest data is at Right/End (Index 1024) for RTL/DTU.
        // Newest data is at Left/Start (Index 0) for LTR/UTD.
        let source_x = match local_dir {
            // Keep Right (New), Crop Left (Old)
            ScrollDirection::RTL | ScrollDirection::DownToUp => (HISTORY_LEN - visible_bins) as f32,
            // Keep Left (New), Crop Right (Old)
            ScrollDirection::LTR | ScrollDirection::UpToDown => 0.0,
        };

        let source = Rect::new(source_x, 0.0, visible_bins as f32, bins as f32);

        // Determine DEST SIZE (How big on screen)
        // If window > HISTORY_LEN, dest fills window (Stretch).
        // If window < HISTORY_LEN, dest fills window (1:1 pixel mapping).
        let dest_size = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => vec2(sw, sh),
            ScrollDirection::DownToUp | ScrollDirection::UpToDown => vec2(sh, sw), // Swapped for rotation
        };

        let draw_params = DrawTextureParams {
            dest_size: Some(dest_size),
            source: Some(source),
            rotation,
            flip_x,
            flip_y,
            pivot: if rotation != 0.0 { Some(vec2(sw/2.0, sh/2.0)) } else { None },
            ..Default::default()
        };

        // Centering logic for rotation
        let (draw_x, draw_y) = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0),
            _ => ( (sw - sh) / 2.0, (sh - sw) / 2.0 )
        };

        draw_texture_ex(&texture, draw_x, draw_y, WHITE, draw_params);

        draw_note_ruler(local_mel, local_dir, RESOLUTIONS[current_fft_idx]);
        draw_ui_overlay(local_mel, local_cmap, local_dir, RESOLUTIONS[current_fft_idx]);
        
        next_frame().await
    }
}

// --- HELPERS ---

fn paint_slice(pixels: &mut Vec<u8>, pos_index: usize, data: &[f32], settings: &AppSettings, freq_bins: usize, fft_size: usize) {
    let gradient = get_gradient(settings.colormap);
    let max_freq_idx = data.len();
    let mel_min = 0.0;
    let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();

    for i in 0..freq_bins {
        let norm_i = i as f32 / freq_bins as f32;
        
        let target_idx = if settings.mel_scale {
            let mel = norm_i * (mel_max - mel_min) + mel_min;
            let freq = 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
            let idx = freq / (SAMPLE_RATE as f32 / fft_size as f32); 
            idx as usize
        } else {
            (norm_i * max_freq_idx as f32) as usize
        };

        let safe_idx = target_idx.min(max_freq_idx - 1);
        let magnitude = data[safe_idx];
        let intensity = (magnitude * 2000.0).ln() / 8.0; 
        let c = gradient.eval_continuous(intensity.clamp(0.0, 1.0) as f64);

        let x = pos_index; 
        let y = (freq_bins - 1) - i; 
        let idx = (y * HISTORY_LEN + x) * 4;

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

fn freq_to_screen_pos(freq: f32, mel_scale: bool) -> f32 {
    let max_freq = SAMPLE_RATE as f32 / 2.0;
    if mel_scale {
        let mel_val = 2595.0 * (1.0 + freq / 700.0).log10();
        let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
        mel_val / mel_max
    } else {
        freq / max_freq
    }
}

fn draw_note_ruler(mel_scale: bool, dir: ScrollDirection, fft_size: usize) {
    let w = screen_width();
    let h = screen_height();
    if fft_size == 0 { return; }

    for midi in 21..109 {
        let freq = 440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0);
        if freq > (SAMPLE_RATE as f32 / 2.0) { break; }
        let norm_pos = freq_to_screen_pos(freq, mel_scale);
        
        let (x, y, is_c) = match dir {
            ScrollDirection::RTL => (w, h * (1.0 - norm_pos), midi % 12 == 0),
            ScrollDirection::LTR => (0.0, h * (1.0 - norm_pos), midi % 12 == 0),
            ScrollDirection::UpToDown => (w * norm_pos, 0.0, midi % 12 == 0),
            ScrollDirection::DownToUp => (w * norm_pos, h, midi % 12 == 0),
        };

        if dir == ScrollDirection::RTL || dir == ScrollDirection::LTR {
            let tick_len = if is_c { 15.0 } else { 5.0 };
            let start_x = if dir == ScrollDirection::RTL { x - tick_len } else { x };
            draw_line(start_x, y, start_x + tick_len, y, 1.0, WHITE);
            if is_c {
                let octave = (midi / 12) - 1;
                let text_x = if dir == ScrollDirection::RTL { x - 35.0 } else { x + 20.0 };
                draw_text(&format!("C{}", octave), text_x, y + 4.0, 15.0, WHITE);
            }
        } else {
            let tick_len = if is_c { 15.0 } else { 5.0 };
            let start_y = if dir == ScrollDirection::DownToUp { y - tick_len } else { y };
            draw_line(x, start_y, x, start_y + tick_len, 1.0, WHITE);
            if is_c {
                let octave = (midi / 12) - 1;
                let text_y = if dir == ScrollDirection::DownToUp { y - 20.0 } else { y + 30.0 };
                draw_text(&format!("C{}", octave), x - 5.0, text_y, 15.0, WHITE);
            }
        }
    }
}

struct UiStat {
    label: &'static str,
    hotkey: Option<char>,
    value: String,
    color: Color,
}

fn draw_ui_overlay(mel: bool, cmap: ColorMapType, dir: ScrollDirection, fft: usize) {
    let scale_str = if mel { "Mel" } else { "Linear" };
    let map_str = format!("{:?}", cmap); 
    let dir_str = match dir {
        ScrollDirection::RTL => "RTL", 
        ScrollDirection::LTR => "LTR",
        ScrollDirection::DownToUp => "Fire", 
        ScrollDirection::UpToDown => "Rain",
    };
    let res_str = format!("{} bins ({} FFT)", fft/2, fft);

    let stats = vec![
        UiStat { label: "FPS",   hotkey: None,      value: get_fps().to_string(), color: GREEN },
        UiStat { label: "Scale", hotkey: Some('S'), value: scale_str.to_string(), color: ORANGE },
        UiStat { label: "Color", hotkey: Some('C'), value: map_str,               color: YELLOW },
        UiStat { label: "Flow",  hotkey: Some('F'), value: dir_str.to_string(),   color: SKYBLUE },
        UiStat { label: "Res",   hotkey: Some('R'), value: res_str,               color: VIOLET },
    ];

    let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match dir {
        ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, screen_width(), 30.0, false),
        _ => (screen_width() - 270.0, 10.0, 260.0, 120.0, true),
    };

    draw_rectangle(bg_x, bg_y, bg_w, bg_h, Color::new(0.0, 0.0, 0.0, 0.5));

    let mut cursor_x = bg_x + 10.0;
    let mut cursor_y = bg_y + 20.0;

    for stat in stats {
        let full_label = format!("{}:", stat.label);
        
        draw_text(&full_label, cursor_x, cursor_y, 20.0, WHITE);

        if let Some(c) = stat.hotkey {
            let char_dims = measure_text(&c.to_string(), None, 20, 1.0);
            draw_line(cursor_x, cursor_y + 2.0, cursor_x + char_dims.width, cursor_y + 2.0, 1.0, WHITE);
        }

        if is_vertical {
            let label_width = measure_text(&full_label, None, 20, 1.0).width;
            draw_text(&stat.value, cursor_x + label_width + 10.0, cursor_y, 20.0, stat.color);
            cursor_y += 20.0;
        } else {
            let label_width = measure_text(&full_label, None, 20, 1.0).width;
            draw_text(&stat.value, cursor_x + label_width + 5.0, cursor_y, 20.0, stat.color);
            cursor_x += 140.0; 
        }
    }
}