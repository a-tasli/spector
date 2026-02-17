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
const HOP_SIZE: usize = 512;

const RESOLUTIONS: [usize; 3] = [2048, 4096, 8192];
const MAX_HISTORY: usize = 2520; 
const TARGET_DISPLAY_WIDTH: f32 = 2520.0;

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

// Optimization: Pre-calculated Color Lookup Table
struct ColorLut {
    bytes: Vec<[u8; 3]>, // Stores R,G,B for inputs 0..=255
}

impl ColorLut {
    fn new(map_type: ColorMapType) -> Self {
        let gradient = match map_type {
            ColorMapType::Magma => colorous::MAGMA,
            ColorMapType::Inferno => colorous::INFERNO,
            ColorMapType::Viridis => colorous::VIRIDIS,
            ColorMapType::Plasma => colorous::PLASMA,
            ColorMapType::Turbo => colorous::TURBO,
            ColorMapType::Cubehelix => colorous::CUBEHELIX,
        };

        let mut bytes = Vec::with_capacity(256);
        for i in 0..=255 {
            let val = i as f64 / 255.0;
            let c = gradient.eval_continuous(val);
            bytes.push([c.r, c.g, c.b]);
        }
        Self { bytes }
    }

    #[inline(always)]
    fn get_color(&self, intensity: f32) -> [u8; 3] {
        let idx = (intensity.clamp(0.0, 1.0) * 255.0) as usize;
        self.bytes[idx]
    }
}

struct SpectrogramLayer {
    fft_size: usize,
    freq_bins: usize,
    pixels: Vec<u8>,
    head: usize,         
    total_updates: u64, 
}

impl SpectrogramLayer {
    fn new(fft_size: usize) -> Self {
        let bins = fft_size / 2;
        Self {
            fft_size,
            freq_bins: bins,
            pixels: vec![0u8; MAX_HISTORY * bins * 4],
            head: 0,
            total_updates: 0,
        }
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "spector-bg".to_owned(),
        window_width: 1024,
        window_height: 768,
        high_dpi: true,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
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
        let frag_size = (SAMPLE_RATE as u32 * 4 * 15) / 1000;
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
        let mut pending_buffer = Vec::with_capacity(4096);

        // Store float history for redraws
        let mut float_history: Vec<Vec<f32>> = RESOLUTIONS.iter().map(|&size| {
            vec![0.0f32; MAX_HISTORY * (size / 2)]
        }).collect();

        // Local LUT cache to avoid creating it every pixel
        let mut current_lut_type = ColorMapType::Magma;
        let mut lut = ColorLut::new(current_lut_type);

        loop {
            // 1. Check Flags & Redraw
            let mut local_settings = None;
            let mut redraw_requested = false;

            {
                let mut s = settings_ref.lock().unwrap();
                if s.redraw_flag {
                    redraw_requested = true;
                    s.redraw_flag = false;
                    // Update local LUT if needed
                    if s.colormap != current_lut_type {
                        current_lut_type = s.colormap;
                        lut = ColorLut::new(current_lut_type);
                    }
                }
                // Clone settings for use outside lock
                local_settings = Some(s.clone());
            }

            let settings = local_settings.unwrap();

            // Handle Redraw (Expensive but infrequent)
            if redraw_requested {
                if let Ok(mut layers) = layers_ref.lock() {
                    for (i, layer) in layers.iter_mut().enumerate() {
                        for t in 0..MAX_HISTORY {
                            let start = t * layer.freq_bins;
                            let slice = &float_history[i][start..start + layer.freq_bins];
                            paint_column_fast(&mut layer.pixels, t, slice, &settings, &lut, layer.freq_bins, layer.fft_size);
                        }
                    }
                }
            }

            // 2. Audio Ingest
            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                pending_buffer.extend(temp);
            }

            // 3. Processing
            while pending_buffer.len() >= HOP_SIZE {
                let chunk: Vec<f32> = pending_buffer.drain(0..HOP_SIZE).collect();
                
                rolling_audio.extend(chunk);
                if rolling_audio.len() > max_fft {
                    let remove = rolling_audio.len() - max_fft;
                    rolling_audio.drain(0..remove);
                }

                // Prepare Data Containers to avoid holding lock during FFT
                struct UpdateData {
                    idx: usize,
                    raw_col: Vec<f32>,
                    bins: usize,
                    fft: usize,
                }
                let mut updates: Vec<UpdateData> = Vec::with_capacity(RESOLUTIONS.len());

                // STEP A: Calculate FFTs (No Lock)
                for (i, size) in RESOLUTIONS.iter().enumerate() {
                    if rolling_audio.len() < *size { continue; }
                    
                    let start_sample = rolling_audio.len() - size;
                    let audio_slice = &rolling_audio[start_sample..];
                    let windowed = hann_window(audio_slice);

                    if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                        let raw_col: Vec<f32> = spectrum.data().iter().map(|(_, v)| v.val()).collect();
                        updates.push(UpdateData { 
                            idx: i, 
                            raw_col, 
                            bins: *size / 2, 
                            fft: *size 
                        });
                    }
                }

                // STEP B: Update State (Lock)
                // We only lock when we are ready to write pixels
                if !updates.is_empty() {
                    if let Ok(mut layers) = layers_ref.lock() {
                        for up in updates {
                            let layer = &mut layers[up.idx];
                            let head_idx = layer.head;

                            // Save Floats to history
                            let float_start = head_idx * layer.freq_bins;
                            let copy_len = up.raw_col.len().min(layer.freq_bins);
                            if float_start + copy_len <= float_history[up.idx].len() {
                                float_history[up.idx][float_start..float_start+copy_len].copy_from_slice(&up.raw_col[0..copy_len]);
                            }

                            // Paint Pixels using LUT
                            paint_column_fast(&mut layer.pixels, head_idx, &up.raw_col, &settings, &lut, layer.freq_bins, layer.fft_size);

                            layer.head = (layer.head + 1) % MAX_HISTORY;
                            layer.total_updates += 1;
                        }
                    }
                }
            }
            
            if pending_buffer.len() < HOP_SIZE {
                thread::sleep(Duration::from_millis(1));
            }
        }
    });

    // --- THREAD C: FACE ---
    let mut current_fft_idx = 1;
    let mut last_fft_idx = 1; 
    let mut current_view_len = MAX_HISTORY; 
    let mut current_height = RESOLUTIONS[current_fft_idx] / 2;

    let mut texture = Texture2D::from_image(&Image {
        width: MAX_HISTORY as u16, height: current_height as u16,
        bytes: vec![0; MAX_HISTORY * current_height * 4],
    });
    texture.set_filter(FilterMode::Linear);

    let mut local_mel = true;
    let mut local_cmap = ColorMapType::Magma;
    let mut local_dir = ScrollDirection::RTL;
    let mut smooth_head_pos: f64 = 0.0;

    loop {
        let mut visual_changed = false;

        if is_key_pressed(KeyCode::S) { local_mel = !local_mel; visual_changed = true; }
        if is_key_pressed(KeyCode::C) { local_cmap = cycle_colormap(local_cmap); visual_changed = true; }
        if is_key_pressed(KeyCode::F) { local_dir = cycle_direction(local_dir); }
        if is_key_pressed(KeyCode::R) { current_fft_idx = (current_fft_idx + 1) % RESOLUTIONS.len(); }

        if is_key_pressed(KeyCode::W) {
            current_view_len = if current_view_len == MAX_HISTORY { MAX_HISTORY / 2 } else { MAX_HISTORY };
        }

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
                width: MAX_HISTORY as u16, height: current_height as u16,
                bytes: vec![0; MAX_HISTORY * current_height * 4],
            });
            texture.set_filter(FilterMode::Linear);
        }

        let mut actual_total_updates = 0u64;

        // Scope the lock to just data extraction to free it up for the audio thread
        let pixels_clone = {
            if let Ok(layers) = shared_layers.lock() {
                let layer = &layers[current_fft_idx];
                actual_total_updates = layer.total_updates;
                
                if current_fft_idx != last_fft_idx {
                    let prev_updates = layers[last_fft_idx].total_updates;
                    let diff = actual_total_updates as f64 - prev_updates as f64;
                    smooth_head_pos += diff;
                    last_fft_idx = current_fft_idx;
                }
                // We must clone here because Texture update needs an Image which needs ownership.
                // Optimizing this would require using raw miniquad bindings or unsafe, 
                // but by releasing the lock immediately, we prevent audio stutter.
                Some(layer.pixels.clone())
            } else {
                None
            }
        };

        if let Some(px) = pixels_clone {
             let img = Image {
                width: MAX_HISTORY as u16, height: current_height as u16,
                bytes: px,
            };
            texture.update(&img);
        }

        clear_background(BLACK);
        let sw = screen_width();
        let sh = screen_height();

        // --- SMOOTH SCROLL ---
        let target_pos = actual_total_updates as f64;
        smooth_head_pos += (target_pos - smooth_head_pos) * 0.5;
        if (target_pos - smooth_head_pos).abs() > 50.0 { smooth_head_pos = target_pos; }
        
        let head_snapped = (smooth_head_pos % MAX_HISTORY as f64).floor() as f32;
        let head_offset = head_snapped;

        let (screen_time_dim, screen_freq_dim) = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (sw, sh),
            ScrollDirection::DownToUp | ScrollDirection::UpToDown => (sh, sw),
        };

        let scale_factor = TARGET_DISPLAY_WIDTH / (current_view_len as f32);
        let needed_source_w = screen_time_dim / scale_factor;

        let (final_source_w, _) = if needed_source_w <= current_view_len as f32 {
            (needed_source_w, screen_time_dim)
        } else {
            (current_view_len as f32, screen_time_dim)
        };

        let final_source_w_snapped = final_source_w.round();
        let start_pos_unwrapped = head_offset - final_source_w_snapped;
        let tex_w = MAX_HISTORY as f32;
        let tex_h = current_height as f32;

        let is_horizontal = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => true,
            _ => false
        };

        let (src1, src2) = if start_pos_unwrapped < 0.0 {
            let overflow = start_pos_unwrapped.abs();
            let s1 = Rect::new(tex_w - overflow, 0.0, overflow, tex_h);
            let s2 = Rect::new(0.0, 0.0, head_offset, tex_h);
            (Some(s1), Some(s2))
        } else {
            let s1 = Rect::new(start_pos_unwrapped, 0.0, final_source_w_snapped, tex_h);
            (Some(s1), None)
        };

        let dst1_len = src1.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let dst2_len = src2.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let total_draw_len = dst1_len + dst2_len;
        let start_pos_screen = (screen_time_dim - total_draw_len) / 2.0;

        // --- DRAWING ---
        if is_horizontal {
            let is_ltr = local_dir == ScrollDirection::LTR;
            let flip_x = is_ltr;

            if is_ltr {
                if let Some(s2) = src2 {
                    draw_texture_ex(&texture, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)), source: Some(s2), flip_x, ..Default::default()
                    });
                }
                if let Some(s1) = src1 {
                    draw_texture_ex(&texture, start_pos_screen + dst2_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), source: Some(s1), flip_x, ..Default::default()
                    });
                }
            } else {
                if let Some(s1) = src1 {
                    draw_texture_ex(&texture, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), source: Some(s1), flip_x, ..Default::default()
                    });
                }
                if let Some(s2) = src2 {
                    draw_texture_ex(&texture, start_pos_screen + dst1_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)), source: Some(s2), flip_x, ..Default::default()
                    });
                }
            }
        } else {
            let is_fire = local_dir == ScrollDirection::DownToUp;
            if is_fire {
                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;
                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s1), rotation: std::f32::consts::FRAC_PI_2, ..Default::default()
                    });
                }
                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let y_offset = start_pos_screen + dst1_len;
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;
                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s2), rotation: std::f32::consts::FRAC_PI_2, ..Default::default()
                    });
                }
            } else {
                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;
                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s2), rotation: -std::f32::consts::FRAC_PI_2, flip_y: true, ..Default::default()
                    });
                }
                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let y_offset = start_pos_screen + dst2_len;
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;
                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s1), rotation: -std::f32::consts::FRAC_PI_2, flip_y: true, ..Default::default()
                    });
                }
            }
        }

        draw_note_ruler(local_mel, local_dir, RESOLUTIONS[current_fft_idx]);
        draw_ui_overlay(local_mel, local_cmap, local_dir, RESOLUTIONS[current_fft_idx], current_view_len);

        next_frame().await
    }
}

// --- FAST PAINTER (Uses LUT) ---
fn paint_column_fast(
    pixels: &mut Vec<u8>, 
    col_idx: usize, 
    data: &[f32], 
    settings: &AppSettings, 
    lut: &ColorLut, // Use LUT instead of re-calculating
    freq_bins: usize, 
    fft_size: usize
) {
    let max_freq_idx = data.len();
    let mel_min = 0.0;
    let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();
    let mel_normalization_factor = 2595.0; // pre-calc optimization constant could go deeper but this is fine

    let x_offset = col_idx * 4;
    // Pre-calculate constants for loop
    let sample_rate_over_fft = SAMPLE_RATE as f32 / fft_size as f32;

    for i in 0..freq_bins {
        // Invert Y axis here (standard spectrogram view)
        let y = (freq_bins - 1) - i;
        
        let target_idx = if settings.mel_scale {
            let norm_i = i as f32 / freq_bins as f32;
            let mel = norm_i * (mel_max - mel_min) + mel_min;
            let freq = 700.0 * (10.0f32.powf(mel / mel_normalization_factor) - 1.0);
            (freq / sample_rate_over_fft) as usize
        } else {
            // Linear scale
            let norm_i = i as f32 / freq_bins as f32;
            (norm_i * max_freq_idx as f32) as usize
        };

        let safe_idx = target_idx.min(max_freq_idx - 1);
        let magnitude = data[safe_idx];
        
        // Intensity mapping
        let intensity = (magnitude * 2000.0).ln() / 8.0;
        
        // LUT Lookup (Fast)
        let rgb = lut.get_color(intensity);

        let idx = (y * MAX_HISTORY * 4) + x_offset;

        // Unsafe check skipped for speed, logic guarantees bounds
        if idx + 3 < pixels.len() {
            pixels[idx] = rgb[0];
            pixels[idx+1] = rgb[1];
            pixels[idx+2] = rgb[2];
            pixels[idx+3] = 255;
        }
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