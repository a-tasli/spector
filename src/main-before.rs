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
const MAX_HISTORY: usize = 1260; // Matches target width for 1:1 pixel mapping potential

// The history bins will be scaled to match this target pixel width on screen.
// 2048.0 means 1024 bins are stretched 2x to fill 2048 pixels.
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

struct SpectrogramLayer {
    fft_size: usize,
    freq_bins: usize,
    pixels: Vec<u8>,
    head: usize,         // Ring buffer write head
    total_updates: u64,  // For smooth scrolling synchronization
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
        window_title: "spector".to_owned(),
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
        
        let mut pending_buffer = Vec::with_capacity(4096);

        // Store float history to enable "redraws" (color changes) on existing data
        let mut float_history: Vec<Vec<f32>> = RESOLUTIONS.iter().map(|&size| {
            vec![0.0f32; MAX_HISTORY * (size / 2)]
        }).collect();

        loop {
            // 1. Check Flags
            let mut redraw_needed = false;
            let mut current_settings = None;

            {
                let mut s = settings_ref.lock().unwrap();
                if s.redraw_flag {
                    redraw_needed = true;
                    s.redraw_flag = false;
                    current_settings = Some(s.clone());
                }
            }

            // 2. Handle Redraw
            if redraw_needed {
                if let Some(settings) = current_settings {
                    if let Ok(mut layers) = layers_ref.lock() {
                        for (i, layer) in layers.iter_mut().enumerate() {
                            for t in 0..MAX_HISTORY {
                                let start = t * layer.freq_bins;
                                let slice = &float_history[i][start..start + layer.freq_bins];
                                paint_column_in_place(&mut layer.pixels, t, slice, &settings, layer.freq_bins, layer.fft_size);
                            }
                        }
                    }
                }
            }

            // 3. Audio Ingest
            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                pending_buffer.extend(temp);
            }

            // Process ALL available chunks of HOP_SIZE
            while pending_buffer.len() >= HOP_SIZE {
                let chunk: Vec<f32> = pending_buffer.drain(0..HOP_SIZE).collect();
                
                rolling_audio.extend(chunk);
                if rolling_audio.len() > max_fft {
                    let remove = rolling_audio.len() - max_fft;
                    rolling_audio.drain(0..remove);
                }

                // 4. Processing
                if let (Ok(mut layers), Ok(settings)) = (layers_ref.lock(), settings_ref.lock()) {
                    for (i, layer) in layers.iter_mut().enumerate() {
                        if rolling_audio.len() < layer.fft_size { continue; }

                        let start_sample = rolling_audio.len() - layer.fft_size;
                        let audio_slice = &rolling_audio[start_sample..];
                        let windowed = hann_window(audio_slice);

                        if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                            let raw_col: Vec<f32> = spectrum.data().iter().map(|(_, v)| v.val()).collect();

                            let head_idx = layer.head;

                            // Save Floats
                            let float_start = head_idx * layer.freq_bins;
                            let copy_len = raw_col.len().min(layer.freq_bins);
                            if float_start + copy_len <= float_history[i].len() {
                                float_history[i][float_start..float_start+copy_len].copy_from_slice(&raw_col[0..copy_len]);
                            }

                            // Paint Pixels
                            paint_column_in_place(&mut layer.pixels, head_idx, &raw_col, &settings, layer.freq_bins, layer.fft_size);

                            // Advance Head
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
    
    // Zoom State
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

    // Smooth Scroll State
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

        if let Ok(layers) = shared_layers.lock() {
            let layer = &layers[current_fft_idx];
            actual_total_updates = layer.total_updates;
            
            let img = Image {
                width: MAX_HISTORY as u16, height: current_height as u16,
                bytes: layer.pixels.clone(),
            };
            texture.update(&img);
        }

        if current_fft_idx != last_fft_idx {
            smooth_head_pos = actual_total_updates as f64;
            last_fft_idx = current_fft_idx;
        }

        clear_background(BLACK);
        let sw = screen_width();
        let sh = screen_height();

        // --- SMOOTH SCROLL LERP ---
        let target_pos = actual_total_updates as f64;
        smooth_head_pos += (target_pos - smooth_head_pos) * 0.15;
        if (target_pos - smooth_head_pos).abs() > 50.0 {
            smooth_head_pos = target_pos; 
        }
        
        let head_offset = (smooth_head_pos % MAX_HISTORY as f64) as f32;

        // --- VIEWPORT & CROP LOGIC ---
        let (screen_time_dim, screen_freq_dim) = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (sw, sh),
            ScrollDirection::DownToUp | ScrollDirection::UpToDown => (sh, sw),
        };

        let scale_factor = TARGET_DISPLAY_WIDTH / (current_view_len as f32);
        let needed_source_w = screen_time_dim / scale_factor;

        let (final_source_w, final_dest_w) = if needed_source_w <= current_view_len as f32 {
            (needed_source_w, screen_time_dim)
        } else {
            (current_view_len as f32, screen_time_dim)
        };

        let start_pos_unwrapped = head_offset - final_source_w;
        let tex_w = MAX_HISTORY as f32;
        let tex_h = current_height as f32;

        // --- RENDERING ---
        let is_horizontal = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => true,
            _ => false
        };

        // Determine segments based on wrap-around
        let (src1, src2) = if start_pos_unwrapped < 0.0 {
            let overflow = start_pos_unwrapped.abs();
            let s1 = Rect::new(tex_w - overflow, 0.0, overflow, tex_h);
            let s2 = Rect::new(0.0, 0.0, head_offset, tex_h);
            (Some(s1), Some(s2))
        } else {
            let s1 = Rect::new(start_pos_unwrapped, 0.0, final_source_w, tex_h);
            (Some(s1), None)
        };

        let dst1_len = src1.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let dst2_len = src2.map(|r| r.w * scale_factor).unwrap_or(0.0);
        
        // Center the visualization in the time axis
        let total_draw_len = dst1_len + dst2_len;
        let start_pos_screen = (screen_time_dim - total_draw_len) / 2.0;

        if is_horizontal {
            // --- HORIZONTAL MODE ---
            let is_ltr = local_dir == ScrollDirection::LTR;
            let flip_x = is_ltr;

            if is_ltr {
                // LTR: Swap order -> Draw s2 (New/Left) then s1 (Old/Right)
                if let Some(s2) = src2 {
                    draw_texture_ex(&texture, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)),
                        source: Some(s2),
                        flip_x,
                        ..Default::default()
                    });
                }

                if let Some(s1) = src1 {
                    // Draw s1 after s2
                    draw_texture_ex(&texture, start_pos_screen + dst2_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), 
                        source: Some(s1),
                        flip_x,
                        ..Default::default()
                    });
                }
            } else {
                // RTL (Standard): Draw s1 (Old/Left) then s2 (New/Right)
                if let Some(s1) = src1 {
                    draw_texture_ex(&texture, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), 
                        source: Some(s1),
                        flip_x,
                        ..Default::default()
                    });
                }

                if let Some(s2) = src2 {
                    draw_texture_ex(&texture, start_pos_screen + dst1_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)),
                        source: Some(s2),
                        flip_x,
                        ..Default::default()
                    });
                }
            }

        } else {
            // --- VERTICAL MODE ---
            // Manual rotation math to prevent scaling bugs.
            
            let is_fire = local_dir == ScrollDirection::DownToUp;

            if is_fire {
                // DownToUp (Fire)
                // Newest data appears at Bottom, pushes Up.
                // Time axis: Top (Old) -> Bottom (New).
                // Rotation: 90 deg (PI/2).
                
                // Seg1 (Old) first.
                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;
                    
                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)),
                        source: Some(s1),
                        rotation: std::f32::consts::FRAC_PI_2,
                        ..Default::default()
                    });
                }
                
                // Seg2 (New) second.
                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let y_offset = start_pos_screen + dst1_len;
                    
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;

                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)),
                        source: Some(s2),
                        rotation: std::f32::consts::FRAC_PI_2,
                        ..Default::default()
                    });
                }
            } else {
                // UpToDown (Rain)
                // Newest at Top. Oldest at Bottom.
                // Seg2 (New) first (Top). Seg1 (Old) second (Bottom).
                // Rotation: -90 deg (-PI/2). FlipY: True.

                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;

                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)),
                        source: Some(s2),
                        rotation: -std::f32::consts::FRAC_PI_2,
                        flip_y: true,
                        ..Default::default()
                    });
                }

                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let y_offset = start_pos_screen + dst2_len;
                    
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;

                    draw_texture_ex(&texture, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)),
                        source: Some(s1),
                        rotation: -std::f32::consts::FRAC_PI_2,
                        flip_y: true,
                        ..Default::default()
                    });
                }
            }
        }

        draw_note_ruler(local_mel, local_dir, RESOLUTIONS[current_fft_idx]);
        draw_ui_overlay(local_mel, local_cmap, local_dir, RESOLUTIONS[current_fft_idx], current_view_len);

        next_frame().await
    }
}

// --- HELPERS ---
fn paint_column_in_place(pixels: &mut Vec<u8>, col_idx: usize, data: &[f32], settings: &AppSettings, freq_bins: usize, fft_size: usize) {
    let gradient = get_gradient(settings.colormap);
    let max_freq_idx = data.len();
    let mel_min = 0.0;
    let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();

    let x_offset = col_idx * 4;

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

        let y = (freq_bins - 1) - i;
        let idx = (y * MAX_HISTORY * 4) + x_offset;

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

fn draw_ui_overlay(mel: bool, cmap: ColorMapType, dir: ScrollDirection, fft: usize, history: usize) {
    let scale_str = if mel { "Mel" } else { "Linear" };
    let map_str = format!("{:?}", cmap); 
    let dir_str = match dir {
        ScrollDirection::RTL => "RTL", 
        ScrollDirection::LTR => "LTR",
        ScrollDirection::DownToUp => "Fire", 
        ScrollDirection::UpToDown => "Rain",
    };
    let res_str = format!("{} bins", fft/2);
    let hist_str = format!("{}", history);

    let stats = vec![
        UiStat { label: "Scale",  hotkey: Some('S'), value: scale_str.to_string(), color: ORANGE },
        UiStat { label: "Colour", hotkey: Some('C'), value: map_str,               color: YELLOW },
        UiStat { label: "Flow",   hotkey: Some('F'), value: dir_str.to_string(),   color: SKYBLUE },
        UiStat { label: "Resolution",    hotkey: Some('R'), value: res_str,               color: VIOLET },
        UiStat { label: "Window",    hotkey: Some('W'), value: hist_str,              color: PINK },
    ];

    let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match dir {
        ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, screen_width(), 30.0, false),
        _ => (screen_width() - 200.0, 0.0, 220.0, 112.0, true),
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

        let label_width = measure_text(&full_label, None, 20, 1.0).width;
        
        if is_vertical {
            draw_text(&stat.value, cursor_x + label_width + 10.0, cursor_y, 20.0, stat.color);
            cursor_y += 20.0;
        } else {
            draw_text(&stat.value, cursor_x + label_width + 5.0, cursor_y, 20.0, stat.color);
            
            let val_width = measure_text(&stat.value, None, 20, 1.0).width;
            cursor_x += label_width + 5.0 + val_width + 40.0; 
        }
    }
}