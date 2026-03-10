use macroquad::prelude::*;
use ringbuf::HeapRb;
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};
use spectrum_analyzer::scaling::divide_by_N_sqrt;
use spectrum_analyzer::windows::hann_window;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// PulseAudio bindings for audio capture
use libpulse_binding::sample::{Spec, Format};
use libpulse_binding::stream::Direction;
use libpulse_binding::def::BufferAttr;
use libpulse_simple_binding::Simple;

// --- CONFIG ---
const SAMPLE_RATE: u32 = 44100;
const HOP_SIZE: usize = 512;

const RESOLUTIONS: [usize; 3] = [2048, 4096, 8192];
// View length is 2520, but the Ring Buffer is 2800 to give the write-head an invisible margin!
const MAX_VIEW_LEN: usize = 2520; 
const MAX_HISTORY: usize = 2800; 
const TARGET_DISPLAY_WIDTH: f32 = 2520.0;

#[cfg(feature = "bg")]
const DRAW_UI: bool = false;

#[cfg(not(feature = "bg"))]
const DRAW_UI: bool = true;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix }

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScrollDirection { RTL, LTR, DTU, UTD }

#[derive(Debug, Clone, Copy, PartialEq)]
enum AudioSource { SinkMonitor, Microphone }

/// Shared state between the Brain (processing) and Face (UI) threads.
#[derive(Clone)]
struct AppSettings {
    mel_scale: bool,
    colormap: ColorMapType,
    redraw_flag: bool,
    audio_source: AudioSource,
}

// 1D Colormap Generator for the GPU Shader
struct ColorLut {
    bytes: Vec<[u8; 3]>,
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
}

fn create_colormap_texture(lut: &ColorLut) -> Texture2D {
    let mut bytes = Vec::with_capacity(256 * 4);
    for i in 0..256 {
        let rgb = lut.bytes[i];
        bytes.push(rgb[0]); bytes.push(rgb[1]); bytes.push(rgb[2]); bytes.push(255);
    }
    let img = Image { width: 256, height: 1, bytes };
    let tex = Texture2D::from_image(&img);
    tex.set_filter(FilterMode::Linear);
    tex
}

// Optimization: Pre-calculated Branchless Crossfade Map
#[derive(Clone, Copy)]
struct CrossfadeInstruction {
    b_8192: usize, w_8192: f32,
    b_4096: usize, w_4096: f32,
    b_2048: usize, w_2048: f32,
}

fn build_crossfade_map() -> Vec<CrossfadeInstruction> {
    let bins = RESOLUTIONS[2] / 2;
    let mut map = Vec::with_capacity(bins);
    
    let freq_res_8192 = SAMPLE_RATE as f32 / 8192.0; 
    let freq_res_4096 = SAMPLE_RATE as f32 / 4096.0; 
    let freq_res_2048 = SAMPLE_RATE as f32 / 2048.0; 

    for bin in 0..bins {
        let freq = bin as f32 * freq_res_8192;
        
        let bin_4096 = ((freq / freq_res_4096) as usize).min(2047);
        let bin_2048 = ((freq / freq_res_2048) as usize).min(1023);

        let mut inst = CrossfadeInstruction {
            b_8192: bin, w_8192: 0.0,
            b_4096: bin_4096, w_4096: 0.0,
            b_2048: bin_2048, w_2048: 0.0,
        };

        if freq < 200.0 {
            inst.w_8192 = 1.0;
        } else if freq < 300.0 {
            let t = (freq - 200.0) / 100.0;
            inst.w_8192 = 1.0 - t;
            inst.w_4096 = t;
        } else if freq < 1200.0 {
            inst.w_4096 = 1.0;
        } else if freq < 2000.0 {
            let t = (freq - 1200.0) / 800.0;
            inst.w_4096 = 1.0 - t;
            inst.w_2048 = t;
        } else {
            inst.w_2048 = 1.0; 
        }
        map.push(inst);
    }
    map
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

const VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
varying lowp vec2 uv;
varying lowp vec4 color;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    color = color0;
    uv = texcoord;
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 100
precision highp float;
varying lowp vec2 uv;
varying lowp vec4 color;
uniform sampler2D Texture;
uniform sampler2D colormap;
uniform float use_mel;
uniform float sample_rate;

void main() {
    // uv.y = 0.0 is the top of the screen (Nyquist). uv.y = 1.0 is the bottom (DC).
    // By inverting uv.y, norm_f becomes 0.0 at the bottom (DC) and 1.0 at the top (Nyquist).
    float norm_f = 1.0 - uv.y; 

    // GPU-accelerated Mel-scale stretching
    if (use_mel > 0.5) {
        float max_freq = sample_rate / 2.0;
        
        float mel_max = 2595.0 * (log(1.0 + max_freq / 700.0) / 2.30258509299);
        float current_mel = norm_f * mel_max;
        
        float current_hz = 700.0 * (exp((current_mel / 2595.0) * 2.30258509299) - 1.0);
        
        norm_f = current_hz / max_freq;
    }

    // The CPU paints DC at the bottom of the memory array (y = max), which translates to OpenGL v = 0.0 natively.
    // Since norm_f is 0.0 at the bottom of the screen, passing it straight through perfectly aligns DC to DC.
    vec2 sample_uv = vec2(uv.x, norm_f);

    // Read the linear grayscale intensity computed by the CPU
    float intensity = texture2D(Texture, sample_uv).r;
    
    // Apply 1D Color LUT natively on GPU
    gl_FragColor = texture2D(colormap, vec2(intensity, 0.5));
}
"#;

fn window_conf() -> Conf {
    Conf {
        window_title: if cfg!(feature = "bg") { "spector-bg" } else { "spector" }.to_owned(),
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
    layers.push(SpectrogramLayer::new(RESOLUTIONS[2])); 

    let shared_layers = Arc::new(Mutex::new(layers));
    let shared_settings = Arc::new(Mutex::new(AppSettings {
        mel_scale: true,
        colormap: ColorMapType::Magma,
        redraw_flag: false,
        audio_source: AudioSource::SinkMonitor,
    }));

    let rb = HeapRb::<f32>::new(65536);
    let (mut producer, mut consumer) = rb.split();

    // ==========================================
    // --- THREAD A: RECORDER (Audio Ingest) ---
    // ==========================================
    let shared_settings_recorder = shared_settings.clone();
    
    thread::spawn(move || {
        let mut current_source = AudioSource::SinkMonitor;

        let open_stream = |source: AudioSource| -> Option<Simple> {
            let device_name = match source {
                AudioSource::SinkMonitor => {
                    if let Ok(output) = std::process::Command::new("pactl").arg("get-default-sink").output() {
                        let sink_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        format!("{}.monitor", sink_name) 
                    } else {
                        return None;
                    }
                },
                AudioSource::Microphone => {
                    if let Ok(output) = std::process::Command::new("pactl").arg("get-default-source").output() {
                        String::from_utf8_lossy(&output.stdout).trim().to_string()
                    } else {
                        return None;
                    }
                }
            };

            let spec = Spec { format: Format::F32le, channels: 1, rate: SAMPLE_RATE };
            let frag_size = (SAMPLE_RATE as u32 * 4 * 15) / 1000; 
            let attr = BufferAttr {
                maxlength: u32::MAX, tlength: u32::MAX, prebuf: u32::MAX, minreq: u32::MAX,
                fragsize: frag_size,
            };

            Simple::new(
                None,
                if cfg!(feature = "bg") { "spector-bg" } else { "spector" },
                Direction::Record,
                Some(&device_name),
                "Recorder", &spec,
                None,
                Some(&attr)
            ).ok()
        };

        let mut stream = open_stream(current_source);
        let mut buf = [0u8; 4096];

        loop {
            if let Ok(settings) = shared_settings_recorder.try_lock() {
                if settings.audio_source != current_source {
                    current_source = settings.audio_source;
                    stream = open_stream(current_source);
                }
            }

            if let Some(ref s) = stream {
                if let Ok(_) = s.read(&mut buf) {
                    let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) };
                    producer.push_slice(floats);
                } else {
                    thread::sleep(Duration::from_millis(100));
                    stream = open_stream(current_source);
                }
            } else {
                thread::sleep(Duration::from_millis(100));
                stream = open_stream(current_source);
            }
        }
    });

    // ====================================================
    // --- THREAD B: BRAIN (FFT & Pixel Processing) ---
    // ====================================================
    let layers_ref = shared_layers.clone();
    let settings_ref = shared_settings.clone();

    thread::spawn(move || {
        let max_fft = *RESOLUTIONS.iter().max().unwrap();
        
        let mut rolling_audio = vec![0.0; max_fft];
        let mut pending_buffer = Vec::with_capacity(4096);

        let mut float_history: Vec<Vec<f32>> = (0..=RESOLUTIONS.len()).map(|i| {
            let size = if i < RESOLUTIONS.len() { RESOLUTIONS[i] } else { RESOLUTIONS[2] };
            vec![0.0f32; MAX_HISTORY * (size / 2)]
        }).collect();

        let mut local_cols = vec![
            vec![0.0; RESOLUTIONS[0] / 2],
            vec![0.0; RESOLUTIONS[1] / 2],
            vec![0.0; RESOLUTIONS[2] / 2],
            vec![0.0; RESOLUTIONS[2] / 2], 
        ];
        let crossfade_map = build_crossfade_map();

        loop {
            let mut redraw_requested = false;

            if let Ok(mut s) = settings_ref.lock() {
                if s.redraw_flag {
                    redraw_requested = true;
                    s.redraw_flag = false;
                }
            }

            if redraw_requested {
                if let Ok(mut layers) = layers_ref.lock() {
                    for (i, layer) in layers.iter_mut().enumerate() {
                        for t in 0..MAX_HISTORY {
                            let start = t * layer.freq_bins;
                            let slice = &float_history[i][start..start + layer.freq_bins];
                            paint_column_fast(&mut layer.pixels, t, slice, layer.freq_bins);
                        }
                        layer.total_updates += (MAX_HISTORY * 2) as u64; 
                    }
                }
            }

            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                pending_buffer.extend(temp);
            }

            while pending_buffer.len() >= HOP_SIZE {
                let chunk: Vec<f32> = pending_buffer.drain(0..HOP_SIZE).collect();
                
                rolling_audio.extend(&chunk);
                
                if rolling_audio.len() > max_fft {
                    let remove = rolling_audio.len() - max_fft;
                    rolling_audio.drain(0..remove);
                }

                let mut computed_all = true;

                for (i, size) in RESOLUTIONS.iter().enumerate() {
                    if rolling_audio.len() < *size { computed_all = false; continue; }
                    
                    let start_sample = rolling_audio.len() - size;
                    let audio_slice = &rolling_audio[start_sample..];
                    let windowed = hann_window(audio_slice);

                    if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                        for (out, val) in local_cols[i].iter_mut().zip(spectrum.data().iter()) {
                            *out = val.1.val();
                        }
                    } else { computed_all = false; }
                }

                if computed_all {
                    for (bin, inst) in crossfade_map.iter().enumerate() {
                        local_cols[3][bin] = 
                            local_cols[2][inst.b_8192] * inst.w_8192 +
                            local_cols[1][inst.b_4096] * inst.w_4096 +
                            local_cols[0][inst.b_2048] * inst.w_2048;
                    }

                    if let Ok(mut layers) = layers_ref.lock() {
                        for i in 0..=RESOLUTIONS.len() {
                            let layer = &mut layers[i];
                            let head_idx = layer.head;

                            let float_start = head_idx * layer.freq_bins;
                            let copy_len = local_cols[i].len().min(layer.freq_bins);
                            if float_start + copy_len <= float_history[i].len() {
                                float_history[i][float_start..float_start+copy_len].copy_from_slice(&local_cols[i][0..copy_len]);
                            }

                            paint_column_fast(&mut layer.pixels, head_idx, &local_cols[i], layer.freq_bins);

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

    // ===============================================
    // --- THREAD C: FACE (UI & GPU Rendering) ---
    // ===============================================
    let mut current_fft_idx = 3; 
    let mut last_fft_idx = 3; 
    let mut current_view_len = MAX_VIEW_LEN; 
    
    let mut render_targets = Vec::new();
    for i in 0..=RESOLUTIONS.len() {
        let size = if i < RESOLUTIONS.len() { RESOLUTIONS[i] } else { RESOLUTIONS[2] };
        let rt = render_target(MAX_HISTORY as u32, (size / 2) as u32);
        rt.texture.set_filter(FilterMode::Linear);
        render_targets.push(rt);
    }
    let mut last_rendered_updates = vec![0u64; RESOLUTIONS.len() + 1];

    let mut local_mel = true;
    let mut local_cmap = ColorMapType::Magma;
    let mut local_dir = ScrollDirection::RTL;
    let mut local_source = AudioSource::SinkMonitor;
    let mut smooth_head_pos: f64 = 0.0;
    
    let mut current_lut_type = local_cmap;
    let lut = ColorLut::new(current_lut_type);
    let mut colormap_texture = create_colormap_texture(&lut);

    let shader_material = load_material(
        ShaderSource::Glsl { vertex: VERTEX_SHADER, fragment: FRAGMENT_SHADER },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("use_mel", UniformType::Float1),
                UniformDesc::new("sample_rate", UniformType::Float1),
            ],
            textures: vec!["colormap".to_string()],
            ..Default::default()
        },
    ).unwrap();

    let max_possible_height = *RESOLUTIONS.iter().max().unwrap() / 2;
    let mut local_pixels_buffer = vec![0u8; MAX_HISTORY * max_possible_height * 4];

    loop {
        let mut visual_changed = false;
        let mut source_changed = false;

        if is_key_pressed(KeyCode::S) { local_mel = !local_mel; } 
        if is_key_pressed(KeyCode::C) { local_cmap = cycle_colormap(local_cmap); visual_changed = true; }
        if is_key_pressed(KeyCode::F) { local_dir = cycle_direction(local_dir); }
        if is_key_pressed(KeyCode::R) { current_fft_idx = (current_fft_idx + 1) % (RESOLUTIONS.len() + 1); } 
        if is_key_pressed(KeyCode::X) { 
            local_source = match local_source {
                AudioSource::SinkMonitor => AudioSource::Microphone,
                AudioSource::Microphone => AudioSource::SinkMonitor,
            };
            source_changed = true; 
        }

        if is_key_pressed(KeyCode::W) {
            current_view_len = if current_view_len == MAX_VIEW_LEN { MAX_VIEW_LEN / 2 } else { MAX_VIEW_LEN };
        }

        if visual_changed || source_changed {
            if visual_changed && local_cmap != current_lut_type {
                current_lut_type = local_cmap;
                let lut = ColorLut::new(current_lut_type);
                colormap_texture = create_colormap_texture(&lut);
            }
            if let Ok(mut s) = shared_settings.lock() {
                s.mel_scale = local_mel;
                s.colormap = local_cmap;
                s.audio_source = local_source;
            }
        }

        let display_fft_size = if current_fft_idx == RESOLUTIONS.len() { RESOLUTIONS[2] } else { RESOLUTIONS[current_fft_idx] };
        let current_height = display_fft_size / 2;

        let mut actual_total_updates = 0u64;
        let mut full_redraw_needed = false;
        let mut delta_images: Vec<(f32, Image)> = Vec::new();

        {
            if let Ok(layers) = shared_layers.lock() {
                let layer = &layers[current_fft_idx];
                actual_total_updates = layer.total_updates;
                
                let prev_updates = last_rendered_updates[current_fft_idx];
                let diff = if actual_total_updates > prev_updates { actual_total_updates - prev_updates } else { 0 };
                
                let full_redraw_threshold = (MAX_HISTORY / 2) as u64;

                if diff >= full_redraw_threshold {
                    let expected_len = layer.pixels.len();
                    if local_pixels_buffer.len() < expected_len {
                        local_pixels_buffer.resize(expected_len, 0);
                    }
                    local_pixels_buffer[..expected_len].copy_from_slice(&layer.pixels);
                    full_redraw_needed = true;
                } else if diff > 0 {
                    let diff_usize = diff as usize;
                    let head = layer.head;
                    let height = layer.freq_bins;

                    if head >= diff_usize {
                        let start_x = head - diff_usize;
                        let mut img = Image { width: diff as u16, height: height as u16, bytes: vec![0; diff_usize * height * 4] };
                        for y in 0..height {
                            let src_row = y * MAX_HISTORY * 4;
                            let dst_row = y * diff_usize * 4;
                            img.bytes[dst_row .. dst_row + diff_usize * 4]
                                .copy_from_slice(&layer.pixels[src_row + start_x * 4 .. src_row + (start_x + diff_usize) * 4]);
                        }
                        delta_images.push((start_x as f32, img));
                    } else {
                        let part1_len = diff_usize - head;
                        let start_x = MAX_HISTORY - part1_len;
                        
                        let mut img1 = Image { width: part1_len as u16, height: height as u16, bytes: vec![0; part1_len * height * 4] };
                        for y in 0..height {
                            let src_row = y * MAX_HISTORY * 4;
                            let dst_row = y * part1_len * 4;
                            img1.bytes[dst_row .. dst_row + part1_len * 4]
                                .copy_from_slice(&layer.pixels[src_row + start_x * 4 .. src_row + MAX_HISTORY * 4]);
                        }
                        delta_images.push((start_x as f32, img1));

                        if head > 0 {
                            let mut img2 = Image { width: head as u16, height: height as u16, bytes: vec![0; head * height * 4] };
                            for y in 0..height {
                                let src_row = y * MAX_HISTORY * 4;
                                let dst_row = y * head * 4;
                                img2.bytes[dst_row .. dst_row + head * 4]
                                    .copy_from_slice(&layer.pixels[src_row .. src_row + head * 4]);
                            }
                            delta_images.push((0.0, img2));
                        }
                    }
                }
                
                if current_fft_idx != last_fft_idx {
                    let prev_mode_updates = layers[last_fft_idx].total_updates;
                    let diff_from_last = actual_total_updates as f64 - prev_mode_updates as f64;
                    smooth_head_pos += diff_from_last;
                    last_fft_idx = current_fft_idx;
                }
            }
        }

        if full_redraw_needed {
            let img = Image {
                width: MAX_HISTORY as u16, height: current_height as u16,
                bytes: local_pixels_buffer[..(MAX_HISTORY * current_height * 4)].to_vec(),
            };
            let tex = Texture2D::from_image(&img);
            
            let mut cam = Camera2D::from_display_rect(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32));
            cam.render_target = Some(render_targets[current_fft_idx].clone());
            set_camera(&cam);
            draw_texture(&tex, 0.0, 0.0, WHITE);
            set_default_camera();
            last_rendered_updates[current_fft_idx] = actual_total_updates;
        } else if !delta_images.is_empty() {
            let mut cam = Camera2D::from_display_rect(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32));
            cam.render_target = Some(render_targets[current_fft_idx].clone());
            set_camera(&cam);
            
            for (x, img) in delta_images {
                let tex = Texture2D::from_image(&img);
                tex.set_filter(FilterMode::Nearest); 
                draw_texture(&tex, x, 0.0, WHITE);
            }
            set_default_camera();
            last_rendered_updates[current_fft_idx] = actual_total_updates;
        }

        clear_background(BLACK);
        let sw = screen_width();
        let sh = screen_height();

        let target_pos = actual_total_updates as f64;
        let diff = target_pos - smooth_head_pos;
        let dt = get_frame_time() as f64;
        
        smooth_head_pos += diff * (15.0 * dt).min(1.0); 
        if diff.abs() > 50.0 { smooth_head_pos = target_pos; } 
        
        let head_snapped = (smooth_head_pos.rem_euclid(MAX_HISTORY as f64)).floor() as f32;
        let head_offset = head_snapped;

        let (screen_time_dim, _screen_freq_dim) = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (sw, sh),
            ScrollDirection::DTU | ScrollDirection::UTD => (sh, sw),
        };

        let scale_factor = TARGET_DISPLAY_WIDTH / (current_view_len as f32);
        let needed_source_w = screen_time_dim / scale_factor;

        let (final_source_w, _) = if needed_source_w <= current_view_len as f32 {
            (needed_source_w, screen_time_dim)
        } else {
            (current_view_len as f32, screen_time_dim)
        };

        // Do not round the width here to ensure accurate unwrapping without sub-pixel accumulation
        let final_source_w_snapped = final_source_w;
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
            // Protect against 0-width draw calls which can cause a full-height line artifact
            let s2 = if head_offset > 0.001 {
                Some(Rect::new(0.0, 0.0, head_offset, tex_h))
            } else {
                None
            };
            (Some(s1), s2)
        } else {
            let s1 = Rect::new(start_pos_unwrapped, 0.0, final_source_w_snapped, tex_h);
            (Some(s1), None)
        };

        // Exact unceiled widths to perfectly align dst1 and dst2 edges without pixel overlap
        let dst1_len = src1.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let dst2_len = src2.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let total_draw_len = dst1_len + dst2_len;
        let start_pos_screen = (screen_time_dim - total_draw_len) / 2.0;

        let texture_to_draw = &render_targets[current_fft_idx].texture;

        shader_material.set_uniform("use_mel", if local_mel { 1.0f32 } else { 0.0f32 });
        shader_material.set_uniform("sample_rate", SAMPLE_RATE as f32);
        shader_material.set_texture("colormap", colormap_texture.clone());
        gl_use_material(&shader_material);

        if is_horizontal {
            let is_ltr = local_dir == ScrollDirection::LTR;
            let flip_x = is_ltr;

            if is_ltr {
                if let Some(s2) = src2 {
                    draw_texture_ex(texture_to_draw, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)), source: Some(s2), flip_x, ..Default::default()
                    });
                }
                if let Some(s1) = src1 {
                    draw_texture_ex(texture_to_draw, start_pos_screen + dst2_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), source: Some(s1), flip_x, ..Default::default()
                    });
                }
            } else {
                if let Some(s1) = src1 {
                    draw_texture_ex(texture_to_draw, start_pos_screen, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst1_len, sh)), source: Some(s1), flip_x, ..Default::default()
                    });
                }
                if let Some(s2) = src2 {
                    draw_texture_ex(texture_to_draw, start_pos_screen + dst1_len, 0.0, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(dst2_len, sh)), source: Some(s2), flip_x, ..Default::default()
                    });
                }
            }
        } else {
            let is_fire = local_dir == ScrollDirection::DTU;
            if is_fire {
                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;
                    draw_texture_ex(texture_to_draw, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s1), rotation: std::f32::consts::FRAC_PI_2, ..Default::default()
                    });
                }
                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let y_offset = start_pos_screen + dst1_len;
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;
                    draw_texture_ex(texture_to_draw, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s2), rotation: std::f32::consts::FRAC_PI_2, ..Default::default()
                    });
                }
            } else {
                if let Some(s2) = src2 {
                    let h = dst2_len;
                    let x = (sw - h) / 2.0;
                    let y = start_pos_screen + (h - sw) / 2.0;
                    draw_texture_ex(texture_to_draw, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s2), rotation: -std::f32::consts::FRAC_PI_2, flip_y: true, ..Default::default()
                    });
                }
                if let Some(s1) = src1 {
                    let h = dst1_len;
                    let y_offset = start_pos_screen + dst2_len;
                    let x = (sw - h) / 2.0;
                    let y = y_offset + (h - sw) / 2.0;
                    draw_texture_ex(texture_to_draw, x, y, WHITE, DrawTextureParams {
                        dest_size: Some(vec2(h, sw)), source: Some(s1), rotation: -std::f32::consts::FRAC_PI_2, flip_y: true, ..Default::default()
                    });
                }
            }
        }

        gl_use_default_material();

        draw_note_ruler(local_mel, local_dir, display_fft_size);
        if DRAW_UI { draw_ui_overlay(local_mel, local_cmap, local_dir, current_fft_idx, current_view_len, local_source); }

        if DRAW_UI {
            // --- Interactive Mouse Crosshair (Frequency/Note Peeking) ---
            let (mx, my) = mouse_position();
            
            // Re-map the mouse position to be relative to the *drawn area* (handling scaling and centering)
            let norm_time = if total_draw_len > 0.0 {
                match local_dir {
                    ScrollDirection::RTL => (start_pos_screen + total_draw_len - mx) / total_draw_len,
                    ScrollDirection::LTR => (mx - start_pos_screen) / total_draw_len,
                    ScrollDirection::DTU => (start_pos_screen + total_draw_len - my) / total_draw_len,
                    ScrollDirection::UTD => (my - start_pos_screen) / total_draw_len,
                }
            } else {
                -1.0
            };

            let norm_freq = match local_dir {
                ScrollDirection::RTL | ScrollDirection::LTR => 1.0 - (my / sh),
                ScrollDirection::DTU | ScrollDirection::UTD => mx / sw, 
            };

            // Only show tooltip if the mouse is hovering over the actual drawn area
            if norm_time >= 0.0 && norm_time <= 1.0 && norm_freq >= 0.0 && norm_freq <= 1.0 {
                let max_freq = SAMPLE_RATE as f32 / 2.0;
                let current_hz = if local_mel {
                    let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
                    let current_mel = norm_freq * mel_max;
                    700.0 * (10.0f32.powf(current_mel / 2595.0) - 1.0)
                } else {
                    norm_freq * max_freq
                };

                // The drawn duration depends on the ACTUAL width we rendered, not strictly current_view_len
                let drawn_time_seconds = (final_source_w_snapped * HOP_SIZE as f32) / SAMPLE_RATE as f32;
                let time_ago = norm_time * drawn_time_seconds;
                let (note_name, _) = hz_to_pitch(current_hz);

                draw_line(mx, 0.0, mx, sh, 1.0, Color::new(1.0, 1.0, 1.0, 0.3));
                draw_line(0.0, my, sw, my, 1.0, Color::new(1.0, 1.0, 1.0, 0.3));

                let mut intensity_u8 = 0;
                if let Ok(layers) = shared_layers.lock() {
                    let layer = &layers[current_fft_idx];
                    
                    // Recover the exact column relative to the current subpixel write-head
                    let exact_col = (head_offset - (norm_time * final_source_w_snapped) - 1.0).floor();
                    let ring_buffer_col = exact_col.rem_euclid(MAX_HISTORY as f32) as usize;
                    
                    let bin_idx = ((current_hz / max_freq) * layer.freq_bins as f32) as usize;
                    let bin_idx = bin_idx.clamp(0, layer.freq_bins.saturating_sub(1));
                    
                    let y = (layer.freq_bins - 1) - bin_idx;
                    let pixel_idx = (y * MAX_HISTORY * 4) + (ring_buffer_col * 4);
                    
                    if pixel_idx < layer.pixels.len() {
                        intensity_u8 = layer.pixels[pixel_idx];
                    }
                }

                let exact_mag = ((intensity_u8 as f32 / 255.0) * 8.0).exp() / 2000.0;
                let db = if exact_mag > 0.0001 { 20.0 * exact_mag.log10() } else { -100.0 };

                let tooltip_text = format!("-{:.2}s | {:.1} Hz | {} | {:.1} dB", time_ago, current_hz, note_name, db);
                let text_size = measure_text(&tooltip_text, None, 20, 1.0);
                
                let mut tooltip_x = mx + 15.0;
                let mut tooltip_y = my + 15.0;
                if tooltip_x + text_size.width + 10.0 > sw { tooltip_x = mx - text_size.width - 15.0; }
                if tooltip_y + 25.0 > sh { tooltip_y = my - 25.0; }

                draw_rectangle(tooltip_x, tooltip_y - 20.0, text_size.width + 10.0, 25.0, Color::new(0.0, 0.0, 0.0, 0.8));
                draw_text(&tooltip_text, tooltip_x + 5.0, tooltip_y - 2.0, 20.0, WHITE);
            }
        }

        next_frame().await
    }
}

// Highly optimized lock-free CPU memory prep. Converts f32 -> u8 intensity in a straight linear array.
#[inline(always)]
fn paint_column_fast(
    pixels: &mut [u8], 
    col_idx: usize, 
    data: &[f32], 
    freq_bins: usize, 
) {
    let x_offset = col_idx * 4;
    for i in 0..freq_bins {
        // By mapping i=0 (DC) to y=(max), DC sits at the bottom of the memory buffer
        // This corresponds perfectly with OpenGL's native v=0.0 texture coordinate.
        let y = (freq_bins - 1) - i; 
        
        let magnitude = data[i];
        let intensity = ((magnitude * 2000.0).ln() / 8.0).clamp(0.0, 1.0);
        let val = (intensity * 255.0) as u8;

        let idx = (y * MAX_HISTORY * 4) + x_offset;
        pixels[idx] = val;     // R
        pixels[idx+1] = val;   // G
        pixels[idx+2] = val;   // B
        pixels[idx+3] = 255;   // A
    }
}

fn hz_to_pitch(hz: f32) -> (String, f32) {
    if hz < 16.35 { return ("-".to_string(), 0.0); } 
    let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    
    let midi_float = 69.0 + 12.0 * (hz / 440.0).log2();
    let midi = midi_float.round() as i32;
    let cents = (midi_float - midi as f32) * 100.0;
    
    let octave = (midi / 12) - 1;
    let note_idx = (midi % 12).rem_euclid(12) as usize;
    
    let cent_str = if cents.abs() < 1.0 { "".to_string() } else if cents > 0.0 { format!(" +{:.0}c", cents) } else { format!(" {:.0}c", cents) };
    (format!("{}{}{}", note_names[note_idx], octave, cent_str), cents)
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
        ScrollDirection::LTR => ScrollDirection::DTU,
        ScrollDirection::DTU => ScrollDirection::UTD,
        ScrollDirection::UTD => ScrollDirection::RTL,
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
            ScrollDirection::UTD => (w * norm_pos, 0.0, midi % 12 == 0),
            ScrollDirection::DTU => (w * norm_pos, h, midi % 12 == 0),
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
            let start_y = if dir == ScrollDirection::DTU { y - tick_len } else { y };
            draw_line(x, start_y, x, start_y + tick_len, 1.0, WHITE);
            if is_c {
                let octave = (midi / 12) - 1;
                let text_y = if dir == ScrollDirection::DTU { y - 20.0 } else { y + 30.0 };
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

fn draw_ui_overlay(mel: bool, cmap: ColorMapType, dir: ScrollDirection, fft_idx: usize, history: usize, source: AudioSource) {
    let scale_str = if mel { "Mel" } else { "Linear" };
    let map_str = format!("{:?}", cmap); 
    let dir_str = match dir {
        ScrollDirection::RTL => "RTL", 
        ScrollDirection::LTR => "LTR",
        ScrollDirection::DTU => "Fire", 
        ScrollDirection::UTD => "Rain",
    };
    
    let res_str = if fft_idx == 3 {
        "Multi-Res".to_string()
    } else {
        format!("{} bins", RESOLUTIONS[fft_idx] / 2)
    };
    
    let hist_str = format!("{}", history);
    let src_str = match source {
        AudioSource::SinkMonitor => "Sink",
        AudioSource::Microphone => "Mic",
    };

    let stats = vec![
        UiStat { label: "Scale",  hotkey: Some('S'), value: scale_str.to_string(), color: ORANGE },
        UiStat { label: "Colour", hotkey: Some('C'), value: map_str,               color: YELLOW },
        UiStat { label: "Flow",   hotkey: Some('F'), value: dir_str.to_string(),   color: SKYBLUE },
        UiStat { label: "Res",    hotkey: Some('R'), value: res_str,               color: VIOLET },
        UiStat { label: "Win",    hotkey: Some('W'), value: hist_str,              color: PINK },
        UiStat { label: "X Source",     hotkey: Some('X'), value: src_str.to_string(),   color: GREEN },
    ];

    let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match dir {
        ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, screen_width(), 35.0, false),
        _ => (screen_width() - 220.0, 0.0, 220.0, 155.0, true),
    };

    draw_rectangle(bg_x, bg_y, bg_w, bg_h, Color::new(0.0, 0.0, 0.0, 0.6));

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
        let val_width = measure_text(&stat.value, None, 20, 1.0).width;

        if is_vertical {
            draw_text(&stat.value, cursor_x + label_width + 10.0, cursor_y, 20.0, stat.color);
            cursor_y += 22.0;
        } else {
            draw_text(&stat.value, cursor_x + label_width + 5.0, cursor_y, 20.0, stat.color);
            cursor_x += label_width + val_width + 25.0; 
        }
    }
}