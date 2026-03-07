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
// Number of audio samples to step forward per visual column. Lower = higher temporal resolution.
const HOP_SIZE: usize = 512;

// The different FFT window sizes used for the multi-resolution composite. 
// Smaller = better time accuracy; Larger = better frequency accuracy.
const RESOLUTIONS: [usize; 3] = [2048, 4096, 8192];
// How many pixel columns of history to store in RAM/VRAM.
const MAX_HISTORY: usize = 2520; 
const TARGET_DISPLAY_WIDTH: f32 = 2520.0;
// Massive window used specifically to isolate ultra-low sub-bass frequencies without muddying the rest.
const BASS_WINDOW: usize = 32768; 

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

// Optimization: Pre-calculated Color Lookup Table (LUT)
// Evaluates the continuous color gradient once at startup so the hot render loop 
// only performs a cheap array index lookup per pixel.
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

    #[inline(always)]
    fn get_color(&self, intensity: f32) -> [u8; 3] {
        let idx = (intensity.clamp(0.0, 1.0) * 255.0) as usize;
        self.bytes[idx]
    }
}

// Optimization: Pre-calculated Logarithmic (Mel) and Linear Coordinate Maps
// Maps a physical on-screen Y-pixel directly to the correct FFT frequency bin.
// Prevents calculating logarithmic scales for thousands of pixels every frame.
#[derive(Clone)]
struct YToBinMap {
    mel_map: Vec<usize>,
    linear_map: Vec<usize>,
}

impl YToBinMap {
    fn generate(freq_bins: usize, fft_size: usize) -> Self {
        let mut mel_map = vec![0; freq_bins];
        let mut linear_map = vec![0; freq_bins];
        
        let mel_min = 0.0;
        let mel_max = 2595.0 * (1.0 + (SAMPLE_RATE as f32 / 2.0) / 700.0).log10();
        let mel_normalization_factor = 2595.0;
        let sample_rate_over_fft = SAMPLE_RATE as f32 / fft_size as f32;

        for i in 0..freq_bins {
            // Calculate Mel (perceptual) bin index
            let norm_i = i as f32 / freq_bins as f32;
            let mel = norm_i * (mel_max - mel_min) + mel_min;
            let freq = 700.0 * (10.0f32.powf(mel / mel_normalization_factor) - 1.0);
            mel_map[i] = ((freq / sample_rate_over_fft) as usize).min(freq_bins - 1);
            
            // Calculate Linear bin index
            linear_map[i] = ((i as f32 / freq_bins as f32) * freq_bins as f32) as usize;
        }

        Self { mel_map, linear_map }
    }
}

// Optimization: Pre-calculated Branchless Crossfade Map
// Dictates how to blend the 3 standard FFTs + Bass FFT into a single composite view.
#[derive(Clone, Copy)]
struct CrossfadeInstruction {
    b_8192: usize, w_8192: f32,
    b_4096: usize, w_4096: f32,
    b_2048: usize, w_2048: f32,
    b_bass: usize, w_bass: f32,
}

fn build_crossfade_map() -> Vec<CrossfadeInstruction> {
    let bins = RESOLUTIONS[2] / 2;
    let mut map = Vec::with_capacity(bins);
    
    // Frequency resolution (Hz per bin) for each FFT size
    let freq_res_bass = (SAMPLE_RATE as f32 / 8.0) / 4096.0; 
    let freq_res_8192 = SAMPLE_RATE as f32 / 8192.0; 
    let freq_res_4096 = SAMPLE_RATE as f32 / 4096.0; 
    let freq_res_2048 = SAMPLE_RATE as f32 / 2048.0; 

    // Build blending weights for every vertical pixel/bin
    for bin in 0..bins {
        let freq = bin as f32 * freq_res_8192;
        
        // Find corresponding bins in the smaller FFTs
        let bin_4096 = ((freq / freq_res_4096) as usize).min(2047);
        let bin_2048 = ((freq / freq_res_2048) as usize).min(1023);
        let bin_bass = ((freq / freq_res_bass) as usize).min(2047);

        let mut inst = CrossfadeInstruction {
            b_8192: bin, w_8192: 0.0,
            b_4096: bin_4096, w_4096: 0.0,
            b_2048: bin_2048, w_2048: 0.0,
            b_bass: bin_bass, w_bass: 0.0,
        };

        // Define crossover frequency bands using SQRT_2 for equal-power crossfading
        if freq < 60.0 {
            inst.w_bass = std::f32::consts::SQRT_2; // Sub-bass exclusively uses the decimated bass FFT
        } else if freq < 100.0 {
            let t = (freq - 60.0) / 40.0;
            inst.w_bass = std::f32::consts::SQRT_2 * (1.0 - t);
            inst.w_8192 = 1.0 * t;
        } else if freq < 200.0 {
            inst.w_8192 = 1.0;
        } else if freq < 300.0 {
            let t = (freq - 200.0) / 100.0;
            inst.w_8192 = 1.0 * (1.0 - t);
            inst.w_4096 = std::f32::consts::SQRT_2 * t;
        } else if freq < 1200.0 {
            inst.w_4096 = std::f32::consts::SQRT_2;
        } else if freq < 2000.0 {
            let t = (freq - 1200.0) / 800.0;
            inst.w_4096 = std::f32::consts::SQRT_2 * (1.0 - t);
            inst.w_2048 = 2.0 * t;
        } else {
            inst.w_2048 = 2.0; // High frequencies exclusively use the small 2048 window
        }
        map.push(inst);
    }
    map
}

/// Represents the historical pixel state for one specific FFT resolution.
struct SpectrogramLayer {
    fft_size: usize,
    freq_bins: usize,
    pixels: Vec<u8>,     // Flattened RGBA buffer
    head: usize,         // Current write index (column) in the ring buffer
    total_updates: u64,  // Monotonically increasing counter to sync CPU/GPU state
    y_map: YToBinMap,
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
            y_map: YToBinMap::generate(bins, fft_size),
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
    // Setup 4 layers: The 3 raw resolutions + 1 Composite layer
    let mut layers = Vec::new();
    for &size in RESOLUTIONS.iter() {
        layers.push(SpectrogramLayer::new(size));
    }
    layers.push(SpectrogramLayer::new(RESOLUTIONS[2])); // Composite layer uses largest height

    let shared_layers = Arc::new(Mutex::new(layers));
    let shared_settings = Arc::new(Mutex::new(AppSettings {
        mel_scale: true,
        colormap: ColorMapType::Magma,
        redraw_flag: false,
        audio_source: AudioSource::SinkMonitor,
    }));

    // Lock-free ring buffer for passing raw f32 audio from Recorder thread to Brain thread
    let rb = HeapRb::<f32>::new(65536);
    let (mut producer, mut consumer) = rb.split();

    // ==========================================
    // --- THREAD A: RECORDER (Audio Ingest) ---
    // ==========================================
    let shared_settings_recorder = shared_settings.clone();
    
    thread::spawn(move || {
        let mut current_source = AudioSource::SinkMonitor;

        // Helper to dynamically find and connect to PulseAudio sources using `pactl`
        let open_stream = |source: AudioSource| -> Option<Simple> {
            let device_name = match source {
                AudioSource::SinkMonitor => {
                    if let Ok(output) = std::process::Command::new("pactl").arg("get-default-sink").output() {
                        let sink_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        format!("{}.monitor", sink_name) // Append .monitor to capture system output
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
            let frag_size = (SAMPLE_RATE as u32 * 4 * 15) / 1000; // ~15ms fragments
            let attr = BufferAttr {
                maxlength: u32::MAX, tlength: u32::MAX, prebuf: u32::MAX, minreq: u32::MAX,
                fragsize: frag_size,
            };

            Simple::new(None, "spector", Direction::Record, Some(&device_name), "Recorder", &spec, None, Some(&attr)).ok()
        };

        let mut stream = open_stream(current_source);
        let mut buf = [0u8; 4096];

        loop {
            // Check if the user swapped input sources via UI
            if let Ok(settings) = shared_settings_recorder.try_lock() {
                if settings.audio_source != current_source {
                    current_source = settings.audio_source;
                    stream = open_stream(current_source);
                }
            }

            // Read raw bytes from PulseAudio, cast to f32, and push to lock-free consumer
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
        let audio_buffer_size = BASS_WINDOW.max(max_fft);
        
        let mut rolling_audio = vec![0.0; audio_buffer_size];
        let mut pending_buffer = Vec::with_capacity(4096);

        // Keep a history of pure floating-point FFT magnitudes to allow instant recoloring
        let mut float_history: Vec<Vec<f32>> = (0..=RESOLUTIONS.len()).map(|i| {
            let size = if i < RESOLUTIONS.len() { RESOLUTIONS[i] } else { RESOLUTIONS[2] };
            vec![0.0f32; MAX_HISTORY * (size / 2)]
        }).collect();

        let mut current_lut_type = ColorMapType::Magma;
        let mut lut = ColorLut::new(current_lut_type);

        // Optimization: Pre-allocate hot loop buffers to avoid memory allocation during FFT crunching
        let mut local_cols = vec![
            vec![0.0; RESOLUTIONS[0] / 2],
            vec![0.0; RESOLUTIONS[1] / 2],
            vec![0.0; RESOLUTIONS[2] / 2],
            vec![0.0; RESOLUTIONS[2] / 2], // Composite buffer
        ];
        let mut bass_col = vec![0.0; 4096 / 2];
        let mut downsampled = vec![0.0; 4096];
        let crossfade_map = build_crossfade_map();

        loop {
            // 1. Process UI Settings / Requests
            let mut local_settings = None;
            let mut redraw_requested = false;

            {
                let mut s = settings_ref.lock().unwrap();
                if s.redraw_flag {
                    redraw_requested = true;
                    s.redraw_flag = false;
                    if s.colormap != current_lut_type {
                        current_lut_type = s.colormap;
                        lut = ColorLut::new(current_lut_type);
                    }
                }
                local_settings = Some(s.clone());
            }

            let settings = local_settings.unwrap();

            // Handle Full Redraw (e.g., Color map change or Log/Linear scale flip)
            if redraw_requested {
                if let Ok(mut layers) = layers_ref.lock() {
                    for (i, layer) in layers.iter_mut().enumerate() {
                        for t in 0..MAX_HISTORY {
                            let start = t * layer.freq_bins;
                            let slice = &float_history[i][start..start + layer.freq_bins];
                            paint_column_fast(&mut layer.pixels, t, slice, settings.mel_scale, &lut, &layer.y_map, layer.freq_bins);
                        }
                        // Signal to the Face thread that a massive change occurred, forcing a full GPU upload
                        layer.total_updates += (MAX_HISTORY * 2) as u64; 
                    }
                }
            }

            // 2. Audio Ingest from Ring Buffer
            let num_new = consumer.len();
            if num_new > 0 {
                let mut temp = vec![0.0; num_new];
                consumer.pop_slice(&mut temp);
                pending_buffer.extend(temp);
            }

            // 3. Processing Hot Loop (Runs whenever enough audio is buffered)
            while pending_buffer.len() >= HOP_SIZE {
                let chunk: Vec<f32> = pending_buffer.drain(0..HOP_SIZE).collect();
                
                rolling_audio.extend(chunk);
                if rolling_audio.len() > audio_buffer_size {
                    let remove = rolling_audio.len() - audio_buffer_size;
                    rolling_audio.drain(0..remove);
                }

                let mut computed_all = true;

                // WAVECANDY TRICK: Sub-bass Decimation
                // Standard FFTs can't resolve sub-bass without massive latency. 
                // We downsample the audio by 8x, then run a smaller FFT to get extreme low-end accuracy.
                if rolling_audio.len() >= BASS_WINDOW {
                    let bass_slice = &rolling_audio[rolling_audio.len() - BASS_WINDOW..];
                    for (i, chunk) in bass_slice.chunks_exact(8).enumerate() {
                        downsampled[i] = chunk.iter().sum::<f32>() / 8.0; // Naive decimation
                    }
                    let windowed = hann_window(&downsampled);
                    if let Ok(spectrum) = samples_fft_to_spectrum(&windowed, SAMPLE_RATE / 8, FrequencyLimit::All, Some(&divide_by_N_sqrt)) {
                        for (out, val) in bass_col.iter_mut().zip(spectrum.data().iter()) {
                            *out = val.1.val();
                        }
                    } else { computed_all = false; }
                } else { computed_all = false; }

                // Process Standard Multi-Resolution FFTs
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

                // Compile everything into the final pixels
                if computed_all {
                    // Blend all 4 FFTs based on pre-calculated weights
                    for (bin, inst) in crossfade_map.iter().enumerate() {
                        local_cols[3][bin] = 
                            local_cols[2][inst.b_8192] * inst.w_8192 +
                            local_cols[1][inst.b_4096] * inst.w_4096 +
                            local_cols[0][inst.b_2048] * inst.w_2048 +
                            bass_col[inst.b_bass]      * inst.w_bass;
                    }

                    // Update Shared State (Lock briefly)
                    if let Ok(mut layers) = layers_ref.lock() {
                        for i in 0..=RESOLUTIONS.len() {
                            let layer = &mut layers[i];
                            let head_idx = layer.head;

                            // Save float history for instant recoloring
                            let float_start = head_idx * layer.freq_bins;
                            let copy_len = local_cols[i].len().min(layer.freq_bins);
                            if float_start + copy_len <= float_history[i].len() {
                                float_history[i][float_start..float_start+copy_len].copy_from_slice(&local_cols[i][0..copy_len]);
                            }

                            // Convert raw float magnitudes to RGB pixels
                            paint_column_fast(&mut layer.pixels, head_idx, &local_cols[i], settings.mel_scale, &lut, &layer.y_map, layer.freq_bins);

                            // Advance ring buffer
                            layer.head = (layer.head + 1) % MAX_HISTORY;
                            layer.total_updates += 1;
                        }
                    }
                }
            }
            
            // Yield briefly if starving
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
    let mut current_view_len = MAX_HISTORY; 
    
    // Setup GPU RenderTargets. We treat VRAM as a 2D ring buffer texture.
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
    
    let max_possible_height = *RESOLUTIONS.iter().max().unwrap() / 2;
    let mut local_pixels_buffer = vec![0u8; MAX_HISTORY * max_possible_height * 4];

    loop {
        let mut visual_changed = false;
        let mut source_changed = false;

        // Input Handling
        if is_key_pressed(KeyCode::S) { local_mel = !local_mel; visual_changed = true; }
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
            current_view_len = if current_view_len == MAX_HISTORY { MAX_HISTORY / 2 } else { MAX_HISTORY };
        }

        if visual_changed || source_changed {
            if let Ok(mut s) = shared_settings.lock() {
                s.mel_scale = local_mel;
                s.colormap = local_cmap;
                s.audio_source = local_source;
                if visual_changed {
                    s.redraw_flag = true;
                }
            }
        }

        let display_fft_size = if current_fft_idx == RESOLUTIONS.len() { RESOLUTIONS[2] } else { RESOLUTIONS[current_fft_idx] };
        let current_height = display_fft_size / 2;

        let mut actual_total_updates = 0u64;
        let mut full_redraw_needed = false;
        let mut delta_images: Vec<(f32, Image)> = Vec::new();

        {
            // Acquire lock to read pixel updates from Brain thread
            if let Ok(layers) = shared_layers.lock() {
                let layer = &layers[current_fft_idx];
                actual_total_updates = layer.total_updates;
                
                let prev_updates = last_rendered_updates[current_fft_idx];
                let diff = if actual_total_updates > prev_updates { actual_total_updates - prev_updates } else { 0 };
                
                let full_redraw_threshold = (MAX_HISTORY / 2) as u64;

                // VRAM Synchronization Logic
                if diff >= full_redraw_threshold {
                    // Hot Path 1: Mass update (e.g., UI change). Copies full state to CPU buffer once.
                    let expected_len = layer.pixels.len();
                    if local_pixels_buffer.len() < expected_len {
                        local_pixels_buffer.resize(expected_len, 0);
                    }
                    local_pixels_buffer[..expected_len].copy_from_slice(&layer.pixels);
                    full_redraw_needed = true;
                } else if diff > 0 {
                    // Hot Path 2: Continuous Pipeline (60 FPS). Extracts ONLY the new micro-strip of data.
                    // This prevents sending 20MB of texture data to the GPU every single frame.
                    let diff_usize = diff as usize;
                    let head = layer.head;
                    let height = layer.freq_bins;

                    if head >= diff_usize {
                        // Normal case: data is contiguous
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
                        // Wrapping case: data crosses the ring buffer boundary. Split into two images.
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
                
                // Track visual drift when switching between multi-resolution modes
                if current_fft_idx != last_fft_idx {
                    let prev_mode_updates = layers[last_fft_idx].total_updates;
                    let diff_from_last = actual_total_updates as f64 - prev_mode_updates as f64;
                    smooth_head_pos += diff_from_last;
                    last_fft_idx = current_fft_idx;
                }
            }
        }

        // Apply VRAM Ring Buffer update commands outside the lock
        if full_redraw_needed {
            let img = Image {
                width: MAX_HISTORY as u16, height: current_height as u16,
                bytes: local_pixels_buffer[..(MAX_HISTORY * current_height * 4)].to_vec(),
            };
            let tex = Texture2D::from_image(&img);
            
            // Standard pixel-perfect camera for drawing to RenderTarget (Y points down for proper 1:1 orientation)
            let cam = Camera2D {
                render_target: Some(render_targets[current_fft_idx].clone()),
                zoom: vec2(2.0 / MAX_HISTORY as f32, -2.0 / current_height as f32),
                target: vec2(MAX_HISTORY as f32 / 2.0, current_height as f32 / 2.0),
                ..Default::default()
            };
            set_camera(&cam);
            draw_texture(&tex, 0.0, 0.0, WHITE);
            set_default_camera();
            last_rendered_updates[current_fft_idx] = actual_total_updates;
        } else if !delta_images.is_empty() {
            let cam = Camera2D {
                render_target: Some(render_targets[current_fft_idx].clone()),
                zoom: vec2(2.0 / MAX_HISTORY as f32, -2.0 / current_height as f32),
                target: vec2(MAX_HISTORY as f32 / 2.0, current_height as f32 / 2.0),
                ..Default::default()
            };
            set_camera(&cam);
            
            // Draw only the new micro-strips onto the VRAM texture
            for (x, img) in delta_images {
                let tex = Texture2D::from_image(&img);
                tex.set_filter(FilterMode::Nearest); // Crisp pixel pasting
                draw_texture(&tex, x, 0.0, WHITE);
            }
            set_default_camera();
            last_rendered_updates[current_fft_idx] = actual_total_updates;
        }

        clear_background(BLACK);
        let sw = screen_width();
        let sh = screen_height();

        // Smooth camera panning logic so the waterfall moves smoothly instead of stuttering
        let target_pos = actual_total_updates as f64;
        let diff = target_pos - smooth_head_pos;
        let dt = get_frame_time() as f64;
        
        smooth_head_pos += diff * (15.0 * dt).min(1.0); 
        if diff.abs() > 50.0 { smooth_head_pos = target_pos; } 
        
        // Use rem_euclid for perfect bounds safety when mathematical inflation pushes tracking negative
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

        let final_source_w_snapped = final_source_w.round();
        let start_pos_unwrapped = head_offset - final_source_w_snapped;
        let tex_w = MAX_HISTORY as f32;
        let tex_h = current_height as f32;

        let is_horizontal = match local_dir {
            ScrollDirection::RTL | ScrollDirection::LTR => true,
            _ => false
        };

        // Texture Unwrapping Logic:
        // Because the texture is a ring buffer, the "view window" might cross the end of the texture
        // and wrap around to 0. We slice it into up to two `src` rects to handle this natively.
        let (src1, src2) = if start_pos_unwrapped < 0.0 {
            let overflow = start_pos_unwrapped.abs();
            let s1 = Rect::new(tex_w - overflow, 0.0, overflow, tex_h); // Tail end of texture
            let s2 = Rect::new(0.0, 0.0, head_offset, tex_h);           // Head end of texture
            (Some(s1), Some(s2))
        } else {
            let s1 = Rect::new(start_pos_unwrapped, 0.0, final_source_w_snapped, tex_h);
            (Some(s1), None)
        };

        let dst1_len = src1.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let dst2_len = src2.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let total_draw_len = dst1_len + dst2_len;
        let start_pos_screen = (screen_time_dim - total_draw_len) / 2.0;

        let texture_to_draw = &render_targets[current_fft_idx].texture;

        // --- FINAL DRAW CALLS ---
        // Based on the direction, we stitch the two texture slices (src1, src2) together on screen.
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

        draw_note_ruler(local_mel, local_dir, display_fft_size);

        if DRAW_UI { draw_ui_overlay(local_mel, local_cmap, local_dir, current_fft_idx, current_view_len, local_source); }

        next_frame().await
    }
}

// --- FAST PAINTER ---
// Blits a vertical column of FFT data directly into a 1D pixel array.
// Avoids `f32.log()` and `f32.pow()` by utilizing the pre-calculated Map and LUT.
#[inline(always)]
fn paint_column_fast(
    pixels: &mut [u8], 
    col_idx: usize, 
    data: &[f32], 
    use_mel: bool, 
    lut: &ColorLut,
    y_map: &YToBinMap,
    freq_bins: usize, 
) {
    let x_offset = col_idx * 4;

    for i in 0..freq_bins {
        let y = i;
        
        // Instant array lookups instead of calculating logarithmic math for every pixel
        let target_idx = if use_mel { y_map.mel_map[i] } else { y_map.linear_map[i] };
        
        let magnitude = data[target_idx];
        let intensity = (magnitude * 2000.0).ln() / 8.0;
        let rgb = lut.get_color(intensity);

        let idx = (y * MAX_HISTORY * 4) + x_offset;

        pixels[idx] = rgb[0];
        pixels[idx+1] = rgb[1];
        pixels[idx+2] = rgb[2];
        pixels[idx+3] = 255;
    }
}

// Below are standard UI and Math helper functions

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