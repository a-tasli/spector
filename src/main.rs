use macroquad::prelude::*;
use ringbuf::HeapRb;
use rustfft::{FftPlanner, num_complex::Complex};
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
const MIN_HOP_SIZE: usize = 256; 
const CQT_HOP_SIZE: usize = 512; 

// Professional Visualizer Tweaks
const PINK_NOISE_TILT_DB_PER_OCTAVE: f32 = 0.0; // 3.0 is standard for mastered music
const PEAK_WEIGHT: f32 = 0.5; // Hybrid Peak-RMS balance
const RMS_WEIGHT: f32 = 0.5;

// --- NEW CONFIGS: PSD & Adaptive Smoothing ---
const PSD_NORMALIZATION: bool = true;  // Flattens white noise by dividing by bandwidth (energy density)

const DECAY_LOW_FREQ: f32 = 0.01;      // Heavy phosphor in the bass for stability
const DECAY_HIGH_FREQ: f32 = 0.005;     // Fast release in the treble for transients

const SPLAT_SPREAD_LOW: f32 = 3.0;     // Wide interpolation in the bass to fill sparse STFT gaps
const SPLAT_SPREAD_HIGH: f32 = 0.01;    // Narrow splatting in the treble to maintain razor sharpness

const HALO_RAW_BLEND: f32 = 0.0;       // Amount of the 'blurry' original FFT background (fills gaps)
const HALO_SHARP_BLEND: f32 = 10.0;     // Amount of the razor-sharp reassigned pitch cores
const STFT_BOOST_GAIN: f32 = 5.0;     // Restores brightness lost to PSD and pitch-core condensation
const IIR_BOOST_GAIN: f32 = 5.0;       // Matches the sub-bass brightness to the STFT layer

// Variable resolutions and their corresponding Variable Hop Sizes
const RESOLUTIONS: [usize; 4] = [2048, 4096, 8192, 16384];
const HOP_SIZES: [usize; 4] = [256, 512, 1024, 2048];

const CQT_BINS: usize = 1200; // The HD resolution of our Constant-Q Transform output
const IIR_CROSSOVER_LOWER_HZ: f32 = 100.0; // Start crossfading from IIR to STFT here
const IIR_CROSSOVER_UPPER_HZ: f32 = 261.5; // IIR fully faded out by here

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

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScaleType { Linear, Mel, Logarithmic, Chromatic }

/// Shared state between the Brain (processing) and Face (UI) threads.
#[derive(Clone)]
struct AppSettings {
    scale_type: ScaleType,
    colormap: ColorMapType,
    redraw_flag: bool,
    audio_source: AudioSource,
    iir_enabled: bool,
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

fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}

// --- Professional IIR Biquad Filter (Direct Form II Transposed) ---
// Used for zero-latency, sub-bass transient resolution
#[derive(Clone)]
struct Biquad {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    z1: f32, z2: f32,
}

impl Biquad {
    fn bandpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * w0.cos()) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline(always)]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

// --- Professional Multi-Rate Sparse Matrix Kernel ---
#[derive(Clone)]
struct CqtInstruction {
    cqt_bin_idx: usize,
    fft_idx: usize,
    b_start: usize,
    weights: Vec<f32>,
    weight_sum: f32,
}

// Generates dynamic Gaussian/Hann kernels to perfectly blend sparse bass bins
struct SplatKernel {
    half_width: isize,
    weights: Vec<f32>,
}

fn build_splat_kernels() -> Vec<SplatKernel> {
    (0..CQT_BINS).map(|bin| {
        let norm = bin as f32 / (CQT_BINS - 1) as f32;
        let spread = SPLAT_SPREAD_LOW + (SPLAT_SPREAD_HIGH - SPLAT_SPREAD_LOW) * norm;
        let mut half_width = spread.ceil() as isize;
        
        let mut weights = Vec::new();
        let mut sum = 0.0;
        
        for s in -half_width..=half_width {
            let d = s.abs() as f32;
            let w = if d <= spread {
                0.5 * (1.0 + (std::f32::consts::PI * d / spread.max(0.0001)).cos())
            } else {
                0.0
            };
            weights.push(w);
            sum += w;
        }
        
        if sum > 0.0 {
            for w in &mut weights { *w /= sum; }
        } else {
            weights = vec![1.0];
            half_width = 0;
        }
        
        SplatKernel { half_width, weights }
    }).collect()
}

fn build_cqt_map(sample_rate: u32) -> (Vec<CqtInstruction>, Vec<(usize, Biquad, f32)>) {
    let mut stft_map = Vec::with_capacity(CQT_BINS);
    let mut iir_filters = Vec::new();
    
    let min_freq = 20.0_f32;
    let max_freq = sample_rate as f32 / 2.0;
    let log_min = min_freq.log2();
    let log_max = max_freq.log2();
    
    for bin in 0..CQT_BINS {
        let norm = bin as f32 / (CQT_BINS - 1) as f32;
        let freq = 2.0_f32.powf(log_min + norm * (log_max - log_min));
        
        let musical_bw = freq / 24.0;
        let erb = 24.7 * (4.37 * (freq / 1000.0) + 1.0);
        let mut bw_hz = if freq < 250.0 {
            musical_bw
        } else {
            let t = ((freq - 250.0) / 1000.0).clamp(0.0, 1.0);
            musical_bw * (1.0 - t) + (erb * 1.2) * t
        };

        // --- IIR Sub-Bass Routing ---
        if freq < IIR_CROSSOVER_UPPER_HZ {
            let q_factor = freq / bw_hz;
            iir_filters.push((bin, Biquad::bandpass(sample_rate as f32, freq, q_factor), bw_hz));
        }
        
        let min_stft_bw = sample_rate as f32 / RESOLUTIONS[RESOLUTIONS.len()-1] as f32;
        if bw_hz < min_stft_bw * 1.5 {
            bw_hz = min_stft_bw * 1.5;
        }

        let ideal_n = (4.0 * sample_rate as f32 / bw_hz) as usize;
        
        let mut best_idx = 0;
        let mut min_diff = usize::MAX;
        for (i, &res) in RESOLUTIONS.iter().enumerate() {
            let diff = (res as isize - ideal_n as isize).unsigned_abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i;
            }
        }
        
        let fft_size = RESOLUTIONS[best_idx];
        let freq_res = sample_rate as f32 / fft_size as f32;
        let center_bin = freq / freq_res;
        let mut bw_bins = bw_hz / freq_res;
        
        if bw_bins < 2.0 { bw_bins = 2.0; } 
        
        let start_bin = (center_bin - bw_bins / 2.0).floor() as usize;
        let end_bin = (center_bin + bw_bins / 2.0).ceil() as usize;
        
        // Clamping ensures weights perfectly match the slice, enabling safe SIMD auto-vectorization
        let start_bin = start_bin.clamp(0, fft_size / 2 - 1);
        let end_bin = end_bin.clamp(start_bin, fft_size / 2 - 1);
        
        let mut weights = Vec::new();
        let mut max_w = 0.0;
        
        for b in start_bin..=end_bin {
            let dist = (b as f32 - center_bin).abs();
            let w = if dist <= bw_bins / 2.0 {
                0.5 * (1.0 + (std::f32::consts::PI * dist / (bw_bins / 2.0)).cos())
            } else {
                0.0
            };
            weights.push(w);
            if w > max_w { max_w = w; }
        }
        
        if max_w > 0.0 {
            for w in &mut weights { *w /= max_w; }
        } else if weights.is_empty() && start_bin < fft_size/2 {
            weights.push(1.0);
        }

        let weight_sum = weights.iter().sum::<f32>().max(0.0001);

        stft_map.push(CqtInstruction {
            cqt_bin_idx: bin,
            fft_idx: best_idx,
            b_start: start_bin,
            weights,
            weight_sum,
        });
    }
    (stft_map, iir_filters)
}

struct SpectrogramLayer {
    freq_bins: usize,
    pixels: Vec<u8>,     
    head: usize,         
    total_updates: u64,  
}

impl SpectrogramLayer {
    fn new(freq_bins: usize) -> Self {
        Self {
            freq_bins,
            pixels: vec![0u8; MAX_HISTORY * freq_bins * 4],
            head: 0,
            total_updates: 0,
        }
    }
}

// State container for our Phase Vocoder
struct StftState {
    fft_size: usize,
    hop_size: usize,
    samples_since_last: usize,
    prev_phases: Vec<f32>,
    last_mags: Vec<f32>,
    display_mags: Vec<f32>, 
    last_true_freqs: Vec<f32>,
    decays: Vec<f32>, // Frequency-mapped phosphor decays
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
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
uniform float scale_type;
uniform float sample_rate;
uniform float is_cqt_texture;

void main() {
    // uv.y = 0.0 is the top of the screen (Nyquist). uv.y = 1.0 is the bottom (DC).
    // By inverting uv.y, norm_f becomes 0.0 at the bottom (DC) and 1.0 at the top (Nyquist).
    float norm_f = 1.0 - uv.y; 
    
    float min_log_freq = 20.0;
    float max_freq = sample_rate / 2.0;

    if (is_cqt_texture > 0.5) {
        // The CQT texture natively stores data logarithmically from min_log_freq to max_freq.
        // We must map the requested screen scale to a frequency, then to the CQT coordinate.
        float current_hz;
        
        if (scale_type > 2.5) {
            // Chromatic
            float min_freq = 16.35159;
            float log_min = log2(min_freq);
            float log_max = log2(max_freq);
            current_hz = exp2(log_min + norm_f * (log_max - log_min));
        } else if (scale_type > 1.5) {
            // Logarithmic
            float log_min = log2(min_log_freq);
            float log_max = log2(max_freq);
            current_hz = exp2(log_min + norm_f * (log_max - log_min));
        } else if (scale_type > 0.5) {
            // Mel
            float mel_max = 2595.0 * (log(1.0 + max_freq / 700.0) / 2.30258509299);
            float current_mel = norm_f * mel_max;
            current_hz = 700.0 * (exp((current_mel / 2595.0) * 2.30258509299) - 1.0);
        } else {
            // Linear
            current_hz = norm_f * max_freq;
        }
        
        // Map the resolved frequency to the CQT texture's native log mapping array
        if (current_hz < min_log_freq) {
            norm_f = 0.0; 
        } else {
            norm_f = (log2(current_hz) - log2(min_log_freq)) / (log2(max_freq) - log2(min_log_freq));
            norm_f = clamp(norm_f, 0.0, 1.0);
        }
    } else {
        // Standard linear STFT texture mapping
        if (scale_type > 2.5) {
            float min_freq = 16.35159;
            float log_min = log2(min_freq);
            float log_max = log2(max_freq);
            float current_log = log_min + norm_f * (log_max - log_min);
            float current_hz = exp2(current_log);
            norm_f = clamp(current_hz / max_freq, 0.0, 1.0);
        } else if (scale_type > 1.5) {
            float min_freq = 20.0;
            float log_min = log2(min_freq);
            float log_max = log2(max_freq);
            float current_log = log_min + norm_f * (log_max - log_min);
            float current_hz = exp2(current_log);
            norm_f = clamp(current_hz / max_freq, 0.0, 1.0);
        } else if (scale_type > 0.5) {
            float mel_max = 2595.0 * (log(1.0 + max_freq / 700.0) / 2.30258509299);
            float current_mel = norm_f * mel_max;
            float current_hz = 700.0 * (exp((current_mel / 2595.0) * 2.30258509299) - 1.0);
            norm_f = clamp(current_hz / max_freq, 0.0, 1.0);
        }
    }

    vec2 sample_uv = vec2(uv.x, norm_f);

    // Read the grayscale intensity computed by the CPU
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
        layers.push(SpectrogramLayer::new(size / 2));
    }
    // We now maintain TWO parallel CQT history buffers: one WITH IIR, one WITHOUT IIR.
    layers.push(SpectrogramLayer::new(CQT_BINS)); 
    layers.push(SpectrogramLayer::new(CQT_BINS)); 

    let shared_layers = Arc::new(Mutex::new(layers));
    let shared_settings = Arc::new(Mutex::new(AppSettings {
        scale_type: ScaleType::Mel,
        colormap: ColorMapType::Magma,
        redraw_flag: false,
        audio_source: AudioSource::SinkMonitor,
        iir_enabled: false,
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

        let mut float_history: Vec<Vec<f32>> = (0..=RESOLUTIONS.len() + 1).map(|i| {
            let size = if i < RESOLUTIONS.len() { RESOLUTIONS[i] / 2 } else { CQT_BINS };
            vec![0.0f32; MAX_HISTORY * size]
        }).collect();

        let mut local_cqt_col_no_iir: Vec<f32> = vec![0.0; CQT_BINS];
        let mut local_cqt_col_with_iir: Vec<f32> = vec![0.0; CQT_BINS];

        // --- Generate Visual Tilt Curves ---
        let mut tilt_curves: Vec<Vec<f32>> = Vec::new();
        for &fft_size in RESOLUTIONS.iter() {
            let half_size = fft_size / 2;
            let mut curve = vec![1.0f32; half_size];
            let freq_res = SAMPLE_RATE as f32 / fft_size as f32;
            let min_log = 20.0f32.log2();
            
            for bin in 0..half_size {
                let freq = bin as f32 * freq_res;
                if freq > 20.0 {
                    let octaves_above_min = freq.log2() - min_log;
                    let tilt_db = octaves_above_min * PINK_NOISE_TILT_DB_PER_OCTAVE;
                    curve[bin] = 10.0f32.powf(tilt_db / 20.0);
                }
            }
            tilt_curves.push(curve);
        }

        let mut cqt_tilt_curve = vec![1.0f32; CQT_BINS];
        let min_log = 20.0f32.log2();
        let max_log = (SAMPLE_RATE as f32 / 2.0).log2();
        let log_range = max_log - min_log;
        
        let mut cqt_decays = vec![0.0f32; CQT_BINS];

        for bin in 0..CQT_BINS {
            let norm = bin as f32 / (CQT_BINS - 1) as f32;
            let octaves_above_min = norm * log_range;
            let tilt_db = octaves_above_min * PINK_NOISE_TILT_DB_PER_OCTAVE;
            cqt_tilt_curve[bin] = 10.0f32.powf(tilt_db / 20.0);
            
            // Map the adaptive decay frequency-wise
            cqt_decays[bin] = DECAY_LOW_FREQ + (DECAY_HIGH_FREQ - DECAY_LOW_FREQ) * norm;
        }
        tilt_curves.push(cqt_tilt_curve.clone()); // Add tilt for CQT (No IIR) layer
        tilt_curves.push(cqt_tilt_curve);         // Add tilt for CQT (With IIR) layer

        // Setup Phase Vocoder states for each resolution tier
        let mut planner = FftPlanner::new();
        let mut stft_states: Vec<StftState> = RESOLUTIONS.iter().zip(HOP_SIZES.iter()).map(|(&fft_size, &hop_size)| {
            let freq_res = SAMPLE_RATE as f32 / fft_size as f32;
            let mut decays = vec![0.0f32; fft_size / 2];
            for bin in 0..(fft_size / 2) {
                let freq = bin as f32 * freq_res;
                let norm = if freq >= 20.0 { ((freq.log2() - min_log) / log_range).clamp(0.0, 1.0) } else { 0.0 };
                decays[bin] = DECAY_LOW_FREQ + (DECAY_HIGH_FREQ - DECAY_LOW_FREQ) * norm;
            }

            StftState {
                fft_size,
                hop_size,
                samples_since_last: hop_size, // Trigger immediately on start
                prev_phases: vec![0.0; fft_size / 2],
                last_mags: vec![0.0; fft_size / 2],
                display_mags: vec![0.0; fft_size / 2],
                last_true_freqs: vec![0.0; fft_size / 2],
                decays,
                fft: planner.plan_fft_forward(fft_size),
                window: generate_hann_window(fft_size),
            }
        }).collect();

        let (stft_cqt_map, mut iir_filters) = build_cqt_map(SAMPLE_RATE);
        let splat_kernels = build_splat_kernels();
        
        let min_stft_weight_sum = stft_cqt_map.first().map(|inst| inst.weight_sum).unwrap_or(1.0);
        let min_iir_bw = iir_filters.first().map(|&(_, _, bw)| bw).unwrap_or(1.0);

        // --- Calculate IIR / STFT Blend Weights ---
        let mut iir_blend_weights = vec![0.0f32; CQT_BINS];
        for bin in 0..CQT_BINS {
            let norm = bin as f32 / (CQT_BINS - 1) as f32;
            let freq = 2.0_f32.powf(min_log + norm * log_range);
            if freq <= IIR_CROSSOVER_LOWER_HZ {
                iir_blend_weights[bin] = 1.0;
            } else if freq >= IIR_CROSSOVER_UPPER_HZ {
                iir_blend_weights[bin] = 0.0;
            } else {
                let t = (freq - IIR_CROSSOVER_LOWER_HZ) / (IIR_CROSSOVER_UPPER_HZ - IIR_CROSSOVER_LOWER_HZ);
                iir_blend_weights[bin] = 0.5 * (1.0 + (std::f32::consts::PI * t).cos()); // Smooth fade 1 to 0
            }
        }

        // IIR and Temporal state accumulators
        let mut iir_power_accum = vec![0.0f32; CQT_BINS];
        let mut iir_peak_accum = vec![0.0f32; CQT_BINS];
        let mut iir_samples_accum = 0;
        let mut cqt_samples_since_last = 0;
        
        let mut prev_cqt_col_no_iir = vec![0.0f32; CQT_BINS];
        let mut prev_cqt_col_with_iir = vec![0.0f32; CQT_BINS];

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
                            paint_column_fast(&mut layer.pixels, t, slice, layer.freq_bins, &tilt_curves[i]);
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

            // Process audio in steady multi-rate intervals
            while pending_buffer.len() >= MIN_HOP_SIZE {
                let chunk: Vec<f32> = pending_buffer.drain(0..MIN_HOP_SIZE).collect();

                // --- Instantaneous IIR Processing (Vectorized Inner Loop) ---
                // We always compute this so the background history remains completely populated and accurate.
                for &sample in &chunk {
                    for (bin_idx, biquad, _) in iir_filters.iter_mut() {
                        let filtered = biquad.process(sample);
                        iir_power_accum[*bin_idx] += filtered * filtered;
                        
                        let abs_f = filtered.abs();
                        if abs_f > iir_peak_accum[*bin_idx] {
                            iir_peak_accum[*bin_idx] = abs_f;
                        }
                    }
                }
                iir_samples_accum += chunk.len();
                
                rolling_audio.rotate_left(MIN_HOP_SIZE);
                let len = rolling_audio.len();
                rolling_audio[len - MIN_HOP_SIZE..].copy_from_slice(&chunk);
                
                // 1. Process STFT layers at their independent hop rates
                for state in stft_states.iter_mut() {
                    state.samples_since_last += MIN_HOP_SIZE;
                    
                    if state.samples_since_last >= state.hop_size {
                        let start_sample = len - state.fft_size;
                        let audio_slice = &rolling_audio[start_sample..];
                        
                        let mut buffer: Vec<Complex<f32>> = audio_slice.iter().zip(state.window.iter())
                            .map(|(&a, &w)| Complex { re: a * w, im: 0.0 })
                            .collect();
                        
                        state.fft.process(&mut buffer);
                        
                        // --- STFT PARITY MATH ---
                        let scale = 2.0 / state.fft_size as f32; // mathematically pure (read by CQT layer)
                        let broadband_comp = (state.fft_size as f32 / 2048.0).sqrt(); // fix energy spreading
                        let cqt_makeup_gain = STFT_BOOST_GAIN.sqrt() * HALO_SHARP_BLEND; // fix absolute brightness
                        
                        let half_size = state.fft_size / 2;
                        let freq_res = SAMPLE_RATE as f32 / state.fft_size as f32;
                        let hop_advance = 2.0 * std::f32::consts::PI * state.hop_size as f32 / state.fft_size as f32;
                        let sr_over_hop = SAMPLE_RATE as f32 / (2.0 * std::f32::consts::PI * state.hop_size as f32);
                        
                        // --- SIMD Optimized Phase Vocoder Analysis w/ Decay ---
                        for (bin, (((&c, mag), prev_phase), true_freq)) in buffer[0..half_size].iter()
                            .zip(&mut state.last_mags)
                            .zip(&mut state.prev_phases)
                            .zip(&mut state.last_true_freqs)
                            .enumerate() 
                        {
                            *mag = c.norm() * scale;
                            let phase = c.im.atan2(c.re);
                            
                            let phase_diff = phase - *prev_phase;
                            *prev_phase = phase;
                            
                            let bin_freq = bin as f32 * freq_res;
                            let expected_advance = bin as f32 * hop_advance;
                            
                            let unwrapped_diff = (phase_diff - expected_advance + std::f32::consts::PI)
                                .rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI;
                                
                            *true_freq = bin_freq + unwrapped_diff * sr_over_hop;
                            
                            // Apply visual makeup gain *only* for the direct STFT display to perfectly match CQT
                            let display_mag = *mag * broadband_comp * cqt_makeup_gain;
                            
                            // Apply adaptive Phosphor-decay
                            let prev_disp = state.display_mags[bin];
                            let decay = state.decays[bin];
                            state.display_mags[bin] = if display_mag > prev_disp {
                                display_mag
                            } else {
                                (display_mag * (1.0 - decay)) + (prev_disp * decay)
                            };
                        }
                        
                        state.samples_since_last -= state.hop_size;
                    }
                }

                cqt_samples_since_last += MIN_HOP_SIZE;

                // 2. Compute CQT and commit to visualizer history grids
                if cqt_samples_since_last >= CQT_HOP_SIZE {
                    let mut raw_cqt_power = vec![0.0f32; CQT_BINS];
                    let mut raw_cqt_peak = vec![0.0f32; CQT_BINS];
                    let mut sharp_cqt_power = vec![0.0f32; CQT_BINS];
                    let mut sharp_cqt_peak = vec![0.0f32; CQT_BINS];
                    
                    let mut stft_amplitudes = vec![0.0f32; CQT_BINS];
                    let mut iir_amplitudes = vec![0.0f32; CQT_BINS];
                    
                    // --- Inject Hybrid IIR Bass Energy w/ PSD Normalization ---
                    if iir_samples_accum > 0 {
                        let inv_samples = 1.0 / iir_samples_accum as f32;
                        for &(bin, _, bw) in iir_filters.iter() {
                            let norm_factor = if PSD_NORMALIZATION { bw / min_iir_bw } else { 1.0 };
                            let rms = ((iir_power_accum[bin] * inv_samples) / norm_factor).sqrt();
                            let peak = iir_peak_accum[bin]; 
                            
                            let hybrid = (peak * PEAK_WEIGHT) + (rms * RMS_WEIGHT);
                            iir_amplitudes[bin] = hybrid * IIR_BOOST_GAIN; 
                        }
                        for &(bin, _, _) in iir_filters.iter() {
                            iir_power_accum[bin] = 0.0;
                            iir_peak_accum[bin] = 0.0;
                        }
                        iir_samples_accum = 0;
                    }
                    
                    // --- Dual-Pass Sparse Matrix (Raw Halo + Sharp Splatting) ---
                    for inst in stft_cqt_map.iter() {
                        let state = &stft_states[inst.fft_idx];
                        let default_bin = inst.cqt_bin_idx;
                        let norm_factor = if PSD_NORMALIZATION { inst.weight_sum / min_stft_weight_sum } else { 1.0 };
                        
                        let start = inst.b_start;
                        let end = start + inst.weights.len();
                        
                        let mags = &state.last_mags[start..end];
                        let freqs = &state.last_true_freqs[start..end];
                        
                        for ((&mag, &true_freq), &w) in mags.iter().zip(freqs).zip(&inst.weights) {
                            let energy = ((mag * mag) * w) / norm_factor;
                            
                            // 1. Accumulate Raw CQT 'Halo' Energy (Bandwidth Normalized)
                            raw_cqt_power[default_bin] += energy;
                            if mag > raw_cqt_peak[default_bin] {
                                raw_cqt_peak[default_bin] = mag;
                            }
                            
                            // 2. Accumulate Sharp Reassigned Energy with Dynamic Gaussian Splat
                            if true_freq >= 20.0 {
                                let norm = (true_freq.log2() - min_log) / log_range;
                                let target_bin = (norm * (CQT_BINS - 1) as f32).round() as isize;
                                
                                if target_bin >= 0 && target_bin < CQT_BINS as isize {
                                    let boosted_energy = energy * STFT_BOOST_GAIN;
                                    let boosted_mag = mag * STFT_BOOST_GAIN.sqrt(); 
                                    
                                    let splat = &splat_kernels[target_bin as usize];
                                    
                                    for (s, &s_weight) in (-splat.half_width..=splat.half_width).zip(&splat.weights) {
                                        let offset_bin = target_bin + s;
                                        if offset_bin >= 0 && offset_bin < CQT_BINS as isize {
                                            let ob = offset_bin as usize;
                                            sharp_cqt_power[ob] += boosted_energy * s_weight;
                                            
                                            let s_mag = boosted_mag * s_weight.sqrt();
                                            if s_mag > sharp_cqt_peak[ob] {
                                                sharp_cqt_peak[ob] = s_mag;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    for bin in 0..CQT_BINS {
                        let raw_hybrid = (raw_cqt_peak[bin] * PEAK_WEIGHT) + (raw_cqt_power[bin].sqrt() * RMS_WEIGHT);
                        let sharp_hybrid = (sharp_cqt_peak[bin] * PEAK_WEIGHT) + (sharp_cqt_power[bin].sqrt() * RMS_WEIGHT);
                        
                        stft_amplitudes[bin] = (raw_hybrid * HALO_RAW_BLEND) + (sharp_hybrid * HALO_SHARP_BLEND);
                        
                        let iir_w = iir_blend_weights[bin];
                        let stft_w = 1.0 - iir_w;

                        // Calculate BOTH paths independently
                        let current_no_iir = stft_amplitudes[bin];
                        let current_with_iir = (iir_amplitudes[bin] * iir_w) + (stft_amplitudes[bin] * stft_w);
                        
                        let decay = cqt_decays[bin];
                        
                        // --- Temporal Persistence (Adaptive Phosphor Decay) ---
                        // Persist NO IIR path
                        let prev_no = prev_cqt_col_no_iir[bin];
                        let final_no = if current_no_iir > prev_no { current_no_iir } else { (current_no_iir * (1.0 - decay)) + (prev_no * decay) };
                        prev_cqt_col_no_iir[bin] = final_no;
                        local_cqt_col_no_iir[bin] = final_no;
                        
                        // Persist WITH IIR path
                        let prev_with = prev_cqt_col_with_iir[bin];
                        let final_with = if current_with_iir > prev_with { current_with_iir } else { (current_with_iir * (1.0 - decay)) + (prev_with * decay) };
                        prev_cqt_col_with_iir[bin] = final_with;
                        local_cqt_col_with_iir[bin] = final_with;
                    }

                    if let Ok(mut layers) = layers_ref.lock() {
                        for i in 0..=RESOLUTIONS.len() + 1 {
                            let layer = &mut layers[i];
                            let head_idx = layer.head;
                            let float_start = head_idx * layer.freq_bins;
                            
                            let data_source = if i < RESOLUTIONS.len() {
                                &stft_states[i].display_mags
                            } else if i == RESOLUTIONS.len() {
                                &local_cqt_col_no_iir
                            } else {
                                &local_cqt_col_with_iir
                            };

                            let copy_len = data_source.len().min(layer.freq_bins);
                            if float_start + copy_len <= float_history[i].len() {
                                float_history[i][float_start..float_start+copy_len].copy_from_slice(&data_source[0..copy_len]);
                            }

                            // Pass the visual Tilt curves to the painting function
                            paint_column_fast(&mut layer.pixels, head_idx, data_source, layer.freq_bins, &tilt_curves[i]);

                            layer.head = (layer.head + 1) % MAX_HISTORY;
                            layer.total_updates += 1;
                        }
                    }
                    
                    cqt_samples_since_last -= CQT_HOP_SIZE;
                }
            }
            
            if pending_buffer.len() < MIN_HOP_SIZE {
                thread::sleep(Duration::from_millis(1));
            }
        }
    });

    // ===============================================
    // --- THREAD C: FACE (UI & GPU Rendering) ---
    // ===============================================
    let mut current_fft_idx = RESOLUTIONS.len(); // Default to CQT HD
    let mut last_fft_idx = RESOLUTIONS.len(); 
    let mut current_view_len = MAX_VIEW_LEN; 
    
    let mut render_targets = Vec::new();
    for i in 0..=RESOLUTIONS.len() + 1 {
        let height = if i < RESOLUTIONS.len() { RESOLUTIONS[i] / 2 } else { CQT_BINS };
        let rt = render_target(MAX_HISTORY as u32, height as u32);
        rt.texture.set_filter(FilterMode::Linear);
        render_targets.push(rt);
    }
    let mut last_rendered_updates = vec![0u64; RESOLUTIONS.len() + 2];

    let mut local_scale = ScaleType::Mel;
    let mut local_cmap = ColorMapType::Magma;
    let mut local_dir = ScrollDirection::RTL;
    let mut local_source = AudioSource::SinkMonitor;
    let mut local_iir = false;
    let mut smooth_head_pos: f64 = 0.0;
    
    let mut current_lut_type = local_cmap;
    let lut = ColorLut::new(current_lut_type);
    let mut colormap_texture = create_colormap_texture(&lut);

    let shader_material = load_material(
        ShaderSource::Glsl { vertex: VERTEX_SHADER, fragment: FRAGMENT_SHADER },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("scale_type", UniformType::Float1),
                UniformDesc::new("sample_rate", UniformType::Float1),
                UniformDesc::new("is_cqt_texture", UniformType::Float1),
            ],
            textures: vec!["colormap".to_string()],
            ..Default::default()
        },
    ).unwrap();

    let max_possible_height = (*RESOLUTIONS.iter().max().unwrap() / 2).max(CQT_BINS);
    let mut local_pixels_buffer = vec![0u8; MAX_HISTORY * max_possible_height * 4];

    loop {
        let mut visual_changed = false;
        let mut source_changed = false;

        if is_key_pressed(KeyCode::S) { local_scale = cycle_scale(local_scale); visual_changed = true; } 
        if is_key_pressed(KeyCode::C) { local_cmap = cycle_colormap(local_cmap); visual_changed = true; }
        if is_key_pressed(KeyCode::F) { local_dir = cycle_direction(local_dir); }
        if is_key_pressed(KeyCode::R) { current_fft_idx = (current_fft_idx + 1) % (RESOLUTIONS.len() + 1); } 
        if is_key_pressed(KeyCode::I) { local_iir = !local_iir; visual_changed = true; }
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
                s.scale_type = local_scale;
                s.colormap = local_cmap;
                s.audio_source = local_source;
                s.iir_enabled = local_iir;
            }
        }

        // Dynamically resolve which texture layer we are rendering based on the IIR toggle
        let actual_fft_idx = if current_fft_idx == RESOLUTIONS.len() {
            if local_iir { RESOLUTIONS.len() + 1 } else { RESOLUTIONS.len() }
        } else {
            current_fft_idx
        };

        let current_height = if actual_fft_idx >= RESOLUTIONS.len() { CQT_BINS } else { RESOLUTIONS[actual_fft_idx] / 2 };

        let mut actual_total_updates = 0u64;
        let mut full_redraw_needed = false;
        let mut delta_images: Vec<(f32, Image)> = Vec::new();

        {
            if let Ok(layers) = shared_layers.lock() {
                let layer = &layers[actual_fft_idx];
                actual_total_updates = layer.total_updates;
                
                let prev_updates = last_rendered_updates[actual_fft_idx];
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
                
                if actual_fft_idx != last_fft_idx {
                    let prev_mode_updates = layers[last_fft_idx].total_updates;
                    let diff_from_last = actual_total_updates as f64 - prev_mode_updates as f64;
                    smooth_head_pos += diff_from_last;
                    last_fft_idx = actual_fft_idx;
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
            cam.render_target = Some(render_targets[actual_fft_idx].clone());
            set_camera(&cam);
            draw_texture(&tex, 0.0, 0.0, WHITE);
            set_default_camera();
            last_rendered_updates[actual_fft_idx] = actual_total_updates;
        } else if !delta_images.is_empty() {
            let mut cam = Camera2D::from_display_rect(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32));
            cam.render_target = Some(render_targets[actual_fft_idx].clone());
            set_camera(&cam);
            
            for (x, img) in delta_images {
                let tex = Texture2D::from_image(&img);
                tex.set_filter(FilterMode::Nearest); 
                draw_texture(&tex, x, 0.0, WHITE);
            }
            set_default_camera();
            last_rendered_updates[actual_fft_idx] = actual_total_updates;
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

        let dst1_len = src1.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let dst2_len = src2.map(|r| r.w * scale_factor).unwrap_or(0.0);
        let total_draw_len = dst1_len + dst2_len;
        let start_pos_screen = (screen_time_dim - total_draw_len) / 2.0;

        let texture_to_draw = &render_targets[actual_fft_idx].texture;

        let scale_uniform_val = match local_scale {
            ScaleType::Linear => 0.0f32,
            ScaleType::Mel => 1.0f32,
            ScaleType::Logarithmic => 2.0f32,
            ScaleType::Chromatic => 3.0f32,
        };
        let is_cqt = if actual_fft_idx >= RESOLUTIONS.len() { 1.0f32 } else { 0.0f32 };
        
        shader_material.set_uniform("scale_type", scale_uniform_val);
        shader_material.set_uniform("sample_rate", SAMPLE_RATE as f32);
        shader_material.set_uniform("is_cqt_texture", is_cqt);
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

        draw_note_ruler(local_scale, local_dir);
        if DRAW_UI { draw_ui_overlay(local_scale, local_cmap, local_dir, current_fft_idx, current_view_len, local_source, local_iir); }

        if DRAW_UI {
            // --- Interactive Mouse Crosshair (Frequency/Note Peeking) ---
            let (mx, my) = mouse_position();
            
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

            if norm_time >= 0.0 && norm_time <= 1.0 && norm_freq >= 0.0 && norm_freq <= 1.0 {
                let max_freq = SAMPLE_RATE as f32 / 2.0;
                let current_hz = match local_scale {
                    ScaleType::Linear => norm_freq * max_freq,
                    ScaleType::Mel => {
                        let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
                        let current_mel = norm_freq * mel_max;
                        700.0 * (10.0f32.powf(current_mel / 2595.0) - 1.0)
                    },
                    ScaleType::Logarithmic => {
                        let min_freq = 20.0f32;
                        let log_min = min_freq.log2();
                        let log_max = max_freq.log2();
                        2.0f32.powf(log_min + norm_freq * (log_max - log_min))
                    },
                    ScaleType::Chromatic => {
                        let min_freq = 16.35159f32;
                        let log_min = min_freq.log2();
                        let log_max = max_freq.log2();
                        2.0f32.powf(log_min + norm_freq * (log_max - log_min))
                    }
                };

                // With the new temporal multi-rate structure, we use the CQT hop size to determine UI time intervals
                let drawn_time_seconds = (final_source_w_snapped * CQT_HOP_SIZE as f32) / SAMPLE_RATE as f32;
                let time_ago = norm_time * drawn_time_seconds;
                let (note_name, _) = hz_to_pitch(current_hz);

                draw_line(mx, 0.0, mx, sh, 1.0, Color::new(1.0, 1.0, 1.0, 0.3));
                draw_line(0.0, my, sw, my, 1.0, Color::new(1.0, 1.0, 1.0, 0.3));

                let mut intensity_u8 = 0;
                if let Ok(layers) = shared_layers.lock() {
                    let layer = &layers[actual_fft_idx];
                    
                    let exact_col = (head_offset - (norm_time * final_source_w_snapped) - 1.0).floor();
                    let ring_buffer_col = exact_col.rem_euclid(MAX_HISTORY as f32) as usize;
                    
                    let bin_idx = if actual_fft_idx >= RESOLUTIONS.len() {
                        let min_f = 20.0f32;
                        if current_hz <= min_f { 0 } else {
                            let norm_cqt = (current_hz.log2() - min_f.log2()) / (max_freq.log2() - min_f.log2());
                            (norm_cqt * layer.freq_bins as f32) as usize
                        }
                    } else {
                        ((current_hz / max_freq) * layer.freq_bins as f32) as usize
                    };
                    
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
    tilt_curve: &[f32],
) {
    let x_offset = col_idx * 4;
    for i in 0..freq_bins {
        // By mapping i=0 (DC) to y=(max), DC sits at the bottom of the memory buffer
        // This corresponds perfectly with OpenGL's native v=0.0 texture coordinate.
        let y = (freq_bins - 1) - i; 
        
        let magnitude = data[i] * tilt_curve[i]; // Apply Pink Noise Slope
        
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

fn cycle_scale(s: ScaleType) -> ScaleType {
    match s {
        ScaleType::Linear => ScaleType::Mel,
        ScaleType::Mel => ScaleType::Logarithmic,
        ScaleType::Logarithmic => ScaleType::Chromatic,
        ScaleType::Chromatic => ScaleType::Linear,
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
        ScrollDirection::LTR => ScrollDirection::DTU,
        ScrollDirection::DTU => ScrollDirection::UTD,
        ScrollDirection::UTD => ScrollDirection::RTL,
    }
}

fn freq_to_screen_pos(freq: f32, scale: ScaleType) -> f32 {
    let max_freq = SAMPLE_RATE as f32 / 2.0;
    match scale {
        ScaleType::Linear => freq / max_freq,
        ScaleType::Mel => {
            let mel_val = 2595.0 * (1.0 + freq / 700.0).log10();
            let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
            mel_val / mel_max
        },
        ScaleType::Logarithmic => {
            let min_freq = 20.0f32;
            if freq <= min_freq { return -1.0; } // Ensure off-screen handling
            let log_f = freq.log2();
            let log_min = min_freq.log2();
            let log_max = max_freq.log2();
            (log_f - log_min) / (log_max - log_min)
        },
        ScaleType::Chromatic => {
            let min_freq = 16.35159f32;
            if freq <= min_freq { return -1.0; } // Ensure off-screen handling
            let log_f = freq.log2();
            let log_min = min_freq.log2();
            let log_max = max_freq.log2();
            (log_f - log_min) / (log_max - log_min)
        }
    }
}

fn draw_note_ruler(scale: ScaleType, dir: ScrollDirection) {
    let w = screen_width();
    let h = screen_height();

    for midi in 21..109 {
        let freq = 440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0);
        if freq > (SAMPLE_RATE as f32 / 2.0) { break; }
        let norm_pos = freq_to_screen_pos(freq, scale);
        if norm_pos < 0.0 || norm_pos > 1.0 { continue; } // Exclude notes outside displayable bounds
        
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

fn draw_ui_overlay(scale: ScaleType, cmap: ColorMapType, dir: ScrollDirection, fft_idx: usize, history: usize, source: AudioSource, iir_enabled: bool) {
    let scale_str = format!("{:?}", scale);
    let map_str = format!("{:?}", cmap); 
    let dir_str = match dir {
        ScrollDirection::RTL => "RTL", 
        ScrollDirection::LTR => "LTR",
        ScrollDirection::DTU => "Fire", 
        ScrollDirection::UTD => "Rain",
    };
    
    let res_str = if fft_idx == RESOLUTIONS.len() {
        "CQT (HD)".to_string()
    } else {
        format!("{} bins", RESOLUTIONS[fft_idx] / 2)
    };
    
    let hist_str = format!("{}", history);
    let src_str = match source {
        AudioSource::SinkMonitor => "Sink",
        AudioSource::Microphone => "Mic",
    };
    let iir_str = if iir_enabled { "On" } else { "Off" };

    let stats = vec![
        UiStat { label: "Scale",  hotkey: Some('S'), value: scale_str.to_string(), color: ORANGE },
        UiStat { label: "Colour", hotkey: Some('C'), value: map_str,               color: YELLOW },
        UiStat { label: "Flow",   hotkey: Some('F'), value: dir_str.to_string(),   color: SKYBLUE },
        UiStat { label: "Resolution",    hotkey: Some('R'), value: res_str,               color: VIOLET },
        UiStat { label: "Window",    hotkey: Some('W'), value: hist_str,              color: PINK },
        UiStat { label: "Audio Src",     hotkey: Some('A'), value: src_str.to_string(),   color: GREEN },
        UiStat { label: "IIR Bass",     hotkey: Some('I'), value: iir_str.to_string(),   color: LIME },
    ];

    let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match dir {
        ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, screen_width(), 35.0, false),
        _ => (screen_width() - 220.0, 0.0, 220.0, 180.0, true),
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