use macroquad::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::net::UdpSocket;
use std::sync::mpsc::sync_channel;

// PulseAudio bindings for audio capture
use libpulse_binding::sample::{Spec, Format};
use libpulse_binding::stream::Direction;
use libpulse_binding::def::BufferAttr;
use libpulse_simple_binding::Simple;

// --- CONFIG ---
const SAMPLE_RATE: u32 = 44100;
const MIN_HOP_SIZE: usize = 256; 
const CQT_HOP_SIZE: usize = 512; 
const USE_PHASE_CONFIDENCE_FILTER: bool = true;

// --- SPECTRAL OVERSAMPLING (Zero-Padding) ---
const SPECTRAL_OVERSAMPLING: bool = false;
const OVERSAMPLE_TARGET: usize = 16384; 

// Variable resolutions and their corresponding Variable Hop Sizes
const RESOLUTIONS: [usize; 4] = [1024, 2048, 4096, 8192];
const HOP_SIZES: [usize; 4] = [256, 512, 512, 512];

const CQT_BINS: usize = 1200; 
const IIR_CROSSOVER_LOWER_HZ: f32 = 100.0; 
const IIR_CROSSOVER_UPPER_HZ: f32 = 350.0; 

// View length is 2520, but the Ring Buffer is 2800 to give the write-head an invisible margin!
const MAX_VIEW_LEN: usize = 2520; 
const MAX_HISTORY: usize = 2800; 
const TARGET_DISPLAY_WIDTH: f32 = 2520.0;

#[cfg(feature = "bg")]
const DRAW_UI: bool = false;

#[cfg(not(feature = "bg"))]
const DRAW_UI: bool = true;

// --- EXPLICIT MEMORY LAYOUTS FOR RAW IPC TRANSMISSION ---
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColorMapType { 
    Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix, 
    Cividis, Warm, Cool, Sinebow, 
    Greys, InvertedGreys, InvertedMagma 
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScrollDirection { RTL, LTR, DTU, UTD }

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum AudioSource { SinkMonitor, Microphone }

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScaleType { Linear, Mel, Logarithmic, Bark }

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
struct DspConfig {
    pink_noise_tilt: f32,
    peak_weight: f32,
    rms_weight: f32,
    psd_normalization: bool,
    peak_density_dampening: f32,
    decay_low: f32,
    decay_high: f32,
    splat_low: f32,
    splat_high: f32,
    halo_raw: f32,
    halo_sharp: f32,
    stft_boost: f32,
    iir_boost: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            pink_noise_tilt: 0.0,
            peak_weight: 0.5,
            rms_weight: 0.5,
            psd_normalization: false,
            peak_density_dampening: 0.0,
            decay_low: 0.0,
            decay_high: 0.0,
            splat_low: 3.0,
            splat_high: 0.0,
            halo_raw: 0.0,
            halo_sharp: 1.0,
            stft_boost: 5.0,
            iir_boost: 5.0,
        }
    }
}

/// Shared state synchronized via Mutex locally, and UDP globally.
#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
struct AppSettings {
    scale_type: ScaleType,
    colormap: ColorMapType,
    audio_source: AudioSource,
    dir: ScrollDirection,
    iir_enabled: bool,
    redraw_flag: bool,
    fft_idx: u32,
    view_len: u32,
    dsp_config: DspConfig,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            scale_type: ScaleType::Logarithmic,
            colormap: ColorMapType::Magma,
            audio_source: AudioSource::SinkMonitor,
            dir: ScrollDirection::RTL,
            iir_enabled: false,
            redraw_flag: false,
            fft_idx: RESOLUTIONS.len() as u32,
            view_len: MAX_VIEW_LEN as u32,
            dsp_config: DspConfig::default(),
        }
    }
}

// Memory-safe raw byte conversion for the UDP socket
fn to_bytes(settings: &AppSettings) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (settings as *const AppSettings) as *const u8,
            std::mem::size_of::<AppSettings>(),
        )
    }
}

fn from_bytes(bytes: &[u8]) -> Option<AppSettings> {
    if bytes.len() == std::mem::size_of::<AppSettings>() {
        let mut settings: AppSettings = unsafe { std::mem::zeroed() };
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (&mut settings as *mut AppSettings) as *mut u8,
                bytes.len(),
            );
        }
        Some(settings)
    } else {
        None
    }
}

// 1D Colormap Generator for the GPU Shader
struct ColorLut {
    bytes: Vec<[u8; 3]>,
}

impl ColorLut {
    fn new(map_type: ColorMapType) -> Self {
        let mut bytes = Vec::with_capacity(256);
        for i in 0..=255 {
            let mut val = i as f64 / 255.0;
            
            let gradient = match map_type {
                ColorMapType::Magma => colorous::MAGMA,
                ColorMapType::Inferno => colorous::INFERNO,
                ColorMapType::Viridis => colorous::VIRIDIS,
                ColorMapType::Plasma => colorous::PLASMA,
                ColorMapType::Turbo => colorous::TURBO,
                ColorMapType::Cubehelix => colorous::CUBEHELIX,
                ColorMapType::Cividis => colorous::CIVIDIS,
                ColorMapType::Warm => colorous::WARM,
                ColorMapType::Cool => colorous::COOL,
                ColorMapType::Sinebow => colorous::SINEBOW,
                // Greys naturally goes from White (0.0) to Black (1.0).
                ColorMapType::Greys => colorous::GREYS,
                // Inverted versions flip the intensity input so 0.0 accesses the 'loud' end of the map
                ColorMapType::InvertedGreys => { val = 1.0 - val; colorous::GREYS },
                ColorMapType::InvertedMagma => { val = 1.0 - val; colorous::MAGMA },
            };

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

// DSP Intensity Pre-computation Look-Up Table
struct IntensityLut {
    table: Vec<u8>,
}

impl IntensityLut {
    fn new() -> Self {
        let mut table = vec![0; 65536];
        for i in 0..65536 {
            let mag = (i as f32 / 65535.0) * 2.0; 
            let val_in = mag * 2000.0;
            let intensity = if val_in > 1.0 { (val_in.ln() / 8.0).clamp(0.0, 1.0) } else { 0.0 };
            table[i] = (intensity * 255.0) as u8;
        }
        Self { table }
    }

    #[inline(always)]
    fn get(&self, mag: f32) -> u8 {
        if mag <= 0.0 { return 0; }
        if mag >= 2.0 { return 255; }
        let norm = mag / 2.0;
        let idx = (norm * 65535.0) as usize;
        self.table[idx]
    }
}

// Reverted to Hann window for tight main-lobe frequency resolution.
fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}

// --- Professional IIR Biquad Filter (Direct Form II Transposed) ---
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
    peak_dampening: f32,
}

struct SplatKernel {
    half_width: isize,
    weights: Vec<f32>,
    weights_sqrt: Vec<f32>, // Precalculated to save 1,000,000+ sqrt() calls/sec
}

fn build_splat_kernels(splat_low: f32, splat_high: f32) -> Vec<SplatKernel> {
    (0..CQT_BINS).map(|bin| {
        let norm = bin as f32 / (CQT_BINS - 1) as f32;
        let spread = splat_low + (splat_high - splat_low) * norm;
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
        
        let mut weights_sqrt = Vec::with_capacity(weights.len());
        for w in &weights {
            weights_sqrt.push(w.sqrt());
        }
        
        SplatKernel { half_width, weights, weights_sqrt }
    }).collect()
}

fn build_cqt_map(sample_rate: u32, stft_specs: &[(usize, usize)], peak_damp_amount: f32) -> (Vec<CqtInstruction>, Vec<(usize, Biquad, f32)>) {
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

        if freq < IIR_CROSSOVER_UPPER_HZ {
            let q_factor = freq / bw_hz;
            iir_filters.push((bin, Biquad::bandpass(sample_rate as f32, freq, q_factor), bw_hz));
        }
        
        let (max_win_size, _) = stft_specs[stft_specs.len() - 1];
        let min_stft_bw = sample_rate as f32 / max_win_size as f32;
        if bw_hz < min_stft_bw * 1.5 {
            bw_hz = min_stft_bw * 1.5;
        }

        let ideal_n_float = 4.0 * sample_rate as f32 / bw_hz;
        let ideal_n = ideal_n_float as usize;
        
        let mut best_idx = 0;
        let mut min_diff = usize::MAX;
        for (i, &(win_res, _)) in stft_specs.iter().enumerate() {
            let diff = (win_res as isize - ideal_n as isize).unsigned_abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i;
            }
        }
        
        let (win_res, fft_size) = stft_specs[best_idx];
        
        let density_mismatch = win_res as f32 / ideal_n_float;
        let target_dampening = 1.0 / density_mismatch.powf(0.5);
        let peak_dampening = 1.0 + (target_dampening - 1.0) * peak_damp_amount;

        let freq_res = sample_rate as f32 / fft_size as f32;
        let center_bin = freq / freq_res;
        let mut bw_bins = bw_hz / freq_res;
        
        if bw_bins < 2.0 { bw_bins = 2.0; } 
        
        let start_bin = (center_bin - bw_bins / 2.0).floor() as usize;
        let end_bin = (center_bin + bw_bins / 2.0).ceil() as usize;
        
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
            peak_dampening,
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

struct StftState {
    window_size: usize,
    fft_size: usize,
    hop_size: usize,
    samples_since_last: usize,
    bin_freqs: Vec<f32>,          // Precomputed array eliminates inner-loop multiplications
    expected_advances: Vec<f32>,  // Precomputed array eliminates inner-loop multiplications
    prev_phases: Vec<f32>,
    last_mags: Vec<f32>,
    display_mags: Vec<f32>, 
    last_true_freqs: Vec<f32>,
    decays: Vec<f32>, 
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    scratch_buffer: Vec<Complex<f32>>, 
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
    float norm_f = 1.0 - uv.y; 
    float min_log_freq = 20.0;
    float max_freq = sample_rate / 2.0;

    if (is_cqt_texture > 0.5) {
        float current_hz;
        if (scale_type > 2.5) {
            float arg = max_freq / 600.0;
            float bark_max = 6.0 * log(arg + sqrt(arg * arg + 1.0));
            float current_bark = norm_f * bark_max;
            float z_over_6 = current_bark / 6.0;
            current_hz = 600.0 * (exp(z_over_6) - exp(-z_over_6)) / 2.0;
        } else if (scale_type > 1.5) {
            float log_min = log2(min_log_freq);
            float log_max = log2(max_freq);
            current_hz = exp2(log_min + norm_f * (log_max - log_min));
        } else if (scale_type > 0.5) {
            float mel_max = 2595.0 * (log(1.0 + max_freq / 700.0) / 2.30258509299);
            float current_mel = norm_f * mel_max;
            current_hz = 700.0 * (exp((current_mel / 2595.0) * 2.30258509299) - 1.0);
        } else {
            current_hz = norm_f * max_freq;
        }
        
        if (current_hz < min_log_freq) {
            norm_f = 0.0; 
        } else {
            norm_f = (log2(current_hz) - log2(min_log_freq)) / (log2(max_freq) - log2(min_log_freq));
            norm_f = clamp(norm_f, 0.0, 1.0);
        }
    } else {
        if (scale_type > 2.5) {
            float arg = max_freq / 600.0;
            float bark_max = 6.0 * log(arg + sqrt(arg * arg + 1.0));
            float current_bark = norm_f * bark_max;
            float z_over_6 = current_bark / 6.0;
            float current_hz = 600.0 * (exp(z_over_6) - exp(-z_over_6)) / 2.0;
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
    float intensity = texture2D(Texture, sample_uv).r;
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

// Delta chunker helper
struct DeltaUpdate {
    start_x: f32,
    width: usize,
}

fn push_delta(updates: &mut Vec<DeltaUpdate>, start_x: usize, width: usize, max_delta: usize) {
    let mut remaining = width;
    let mut curr_x = start_x;
    while remaining > 0 {
        let chunk = remaining.min(max_delta);
        updates.push(DeltaUpdate { start_x: curr_x as f32, width: chunk });
        curr_x += chunk;
        remaining -= chunk;
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut layers = Vec::new();
    for &size in RESOLUTIONS.iter() {
        let actual_fft = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET.max(size) } else { size };
        layers.push(SpectrogramLayer::new(actual_fft / 2));
    }
    layers.push(SpectrogramLayer::new(CQT_BINS)); 
    layers.push(SpectrogramLayer::new(CQT_BINS)); 

    let shared_settings = Arc::new(Mutex::new(AppSettings::default()));

    // ==========================================
    // --- UDP IPC LISTENER (Cross-Instance Sync)
    // ==========================================
    let shared_settings_recv = shared_settings.clone();
    let mut bind_port = 44101;
    let mut receiver = None;
    
    // Finds the first available port allowing up to 10 instances to run
    for p in 44101..=44110 {
        if let Ok(socket) = UdpSocket::bind(format!("127.0.0.1:{}", p)) {
            socket.set_nonblocking(false).ok();
            bind_port = p;
            receiver = Some(socket);
            break;
        }
    }

    if let Some(socket) = receiver {
        thread::spawn(move || {
            let mut buf = [0u8; 1024];
            loop {
                if let Ok((len, _)) = socket.recv_from(&mut buf) {
                    if let Some(new_settings) = from_bytes(&buf[..len]) {
                        if let Ok(mut s) = shared_settings_recv.lock() {
                            *s = new_settings;
                        }
                    }
                }
            }
        });
    }

    // UDP Sender allows the UI thread to push its changes out
    let udp_sender = UdpSocket::bind("127.0.0.1:0").ok();

    let shared_layers = Arc::new(Mutex::new(layers));
    
    // --- REPLACED HEAP_RB WITH BLOCKING CHANNEL ---
    let (audio_tx, audio_rx) = sync_channel::<Vec<f32>>(100);

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
                    // Push chunk to blocking channel! This wakes up the waiting Brain Thread.
                    let _ = audio_tx.send(floats.to_vec());
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
        let lut = IntensityLut::new(); // Optimized Pre-computed lookups

        let mut local_dsp_config = DspConfig::default();

        let stft_specs: Vec<(usize, usize)> = RESOLUTIONS.iter().map(|&res| {
            let actual_fft = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET.max(res) } else { res };
            (res, actual_fft)
        }).collect();

        let mut planner = FftPlanner::new();
        let mut stft_states: Vec<StftState> = stft_specs.iter().zip(HOP_SIZES.iter()).map(|(&(win_size, fft_size), &hop_size)| {
            let freq_res = SAMPLE_RATE as f32 / fft_size as f32;
            let hop_advance = 2.0 * std::f32::consts::PI * hop_size as f32 / fft_size as f32;
            StftState {
                window_size: win_size,
                fft_size,
                hop_size,
                samples_since_last: hop_size,
                bin_freqs: (0..fft_size / 2).map(|b| b as f32 * freq_res).collect(),
                expected_advances: (0..fft_size / 2).map(|b| b as f32 * hop_advance).collect(),
                prev_phases: vec![0.0; fft_size / 2],
                last_mags: vec![0.0; fft_size / 2],
                display_mags: vec![0.0; fft_size / 2],
                last_true_freqs: vec![0.0; fft_size / 2],
                decays: vec![0.0; fft_size / 2], 
                fft: planner.plan_fft_forward(fft_size),
                window: generate_hann_window(win_size),
                scratch_buffer: vec![Complex { re: 0.0, im: 0.0 }; fft_size],
            }
        }).collect();

        let mut float_history: Vec<Vec<f32>> = (0..=RESOLUTIONS.len() + 1).map(|i| {
            let size = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
            vec![0.0f32; MAX_HISTORY * size]
        }).collect();

        let mut local_cqt_col_no_iir = vec![0.0; CQT_BINS];
        let mut local_cqt_col_with_iir = vec![0.0; CQT_BINS];

        let mut tilt_curves: Vec<Vec<f32>> = (0..=RESOLUTIONS.len() + 1).map(|i| {
            let size = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
            vec![1.0f32; size]
        }).collect();
        let mut cqt_decays = vec![0.0f32; CQT_BINS];

        let mut splat_kernels = build_splat_kernels(local_dsp_config.splat_low, local_dsp_config.splat_high);
        let (mut stft_cqt_map, mut iir_filters) = build_cqt_map(SAMPLE_RATE, &stft_specs, local_dsp_config.peak_density_dampening);

        let mut scratch_cols: Vec<Vec<u8>> = (0..=RESOLUTIONS.len() + 1).map(|i| {
            let size = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
            vec![0u8; size * 4]
        }).collect();

        let mut rebuild_dsp_caches = |config: &DspConfig, stft_states: &mut Vec<StftState>, tilt_curves: &mut Vec<Vec<f32>>, cqt_decays: &mut Vec<f32>, splat_kernels: &mut Vec<SplatKernel>, stft_cqt_map: &mut Vec<CqtInstruction>| {
            for (i, state) in stft_states.iter_mut().enumerate() {
                let half_size = state.fft_size / 2;
                let freq_res = SAMPLE_RATE as f32 / state.fft_size as f32;
                let min_log = 20.0f32.log2();
                let log_range = (SAMPLE_RATE as f32 / 2.0).log2() - min_log;
                
                for bin in 0..half_size {
                    let freq = bin as f32 * freq_res;
                    let norm = if freq >= 20.0 { ((freq.log2() - min_log) / log_range).clamp(0.0, 1.0) } else { 0.0 };
                    state.decays[bin] = config.decay_low + (config.decay_high - config.decay_low) * norm;
                    
                    if freq > 20.0 {
                        let octaves_above_min = freq.log2() - min_log;
                        let tilt_db = octaves_above_min * config.pink_noise_tilt;
                        tilt_curves[i][bin] = 10.0f32.powf(tilt_db / 20.0);
                    } else {
                        tilt_curves[i][bin] = 1.0;
                    }
                }
            }
            
            let min_log = 20.0f32.log2();
            let log_range = (SAMPLE_RATE as f32 / 2.0).log2() - min_log;
            for bin in 0..CQT_BINS {
                let norm = bin as f32 / (CQT_BINS - 1) as f32;
                let octaves_above_min = norm * log_range;
                let tilt_db = octaves_above_min * config.pink_noise_tilt;
                let tilt_val = 10.0f32.powf(tilt_db / 20.0);
                tilt_curves[RESOLUTIONS.len()][bin] = tilt_val;
                tilt_curves[RESOLUTIONS.len()+1][bin] = tilt_val;
                cqt_decays[bin] = config.decay_low + (config.decay_high - config.decay_low) * norm;
            }
            
            *splat_kernels = build_splat_kernels(config.splat_low, config.splat_high);
            let (new_map, _) = build_cqt_map(SAMPLE_RATE, &stft_specs, config.peak_density_dampening);
            *stft_cqt_map = new_map;
        };

        rebuild_dsp_caches(&local_dsp_config, &mut stft_states, &mut tilt_curves, &mut cqt_decays, &mut splat_kernels, &mut stft_cqt_map);

        let mut iir_blend_weights = vec![0.0f32; CQT_BINS];
        for bin in 0..CQT_BINS {
            let norm = bin as f32 / (CQT_BINS - 1) as f32;
            let freq = 2.0_f32.powf(20.0f32.log2() + norm * ((SAMPLE_RATE as f32 / 2.0).log2() - 20.0f32.log2()));
            if freq <= IIR_CROSSOVER_LOWER_HZ {
                iir_blend_weights[bin] = 1.0;
            } else if freq >= IIR_CROSSOVER_UPPER_HZ {
                iir_blend_weights[bin] = 0.0;
            } else {
                let t = (freq - IIR_CROSSOVER_LOWER_HZ) / (IIR_CROSSOVER_UPPER_HZ - IIR_CROSSOVER_LOWER_HZ);
                iir_blend_weights[bin] = 0.5 * (1.0 + (std::f32::consts::PI * t).cos()); 
            }
        }

        let mut iir_power_accum = vec![0.0f32; CQT_BINS];
        let mut iir_peak_accum = vec![0.0f32; CQT_BINS];
        let mut iir_samples_accum = 0;
        let mut cqt_samples_since_last = 0;
        
        let mut prev_cqt_col_no_iir = vec![0.0f32; CQT_BINS];
        let mut prev_cqt_col_with_iir = vec![0.0f32; CQT_BINS];

        let min_log_f = 20.0f32.log2();
        let log_range_f = (SAMPLE_RATE as f32 / 2.0).log2() - min_log_f;
        let log_range_inv = 1.0 / log_range_f;
        let two_pi = 2.0 * std::f32::consts::PI;

        loop {
            let mut redraw_requested = false;

            if let Ok(mut s) = settings_ref.lock() {
                if s.redraw_flag {
                    redraw_requested = true;
                    s.redraw_flag = false; // Brain thread correctly resets the flag
                }
                if s.dsp_config != local_dsp_config {
                    local_dsp_config = s.dsp_config;
                    rebuild_dsp_caches(&local_dsp_config, &mut stft_states, &mut tilt_curves, &mut cqt_decays, &mut splat_kernels, &mut stft_cqt_map);
                }
            }

            if redraw_requested {
                if let Ok(mut layers) = layers_ref.lock() {
                    for (i, layer) in layers.iter_mut().enumerate() {
                        for t in 0..MAX_HISTORY {
                            let start = t * layer.freq_bins;
                            let slice = &float_history[i][start..start + layer.freq_bins];
                            paint_column_fast(&mut layer.pixels, t, slice, layer.freq_bins, &tilt_curves[i], &lut);
                        }
                        layer.total_updates += (MAX_HISTORY * 2) as u64; 
                    }
                }
            }

            // --- DEEP SLEEP SYNCHRONIZATION ---
            if pending_buffer.len() < MIN_HOP_SIZE {
                if let Ok(mut new_data) = audio_rx.recv() { // <--- CPU GOES TO SLEEP HERE (0% Usage)
                    pending_buffer.append(&mut new_data);
                }
            }

            while let Ok(mut extra_data) = audio_rx.try_recv() {
                pending_buffer.append(&mut extra_data);
            }

            while pending_buffer.len() >= MIN_HOP_SIZE {
                
                rolling_audio.rotate_left(MIN_HOP_SIZE);
                let start_idx = rolling_audio.len() - MIN_HOP_SIZE;

                let mut i = 0;
                for sample in pending_buffer.drain(0..MIN_HOP_SIZE) {
                    for (bin_idx, biquad, _) in iir_filters.iter_mut() {
                        let filtered = biquad.process(sample);
                        iir_power_accum[*bin_idx] += filtered * filtered;
                        
                        let abs_f = filtered.abs();
                        if abs_f > iir_peak_accum[*bin_idx] {
                            iir_peak_accum[*bin_idx] = abs_f;
                        }
                    }
                    rolling_audio[start_idx + i] = sample;
                    i += 1;
                }
                iir_samples_accum += MIN_HOP_SIZE;
                
                for state in stft_states.iter_mut() {
                    state.samples_since_last += MIN_HOP_SIZE;
                    
                    if state.samples_since_last >= state.hop_size {
                        let start_sample = rolling_audio.len() - state.window_size;
                        let audio_slice = &rolling_audio[start_sample..];
                        
                        let buffer = &mut state.scratch_buffer;
                        buffer.fill(Complex { re: 0.0, im: 0.0 });
                        for (i, (&a, &w)) in audio_slice.iter().zip(state.window.iter()).enumerate() {
                            buffer[i] = Complex { re: a * w, im: 0.0 };
                        }
                        
                        state.fft.process(buffer);
                        
                        let scale = 2.0 / state.window_size as f32; 
                        let broadband_comp = (state.window_size as f32 / 2048.0).sqrt(); 
                        let cqt_makeup_gain = local_dsp_config.stft_boost.sqrt() * local_dsp_config.halo_sharp; 
                        
                        let half_size = state.fft_size / 2;
                        let freq_res = SAMPLE_RATE as f32 / state.fft_size as f32;
                        let sr_over_hop = SAMPLE_RATE as f32 / (2.0 * std::f32::consts::PI * state.hop_size as f32);
                        
                        for (bin, (((&c, mag), prev_phase), true_freq)) in buffer[0..half_size].iter()
                            .zip(&mut state.last_mags)
                            .zip(&mut state.prev_phases)
                            .zip(&mut state.last_true_freqs)
                            .enumerate() 
                        {
                            let raw_mag = c.norm() * scale;
                            let phase = c.im.atan2(c.re);
                            
                            let phase_diff = phase - *prev_phase;
                            *prev_phase = phase;
                            
                            // FAST BRANCHLESS PHASE WRAP
                            let diff = phase_diff - state.expected_advances[bin];
                            let diff_wrapped = diff - two_pi * (diff / two_pi).round();
                                
                            let offset_hz = diff_wrapped * sr_over_hop;
                            *true_freq = state.bin_freqs[bin] + offset_hz;
                            
                            // FAST INLINE CONFIDENCE PENALTY
                            // 1.3333333 is exactly 1.0 / 0.75, completely avoiding division
                            let deviation_norm = (offset_hz / freq_res) * 1.3333333_f32; 
                            let dev_sq = deviation_norm * deviation_norm;
                            let dev_6 = dev_sq * dev_sq * dev_sq;
                            let phase_confidence = if USE_PHASE_CONFIDENCE_FILTER {
                                1.0 / (1.0 + dev_6)
                            } else {
                                1.0
                            };
                            
                            // Instantly crushes the ghost phase magnitude to zero. 
                            // Eliminates the need to carry confidence memory over to the CQT loop!
                            *mag = raw_mag * phase_confidence;
                            
                            let display_mag = *mag * broadband_comp * cqt_makeup_gain;
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

                if cqt_samples_since_last >= CQT_HOP_SIZE {
                    let min_stft_weight_sum = stft_cqt_map.first().map(|inst| inst.weight_sum).unwrap_or(1.0);
                    let min_iir_bw = iir_filters.first().map(|&(_, _, bw)| bw).unwrap_or(1.0);

                    let mut raw_cqt_power = vec![0.0f32; CQT_BINS];
                    let mut raw_cqt_peak = vec![0.0f32; CQT_BINS];
                    let mut sharp_cqt_power = vec![0.0f32; CQT_BINS];
                    let mut sharp_cqt_peak = vec![0.0f32; CQT_BINS];
                    
                    let mut stft_amplitudes = vec![0.0f32; CQT_BINS];
                    let mut iir_amplitudes = vec![0.0f32; CQT_BINS];
                    
                    if iir_samples_accum > 0 {
                        let inv_samples = 1.0 / iir_samples_accum as f32;
                        for &(bin, _, bw) in iir_filters.iter() {
                            let norm_factor = if local_dsp_config.psd_normalization { bw / min_iir_bw } else { 1.0 };
                            let rms = ((iir_power_accum[bin] * inv_samples) / norm_factor).sqrt();
                            let peak = iir_peak_accum[bin]; 
                            
                            let hybrid = (peak * local_dsp_config.peak_weight) + (rms * local_dsp_config.rms_weight);
                            iir_amplitudes[bin] = hybrid * local_dsp_config.iir_boost; 
                        }
                        for &(bin, _, _) in iir_filters.iter() {
                            iir_power_accum[bin] = 0.0;
                            iir_peak_accum[bin] = 0.0;
                        }
                        iir_samples_accum = 0;
                    }

                    let stft_boost = local_dsp_config.stft_boost;
                    let stft_boost_sqrt = stft_boost.sqrt(); // Hoisted Constant!

                    for inst in stft_cqt_map.iter() {
                        let state = &stft_states[inst.fft_idx];
                        let default_bin = inst.cqt_bin_idx;
                        let norm_factor = if local_dsp_config.psd_normalization { inst.weight_sum / min_stft_weight_sum } else { 1.0 };
                        
                        let resolution_comp = state.window_size as f32 / 2048.0;
                        let comp_mag_factor = resolution_comp.sqrt();
                        let peak_dampening = inst.peak_dampening;
                        
                        let start = inst.b_start;
                        let end = start + inst.weights.len();
                        
                        let mags = &state.last_mags[start..end];
                        let freqs = &state.last_true_freqs[start..end];
                        
                        for ((&mag, &true_freq), &w) in mags.iter().zip(freqs).zip(&inst.weights) {
                            let energy = (((mag * mag) * w) / norm_factor) * resolution_comp;
                            raw_cqt_power[default_bin] += energy; 
                            
                            let comp_mag = (mag * comp_mag_factor) * peak_dampening; 
                            if comp_mag > raw_cqt_peak[default_bin] {
                                raw_cqt_peak[default_bin] = comp_mag;
                            }
                            
                            if true_freq >= 20.0 {
                                // Fast log2 scaling matching screen layout
                                let norm = (true_freq.log2() - min_log_f) * log_range_inv;
                                let target_bin = (norm * (CQT_BINS - 1) as f32).round() as isize;
                                
                                if target_bin >= 0 && target_bin < CQT_BINS as isize {
                                    let boosted_energy = energy * stft_boost;
                                    let boosted_mag = comp_mag * stft_boost_sqrt; 
                                    
                                    let splat = &splat_kernels[target_bin as usize];
                                    
                                    // Utilizes Pre-computed s_weight_sqrt array!
                                    // Eliminates the massive 4-array memory zip() bottleneck!
                                    for ((s, &s_weight), &s_weight_sqrt) in (-splat.half_width..=splat.half_width).zip(&splat.weights).zip(&splat.weights_sqrt) {
                                        let offset_bin = target_bin + s;
                                        if offset_bin >= 0 && offset_bin < CQT_BINS as isize {
                                            let ob = offset_bin as usize;
                                            sharp_cqt_power[ob] += boosted_energy * s_weight;
                                            
                                            let s_mag = boosted_mag * s_weight_sqrt;
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
                        let raw_hybrid = (raw_cqt_peak[bin] * local_dsp_config.peak_weight) + (raw_cqt_power[bin].sqrt() * local_dsp_config.rms_weight);
                        let sharp_hybrid = (sharp_cqt_peak[bin] * local_dsp_config.peak_weight) + (sharp_cqt_power[bin].sqrt() * local_dsp_config.rms_weight);
                        
                        stft_amplitudes[bin] = (raw_hybrid * local_dsp_config.halo_raw) + (sharp_hybrid * local_dsp_config.halo_sharp);
                        
                        let iir_w = iir_blend_weights[bin];
                        let stft_w = 1.0 - iir_w;

                        let current_no_iir = stft_amplitudes[bin];
                        let current_with_iir = (iir_amplitudes[bin] * iir_w) + (stft_amplitudes[bin] * stft_w);
                        
                        let decay = cqt_decays[bin];
                        
                        let prev_no = prev_cqt_col_no_iir[bin];
                        let final_no = if current_no_iir > prev_no { current_no_iir } else { (current_no_iir * (1.0 - decay)) + (prev_no * decay) };
                        prev_cqt_col_no_iir[bin] = final_no;
                        local_cqt_col_no_iir[bin] = final_no;
                        
                        let prev_with = prev_cqt_col_with_iir[bin];
                        let final_with = if current_with_iir > prev_with { current_with_iir } else { (current_with_iir * (1.0 - decay)) + (prev_with * decay) };
                        prev_cqt_col_with_iir[bin] = final_with;
                        local_cqt_col_with_iir[bin] = final_with;
                    }

                    for i in 0..=RESOLUTIONS.len() + 1 {
                        let freq_bins = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
                        let data_source = if i < RESOLUTIONS.len() {
                            &stft_states[i].display_mags
                        } else if i == RESOLUTIONS.len() {
                            &local_cqt_col_no_iir
                        } else {
                            &local_cqt_col_with_iir
                        };

                        compute_column_colors(&mut scratch_cols[i], data_source, freq_bins, &tilt_curves[i], &lut);
                    }

                    if let Ok(mut layers) = layers_ref.lock() {
                        for i in 0..=RESOLUTIONS.len() + 1 {
                            let layer = &mut layers[i];
                            let head_idx = layer.head;
                            let freq_bins = layer.freq_bins;
                            
                            let data_source = if i < RESOLUTIONS.len() {
                                &stft_states[i].display_mags
                            } else if i == RESOLUTIONS.len() {
                                &local_cqt_col_no_iir
                            } else {
                                &local_cqt_col_with_iir
                            };

                            let float_start = head_idx * freq_bins;
                            let copy_len = data_source.len().min(freq_bins);
                            if float_start + copy_len <= float_history[i].len() {
                                float_history[i][float_start..float_start+copy_len].copy_from_slice(&data_source[0..copy_len]);
                            }

                            for y in 0..freq_bins {
                                let src_idx = y * 4;
                                let dst_idx = (y * MAX_HISTORY * 4) + (head_idx * 4);
                                layer.pixels[dst_idx..dst_idx+4].copy_from_slice(&scratch_cols[i][src_idx..src_idx+4]);
                            }

                            layer.head = (layer.head + 1) % MAX_HISTORY;
                            layer.total_updates += 1;
                        }
                    }
                    
                    cqt_samples_since_last -= CQT_HOP_SIZE;
                }
            }
        }
    });

    // ===============================================
    // --- THREAD C: FACE (UI & GPU Rendering) ---
    // ===============================================
    let mut render_targets = Vec::new();
    for i in 0..=RESOLUTIONS.len() + 1 {
        let height = if i < RESOLUTIONS.len() {
            let res = RESOLUTIONS[i];
            let actual_fft = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET.max(res) } else { res };
            actual_fft / 2
        } else { CQT_BINS };
        let rt = render_target(MAX_HISTORY as u32, height as u32);
        rt.texture.set_filter(FilterMode::Linear);
        render_targets.push(rt);
    }
    let mut last_rendered_updates = vec![0u64; RESOLUTIONS.len() + 2];

    let mut local_settings = AppSettings::default();
    let mut last_broadcast_settings = local_settings;

    let mut last_fft_idx = RESOLUTIONS.len(); 
    let mut smooth_head_pos: f64 = 0.0;
    
    let mut current_lut_type = local_settings.colormap;
    let mut colormap_texture = create_colormap_texture(&ColorLut::new(current_lut_type));

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

    let max_stft_height = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET / 2 } else { *RESOLUTIONS.iter().max().unwrap() / 2 };
    let max_possible_height = max_stft_height.max(CQT_BINS);
    let mut local_pixels_buffer = vec![0u8; MAX_HISTORY * max_possible_height * 4];

    // --- PRE-ALLOCATED TEXTURES & IMAGES FOR PRECISE UPDATE DRIVER OPTIMIZATIONS ---
    const MAX_DELTA_WIDTH: usize = 256;
    let mut delta_img = Image { 
        width: MAX_DELTA_WIDTH as u16, 
        height: max_possible_height as u16, 
        bytes: vec![0; MAX_DELTA_WIDTH * max_possible_height * 4] 
    };
    let delta_tex = Texture2D::from_image(&delta_img);
    delta_tex.set_filter(FilterMode::Nearest);

    let mut full_img = Image {
        width: MAX_HISTORY as u16,
        height: max_possible_height as u16,
        bytes: vec![0; MAX_HISTORY * max_possible_height * 4]
    };
    let full_tex = Texture2D::from_image(&full_img);
    full_tex.set_filter(FilterMode::Nearest);

    // Native Custom UI State
    let mut show_advanced_ui = false;
    let mut active_slider: Option<usize> = None;
    let mut drag_start_freq: Option<f32> = None;

    loop {
        let mut ui_wants_input = false;

        // --- 1. RECEIVE INCOMING UDP SYNCS ---
        if let Ok(s) = shared_settings.lock() {
            if *s != last_broadcast_settings {
                local_settings = *s;
                last_broadcast_settings = *s;
                if local_settings.colormap != current_lut_type {
                    current_lut_type = local_settings.colormap;
                    colormap_texture = create_colormap_texture(&ColorLut::new(current_lut_type));
                }
            }
        }

        let (mx, my) = mouse_position();
        let m_down = is_mouse_button_down(MouseButton::Left);
        let m_pressed = is_mouse_button_pressed(MouseButton::Left);

        if !m_down { drag_start_freq = None; }
        if is_key_pressed(KeyCode::T) { show_advanced_ui = !show_advanced_ui; }

        let menu_x = 20.0;
        let menu_y = 60.0; 
        let menu_w = 320.0;
        let menu_h = 640.0;

        if show_advanced_ui {
            if mx >= menu_x && mx <= menu_x + menu_w && my >= menu_y && my <= menu_y + menu_h {
                ui_wants_input = true;
            }
            if active_slider.is_some() {
                ui_wants_input = true;
            }
        }

        // --- 2. PROCESS LOCAL INPUTS ---
        if !ui_wants_input {
            if is_key_pressed(KeyCode::S) { local_settings.scale_type = cycle_scale(local_settings.scale_type); } 
            if is_key_pressed(KeyCode::C) { 
                local_settings.colormap = cycle_colormap(local_settings.colormap); 
                current_lut_type = local_settings.colormap;
                colormap_texture = create_colormap_texture(&ColorLut::new(current_lut_type));
            }
            if is_key_pressed(KeyCode::F) { local_settings.dir = cycle_direction(local_settings.dir); }
            if is_key_pressed(KeyCode::R) { local_settings.fft_idx = ((local_settings.fft_idx + 1) % (RESOLUTIONS.len() as u32 + 1)) as u32; } 
            if is_key_pressed(KeyCode::I) { local_settings.iir_enabled = !local_settings.iir_enabled; }
            if is_key_pressed(KeyCode::A) { 
                local_settings.audio_source = match local_settings.audio_source {
                    AudioSource::SinkMonitor => AudioSource::Microphone,
                    AudioSource::Microphone => AudioSource::SinkMonitor,
                };
            }
            if is_key_pressed(KeyCode::W) {
                local_settings.view_len = if local_settings.view_len == MAX_VIEW_LEN as u32 { (MAX_VIEW_LEN / 2) as u32 } else { MAX_VIEW_LEN as u32 };
            }
        }

        let actual_fft_idx = if local_settings.fft_idx as usize == RESOLUTIONS.len() {
            if local_settings.iir_enabled { RESOLUTIONS.len() + 1 } else { RESOLUTIONS.len() }
        } else {
            local_settings.fft_idx as usize
        };

        let current_height = if actual_fft_idx >= RESOLUTIONS.len() {
            CQT_BINS
        } else {
            let res = RESOLUTIONS[actual_fft_idx];
            let actual_fft = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET.max(res) } else { res };
            actual_fft / 2 
        };

        let mut actual_total_updates = 0u64;
        let mut full_redraw_needed = false;
        let mut delta_updates: Vec<DeltaUpdate> = Vec::new();

        {
            if let Ok(layers) = shared_layers.lock() {
                let layer = &layers[actual_fft_idx];
                actual_total_updates = layer.total_updates;
                
                let prev_updates = last_rendered_updates[actual_fft_idx];
                let diff = if actual_total_updates > prev_updates { actual_total_updates - prev_updates } else { 0 };
                
                let full_redraw_threshold = (MAX_HISTORY / 2) as u64;

                // FIX: Force a full redraw if we just switched resolutions
                let resolution_changed = actual_fft_idx != last_fft_idx;

                if resolution_changed || diff >= full_redraw_threshold {
                    let expected_len = MAX_HISTORY * current_height * 4;
                    local_pixels_buffer[..expected_len].copy_from_slice(&layer.pixels[..expected_len]);
                    full_redraw_needed = true;
                } else if diff > 0 {
                    let diff_usize = diff as usize;
                    let head = layer.head;
                    let height = layer.freq_bins;

                    // Inline closure to extract new columns from the shared layer to our local flat buffer
                    let mut copy_delta = |start_x: usize, width: usize| {
                        for y in 0..height {
                            let row_start = y * MAX_HISTORY * 4;
                            let src_idx = row_start + start_x * 4;
                            let len = width * 4;
                            local_pixels_buffer[src_idx .. src_idx + len].copy_from_slice(&layer.pixels[src_idx .. src_idx + len]);
                        }
                        push_delta(&mut delta_updates, start_x, width, MAX_DELTA_WIDTH);
                    };

                    if head >= diff_usize {
                        copy_delta(head - diff_usize, diff_usize);
                    } else {
                        let part1_len = diff_usize - head;
                        copy_delta(MAX_HISTORY - part1_len, part1_len);
                        if head > 0 { copy_delta(0, head); }
                    }
                }
                
                if resolution_changed {
                    let prev_mode_updates = layers[last_fft_idx].total_updates;
                    let diff_from_last = actual_total_updates as f64 - prev_mode_updates as f64;
                    smooth_head_pos += diff_from_last;
                    last_fft_idx = actual_fft_idx;
                }
            }
        }

        // --- Zero-Allocation Precise GPU Update Logic ---
        if full_redraw_needed {
            let total_bytes = MAX_HISTORY * current_height * 4;
            full_img.bytes[..total_bytes].copy_from_slice(&local_pixels_buffer[..total_bytes]);
            full_tex.update(&full_img);
            
            let mut cam = Camera2D::from_display_rect(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32));
            cam.render_target = Some(render_targets[actual_fft_idx].clone());
            set_camera(&cam);
            draw_texture_ex(&full_tex, 0.0, 0.0, WHITE, DrawTextureParams {
                source: Some(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32)),
                dest_size: Some(vec2(MAX_HISTORY as f32, current_height as f32)),
                ..Default::default()
            });
            set_default_camera();
            last_rendered_updates[actual_fft_idx] = actual_total_updates;

        } else if !delta_updates.is_empty() {
            let mut cam = Camera2D::from_display_rect(Rect::new(0.0, 0.0, MAX_HISTORY as f32, current_height as f32));
            cam.render_target = Some(render_targets[actual_fft_idx].clone());
            set_camera(&cam);
            
            for update in &delta_updates {
                for y in 0..current_height {
                    let src_row = y * MAX_HISTORY * 4;
                    let dst_row = y * MAX_DELTA_WIDTH * 4;
                    let start_idx = src_row + (update.start_x as usize) * 4;
                    let len = update.width * 4;
                    // Move only the new data block into the pre-allocated reusable texture Image struct
                    delta_img.bytes[dst_row .. dst_row + len].copy_from_slice(&local_pixels_buffer[start_idx .. start_idx + len]);
                }
                delta_tex.update(&delta_img);
                draw_texture_ex(&delta_tex, update.start_x, 0.0, WHITE, DrawTextureParams {
                    source: Some(Rect::new(0.0, 0.0, update.width as f32, current_height as f32)),
                    dest_size: Some(vec2(update.width as f32, current_height as f32)),
                    ..Default::default()
                });
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

        let (screen_time_dim, _screen_freq_dim) = match local_settings.dir {
            ScrollDirection::RTL | ScrollDirection::LTR => (sw, sh),
            ScrollDirection::DTU | ScrollDirection::UTD => (sh, sw),
        };

        let scale_factor = TARGET_DISPLAY_WIDTH / (local_settings.view_len as f32);
        let needed_source_w = screen_time_dim / scale_factor;

        let (final_source_w, _) = if needed_source_w <= local_settings.view_len as f32 {
            (needed_source_w, screen_time_dim)
        } else {
            (local_settings.view_len as f32, screen_time_dim)
        };

        let final_source_w_snapped = final_source_w;
        let start_pos_unwrapped = head_offset - final_source_w_snapped;
        let tex_w = MAX_HISTORY as f32;
        let tex_h = current_height as f32;

        let is_horizontal = match local_settings.dir {
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

        let scale_uniform_val = match local_settings.scale_type {
            ScaleType::Linear => 0.0f32,
            ScaleType::Mel => 1.0f32,
            ScaleType::Logarithmic => 2.0f32,
            ScaleType::Bark => 3.0f32,
        };
        let is_cqt = if actual_fft_idx >= RESOLUTIONS.len() { 1.0f32 } else { 0.0f32 };
        
        shader_material.set_uniform("scale_type", scale_uniform_val);
        shader_material.set_uniform("sample_rate", SAMPLE_RATE as f32);
        shader_material.set_uniform("is_cqt_texture", is_cqt);
        shader_material.set_texture("colormap", colormap_texture.clone());
        gl_use_material(&shader_material);

        if is_horizontal {
            let is_ltr = local_settings.dir == ScrollDirection::LTR;
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
            let is_fire = local_settings.dir == ScrollDirection::DTU;
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

        draw_note_ruler(local_settings.scale_type, local_settings.dir);
        if DRAW_UI { draw_ui_overlay(&local_settings, show_advanced_ui); }

        if DRAW_UI && !ui_wants_input {
            // --- Interactive Mouse Crosshair (Frequency/Note Peeking) ---
            let norm_time = if total_draw_len > 0.0 {
                match local_settings.dir {
                    ScrollDirection::RTL => (start_pos_screen + total_draw_len - mx) / total_draw_len,
                    ScrollDirection::LTR => (mx - start_pos_screen) / total_draw_len,
                    ScrollDirection::DTU => (start_pos_screen + total_draw_len - my) / total_draw_len,
                    ScrollDirection::UTD => (my - start_pos_screen) / total_draw_len,
                }
            } else {
                -1.0
            };

            let norm_freq = match local_settings.dir {
                ScrollDirection::RTL | ScrollDirection::LTR => 1.0 - (my / sh),
                ScrollDirection::DTU | ScrollDirection::UTD => mx / sw, 
            };

            if norm_time >= 0.0 && norm_time <= 1.0 && norm_freq >= 0.0 && norm_freq <= 1.0 {
                let max_freq = SAMPLE_RATE as f32 / 2.0;
                let current_hz = match local_settings.scale_type {
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
                    ScaleType::Bark => {
                        let bark_max = 6.0 * (max_freq / 600.0).asinh();
                        let current_bark = norm_freq * bark_max;
                        600.0 * (current_bark / 6.0).sinh()
                    }
                };

                if m_pressed {
                    drag_start_freq = Some(current_hz);
                }

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

                // Handle Drag-to-Select Rendering
                if let Some(f1) = drag_start_freq {
                    let norm_start_pos = freq_to_screen_pos(f1, local_settings.scale_type);
                    
                    if local_settings.dir == ScrollDirection::RTL || local_settings.dir == ScrollDirection::LTR {
                        let start_y = sh * (1.0 - norm_start_pos);
                        let min_y = start_y.min(my);
                        let max_y = start_y.max(my);
                        draw_rectangle(0.0, min_y, sw, max_y - min_y, Color::new(1.0, 0.8, 0.3, 0.2));
                        draw_line(0.0, start_y, sw, start_y, 1.0, Color::new(1.0, 0.8, 0.3, 0.8));
                        draw_line(0.0, my, sw, my, 1.0, Color::new(1.0, 0.8, 0.3, 0.8));
                    } else {
                        let start_x = sw * norm_start_pos;
                        let min_x = start_x.min(mx);
                        let max_x = start_x.max(mx);
                        draw_rectangle(min_x, 0.0, max_x - min_x, sh, Color::new(1.0, 0.8, 0.3, 0.2));
                        draw_line(start_x, 0.0, start_x, sh, 1.0, Color::new(1.0, 0.8, 0.3, 0.8));
                        draw_line(mx, 0.0, mx, sh, 1.0, Color::new(1.0, 0.8, 0.3, 0.8));
                    }

                    let f2 = current_hz;
                    let raw_semitones = 12.0 * (f2 / f1).log2();
                    let diff_hz = (f2 - f1).abs();
                    let sign_str = if raw_semitones >= 0.0 { "+" } else { "" };
                    
                    let drag_tooltip = format!("{}{:.2} st | {}{:.1} Hz", sign_str, raw_semitones, sign_str, diff_hz);
                    let drag_text_size = measure_text(&drag_tooltip, None, 20, 1.0);
                    
                    let mut d_tooltip_x = mx + 15.0;
                    let mut d_tooltip_y = my - 25.0;
                    if d_tooltip_x + drag_text_size.width + 10.0 > sw { d_tooltip_x = mx - drag_text_size.width - 15.0; }
                    if d_tooltip_y - 20.0 < 0.0 { d_tooltip_y = my + 45.0; }

                    draw_rectangle(d_tooltip_x, d_tooltip_y - 10.0, drag_text_size.width + 10.0, 25.0, Color::new(0.3, 0.1, 0.0, 0.6));
                    draw_text(&drag_tooltip, d_tooltip_x + 5.0, d_tooltip_y + 8.0, 20.0, Color::new(1.0, 0.9, 0.5, 1.0));
                }

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

        // --- Native Macroquad Advanced UI Overlay ---
        if DRAW_UI && show_advanced_ui {
            draw_rectangle(menu_x, menu_y, menu_w, menu_h, Color::new(0.0, 0.0, 0.0, 0.85));
            draw_rectangle_lines(menu_x, menu_y, menu_w, menu_h, 2.0, Color::new(0.3, 0.3, 0.3, 1.0));
            
            draw_text("DSP Engine Tweaks", menu_x + 15.0, menu_y + 30.0, 24.0, WHITE);
            
            let mut cy = menu_y + 60.0;
            let mut id = 0;
            
            draw_text("Signal Pipeline", menu_x + 15.0, cy, 18.0, ORANGE); cy += 25.0;
            draw_slider("Pink Noise Tilt (dB/Oct)", &mut local_settings.dsp_config.pink_noise_tilt, -6.0, 6.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_checkbox("PSD Normalization", &mut local_settings.dsp_config.psd_normalization, menu_x + 15.0, cy, (mx, my), m_pressed); cy += 30.0;
            draw_slider("Density Dampening", &mut local_settings.dsp_config.peak_density_dampening, 0.0, 2.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 45.0;

            draw_text("Dynamics & Decay", menu_x + 15.0, cy, 18.0, ORANGE); cy += 25.0;
            draw_slider("Peak Weight", &mut local_settings.dsp_config.peak_weight, 0.0, 1.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("RMS Weight", &mut local_settings.dsp_config.rms_weight, 0.0, 1.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("Phosphor Decay (Bass)", &mut local_settings.dsp_config.decay_low, 0.0, 0.1, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("Phosphor Decay (Treble)", &mut local_settings.dsp_config.decay_high, 0.0, 0.1, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 45.0;

            draw_text("CQT Kernel Splatting", menu_x + 15.0, cy, 18.0, ORANGE); cy += 25.0;
            draw_slider("Splat Spread (Bass)", &mut local_settings.dsp_config.splat_low, 0.0, 10.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("Splat Spread (Treble)", &mut local_settings.dsp_config.splat_high, 0.0, 5.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("Halo Raw Blend", &mut local_settings.dsp_config.halo_raw, 0.0, 10.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("Halo Sharp Blend", &mut local_settings.dsp_config.halo_sharp, 0.0, 10.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 45.0;

            draw_text("Makeup Gains", menu_x + 15.0, cy, 18.0, ORANGE); cy += 25.0;
            draw_slider("STFT Boost Gain", &mut local_settings.dsp_config.stft_boost, 1.0, 20.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); id += 1; cy += 35.0;
            draw_slider("IIR Boost Gain", &mut local_settings.dsp_config.iir_boost, 1.0, 20.0, menu_x + 15.0, cy, 180.0, (mx, my), m_down, m_pressed, &mut active_slider, id); 
        }

        // --- 3. BROADCAST LOCAL CHANGES VIA UDP ---
        // Exclude redraw_flag from equality check so we don't trigger infinite loops
        let mut diff = local_settings;
        diff.redraw_flag = false;
        let mut last_diff = last_broadcast_settings;
        last_diff.redraw_flag = false;

        if diff != last_diff {
            last_broadcast_settings = local_settings;
            if let Ok(mut s) = shared_settings.lock() {
                *s = local_settings;
            }

            // Sync with all background threads out in the wild!
            if let Some(ref sock) = udp_sender {
                let bytes = to_bytes(&local_settings);
                for p in 44101..=44110 {
                    if p != bind_port {
                        let _ = sock.send_to(bytes, format!("127.0.0.1:{}", p));
                    }
                }
            }
        }

        next_frame().await
    }
}

// --- Custom Macroquad UI Components ---

fn draw_slider(
    label: &str,
    val: &mut f32,
    min: f32,
    max: f32,
    x: f32,
    y: f32,
    w: f32,
    mouse_pos: (f32, f32),
    mouse_down: bool,
    mouse_pressed: bool,
    active_id: &mut Option<usize>,
    my_id: usize,
) -> bool {
    let mut changed = false;
    draw_text(label, x, y - 4.0, 16.0, WHITE);
    
    let is_hovered = mouse_pos.0 >= x && mouse_pos.0 <= x + w + 30.0 && mouse_pos.1 >= y - 10.0 && mouse_pos.1 <= y + 20.0;

    if mouse_pressed && is_hovered {
        *active_id = Some(my_id);
    }

    if mouse_down && *active_id == Some(my_id) {
        let norm = ((mouse_pos.0 - x) / w).clamp(0.0, 1.0);
        let new_val = min + norm * (max - min);
        if (*val - new_val).abs() > f32::EPSILON {
            *val = new_val;
            changed = true;
        }
    }

    if !mouse_down && *active_id == Some(my_id) {
        *active_id = None;
    }

    draw_line(x, y + 10.0, x + w, y + 10.0, 4.0, GRAY);
    
    let norm = (*val - min) / (max - min);
    let hx = x + norm * w;
    let color = if *active_id == Some(my_id) { YELLOW } else if is_hovered { LIGHTGRAY } else { WHITE };
    draw_circle(hx, y + 10.0, 6.0, color);
    
    let val_str = format!("{:.2}", val);
    draw_text(&val_str, x + w + 10.0, y + 14.0, 16.0, color);

    changed
}

fn draw_checkbox(
    label: &str,
    val: &mut bool,
    x: f32,
    y: f32,
    mouse_pos: (f32, f32),
    mouse_pressed: bool,
) -> bool {
    let mut changed = false;
    let box_size = 14.0;
    
    let is_hovered = mouse_pos.0 >= x && mouse_pos.0 <= x + 200.0 && mouse_pos.1 >= y - box_size && mouse_pos.1 <= y + 5.0;

    if mouse_pressed && is_hovered {
        *val = !*val;
        changed = true;
    }

    let color = if is_hovered { YELLOW } else { WHITE };

    draw_rectangle_lines(x, y - box_size + 2.0, box_size, box_size, 2.0, color);
    if *val {
        draw_rectangle(x + 3.0, y - box_size + 5.0, box_size - 6.0, box_size - 6.0, color);
    }
    draw_text(label, x + box_size + 10.0, y, 16.0, color);

    changed
}

#[inline(always)]
fn compute_column_colors(
    col_buffer: &mut [u8], 
    data: &[f32], 
    freq_bins: usize, 
    tilt_curve: &[f32],
    lut: &IntensityLut,
) {
    for i in 0..freq_bins {
        let y = (freq_bins - 1) - i; 
        let magnitude = data[i] * tilt_curve[i]; 
        let val = lut.get(magnitude);

        let idx = y * 4;
        col_buffer[idx] = val;     // R
        col_buffer[idx+1] = val;   // G
        col_buffer[idx+2] = val;   // B
        col_buffer[idx+3] = 255;   // A
    }
}

#[inline(always)]
fn paint_column_fast(
    pixels: &mut [u8], 
    col_idx: usize, 
    data: &[f32], 
    freq_bins: usize, 
    tilt_curve: &[f32],
    lut: &IntensityLut,
) {
    let x_offset = col_idx * 4;
    for i in 0..freq_bins {
        let y = (freq_bins - 1) - i; 
        let magnitude = data[i] * tilt_curve[i];
        let val = lut.get(magnitude);

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
        ScaleType::Mel => ScaleType::Bark,
        ScaleType::Bark => ScaleType::Logarithmic,
        ScaleType::Logarithmic => ScaleType::Linear,
    }
}

fn cycle_colormap(c: ColorMapType) -> ColorMapType {
    match c {
        ColorMapType::Magma => ColorMapType::Inferno,
        ColorMapType::Inferno => ColorMapType::Viridis,
        ColorMapType::Viridis => ColorMapType::Plasma,
        ColorMapType::Plasma => ColorMapType::Turbo,
        ColorMapType::Turbo => ColorMapType::Cubehelix,
        ColorMapType::Cubehelix => ColorMapType::Cividis,
        ColorMapType::Cividis => ColorMapType::Warm,
        ColorMapType::Warm => ColorMapType::Cool,
        ColorMapType::Cool => ColorMapType::Sinebow,
        ColorMapType::Sinebow => ColorMapType::Greys,
        ColorMapType::Greys => ColorMapType::InvertedGreys,
        ColorMapType::InvertedGreys => ColorMapType::InvertedMagma,
        ColorMapType::InvertedMagma => ColorMapType::Magma,
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
        ScaleType::Bark => {
            let bark_val = 6.0 * (freq / 600.0).asinh();
            let bark_max = 6.0 * (max_freq / 600.0).asinh();
            bark_val / bark_max
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
        if norm_pos < 0.0 || norm_pos > 1.0 { continue; } 
        
        let note_mod = midi % 12;
        let is_c = note_mod == 0;

        let (x, y) = match dir {
            ScrollDirection::RTL => (w, h * (1.0 - norm_pos)),
            ScrollDirection::LTR => (0.0, h * (1.0 - norm_pos)),
            ScrollDirection::UTD => (w * norm_pos, 0.0),
            ScrollDirection::DTU => (w * norm_pos, h),
        };

        // Determine what text label to draw based on the note and current scale
        let label = if is_c {
            let octave = (midi / 12) - 1;
            Some(format!("C{}", octave))
        } else if scale == ScaleType::Logarithmic {
            match note_mod {
                1 | 3 | 6 | 8 | 10 => Some("#".to_string()),
                2 => Some("D".to_string()),
                4 => Some("E".to_string()),
                5 => Some("F".to_string()),
                7 => Some("G".to_string()),
                9 => Some("A".to_string()),
                11 => Some("B".to_string()),
                _ => None,
            }
        } else {
            None
        };

        if dir == ScrollDirection::RTL || dir == ScrollDirection::LTR {
            let tick_len = if is_c { 15.0 } else { 5.0 };
            let start_x = if dir == ScrollDirection::RTL { x - tick_len } else { x };
            draw_line(start_x, y, start_x + tick_len, y, 1.0, WHITE);
            
            if let Some(text) = label {
                // Adjust text 10px closer to the edge for non-C notes
                let text_x = if dir == ScrollDirection::RTL { 
                    x - if is_c { 32.0 } else { 16.0 } 
                } else { 
                    x + if is_c { 20.0 } else { 10.0 } 
                };
                draw_text(&text, text_x, y + 4.0, 15.0, WHITE);
            }
        } else {
            let tick_len = if is_c { 15.0 } else { 5.0 };
            let start_y = if dir == ScrollDirection::DTU { y - tick_len } else { y };
            draw_line(x, start_y, x, start_y + tick_len, 1.0, WHITE);
            
            if let Some(text) = label {
                // Adjust text 10px closer to the edge for non-C notes
                let text_y = if dir == ScrollDirection::DTU { 
                    y - if is_c { 20.0 } else { 10.0 } 
                } else { 
                    y + if is_c { 30.0 } else { 20.0 } 
                };
                draw_text(&text, x - 5.0, text_y, 15.0, WHITE);
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

fn draw_ui_overlay(settings: &AppSettings, show_tweaks: bool) {
    let scale_str = format!("{:?}", settings.scale_type);
    let map_str = format!("{:?}", settings.colormap); 
    let dir_str = match settings.dir {
        ScrollDirection::RTL => "RTL", 
        ScrollDirection::LTR => "LTR",
        ScrollDirection::DTU => "Fire", 
        ScrollDirection::UTD => "Rain",
    };
    
    let res_str = if settings.fft_idx as usize == RESOLUTIONS.len() {
        "CQT (HD)".to_string()
    } else {
        let actual_fft = if SPECTRAL_OVERSAMPLING { OVERSAMPLE_TARGET.max(RESOLUTIONS[settings.fft_idx as usize]) } else { RESOLUTIONS[settings.fft_idx as usize] };
        if SPECTRAL_OVERSAMPLING && actual_fft > RESOLUTIONS[settings.fft_idx as usize] {
            format!("{} bins (x{} Pad)", actual_fft / 2, actual_fft / RESOLUTIONS[settings.fft_idx as usize])
        } else {
            format!("{} bins", actual_fft / 2)
        }
    };
    
    let hist_str = format!("{}", settings.view_len);
    let src_str = match settings.audio_source {
        AudioSource::SinkMonitor => "Sink",
        AudioSource::Microphone => "Mic",
    };
    let iir_str = if settings.iir_enabled { "On" } else { "Off" };
    let tweaks_str = if show_tweaks { "Visible" } else { "Hidden" };

    let stats = vec![
        UiStat { label: "Scale",  hotkey: Some('S'), value: scale_str.to_string(), color: ORANGE },
        UiStat { label: "Colour", hotkey: Some('C'), value: map_str,               color: YELLOW },
        UiStat { label: "Flow",   hotkey: Some('F'), value: dir_str.to_string(),   color: SKYBLUE },
        UiStat { label: "Resolution",    hotkey: Some('R'), value: res_str,               color: VIOLET },
        UiStat { label: "Window",    hotkey: Some('W'), value: hist_str,              color: PINK },
        UiStat { label: "Audio Src",     hotkey: Some('A'), value: src_str.to_string(),   color: GREEN },
        UiStat { label: "IIR Bass",     hotkey: Some('I'), value: iir_str.to_string(),   color: LIME },
        UiStat { label: "Tweaks",       hotkey: Some('T'), value: tweaks_str.to_string(), color: WHITE },
    ];

    let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match settings.dir {
        ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, screen_width(), 35.0, false),
        _ => (screen_width() - 220.0, 0.0, 220.0, 205.0, true),
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