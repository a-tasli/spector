use rustfft::{num_complex::Complex, FftPlanner};
use spector_core::*;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::net::UnixListener;
use std::sync::atomic::Ordering;
use std::sync::{mpsc::sync_channel, Arc, Mutex};
use std::thread;
use std::time::Duration;

// PulseAudio bindings
use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

// --- INTERNAL DSP CONFIG ---
const USE_PHASE_CONFIDENCE_FILTER: bool = false;
const SPECTRAL_OVERSAMPLING: bool = false;
const OVERSAMPLE_TARGET: usize = 16384;
const MANUAL_STFT_MAPPING: bool = true;

struct ManualMapping {
    max_freq: f32,
    res_idx: usize,
}

const MANUAL_MAPPINGS: [ManualMapping; 4] = [
    ManualMapping { max_freq: 130.7, res_idx: 3 },
    ManualMapping { max_freq: 261.5, res_idx: 2 },
    ManualMapping { max_freq: 522.0, res_idx: 1 },
    ManualMapping { max_freq: MAX_DISPLAY_FREQ, res_idx: 0 },
];

const IIR_CROSSOVER_LOWER_HZ: f32 = 65.4;
const IIR_CROSSOVER_UPPER_HZ: f32 = 261.5;

// --- DSP HELPERS ---

struct IntensityLut { table: Vec<u8> }
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

fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())).collect()
}

#[derive(Clone)]
struct Biquad {
    b0: f32, b1: f32, b2: f32, a1: f32, a2: f32, z1: f32, z2: f32,
}
impl Biquad {
    fn bandpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self { b0: alpha / a0, b1: 0.0, b2: -alpha / a0, a1: (-2.0 * w0.cos()) / a0, a2: (1.0 - alpha) / a0, z1: 0.0, z2: 0.0 }
    }
    #[inline(always)]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if self.z1.abs() < 1e-10 { self.z1 = 0.0; }
        if self.z2.abs() < 1e-10 { self.z2 = 0.0; }
        y
    }
}

#[derive(Clone)]
struct CqtInstruction { cqt_bin_idx: usize, fft_idx: usize, b_start: usize, weights: Vec<f32>, weight_sum: f32, peak_dampening: f32 }

struct SplatKernel { half_width: isize, weights: Vec<f32>, weights_sqrt: Vec<f32> }

fn build_splat_kernels(splat_low: f32, splat_high: f32) -> Vec<SplatKernel> {
    (0..CQT_BINS).map(|bin| {
        let norm = bin as f32 / (CQT_BINS - 1) as f32;
        let spread = splat_low + (splat_high - splat_low) * norm;
        let mut half_width = spread.ceil() as isize;
        let mut weights = Vec::new();
        let mut sum = 0.0;
        for s in -half_width..=half_width {
            let d = s.abs() as f32;
            let w = if d <= spread { 0.5 * (1.0 + (std::f32::consts::PI * d / spread.max(0.0001)).cos()) } else { 0.0 };
            weights.push(w);
            sum += w;
        }
        if sum > 0.0 { for w in &mut weights { *w /= sum; } } else { weights = vec![1.0]; half_width = 0; }
        let weights_sqrt = weights.iter().map(|w| w.sqrt()).collect();
        SplatKernel { half_width, weights, weights_sqrt }
    }).collect()
}

fn build_cqt_map(sample_rate: u32, stft_specs: &[(usize, usize)], peak_damp_amount: f32) -> (Vec<CqtInstruction>, Vec<(usize, Biquad, f32)>) {
    let mut stft_map = Vec::with_capacity(CQT_BINS);
    let mut iir_filters = Vec::new();
    let min_freq = 20.0_f32;
    let max_freq = MAX_DISPLAY_FREQ;
    let log_min = min_freq.log2();
    let log_max = max_freq.log2();
    
    for bin in 0..CQT_BINS {
        let norm = bin as f32 / (CQT_BINS - 1) as f32;
        let freq = 2.0_f32.powf(log_min + norm * (log_max - log_min));
        let musical_bw = freq / 24.0;
        let erb = 24.7 * (4.37 * (freq / 1000.0) + 1.0);
        let mut bw_hz = if freq < 250.0 { musical_bw } else {
            let t = ((freq - 250.0) / 1000.0).clamp(0.0, 1.0);
            musical_bw * (1.0 - t) + (erb * 1.2) * t
        };

        if freq < IIR_CROSSOVER_UPPER_HZ {
            let q_factor = freq / bw_hz;
            iir_filters.push((bin, Biquad::bandpass(sample_rate as f32, freq, q_factor), bw_hz));
        }
        
        let (max_win_size, _) = stft_specs[stft_specs.len() - 1];
        let min_stft_bw = sample_rate as f32 / max_win_size as f32;
        if bw_hz < min_stft_bw * 1.5 { bw_hz = min_stft_bw * 1.5; }

        let ideal_n_float = 4.0 * sample_rate as f32 / bw_hz;
        let mut best_idx = 0;

        if MANUAL_STFT_MAPPING {
            for mapping in MANUAL_MAPPINGS.iter() {
                if freq <= mapping.max_freq { best_idx = mapping.res_idx; break; }
            }
            if best_idx >= stft_specs.len() { best_idx = stft_specs.len() - 1; }
        } else {
            let mut min_log_diff = f32::MAX;
            for (i, &(win_res, _)) in stft_specs.iter().enumerate() {
                let actual_n = win_res as f32;
                let log_diff = (actual_n.log2() - ideal_n_float.log2()).abs();
                if log_diff < min_log_diff { min_log_diff = log_diff; best_idx = i; }
            }
        }
        
        let (win_res, fft_size) = stft_specs[best_idx];
        let density_mismatch = win_res as f32 / ideal_n_float;
        let peak_dampening = 1.0 + ((1.0 / density_mismatch.powf(0.5)) - 1.0) * peak_damp_amount;

        let freq_res = sample_rate as f32 / fft_size as f32;
        let center_bin = freq / freq_res;
        let mut bw_bins = bw_hz / freq_res;
        if bw_bins < 2.0 { bw_bins = 2.0; } 
        
        let start_bin = ((center_bin - bw_bins / 2.0).floor() as usize).clamp(0, fft_size / 2 - 1);
        let end_bin = ((center_bin + bw_bins / 2.0).ceil() as usize).clamp(start_bin, fft_size / 2 - 1);
        
        let mut weights = Vec::new();
        let mut max_w = 0.0;
        for b in start_bin..=end_bin {
            let dist = (b as f32 - center_bin).abs();
            let w = if dist <= bw_bins / 2.0 { 0.5 * (1.0 + (std::f32::consts::PI * dist / (bw_bins / 2.0)).cos()) } else { 0.0 };
            weights.push(w);
            if w > max_w { max_w = w; }
        }
        if max_w > 0.0 { for w in &mut weights { *w /= max_w; } } else if weights.is_empty() && start_bin < fft_size/2 { weights.push(1.0); }

        stft_map.push(CqtInstruction {
            cqt_bin_idx: bin, fft_idx: best_idx, b_start: start_bin,
            weight_sum: weights.iter().sum::<f32>().max(0.0001),
            weights, peak_dampening,
        });
    }
    (stft_map, iir_filters)
}

struct StftState {
    window_size: usize, fft_size: usize, hop_size: usize, samples_since_last: usize,
    bin_freqs: Vec<f32>, expected_advances: Vec<f32>, prev_phases: Vec<f32>,
    last_mags: Vec<f32>, display_mags: Vec<f32>, last_true_freqs: Vec<f32>, decays: Vec<f32>, 
    fft: Arc<dyn rustfft::Fft<f32>>, window: Vec<f32>, scratch_buffer: Vec<Complex<f32>>, 
}

#[inline(always)]
fn compute_column_colors(col_buffer: &mut [u8], data: &[f32], freq_bins: usize, tilt_curve: &[f32], lut: &IntensityLut) {
    for i in 0..freq_bins {
        col_buffer[i] = lut.get(data[i] * tilt_curve[i]);
    }
}

// --- MAIN DAEMON ---

fn main() {
    println!("Spector Daemon Starting...");

    // 1. Initialize Shared Memory
    let shm_path = format!("/dev/shm/{}", SHM_PATH);
    let file = OpenOptions::new().read(true).write(true).create(true).open(&shm_path)
        .expect("Failed to create shared memory file in /dev/shm/");
    
    file.set_len(std::mem::size_of::<SharedMemoryBlock>() as u64)
        .expect("Failed to set SHM file size");
    
    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).expect("Failed to map shared memory") };

    // Zero out the block cleanly on startup
    unsafe { (*(mmap.as_mut_ptr() as *mut SharedMemoryBlock)).init_zeroed() };
    println!("Shared memory initialized at {}", shm_path);

    // 2. Shared Control State
    let initial_state = DaemonControlMessage {
        dsp_config: DspConfig::default(),
        audio_source: AudioSource::SinkMonitor as u32,
    };
    let control_state = Arc::new(Mutex::new(initial_state));

    // 3. IPC UDS Listener (Client -> Daemon)
    let ctrl_state_clone = control_state.clone();
    thread::spawn(move || {
        let _ = std::fs::remove_file(CTRL_SOCK_PATH);
        let listener = UnixListener::bind(CTRL_SOCK_PATH).expect("Failed to bind UDS");
        println!("Listening for UI controls on {}", CTRL_SOCK_PATH);

        for stream in listener.incoming() {
            if let Ok(mut stream) = stream {
                let mut buf = [0u8; std::mem::size_of::<DaemonControlMessage>()];
                if let Ok(len) = stream.read(&mut buf) {
                    if len == std::mem::size_of::<DaemonControlMessage>() {
                        let msg: &DaemonControlMessage = bytemuck::from_bytes(&buf);
                        if let Ok(mut state) = ctrl_state_clone.lock() {
                            *state = *msg;
                        }
                    }
                }
            }
        }
    });

    // 4. Audio Capture Thread (PulseAudio)
    let (audio_tx, audio_rx) = sync_channel::<Vec<f32>>(100);
    let (recycle_tx, recycle_rx) = sync_channel::<Vec<f32>>(100);
    let audio_ctrl_state = control_state.clone();

    thread::spawn(move || {
        let mut current_source = AudioSource::SinkMonitor;
        let get_device_name = |source: AudioSource| -> Option<String> {
            match source {
                AudioSource::SinkMonitor => {
                    if let Ok(output) = std::process::Command::new("pactl").arg("get-default-sink").output() {
                        Some(format!("{}.monitor", String::from_utf8_lossy(&output.stdout).trim()))
                    } else { None }
                },
                AudioSource::Microphone => {
                    if let Ok(output) = std::process::Command::new("pactl").arg("get-default-source").output() {
                        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                    } else { None }
                }
            }
        };

        let mut cached_device_name = get_device_name(current_source);
        let open_stream = |device_name: Option<&String>| -> Option<Simple> {
            let spec = Spec { format: Format::F32le, channels: 1, rate: SAMPLE_RATE };
            let frag_size = (SAMPLE_RATE as u32 * 4 * 15) / 1000; 
            let attr = BufferAttr {
                maxlength: frag_size * 4, tlength: u32::MAX, prebuf: u32::MAX, minreq: u32::MAX, fragsize: frag_size,
            };
            Simple::new(None, "spector-daemon", Direction::Record, device_name.map(|s| s.as_str()), "Recorder", &spec, None, Some(&attr)).ok()
        };

        let mut stream = open_stream(cached_device_name.as_ref());
        let mut buf = [0u8; 4096];

        loop {
            if let Ok(settings) = audio_ctrl_state.try_lock() {
                let target_source = AudioSource::from_u32(settings.audio_source);
                if target_source != current_source {
                    current_source = target_source;
                    cached_device_name = get_device_name(current_source);
                    stream = open_stream(cached_device_name.as_ref());
                }
            }

            if let Some(ref s) = stream {
                if let Ok(_) = s.read(&mut buf) {
                    let floats: &[f32] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, buf.len() / 4) };
                    let mut chunk = recycle_rx.try_recv().unwrap_or_else(|_| Vec::with_capacity(1024));
                    chunk.clear(); chunk.extend_from_slice(floats);
                    let _ = audio_tx.send(chunk);
                } else {
                    thread::sleep(Duration::from_millis(50));
                }
            } else {
                thread::sleep(Duration::from_millis(100));
                stream = open_stream(cached_device_name.as_ref());
            }
        }
    });

    // 5. DSP Thread (Heavy Math)
    let dsp_thread = thread::spawn(move || {
        let max_fft = *RESOLUTIONS.iter().max().unwrap();
        let mut rolling_audio = vec![0.0; max_fft * 2];
        let mut audio_head = 0; 
        let mut pending_buffer: VecDeque<f32> = VecDeque::with_capacity(8192);
        let lut = IntensityLut::new();
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
                window_size: win_size, fft_size, hop_size, samples_since_last: hop_size,
                bin_freqs: (0..fft_size / 2).map(|b| b as f32 * freq_res).collect(),
                expected_advances: (0..fft_size / 2).map(|b| b as f32 * hop_advance).collect(),
                prev_phases: vec![0.0; fft_size / 2], last_mags: vec![0.0; fft_size / 2],
                display_mags: vec![0.0; fft_size / 2], last_true_freqs: vec![0.0; fft_size / 2], decays: vec![0.0; fft_size / 2], 
                fft: planner.plan_fft_forward(fft_size), window: generate_hann_window(win_size),
                scratch_buffer: vec![Complex { re: 0.0, im: 0.0 }; fft_size],
            }
        }).collect();

        let mut local_cqt_col_no_iir = vec![0.0; CQT_BINS];
        let mut local_cqt_col_with_iir = vec![0.0; CQT_BINS];

        let mut tilt_curves: Vec<Vec<f32>> = (0..NUM_LAYERS).map(|i| {
            let size = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
            vec![1.0f32; size]
        }).collect();
        let mut cqt_decays = vec![0.0f32; CQT_BINS];

        let mut splat_kernels = build_splat_kernels(local_dsp_config.splat_low, local_dsp_config.splat_high);
        let (mut stft_cqt_map, mut iir_filters) = build_cqt_map(SAMPLE_RATE, &stft_specs, local_dsp_config.peak_density_dampening);

        let mut scratch_cols: Vec<Vec<u8>> = (0..NUM_LAYERS).map(|i| {
            let size = if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS };
            vec![0u8; size] 
        }).collect();

        let rebuild_dsp_caches = |config: &DspConfig, stft_states: &mut Vec<StftState>, tilt_curves: &mut Vec<Vec<f32>>, cqt_decays: &mut Vec<f32>, splat_kernels: &mut Vec<SplatKernel>, stft_cqt_map: &mut Vec<CqtInstruction>| {
            for (i, state) in stft_states.iter_mut().enumerate() {
                let half_size = state.fft_size / 2;
                let freq_res = SAMPLE_RATE as f32 / state.fft_size as f32;
                let min_log = 20.0f32.log2();
                let log_range_stft = (SAMPLE_RATE as f32 / 2.0).log2() - min_log; 
                
                for bin in 0..half_size {
                    let freq = bin as f32 * freq_res;
                    let norm = if freq >= 20.0 { ((freq.log2() - min_log) / log_range_stft).clamp(0.0, 1.0) } else { 0.0 };
                    state.decays[bin] = config.decay_low + (config.decay_high - config.decay_low) * norm;
                    if freq > 20.0 { tilt_curves[i][bin] = 10.0f32.powf((norm * log_range_stft * config.pink_noise_tilt) / 20.0); } 
                    else { tilt_curves[i][bin] = 1.0; }
                }
            }
            let min_log = 20.0f32.log2();
            let log_range_cqt = MAX_DISPLAY_FREQ.log2() - min_log;
            for bin in 0..CQT_BINS {
                let norm = bin as f32 / (CQT_BINS - 1) as f32;
                tilt_curves[RESOLUTIONS.len()][bin] = 10.0f32.powf((norm * log_range_cqt * config.pink_noise_tilt) / 20.0);
                tilt_curves[RESOLUTIONS.len()+1][bin] = tilt_curves[RESOLUTIONS.len()][bin];
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
            let freq = 2.0_f32.powf(20.0f32.log2() + norm * (MAX_DISPLAY_FREQ.log2() - 20.0f32.log2()));
            if freq <= IIR_CROSSOVER_LOWER_HZ { iir_blend_weights[bin] = 0.9; } 
            else if freq >= IIR_CROSSOVER_UPPER_HZ { iir_blend_weights[bin] = 0.0; } 
            else {
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
        let log_range_f = MAX_DISPLAY_FREQ.log2() - min_log_f;
        let log_range_inv = 1.0 / log_range_f;
        let two_pi = 2.0 * std::f32::consts::PI;

        let mut last_recv_time = std::time::Instant::now();
        let cqt_hop_duration = std::time::Duration::from_secs_f32(CQT_HOP_SIZE as f32 / SAMPLE_RATE as f32);
        let mut consecutive_black_hops: u64 = 0;
        let mut cqt_max_sample = 0.0f32;

        // Local tracking of updates so we can atomic store later
        let mut local_heads = [0usize; NUM_LAYERS];
        let mut local_updates = [0u64; NUM_LAYERS];

        // Move mmap into this thread to keep it alive and get the pointer safely
        let mut mmap = mmap;

        // Access the shared memory!
        let shm_block = unsafe { &mut *(mmap.as_mut_ptr() as *mut SharedMemoryBlock) };

        loop {
            // 1. Check for control settings updates
            if let Ok(state) = control_state.try_lock() {
                if state.dsp_config != local_dsp_config {
                    local_dsp_config = state.dsp_config;
                    rebuild_dsp_caches(&local_dsp_config, &mut stft_states, &mut tilt_curves, &mut cqt_decays, &mut splat_kernels, &mut stft_cqt_map);
                }
            }

            if pending_buffer.len() < MIN_HOP_SIZE {
                match audio_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(new_data) => {
                        pending_buffer.extend(new_data.iter());
                        let _ = recycle_tx.try_send(new_data);
                        last_recv_time = std::time::Instant::now();
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        let elapsed = last_recv_time.elapsed();
                        if elapsed >= cqt_hop_duration {
                            let missed_hops = (elapsed.as_secs_f32() / cqt_hop_duration.as_secs_f32()) as u64;
                            let hops_to_push = std::cmp::min(missed_hops, (MAX_HISTORY as u64).saturating_sub(consecutive_black_hops));
                            
                            if hops_to_push > 0 {
                                // SPARSITY MASK FAST-FORWARD (SHM Version)
                                for i in 0..NUM_LAYERS {
                                    let layer = &mut shm_block.layers[i];
                                    let h = local_heads[i];
                                    let h_push = hops_to_push as usize;
                                    
                                    if h + h_push <= MAX_HISTORY {
                                        layer.mask[h .. h + h_push].fill(0);
                                    } else {
                                        let chunk1 = MAX_HISTORY - h;
                                        layer.mask[h .. MAX_HISTORY].fill(0);
                                        layer.mask[0 .. (h_push - chunk1)].fill(0);
                                    }

                                    local_heads[i] = (h + h_push) % MAX_HISTORY;
                                    local_updates[i] += hops_to_push;
                                    
                                    layer.head.store(local_heads[i], Ordering::Release);
                                    layer.total_updates.store(local_updates[i], Ordering::Release);
                                }
                                consecutive_black_hops += hops_to_push;
                            }
                            
                            for state in stft_states.iter_mut() { state.display_mags.fill(0.0); state.last_mags.fill(0.0); }
                            local_cqt_col_no_iir.fill(0.0); local_cqt_col_with_iir.fill(0.0);
                            last_recv_time += cqt_hop_duration * (missed_hops as u32);
                        }
                    }
                    Err(_) => {}
                }
            }

            while let Ok(extra_data) = audio_rx.try_recv() {
                pending_buffer.extend(extra_data.iter());
                let _ = recycle_tx.try_send(extra_data);
                last_recv_time = std::time::Instant::now(); 
            }

            while pending_buffer.len() >= MIN_HOP_SIZE {
                let is_sleeping = consecutive_black_hops >= MAX_HISTORY as u64;

                if is_sleeping {
                    // Check if the next chunk has any valid sound before we skip the math
                    let mut wake_up = false;
                    for i in 0..MIN_HOP_SIZE {
                        if pending_buffer[i].abs() >= 1e-5 {
                            wake_up = true;
                            break;
                        }
                    }

                    if !wake_up {
                        // FAST-FORWARD: Consume the silent audio without doing heavy DSP
                        for _ in 0..MIN_HOP_SIZE {
                            pending_buffer.pop_front();
                        }
                        
                        // Keep the timing grids aligned so STFTs and CQT stay in phase relative to each other
                        iir_samples_accum += MIN_HOP_SIZE;
                        for state in stft_states.iter_mut() {
                            state.samples_since_last += MIN_HOP_SIZE;
                            if state.samples_since_last >= state.hop_size {
                                state.samples_since_last -= state.hop_size;
                            }
                        }
                        
                        cqt_samples_since_last += MIN_HOP_SIZE;
                        if cqt_samples_since_last >= CQT_HOP_SIZE {
                            cqt_samples_since_last -= CQT_HOP_SIZE;
                            // Cap to prevent overflow, we are already past MAX_HISTORY anyway
                            if consecutive_black_hops < u64::MAX - 100 {
                                consecutive_black_hops += 1;
                            }
                        }
                        continue; // Bypass the heavy math loop!
                    } else {
                        // WAKE-UP SEQUENCE!
                        consecutive_black_hops = 0;
                        
                        // Edge Case 1: Wipe the audio ring buffer. 
                        // Prevents any microscopic dither/noise frozen 15 seconds ago from leaking into the new STFT window.
                        rolling_audio.fill(0.0);
                        
                        // Edge Case 2: Zero the IIR delay lines.
                        // Prevents the filters from applying an old, microscopic state to a huge new transient (prevents math clicks).
                        for (_, biquad, _) in iir_filters.iter_mut() {
                            biquad.z1 = 0.0;
                            biquad.z2 = 0.0;
                        }
                        iir_power_accum.fill(0.0);
                        iir_peak_accum.fill(0.0);
                        
                        // Edge Case 3: Zero the phase trackers.
                        // Prevents a massive, incorrect phase differential calculation from jumping across 15 seconds of missing time.
                        for state in stft_states.iter_mut() {
                            state.prev_phases.fill(0.0);
                            state.last_mags.fill(0.0);
                            state.last_true_freqs.fill(0.0);
                            state.display_mags.fill(0.0);
                        }
                        
                        prev_cqt_col_no_iir.fill(0.0);
                        prev_cqt_col_with_iir.fill(0.0);
                        local_cqt_col_no_iir.fill(0.0);
                        local_cqt_col_with_iir.fill(0.0);
                    }
                }

                for _ in 0..MIN_HOP_SIZE {
                    let sample = pending_buffer.pop_front().unwrap();
                    cqt_max_sample = cqt_max_sample.max(sample.abs());

                    for (bin_idx, biquad, _) in iir_filters.iter_mut() {
                        let filtered = biquad.process(sample);
                        iir_power_accum[*bin_idx] += filtered * filtered;
                        let abs_f = filtered.abs();
                        if abs_f > iir_peak_accum[*bin_idx] { iir_peak_accum[*bin_idx] = abs_f; }
                    }
                    rolling_audio[audio_head] = sample;
                    rolling_audio[audio_head + max_fft] = sample;
                    audio_head = (audio_head + 1) % max_fft;
                }
                iir_samples_accum += MIN_HOP_SIZE;
                
                for state in stft_states.iter_mut() {
                    state.samples_since_last += MIN_HOP_SIZE;
                    if state.samples_since_last >= state.hop_size {
                        let start_sample = audio_head + max_fft - state.window_size;
                        let audio_slice = &rolling_audio[start_sample .. start_sample + state.window_size];
                        
                        let (active_buf, pad_buf) = state.scratch_buffer.split_at_mut(state.window_size);
                        for (dst, (&a, &w)) in active_buf.iter_mut().zip(audio_slice.iter().zip(state.window.iter())) {
                            *dst = Complex { re: a * w, im: 0.0 };
                        }
                        pad_buf.fill(Complex { re: 0.0, im: 0.0 });
                        
                        state.fft.process(&mut state.scratch_buffer);
                        
                        let scale = 2.0 / state.window_size as f32; 
                        let broadband_comp = (state.window_size as f32 / 2048.0).sqrt(); 
                        let cqt_makeup_gain = local_dsp_config.stft_boost.sqrt() * local_dsp_config.halo_sharp; 
                        let display_factor = broadband_comp * cqt_makeup_gain; 
                        
                        let half_size = state.fft_size / 2;
                        let freq_res = SAMPLE_RATE as f32 / state.fft_size as f32;
                        let sr_over_hop = SAMPLE_RATE as f32 / (2.0 * std::f32::consts::PI * state.hop_size as f32);
                        let inv_two_pi = 1.0 / two_pi; 
                        
                        for (bin, (((&c, mag), prev_phase), true_freq)) in state.scratch_buffer[0..half_size].iter()
                            .zip(&mut state.last_mags).zip(&mut state.prev_phases).zip(&mut state.last_true_freqs).enumerate() 
                        {
                            let raw_mag = c.norm() * scale;
                            let phase = c.im.atan2(c.re);
                            let phase_diff = phase - *prev_phase;
                            *prev_phase = phase;
                            
                            let diff = phase_diff - state.expected_advances[bin];
                            let diff_wrapped = diff - two_pi * (diff * inv_two_pi).round();
                            let offset_hz = diff_wrapped * sr_over_hop;
                            *true_freq = state.bin_freqs[bin] + offset_hz;
                            
                            let deviation_norm = (offset_hz / freq_res) * 1.3333333_f32; 
                            let dev_sq = deviation_norm * deviation_norm;
                            let phase_confidence = if USE_PHASE_CONFIDENCE_FILTER { 1.0 / (1.0 + dev_sq*dev_sq*dev_sq) } else { 1.0 };
                            
                            *mag = raw_mag * phase_confidence;
                            let display_mag = *mag * display_factor;
                            let prev_disp = state.display_mags[bin];
                            let next_disp = if display_mag > prev_disp { display_mag } else { (display_mag * (1.0 - state.decays[bin])) + (prev_disp * state.decays[bin]) };
                            state.display_mags[bin] = if next_disp < 1e-10 { 0.0 } else { next_disp };
                        }
                        state.samples_since_last -= state.hop_size;
                    }
                }

                cqt_samples_since_last += MIN_HOP_SIZE;

                if cqt_samples_since_last >= CQT_HOP_SIZE {
                    let min_stft_weight_sum = stft_cqt_map.first().map(|inst| inst.weight_sum).unwrap_or(1.0);
                    let min_iir_bw = iir_filters.first().map(|&(_, _, bw)| bw).unwrap_or(1.0);

                    let mut raw_cqt_power = vec![0.0f32; CQT_BINS]; let mut raw_cqt_peak = vec![0.0f32; CQT_BINS];
                    let mut sharp_cqt_power = vec![0.0f32; CQT_BINS]; let mut sharp_cqt_peak = vec![0.0f32; CQT_BINS];
                    let mut stft_amplitudes = vec![0.0f32; CQT_BINS]; let mut iir_amplitudes = vec![0.0f32; CQT_BINS];
                    
                    if iir_samples_accum > 0 {
                        let inv_samples = 1.0 / iir_samples_accum as f32;
                        for &(bin, _, bw) in iir_filters.iter() {
                            let norm_factor = if local_dsp_config.psd_normalization == 1 { bw / min_iir_bw } else { 1.0 };
                            let rms = ((iir_power_accum[bin] * inv_samples) / norm_factor).sqrt();
                            iir_amplitudes[bin] = ((iir_peak_accum[bin] * local_dsp_config.peak_weight) + (rms * local_dsp_config.rms_weight)) * local_dsp_config.iir_boost; 
                        }
                        for &(bin, _, _) in iir_filters.iter() { iir_power_accum[bin] = 0.0; iir_peak_accum[bin] = 0.0; }
                        iir_samples_accum = 0;
                    }

                    let stft_boost_sqrt = local_dsp_config.stft_boost.sqrt();

                    for inst in stft_cqt_map.iter() {
                        let state = &stft_states[inst.fft_idx];
                        let norm_factor = if local_dsp_config.psd_normalization == 1 { inst.weight_sum / min_stft_weight_sum } else { 1.0 };
                        let resolution_comp = state.window_size as f32 / 2048.0;
                        let comp_mag_factor = resolution_comp.sqrt();
                        
                        let start = inst.b_start;
                        let end = start + inst.weights.len();
                        
                        for ((&mag, &true_freq), &w) in state.last_mags[start..end].iter().zip(&state.last_true_freqs[start..end]).zip(&inst.weights) {
                            let energy = (((mag * mag) * w) / norm_factor) * resolution_comp;
                            raw_cqt_power[inst.cqt_bin_idx] += energy; 
                            let comp_mag = (mag * comp_mag_factor) * inst.peak_dampening; 
                            if comp_mag > raw_cqt_peak[inst.cqt_bin_idx] { raw_cqt_peak[inst.cqt_bin_idx] = comp_mag; }
                            
                            if true_freq >= 20.0 {
                                let target_bin = (((true_freq.log2() - min_log_f) * log_range_inv) * (CQT_BINS - 1) as f32).round() as isize;
                                if target_bin >= 0 && target_bin < CQT_BINS as isize {
                                    let splat = &splat_kernels[target_bin as usize];
                                    for ((s, &s_w), &s_w_sqrt) in (-splat.half_width..=splat.half_width).zip(&splat.weights).zip(&splat.weights_sqrt) {
                                        let ob = target_bin + s;
                                        if ob >= 0 && ob < CQT_BINS as isize {
                                            sharp_cqt_power[ob as usize] += energy * local_dsp_config.stft_boost * s_w;
                                            let s_mag = comp_mag * stft_boost_sqrt * s_w_sqrt;
                                            if s_mag > sharp_cqt_peak[ob as usize] { sharp_cqt_peak[ob as usize] = s_mag; }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    for bin in 0..CQT_BINS {
                        let raw_h = (raw_cqt_peak[bin] * local_dsp_config.peak_weight) + (raw_cqt_power[bin].sqrt() * local_dsp_config.rms_weight);
                        let sharp_h = (sharp_cqt_peak[bin] * local_dsp_config.peak_weight) + (sharp_cqt_power[bin].sqrt() * local_dsp_config.rms_weight);
                        stft_amplitudes[bin] = (raw_h * local_dsp_config.halo_raw) + (sharp_h * local_dsp_config.halo_sharp);
                        
                        let iir_w = iir_blend_weights[bin];
                        let current_with_iir = (iir_amplitudes[bin] * iir_w) + (stft_amplitudes[bin] * (1.0 - iir_w));
                        let decay = cqt_decays[bin];
                        
                        let p_no = prev_cqt_col_no_iir[bin];
                        local_cqt_col_no_iir[bin] = if stft_amplitudes[bin] > p_no { stft_amplitudes[bin] } else { (stft_amplitudes[bin] * (1.0 - decay)) + (p_no * decay) };
                        if local_cqt_col_no_iir[bin] < 1e-10 { local_cqt_col_no_iir[bin] = 0.0; }
                        prev_cqt_col_no_iir[bin] = local_cqt_col_no_iir[bin];
                        
                        let p_with = prev_cqt_col_with_iir[bin];
                        local_cqt_col_with_iir[bin] = if current_with_iir > p_with { current_with_iir } else { (current_with_iir * (1.0 - decay)) + (p_with * decay) };
                        if local_cqt_col_with_iir[bin] < 1e-10 { local_cqt_col_with_iir[bin] = 0.0; }
                        prev_cqt_col_with_iir[bin] = local_cqt_col_with_iir[bin];
                    }

                    for i in 0..NUM_LAYERS {
                        let data_source = if i < RESOLUTIONS.len() { &stft_states[i].display_mags } else if i == RESOLUTIONS.len() { &local_cqt_col_no_iir } else { &local_cqt_col_with_iir };
                        compute_column_colors(&mut scratch_cols[i], data_source, if i < RESOLUTIONS.len() { stft_states[i].fft_size / 2 } else { CQT_BINS }, &tilt_curves[i], &lut);
                    }

                    if cqt_max_sample < 1e-5 {
                        consecutive_black_hops += 1;
                    } else {
                        consecutive_black_hops = 0;
                    }
                    cqt_max_sample = 0.0;

                    if consecutive_black_hops < MAX_HISTORY as u64 {
                        // --- WRITE TO SHARED MEMORY ---
                        for i in 0..NUM_LAYERS {
                            let layer = &mut shm_block.layers[i];
                            let head = local_heads[i];
                            
                            layer.mask[head] = 255; // Valid audio marker
                            
                            // Unroll writes to the massive flat pixel array
                            let active_bins = layer.active_freq_bins;
                            for y in 0..active_bins {
                                layer.pixels[y * MAX_HISTORY + head] = scratch_cols[i][y];
                            }
                            
                            local_heads[i] = (head + 1) % MAX_HISTORY;
                            local_updates[i] += 1;
                            
                            // Atomic publish to the UI clients
                            layer.head.store(local_heads[i], Ordering::Release);
                            layer.total_updates.store(local_updates[i], Ordering::Release);
                        }
                    }
                    
                    cqt_samples_since_last -= CQT_HOP_SIZE;
                }
            }
        }
    });

    dsp_thread.join().unwrap();
}