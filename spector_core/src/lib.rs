use std::sync::atomic::{AtomicUsize, AtomicU64};

// --- GLOBAL CONFIG ---
pub const BASE_SAMPLE_RATE: u32 = 48000;
pub const SR_MULT: u32 = 2;
pub const SAMPLE_RATE: u32 = BASE_SAMPLE_RATE * SR_MULT;

pub const MAX_DISPLAY_FREQ: f32 = 22050.0;
pub const MIN_HOP_SIZE: usize = 128;
pub const CQT_HOP_SIZE: usize = (256 * SR_MULT) as usize;

pub const RESOLUTIONS: [usize; 4] = [2*1024, 1*4096, 2*4096, 3*4096];
pub const HOP_SIZES: [usize; 4] = [256, 256, 128, 128];
pub const CQT_BINS: usize = 1200;

pub const MAX_VIEW_LEN: usize = 2520;
pub const MAX_HISTORY: usize = 2800;

// To make a static C-repr shared memory block, we need a hard max for frequency bins.
// Your largest FFT is 3*4096 = 12288. Half of that (bins) is 6144.
pub const MAX_FREQ_BINS: usize = 6144;
pub const NUM_LAYERS: usize = 6; // 4 STFT + 2 CQT layers

// IPC Settings
pub const SHM_PATH: &str = "spector_shm_state";
pub const CTRL_SOCK_PATH: &str = "/tmp/spector_ctrl.sock";

// --- ENUMS ---
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo, Cubehelix, Cividis, Warm, Cool, Sinebow, Greys, InvertedGreys, InvertedMagma }

impl ColorMapType {
    pub fn cycle(self) -> Self {
        match self {
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
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScrollDirection { RTL, LTR, DTU, UTD }

impl ScrollDirection {
    pub fn cycle(self) -> Self {
        match self {
            ScrollDirection::RTL => ScrollDirection::LTR,
            ScrollDirection::LTR => ScrollDirection::DTU,
            ScrollDirection::DTU => ScrollDirection::UTD,
            ScrollDirection::UTD => ScrollDirection::RTL,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioSource { SinkMonitor = 0, Microphone = 1 }

impl AudioSource {
    pub fn toggle(self) -> Self {
        match self {
            AudioSource::SinkMonitor => AudioSource::Microphone,
            AudioSource::Microphone => AudioSource::SinkMonitor,
        }
    }
    
    pub fn from_u32(val: u32) -> Self {
        match val {
            1 => AudioSource::Microphone,
            _ => AudioSource::SinkMonitor,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleType { Linear, Mel, Logarithmic, Bark }

impl ScaleType {
    pub fn cycle(self) -> Self {
        match self {
            ScaleType::Linear => ScaleType::Mel,
            ScaleType::Mel => ScaleType::Bark,
            ScaleType::Bark => ScaleType::Logarithmic,
            ScaleType::Logarithmic => ScaleType::Linear,
        }
    }
}

// --- CONFIG STRUCTS ---

#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DspConfig {
    pub pink_noise_tilt: f32,
    pub peak_weight: f32,
    pub rms_weight: f32,
    pub psd_normalization: u32, // Changed bool to u32 for bytemuck C-repr safety
    pub peak_density_dampening: f32,
    pub decay_low: f32,
    pub decay_high: f32,
    pub splat_low: f32,
    pub splat_high: f32,
    pub halo_raw: f32,
    pub halo_sharp: f32,
    pub stft_boost: f32,
    pub iir_boost: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            pink_noise_tilt: 0.0,
            peak_weight: 0.5,
            rms_weight: 0.5,
            psd_normalization: 1,
            peak_density_dampening: 1.0,
            decay_low: 0.0,
            decay_high: 0.0,
            splat_low: 3.0,
            splat_high: 0.0,
            halo_raw: 0.0,
            halo_sharp: 1.0,
            stft_boost: 5.0,
            iir_boost: 1.0,
        }
    }
}

// Sent via IPC from Client -> Daemon
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DaemonControlMessage {
    pub dsp_config: DspConfig,
    pub audio_source: u32,
}

// --- SHARED MEMORY LAYOUT ---

/// Represents a single FFT or CQT layer in shared memory.
/// Size: ~17.2 MB per layer
#[repr(C)]
pub struct SharedLayer {
    // Atomic counters to allow lock-free reading by clients
    pub total_updates: AtomicU64,
    pub head: AtomicUsize,
    
    // The actual active height of this layer (e.g., 1200 for CQT, 1024 for STFT)
    pub active_freq_bins: usize,
    
    // 1D Sparsity mask (1 = valid audio, 0 = pure silence)
    pub mask: [u8; MAX_HISTORY],
    
    // The raw intensity pixels: [freq_y * MAX_HISTORY + time_x]
    pub pixels: [u8; MAX_HISTORY * MAX_FREQ_BINS],
}

/// The monolithic block mapped into RAM. 
/// Total size: ~103 MB.
#[repr(C)]
pub struct SharedMemoryBlock {
    pub layers: [SharedLayer; NUM_LAYERS],
}

impl SharedMemoryBlock {
    /// Safe initialization for the daemon to zero out the block
    pub fn init_zeroed(&mut self) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.total_updates.store(0, std::sync::atomic::Ordering::SeqCst);
            layer.head.store(0, std::sync::atomic::Ordering::SeqCst);
            
            // Set expected bin heights based on index
            layer.active_freq_bins = if i < RESOLUTIONS.len() {
                RESOLUTIONS[i] / 2
            } else {
                CQT_BINS
            };
            
            layer.mask.fill(0);
            layer.pixels.fill(0);
        }
    }
}