use eframe::egui;
use eframe::epaint::ColorImage;
use ringbuf::{HeapRb, Consumer};
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit}; 
use spectrum_analyzer::scaling::divide_by_N_sqrt;
use spectrum_analyzer::windows::hann_window; 
use std::sync::Arc;
use libpulse_binding::sample::{Spec, Format};
use libpulse_binding::stream::Direction;
use libpulse_binding::def::BufferAttr; 
use libpulse_simple_binding::Simple;

// --- PRO SETTINGS ---
const FFT_SIZE: usize = 4096;      
const SAMPLE_RATE: u32 = 44100;
const HISTORY_LEN: usize = 512;    
const HOP_SIZE: usize = 512;       

#[derive(Debug, PartialEq, Clone, Copy)]
enum ColorMapType { Magma, Inferno, Viridis, Plasma, Turbo }

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); 

    let rb = HeapRb::<f32>::new(16384);
    let (mut producer, consumer) = rb.split();

    // --- AUDIO THREAD ---
    std::thread::spawn(move || {
        let default_sink = std::process::Command::new("pactl")
            .arg("get-default-sink")
            .output()
            .expect("Failed to run pactl");
        let sink_name = String::from_utf8_lossy(&default_sink.stdout).trim().to_string();
        let monitor_source = format!("{}.monitor", sink_name);
        println!(">> Target: {}", monitor_source);

        let spec = Spec {
            format: Format::F32le,
            channels: 1,
            rate: SAMPLE_RATE,
        };

        let frag_size = (SAMPLE_RATE as u32 * 4 * 10) / 1000; 
        let attr = BufferAttr {
            maxlength: u32::MAX,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: frag_size, 
        };

        let s = Simple::new(
            None, "RustySpectrogram", Direction::Record, Some(&monitor_source), 
            "LowLatencyStream", &spec, None, Some(&attr) 
        );

        let stream = match s { 
            Ok(s) => s,
            Err(e) => panic!("PulseAudio Connection Failed: {:?}", e),
        };

        let mut buffer_bytes = [0u8; 2048]; 

        loop {
            match stream.read(&mut buffer_bytes) {
                Ok(_) => {
                    let float_data: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            buffer_bytes.as_ptr() as *const f32,
                            buffer_bytes.len() / 4
                        )
                    };
                    let _ = producer.push_slice(float_data);
                },
                Err(_) => {}
            }
        }
    });

    // --- UI SETUP ---
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Rusty Spectrogram",
        options,
        Box::new(|cc| Box::new(SpectrogramApp::new(cc, consumer))),
    )
}

struct SpectrogramApp {
    consumer: Consumer<f32, Arc<HeapRb<f32>>>,
    rolling_window: Vec<f32>,
    spectrogram_image: ColorImage, 
    texture_handle: Option<egui::TextureHandle>, 
    use_log_scale: bool,
    current_colormap: ColorMapType, 
}

impl SpectrogramApp {
    fn new(_cc: &eframe::CreationContext, consumer: Consumer<f32, Arc<HeapRb<f32>>>) -> Self {
        let height = FFT_SIZE / 2;
        let image = ColorImage::new([HISTORY_LEN, height], egui::Color32::BLACK);

        Self {
            consumer,
            rolling_window: vec![0.0; FFT_SIZE],
            spectrogram_image: image,
            texture_handle: None, 
            use_log_scale: true,
            current_colormap: ColorMapType::Magma,
        }
    }

    fn update_spectrogram(&mut self, spectrum_data: &[(f32, f32)]) {
        let width = self.spectrogram_image.width();
        let height = self.spectrogram_image.height();
        let len = self.spectrogram_image.pixels.len();
        let pixels = &mut self.spectrogram_image.pixels;
        
        pixels.copy_within(1..len, 0);

        let max_freq_idx = spectrum_data.len();
        
        let gradient = match self.current_colormap {
            ColorMapType::Magma => colorous::MAGMA,
            ColorMapType::Inferno => colorous::INFERNO,
            ColorMapType::Viridis => colorous::VIRIDIS,
            ColorMapType::Plasma => colorous::PLASMA,
            ColorMapType::Turbo => colorous::TURBO,
        };

        let min_log = 1.0f32.ln();
        let max_log = (max_freq_idx as f32).ln();
        let log_range = max_log - min_log;

        for y in 0..height {
            let normalized_y = 1.0 - (y as f32 / height as f32);
            
            let freq_idx = if self.use_log_scale {
                let log_pos = min_log + (normalized_y * log_range);
                log_pos.exp() as usize
            } else {
                (normalized_y * max_freq_idx as f32) as usize
            };

            let idx = freq_idx.min(max_freq_idx - 1);
            let magnitude = spectrum_data[idx].1;
            let intensity = (magnitude * 1000.0).ln() / 8.0; 
            let color = gradient.eval_continuous(intensity.clamp(0.0, 1.0) as f64);
            
            pixels[(width - 1) + y * width] = 
                egui::Color32::from_rgb(color.r, color.g, color.b);
        }
    }
}

impl eframe::App for SpectrogramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let num_new = self.consumer.len();
        if num_new > 0 {
            let mut new_samples = vec![0.0; num_new];
            self.consumer.pop_slice(&mut new_samples);
            self.rolling_window.extend(new_samples);
            
            if self.rolling_window.len() > FFT_SIZE {
                let remove = self.rolling_window.len() - FFT_SIZE;
                self.rolling_window.drain(0..remove);
            }
        }

        let mut updates = 0;
        // Limit updates to prevent hanging
        while updates < 3 && num_new >= HOP_SIZE && self.rolling_window.len() == FFT_SIZE {
             if updates == 0 && num_new < HOP_SIZE { break; } 

            let windowed_data = hann_window(&self.rolling_window);
            let spectrum = samples_fft_to_spectrum(
                &windowed_data,
                SAMPLE_RATE,
                FrequencyLimit::All,
                Some(&divide_by_N_sqrt),
            ).unwrap();
            
            let raw_data: Vec<(f32, f32)> = spectrum.data().iter()
                .map(|(freq, val)| (freq.val(), val.val()))
                .collect();

            self.update_spectrogram(&raw_data);
            updates += 1;
            if updates >= 1 { break; } 
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Spectrogram");
                ui.separator();
                ui.checkbox(&mut self.use_log_scale, "Log Scale");
                
                egui::ComboBox::from_label("Colormap")
                .selected_text(format!("{:?}", self.current_colormap).split("::").last().unwrap())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.current_colormap, ColorMapType::Magma, "Magma");
                    ui.selectable_value(&mut self.current_colormap, ColorMapType::Inferno, "Inferno");
                    ui.selectable_value(&mut self.current_colormap, ColorMapType::Viridis, "Viridis");
                    ui.selectable_value(&mut self.current_colormap, ColorMapType::Plasma, "Plasma");
                    ui.selectable_value(&mut self.current_colormap, ColorMapType::Turbo, "Turbo");
                });
            });

            // 1. Get or Create Texture
            let texture = self.texture_handle.get_or_insert_with(|| {
                ctx.load_texture(
                    "spectrogram",
                    self.spectrogram_image.clone(),
                    egui::TextureOptions::NEAREST
                )
            });

            // 2. Update Texture (Requires Clone due to API limits, but still lighter on GPU)
            texture.set(self.spectrogram_image.clone(), egui::TextureOptions::NEAREST);

            // 3. Draw (Convert mutable ref to immutable ref using &*)
            ui.image(&*texture);
        });
        
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}