use eframe::egui;
use eframe::wgpu::util::DeviceExt;
use memmap2::Mmap;
use spector_core::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::atomic::Ordering;

const TARGET_DISPLAY_WIDTH: f32 = 2520.0;

// --- WGSL GPU Shader ---
const SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var ring_tex: texture_2d<f32>;
@group(0) @binding(1) var ring_sampler: sampler;
@group(0) @binding(2) var colormap_tex: texture_2d<f32>;
@group(0) @binding(3) var colormap_sampler: sampler;
@group(0) @binding(4) var ymap_tex: texture_2d<f32>;
@group(0) @binding(5) var ymap_sampler: sampler;
@group(0) @binding(6) var mask_tex: texture_2d<f32>; // 1D Sparsity Mask

struct Uniforms {
    scroll_offset: f32,
    view_width_norm: f32,
    direction: u32,
    _padding: u32,
};
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var time_uv: f32;
    var freq_uv: f32;

    if (uniforms.direction == 0u) { time_uv = in.uv.x; freq_uv = in.uv.y; } 
    else if (uniforms.direction == 1u) { time_uv = 1.0 - in.uv.x; freq_uv = in.uv.y; } 
    else if (uniforms.direction == 2u) { time_uv = in.uv.y; freq_uv = 1.0 - in.uv.x; } 
    else { time_uv = 1.0 - in.uv.y; freq_uv = in.uv.x; }

    let sample_u = fract(uniforms.scroll_offset - uniforms.view_width_norm + (time_uv * uniforms.view_width_norm));

    let mask_val = textureSample(mask_tex, ring_sampler, vec2<f32>(sample_u, 0.5)).r;
    if (mask_val < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); 
    }

    let map_val = textureSample(ymap_tex, ymap_sampler, vec2<f32>(1.0 - freq_uv, 0.5));
    let mapped_y = map_val.r + map_val.g / 256.0;

    let intensity = textureSample(ring_tex, ring_sampler, vec2<f32>(sample_u, mapped_y)).r;
    return textureSample(colormap_tex, colormap_sampler, vec2<f32>(intensity, 0.5));
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ShaderUniforms {
    scroll_offset: f32,
    view_width_norm: f32,
    direction: u32,
    _padding: u32,
}

// ----------------------------------------------------
// LUT Generators
// ----------------------------------------------------

fn generate_colormap_bytes(map_type: ColorMapType) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(256 * 4);
    for i in 0..=255 {
        let mut val = i as f64 / 255.0;
        let gradient = match map_type {
            ColorMapType::Magma => colorous::MAGMA,
            ColorMapType::Inferno => colorous::INFERNO,
            ColorMapType::Viridis => colorous::VIRIDIS,
            ColorMapType::Plasma => colorous::TURBO,
            ColorMapType::Turbo => colorous::TURBO,
            ColorMapType::Cubehelix => colorous::CUBEHELIX,
            ColorMapType::Cividis => colorous::CIVIDIS,
            ColorMapType::Warm => colorous::WARM,
            ColorMapType::Cool => colorous::COOL,
            ColorMapType::Sinebow => colorous::SINEBOW,
            ColorMapType::Greys => colorous::GREYS,
            ColorMapType::InvertedGreys => { val = 1.0 - val; colorous::GREYS },
            ColorMapType::InvertedMagma => { val = 1.0 - val; colorous::MAGMA },
        };
        let c = gradient.eval_continuous(val);
        bytes.extend_from_slice(&[c.r, c.g, c.b, 255]);
    }
    bytes
}

fn generate_y_mapping_bytes(scale_type: ScaleType, is_cqt: bool, max_freq: f32, nyquist: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1024 * 4);
    let min_log_freq: f32 = 20.0;

    for i in 0..1024 {
        let norm_f = i as f32 / 1023.0;
        let current_hz = match scale_type {
            ScaleType::Bark => {
                let arg = max_freq / 600.0;
                let bark_max = 6.0 * (arg + (arg * arg + 1.0).sqrt()).ln();
                600.0 * ((norm_f * bark_max) / 6.0).sinh()
            },
            ScaleType::Logarithmic => {
                2.0f32.powf(min_log_freq.log2() + norm_f * (max_freq.log2() - min_log_freq.log2()))
            },
            ScaleType::Mel => {
                let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
                700.0 * (10.0f32.powf((norm_f * mel_max) / 2595.0) - 1.0)
            },
            ScaleType::Linear => norm_f * max_freq,
        };

        let mut mapped_y = 0.0;
        if is_cqt {
            if current_hz >= min_log_freq {
                mapped_y = (current_hz.log2() - min_log_freq.log2()) / (max_freq.log2() - min_log_freq.log2());
            }
        } else {
            mapped_y = current_hz / nyquist;
        }

        let mapped_clamp = mapped_y.clamp(0.0, 1.0);
        let val_u16 = (mapped_clamp * 65535.0) as u16;
        bytes.extend_from_slice(&[(val_u16 >> 8) as u8, (val_u16 & 0xFF) as u8, 0, 255]);
    }
    bytes
}

// ----------------------------------------------------
// UI Logic
// ----------------------------------------------------

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

fn get_interval_name(semitones: f32) -> String {
    let st = semitones.abs().round() as usize;
    match st {
        0 => "Unison".to_string(), 1 => "Minor 2nd".to_string(), 2 => "Major 2nd".to_string(),
        3 => "Minor 3rd".to_string(), 4 => "Major 3rd".to_string(), 5 => "Perfect 4th".to_string(),
        6 => "Tritone".to_string(), 7 => "Perfect 5th".to_string(), 8 => "Minor 6th".to_string(),
        9 => "Major 6th".to_string(), 10 => "Minor 7th".to_string(), 11 => "Major 7th".to_string(),
        12 => "Octave".to_string(),
        _ => {
            let octaves = st / 12;
            let remainder = st % 12;
            if remainder == 0 { format!("{} Octaves", octaves) } else { format!("{} Octaves +", octaves) }
        }
    }
}

fn freq_to_norm_pos(freq: f32, scale: ScaleType) -> f32 {
    let max_freq = MAX_DISPLAY_FREQ;
    match scale {
        ScaleType::Linear => freq / max_freq,
        ScaleType::Mel => {
            let mel_val = 2595.0 * (1.0 + freq / 700.0).log10();
            let mel_max = 2595.0 * (1.0 + max_freq / 700.0).log10();
            mel_val / mel_max
        },
        ScaleType::Logarithmic => {
            let min_freq = 20.0f32;
            if freq <= min_freq { return -1.0; }
            (freq.log2() - min_freq.log2()) / (max_freq.log2() - min_freq.log2())
        },
        ScaleType::Bark => {
            let bark_val = 6.0 * (freq / 600.0).asinh();
            let bark_max = 6.0 * (max_freq / 600.0).asinh();
            bark_val / bark_max
        }
    }
}

fn draw_note_ruler(ui: &egui::Ui, rect: egui::Rect, scale: ScaleType, dir: ScrollDirection) {
    let max_freq = MAX_DISPLAY_FREQ;
    let painter = ui.painter();
    
    for midi in 21..109 {
        let freq = 440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0);
        if freq > max_freq { break; }
        
        let norm_pos = freq_to_norm_pos(freq, scale);
        if norm_pos < 0.0 || norm_pos > 1.0 { continue; }
        
        let note_mod = midi % 12;
        let is_c = note_mod == 0;
        let tick_len = if is_c { 15.0 } else { 5.0 };
        
        let label = if is_c {
            Some(format!("C{}", (midi / 12) - 1))
        } else if scale == ScaleType::Logarithmic {
            match note_mod {
                1|3|6|8|10 => Some("#".to_string()),
                2 => Some("D".to_string()), 4 => Some("E".to_string()),
                5 => Some("F".to_string()), 7 => Some("G".to_string()),
                9 => Some("A".to_string()), 11 => Some("B".to_string()),
                _ => None,
            }
        } else { None };

        let color = egui::Color32::from_white_alpha(180);

        match dir {
            ScrollDirection::RTL | ScrollDirection::LTR => {
                let y = rect.max.y - norm_pos * rect.height();
                let (start_x, end_x, align) = if dir == ScrollDirection::RTL {
                    (rect.max.x - tick_len, rect.max.x, egui::Align2::RIGHT_CENTER)
                } else {
                    (rect.min.x, rect.min.x + tick_len, egui::Align2::LEFT_CENTER)
                };
                
                painter.line_segment([egui::pos2(start_x, y), egui::pos2(end_x, y)], egui::Stroke::new(1.0, color));
                if let Some(text) = label {
                    let text_x = if dir == ScrollDirection::RTL { start_x - 5.0 } else { end_x + 5.0 };
                    painter.text(egui::pos2(text_x, y), align, text, egui::FontId::proportional(11.0), color);
                }
            },
            ScrollDirection::DTU | ScrollDirection::UTD => {
                let x = rect.min.x + norm_pos * rect.width();
                let (start_y, end_y, align) = if dir == ScrollDirection::DTU {
                    (rect.max.y - tick_len, rect.max.y, egui::Align2::CENTER_BOTTOM)
                } else {
                    (rect.min.y, rect.min.y + tick_len, egui::Align2::CENTER_TOP)
                };

                painter.line_segment([egui::pos2(x, start_y), egui::pos2(x, end_y)], egui::Stroke::new(1.0, color));
                if let Some(text) = label {
                    let text_y = if dir == ScrollDirection::DTU { start_y - 5.0 } else { end_y + 5.0 };
                    painter.text(egui::pos2(x, text_y), align, text, egui::FontId::proportional(11.0), color);
                }
            }
        }
    }
}

// ----------------------------------------------------
// Custom WGPU Resource Holder
// ----------------------------------------------------

struct CustomWgpuState {
    pipeline: eframe::wgpu::RenderPipeline,
    bind_group: eframe::wgpu::BindGroup,
    uniform_buf: eframe::wgpu::Buffer,
    ring_tex: eframe::wgpu::Texture,
    colormap_tex: eframe::wgpu::Texture,
    ymap_tex: eframe::wgpu::Texture,
    mask_tex: eframe::wgpu::Texture,
    current_freq_bins: usize,
}

impl CustomWgpuState {
    fn new(cc: &eframe::CreationContext<'_>, freq_bins: usize) -> Self {
        let wgpu_state = cc.wgpu_render_state.as_ref().unwrap();
        let device = &wgpu_state.device;
        let queue = &wgpu_state.queue;

        let uniforms = ShaderUniforms { scroll_offset: 0.0, view_width_norm: 1.0, direction: 0, _padding: 0 };
        let uniforms_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&uniforms as *const ShaderUniforms) as *const u8, std::mem::size_of::<ShaderUniforms>()) };

        let uniform_buf = device.create_buffer_init(&eframe::wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"), contents: uniforms_bytes, usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST,
        });

        let ring_tex = device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("Ring Buffer"),
            size: eframe::wgpu::Extent3d { width: MAX_HISTORY as u32, height: freq_bins as u32, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::R8Unorm,
            usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });
        
        let mask_tex = device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("Mask Buffer"),
            size: eframe::wgpu::Extent3d { width: MAX_HISTORY as u32, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::R8Unorm,
            usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });

        let ring_sampler = device.create_sampler(&eframe::wgpu::SamplerDescriptor {
            address_mode_u: eframe::wgpu::AddressMode::Repeat, address_mode_v: eframe::wgpu::AddressMode::ClampToEdge,
            mag_filter: eframe::wgpu::FilterMode::Linear, min_filter: eframe::wgpu::FilterMode::Linear, ..Default::default()
        });

        let colormap_tex = device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("Colormap"), size: eframe::wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::Rgba8Unorm, usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });
        let ymap_tex = device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("YMap"), size: eframe::wgpu::Extent3d { width: 1024, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::Rgba8Unorm, usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });
        
        let map_sampler = device.create_sampler(&eframe::wgpu::SamplerDescriptor { mag_filter: eframe::wgpu::FilterMode::Linear, min_filter: eframe::wgpu::FilterMode::Linear, ..Default::default() });

        queue.write_texture(
            eframe::wgpu::TexelCopyTextureInfo { texture: &colormap_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
            &generate_colormap_bytes(ColorMapType::Magma),
            eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: None },
            eframe::wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 }
        );

        queue.write_texture(
            eframe::wgpu::TexelCopyTextureInfo { texture: &ymap_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
            &generate_y_mapping_bytes(ScaleType::Logarithmic, true, MAX_DISPLAY_FREQ, SAMPLE_RATE as f32 / 2.0),
            eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1024 * 4), rows_per_image: None },
            eframe::wgpu::Extent3d { width: 1024, height: 1, depth_or_array_layers: 1 }
        );

        let shader = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
            label: Some("Spector WGSL"), source: eframe::wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
            label: Some("BG Layout"),
            entries: &[
                eframe::wgpu::BindGroupLayoutEntry { binding: 0, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true }, view_dimension: eframe::wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 1, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Sampler(eframe::wgpu::SamplerBindingType::Filtering), count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 2, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true }, view_dimension: eframe::wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 3, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Sampler(eframe::wgpu::SamplerBindingType::Filtering), count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 4, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true }, view_dimension: eframe::wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 5, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Sampler(eframe::wgpu::SamplerBindingType::Filtering), count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 6, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true }, view_dimension: eframe::wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                eframe::wgpu::BindGroupLayoutEntry { binding: 7, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Buffer { ty: eframe::wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bind_group = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor {
            label: Some("Bind Group"), layout: &bind_group_layout,
            entries: &[
                eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::TextureView(&ring_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                eframe::wgpu::BindGroupEntry { binding: 1, resource: eframe::wgpu::BindingResource::Sampler(&ring_sampler) },
                eframe::wgpu::BindGroupEntry { binding: 2, resource: eframe::wgpu::BindingResource::TextureView(&colormap_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                eframe::wgpu::BindGroupEntry { binding: 3, resource: eframe::wgpu::BindingResource::Sampler(&map_sampler) },
                eframe::wgpu::BindGroupEntry { binding: 4, resource: eframe::wgpu::BindingResource::TextureView(&ymap_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                eframe::wgpu::BindGroupEntry { binding: 5, resource: eframe::wgpu::BindingResource::Sampler(&map_sampler) },
                eframe::wgpu::BindGroupEntry { binding: 6, resource: eframe::wgpu::BindingResource::TextureView(&mask_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                eframe::wgpu::BindGroupEntry { binding: 7, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"), bind_group_layouts: &[Some(&bind_group_layout)], immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor {
            label: Some("Spector Pipeline"), layout: Some(&pipeline_layout),
            vertex: eframe::wgpu::VertexState { 
                module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(eframe::wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(eframe::wgpu::ColorTargetState {
                    format: wgpu_state.target_format, blend: Some(eframe::wgpu::BlendState::REPLACE), write_mask: eframe::wgpu::ColorWrites::ALL,
                })],
                compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: eframe::wgpu::PrimitiveState::default(), depth_stencil: None, multisample: eframe::wgpu::MultisampleState::default(),
            multiview_mask: None, cache: None,
        });

        Self { pipeline, bind_group, uniform_buf, ring_tex, colormap_tex, ymap_tex, mask_tex, current_freq_bins: freq_bins }
    }
}

struct SpectorRenderCallback;

impl eframe::egui_wgpu::CallbackTrait for SpectorRenderCallback {
    fn prepare(
        &self, _device: &eframe::wgpu::Device, _queue: &eframe::wgpu::Queue, _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder, _callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> { Vec::new() }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        let state = callback_resources.get::<CustomWgpuState>().unwrap();
        render_pass.set_pipeline(&state.pipeline);
        render_pass.set_bind_group(0, &state.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

// ----------------------------------------------------
// Safe Wrapper for Mmap memory
// ----------------------------------------------------
struct ShmReader(*const SharedMemoryBlock);
unsafe impl Send for ShmReader {}
unsafe impl Sync for ShmReader {}

// ----------------------------------------------------
// Main App Structure
// ----------------------------------------------------

struct SpectorApp {
    _mmap: Mmap, // Kept alive so pointer doesn't dangle
    shm_ptr: ShmReader,

    // Client Local State
    scale_type: ScaleType,
    colormap: ColorMapType,
    dir: ScrollDirection,
    fft_idx: usize,
    view_len: usize,

    // Daemon Remote State (Synced via IPC)
    dsp_config: DspConfig,
    audio_source: AudioSource,

    // Rendering trackers
    smooth_head_pos: f64,
    last_uploaded_updates: u64,
    last_uploaded_head: usize,
    
    current_colormap: ColorMapType,
    current_scale: ScaleType,
    current_is_cqt: bool,
    
    show_menu: bool,
    drag_start_freq: Option<f32>,
}

impl SpectorApp {
    fn send_ipc_update(&self) {
        let msg = DaemonControlMessage {
            dsp_config: self.dsp_config,
            audio_source: self.audio_source as u32,
        };
        if let Ok(mut stream) = UnixStream::connect(CTRL_SOCK_PATH) {
            let _ = stream.write_all(bytemuck::bytes_of(&msg));
        }
    }
}

impl eframe::App for SpectorApp {
    #[allow(deprecated)]
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut ui_needs_ipc_sync = false;

        ctx.input(|i| {
            if i.key_pressed(egui::Key::T) { self.show_menu = !self.show_menu; }
            if i.key_pressed(egui::Key::S) { self.scale_type = self.scale_type.cycle(); }
            if i.key_pressed(egui::Key::C) { self.colormap = self.colormap.cycle(); }
            if i.key_pressed(egui::Key::F) { self.dir = self.dir.cycle(); }
            if i.key_pressed(egui::Key::R) { self.fft_idx = (self.fft_idx + 1) % NUM_LAYERS; }
            if i.key_pressed(egui::Key::A) { self.audio_source = self.audio_source.toggle(); ui_needs_ipc_sync = true; }
            if i.key_pressed(egui::Key::W) { 
                self.view_len = if self.view_len == MAX_VIEW_LEN { MAX_VIEW_LEN / 2 } else { MAX_VIEW_LEN };
            }
        });

        // Ensure we don't index out of bounds
        let actual_fft_idx = self.fft_idx.min(NUM_LAYERS - 1);

        // --- 1. MEMORY DELTA BATCHING FROM SHM ---
        let shm_block = unsafe { &*self.shm_ptr.0 };
        let layer = &shm_block.layers[actual_fft_idx];
        
        // Read the atomics safely!
        let target_updates = layer.total_updates.load(Ordering::Acquire);
        let freq_bins = layer.active_freq_bins;

        if target_updates > self.last_uploaded_updates {
            let target_head = layer.head.load(Ordering::Acquire); 
            let wgpu_state = frame.wgpu_render_state().unwrap();
            let mut custom_state = wgpu_state.renderer.write();
            let state = custom_state.callback_resources.get_mut::<CustomWgpuState>().unwrap();

            if freq_bins != state.current_freq_bins {
                let ring_tex = wgpu_state.device.create_texture(&eframe::wgpu::TextureDescriptor {
                    label: Some("Ring Buffer"),
                    size: eframe::wgpu::Extent3d { width: MAX_HISTORY as u32, height: freq_bins as u32, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2,
                    format: eframe::wgpu::TextureFormat::R8Unorm,
                    usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[],
                });
                
                let bind_group_layout = state.pipeline.get_bind_group_layout(0);
                let ring_sampler = wgpu_state.device.create_sampler(&eframe::wgpu::SamplerDescriptor {
                    address_mode_u: eframe::wgpu::AddressMode::Repeat, address_mode_v: eframe::wgpu::AddressMode::ClampToEdge,
                    mag_filter: eframe::wgpu::FilterMode::Linear, min_filter: eframe::wgpu::FilterMode::Linear, ..Default::default()
                });
                let map_sampler = wgpu_state.device.create_sampler(&eframe::wgpu::SamplerDescriptor { mag_filter: eframe::wgpu::FilterMode::Linear, min_filter: eframe::wgpu::FilterMode::Linear, ..Default::default() });

                state.bind_group = wgpu_state.device.create_bind_group(&eframe::wgpu::BindGroupDescriptor {
                    label: Some("Bind Group"), layout: &bind_group_layout,
                    entries: &[
                        eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::TextureView(&ring_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                        eframe::wgpu::BindGroupEntry { binding: 1, resource: eframe::wgpu::BindingResource::Sampler(&ring_sampler) },
                        eframe::wgpu::BindGroupEntry { binding: 2, resource: eframe::wgpu::BindingResource::TextureView(&state.colormap_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                        eframe::wgpu::BindGroupEntry { binding: 3, resource: eframe::wgpu::BindingResource::Sampler(&map_sampler) },
                        eframe::wgpu::BindGroupEntry { binding: 4, resource: eframe::wgpu::BindingResource::TextureView(&state.ymap_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                        eframe::wgpu::BindGroupEntry { binding: 5, resource: eframe::wgpu::BindingResource::Sampler(&map_sampler) },
                        eframe::wgpu::BindGroupEntry { binding: 6, resource: eframe::wgpu::BindingResource::TextureView(&state.mask_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default())) },
                        eframe::wgpu::BindGroupEntry { binding: 7, resource: state.uniform_buf.as_entire_binding() },
                    ],
                });
                state.ring_tex = ring_tex;
                state.current_freq_bins = freq_bins;
                self.last_uploaded_head = 0;
            }

            if target_head > self.last_uploaded_head {
                let width = target_head - self.last_uploaded_head;
                
                wgpu_state.queue.write_texture(
                    eframe::wgpu::TexelCopyTextureInfo { texture: &state.mask_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: self.last_uploaded_head as u32, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                    &layer.mask[self.last_uploaded_head..target_head],
                    eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width as u32), rows_per_image: None },
                    eframe::wgpu::Extent3d { width: width as u32, height: 1, depth_or_array_layers: 1 }
                );

                if layer.mask[target_head.saturating_sub(1)] == 255 {
                    wgpu_state.queue.write_texture(
                        eframe::wgpu::TexelCopyTextureInfo { texture: &state.ring_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: self.last_uploaded_head as u32, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                        &layer.pixels[.. MAX_HISTORY * freq_bins],
                        eframe::wgpu::TexelCopyBufferLayout { offset: self.last_uploaded_head as u64, bytes_per_row: Some(MAX_HISTORY as u32), rows_per_image: None },
                        eframe::wgpu::Extent3d { width: width as u32, height: freq_bins as u32, depth_or_array_layers: 1 }
                    );
                }
            } else if target_head < self.last_uploaded_head {
                let width1 = MAX_HISTORY - self.last_uploaded_head;
                
                wgpu_state.queue.write_texture(
                    eframe::wgpu::TexelCopyTextureInfo { texture: &state.mask_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: self.last_uploaded_head as u32, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                    &layer.mask[self.last_uploaded_head..MAX_HISTORY],
                    eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width1 as u32), rows_per_image: None },
                    eframe::wgpu::Extent3d { width: width1 as u32, height: 1, depth_or_array_layers: 1 }
                );

                if layer.mask[MAX_HISTORY - 1] == 255 {
                    wgpu_state.queue.write_texture(
                        eframe::wgpu::TexelCopyTextureInfo { texture: &state.ring_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: self.last_uploaded_head as u32, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                        &layer.pixels[.. MAX_HISTORY * freq_bins],
                        eframe::wgpu::TexelCopyBufferLayout { offset: self.last_uploaded_head as u64, bytes_per_row: Some(MAX_HISTORY as u32), rows_per_image: None },
                        eframe::wgpu::Extent3d { width: width1 as u32, height: freq_bins as u32, depth_or_array_layers: 1 }
                    );
                }

                if target_head > 0 {
                    wgpu_state.queue.write_texture(
                        eframe::wgpu::TexelCopyTextureInfo { texture: &state.mask_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: 0, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                        &layer.mask[0..target_head],
                        eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(target_head as u32), rows_per_image: None },
                        eframe::wgpu::Extent3d { width: target_head as u32, height: 1, depth_or_array_layers: 1 }
                    );

                    if layer.mask[target_head - 1] == 255 {
                        wgpu_state.queue.write_texture(
                            eframe::wgpu::TexelCopyTextureInfo { texture: &state.ring_tex, mip_level: 0, origin: eframe::wgpu::Origin3d { x: 0, y: 0, z: 0 }, aspect: eframe::wgpu::TextureAspect::All },
                            &layer.pixels[.. MAX_HISTORY * freq_bins],
                            eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(MAX_HISTORY as u32), rows_per_image: None },
                            eframe::wgpu::Extent3d { width: target_head as u32, height: freq_bins as u32, depth_or_array_layers: 1 }
                        );
                    }
                }
            }

            self.last_uploaded_head = target_head;
            self.last_uploaded_updates = target_updates;
        }

        // --- 2. UPDATE STATIC LUTS IF CHANGED ---
        let is_cqt = actual_fft_idx >= RESOLUTIONS.len();
        if self.colormap != self.current_colormap || self.scale_type != self.current_scale || is_cqt != self.current_is_cqt {
            self.current_colormap = self.colormap;
            self.current_scale = self.scale_type;
            self.current_is_cqt = is_cqt;

            let wgpu_state = frame.wgpu_render_state().unwrap();
            let mut custom_state = wgpu_state.renderer.write();
            let state = custom_state.callback_resources.get_mut::<CustomWgpuState>().unwrap();

            wgpu_state.queue.write_texture(
                eframe::wgpu::TexelCopyTextureInfo { texture: &state.colormap_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                &generate_colormap_bytes(self.current_colormap),
                eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: None },
                eframe::wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 }
            );

            wgpu_state.queue.write_texture(
                eframe::wgpu::TexelCopyTextureInfo { texture: &state.ymap_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                &generate_y_mapping_bytes(self.current_scale, self.current_is_cqt, MAX_DISPLAY_FREQ, SAMPLE_RATE as f32 / 2.0),
                eframe::wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1024 * 4), rows_per_image: None },
                eframe::wgpu::Extent3d { width: 1024, height: 1, depth_or_array_layers: 1 }
            );
        }

        // --- 3. ANIMATE FRACTIONAL SCROLL ---
        let diff = (target_updates as f64) - self.smooth_head_pos;
        let dt = ctx.input(|i| i.unstable_dt) as f64;
        
        self.smooth_head_pos += diff * (15.0 * dt).min(1.0);
        if diff.abs() > 50.0 { self.smooth_head_pos = target_updates as f64; }
        
        // Save battery: only request repaint if we're visibly scrolling
        if diff.abs() > 0.05 {
            ctx.request_repaint();
        }

        // --- 4. BIND SHADER UNIFORMS ---
        let rect = ctx.available_rect();
        let ppp = ctx.pixels_per_point();
        
        let screen_time_dim = match self.dir {
            ScrollDirection::RTL | ScrollDirection::LTR => rect.width() * ppp,
            ScrollDirection::DTU | ScrollDirection::UTD => rect.height() * ppp,
        };
        
        let scale_factor = TARGET_DISPLAY_WIDTH / (self.view_len as f32);
        let actual_source_columns = (screen_time_dim / scale_factor).min(self.view_len as f32);
        let dynamic_view_width_norm = actual_source_columns / (MAX_HISTORY as f32);

        {
            let wgpu_state = frame.wgpu_render_state().unwrap();
            let mut custom_state = wgpu_state.renderer.write();
            let state = custom_state.callback_resources.get_mut::<CustomWgpuState>().unwrap();

            let uniforms = ShaderUniforms {
                scroll_offset: (self.smooth_head_pos.rem_euclid(MAX_HISTORY as f64) / MAX_HISTORY as f64) as f32,
                view_width_norm: dynamic_view_width_norm,
                direction: self.dir as u32,
                _padding: 0,
            };
            
            let uniforms_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&uniforms as *const ShaderUniforms) as *const u8, std::mem::size_of::<ShaderUniforms>()) };
            wgpu_state.queue.write_buffer(&state.uniform_buf, 0, uniforms_bytes);
        }

        // --- 5. EGUI DRAWING ---
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                
            let (rect, _response) = ui.allocate_exact_size(rect.size(), egui::Sense::hover());
            
            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(rect, SpectorRenderCallback));
            draw_note_ruler(ui, rect, self.scale_type, self.dir);

            if ctx.input(|i| i.pointer.primary_down()) == false { self.drag_start_freq = None; }

            if let Some(mouse_pos) = ctx.pointer_hover_pos() {
                if rect.contains(mouse_pos) {
                    let norm_x = (mouse_pos.x - rect.min.x) / rect.width();
                    let norm_y = (mouse_pos.y - rect.min.y) / rect.height();
                    
                    let norm_freq = match self.dir {
                        ScrollDirection::RTL | ScrollDirection::LTR => 1.0 - norm_y,
                        ScrollDirection::DTU | ScrollDirection::UTD => norm_x,
                    };
                    
                    let max_f = MAX_DISPLAY_FREQ;
                    let current_hz = match self.scale_type {
                        ScaleType::Linear => norm_freq * max_f,
                        ScaleType::Mel => {
                            let mel_max = 2595.0 * (1.0 + max_f / 700.0).log10();
                            700.0 * (10.0f32.powf((norm_freq * mel_max) / 2595.0) - 1.0)
                        },
                        ScaleType::Logarithmic => {
                            let min_log = 20.0f32.log2();
                            2.0f32.powf(min_log + norm_freq * (max_f.log2() - min_log))
                        },
                        ScaleType::Bark => {
                            let bark_max = 6.0 * (max_f / 600.0).asinh();
                            600.0 * ((norm_freq * bark_max) / 6.0).sinh()
                        }
                    };

                    if ctx.input(|i| i.pointer.primary_pressed()) { self.drag_start_freq = Some(current_hz); }

                    let painter = ui.painter();
                    
                    let crosshair_color = egui::Color32::from_white_alpha(100);
                    painter.line_segment([egui::pos2(mouse_pos.x, rect.min.y), egui::pos2(mouse_pos.x, rect.max.y)], egui::Stroke::new(1.0, crosshair_color));
                    painter.line_segment([egui::pos2(rect.min.x, mouse_pos.y), egui::pos2(rect.max.x, mouse_pos.y)], egui::Stroke::new(1.0, crosshair_color));

                    if let Some(f1) = self.drag_start_freq {
                        let norm_start = freq_to_norm_pos(f1, self.scale_type);
                        if norm_start >= 0.0 && norm_start <= 1.0 {
                            let drag_color = egui::Color32::from_rgba_unmultiplied(255, 204, 76, 200);
                            let fill_color = egui::Color32::from_rgba_unmultiplied(255, 204, 76, 50);

                            match self.dir {
                                ScrollDirection::RTL | ScrollDirection::LTR => {
                                    let start_y = rect.max.y - norm_start * rect.height();
                                    let min_y = start_y.min(mouse_pos.y);
                                    let max_y = start_y.max(mouse_pos.y);
                                    
                                    painter.rect_filled(egui::Rect::from_x_y_ranges(rect.min.x..=rect.max.x, min_y..=max_y), 0.0, fill_color);
                                    painter.line_segment([egui::pos2(rect.min.x, start_y), egui::pos2(rect.max.x, start_y)], egui::Stroke::new(1.0, drag_color));
                                    painter.line_segment([egui::pos2(rect.min.x, mouse_pos.y), egui::pos2(rect.max.x, mouse_pos.y)], egui::Stroke::new(1.0, drag_color));
                                },
                                ScrollDirection::DTU | ScrollDirection::UTD => {
                                    let start_x = rect.min.x + norm_start * rect.width();
                                    let min_x = start_x.min(mouse_pos.x);
                                    let max_x = start_x.max(mouse_pos.x);

                                    painter.rect_filled(egui::Rect::from_x_y_ranges(min_x..=max_x, rect.min.y..=rect.max.y), 0.0, fill_color);
                                    painter.line_segment([egui::pos2(start_x, rect.min.y), egui::pos2(start_x, rect.max.y)], egui::Stroke::new(1.0, drag_color));
                                    painter.line_segment([egui::pos2(mouse_pos.x, rect.min.y), egui::pos2(mouse_pos.x, rect.max.y)], egui::Stroke::new(1.0, drag_color));
                                }
                            }
                        }
                    }

                    let (note_name, _) = hz_to_pitch(current_hz);

                    let drawn_time_seconds = (actual_source_columns * CQT_HOP_SIZE as f32) / SAMPLE_RATE as f32;
                    let time_ago = match self.dir {
                        ScrollDirection::RTL => (1.0 - norm_x) * drawn_time_seconds,
                        ScrollDirection::LTR => norm_x * drawn_time_seconds,
                        ScrollDirection::DTU => (1.0 - norm_y) * drawn_time_seconds,
                        ScrollDirection::UTD => norm_y * drawn_time_seconds,
                    };

                    if let Some(f1) = self.drag_start_freq {
                        let f2 = current_hz;
                        let raw_semitones = 12.0 * (f2 / f1).log2();
                        let diff_hz = (f2 - f1).abs();
                        let sign_str = if raw_semitones >= 0.0 { "+" } else { "" };
                        
                        let tooltip = format!("-{:.2}s | {}{:.2} st ({}) | {}{:.1} Hz", time_ago, sign_str, raw_semitones, get_interval_name(raw_semitones), sign_str, diff_hz);
                        let layer_id = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("drag_tooltip_layer"));
                        egui::show_tooltip_at_pointer(ctx, layer_id, egui::Id::new("drag_tooltip"), |ui| { ui.label(tooltip); });
                    } else {
                        let tooltip = format!("-{:.2}s | {:.1} Hz | {}", time_ago, current_hz, note_name);
                        let layer_id = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("hover_tooltip_layer"));
                        egui::show_tooltip_at_pointer(ctx, layer_id, egui::Id::new("hover_tooltip"), |ui| { ui.label(tooltip); });
                    }
                }
            }

            // --- 6. MACROQUAD-STYLE HUD BAR ---
            let ui_painter = ui.painter();
            
            let (bg_x, bg_y, bg_w, bg_h, is_vertical) = match self.dir {
                ScrollDirection::RTL | ScrollDirection::LTR => (0.0, 0.0, rect.width(), 35.0, false),
                _ => (rect.width() - 220.0, 0.0, 220.0, 205.0, true),
            };

            ui_painter.rect_filled(
                egui::Rect::from_min_size(rect.min + egui::vec2(bg_x, bg_y), egui::vec2(bg_w, bg_h)),
                0.0,
                egui::Color32::from_black_alpha(153),
            );

            let mut cursor_x = rect.min.x + bg_x + 10.0;
            let mut cursor_y = rect.min.y + bg_y + 10.0;
            let font_id = egui::FontId::proportional(11.0);

            let res_str = if self.fft_idx as usize >= RESOLUTIONS.len() {
                "CQT (HD)".to_string()
            } else {
                format!("{} bins", freq_bins)
            };

            let stats = [
                ("Scale", format!("{:?}", self.scale_type), egui::Color32::from_rgb(255, 165, 0)),
                ("Colour", format!("{:?}", self.colormap), egui::Color32::YELLOW),
                ("Flow", format!("{:?}", self.dir), egui::Color32::LIGHT_BLUE),
                ("Resolution", res_str, egui::Color32::from_rgb(238, 130, 238)),
                ("Window", self.view_len.to_string(), egui::Color32::LIGHT_RED),
                ("Audio Src", format!("{:?}", self.audio_source), egui::Color32::GREEN),
                ("Tweaks", if self.show_menu { "Visible".to_string() } else { "Hidden".to_string() }, egui::Color32::WHITE),
            ];

            for (label, val, color) in &stats {
                let full_label = format!("{}:", label);
                let label_galley = ui_painter.layout_no_wrap(full_label.clone(), font_id.clone(), egui::Color32::WHITE);
                let first_char_galley = ui_painter.layout_no_wrap(label.chars().next().unwrap().to_string(), font_id.clone(), egui::Color32::WHITE);

                if is_vertical {
                    ui_painter.galley(egui::pos2(cursor_x, cursor_y), label_galley.clone(), egui::Color32::WHITE);
                    ui_painter.line_segment([
                        egui::pos2(cursor_x, cursor_y + first_char_galley.size().y + 1.0),
                        egui::pos2(cursor_x + first_char_galley.size().x, cursor_y + first_char_galley.size().y + 1.0)
                    ], egui::Stroke::new(1.0, egui::Color32::WHITE));

                    let val_galley = ui_painter.layout_no_wrap(val.clone(), font_id.clone(), *color);
                    ui_painter.galley(egui::pos2(cursor_x + label_galley.size().x + 10.0, cursor_y), val_galley, *color);
                    cursor_y += 18.0;
                } else {
                    ui_painter.galley(egui::pos2(cursor_x, cursor_y), label_galley.clone(), egui::Color32::WHITE);
                    ui_painter.line_segment([
                        egui::pos2(cursor_x, cursor_y + first_char_galley.size().y + 1.0),
                        egui::pos2(cursor_x + first_char_galley.size().x, cursor_y + first_char_galley.size().y + 1.0)
                    ], egui::Stroke::new(1.0, egui::Color32::WHITE));

                    let val_galley = ui_painter.layout_no_wrap(val.clone(), font_id.clone(), *color);
                    ui_painter.galley(egui::pos2(cursor_x + label_galley.size().x + 5.0, cursor_y), val_galley.clone(), *color);
                    cursor_x += label_galley.size().x + val_galley.size().x + 25.0;
                }
            }
        });

        // --- 7. DSP TWEAKS SETTINGS SYNC ---
        if self.show_menu {
            egui::Window::new("DSP Tweaks")
                .frame(egui::Frame::window(&ctx.global_style()).fill(egui::Color32::from_black_alpha(200)))
                .show(ctx, |ui| {
                    let mut s = self.dsp_config;
                    
                    ui.label(egui::RichText::new("Signal Pipeline").color(egui::Color32::from_rgb(255, 165, 0)));
                    if ui.add(egui::Slider::new(&mut s.pink_noise_tilt, -6.0..=6.0).text("Pink Noise Tilt (dB/Oct)")).changed() { ui_needs_ipc_sync = true; }
                    
                    let mut is_psd_norm = s.psd_normalization == 1;
                    if ui.checkbox(&mut is_psd_norm, "PSD Normalization").changed() { 
                        s.psd_normalization = if is_psd_norm { 1 } else { 0 }; 
                        ui_needs_ipc_sync = true; 
                    }
                    
                    if ui.add(egui::Slider::new(&mut s.peak_density_dampening, 0.0..=2.0).text("Density Dampening")).changed() { ui_needs_ipc_sync = true; }
                    
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Dynamics & Decay").color(egui::Color32::from_rgb(255, 165, 0)));
                    if ui.add(egui::Slider::new(&mut s.peak_weight, 0.0..=1.0).text("Peak Weight")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.rms_weight, 0.0..=1.0).text("RMS Weight")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.decay_low, 0.0..=0.1).text("Phosphor Decay (Bass)")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.decay_high, 0.0..=0.1).text("Phosphor Decay (Treble)")).changed() { ui_needs_ipc_sync = true; }

                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("CQT Kernel Splatting").color(egui::Color32::from_rgb(255, 165, 0)));
                    if ui.add(egui::Slider::new(&mut s.splat_low, 0.0..=10.0).text("Splat Spread (Bass)")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.splat_high, 0.0..=5.0).text("Splat Spread (Treble)")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.halo_raw, 0.0..=10.0).text("Halo Raw Blend")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.halo_sharp, 0.0..=10.0).text("Halo Sharp Blend")).changed() { ui_needs_ipc_sync = true; }

                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Makeup Gains").color(egui::Color32::from_rgb(255, 165, 0)));
                    if ui.add(egui::Slider::new(&mut s.stft_boost, 1.0..=20.0).text("STFT Boost Gain")).changed() { ui_needs_ipc_sync = true; }
                    if ui.add(egui::Slider::new(&mut s.iir_boost, 1.0..=20.0).text("IIR Boost Gain")).changed() { ui_needs_ipc_sync = true; }

                    self.dsp_config = s;
                });
        }

        if ui_needs_ipc_sync {
            self.send_ipc_update();
        }
    }

    fn ui(&mut self, _ui: &mut egui::Ui, _frame: &mut eframe::Frame) {}
}

fn setup_custom_fonts(ctx: &egui::Context) {
    let fonts = egui::FontDefinitions::default();
    ctx.set_fonts(fonts);
    
    let mut style = (*ctx.global_style()).clone();
    for (_text_style, font_id) in style.text_styles.iter_mut() {
        font_id.size *= 0.85; 
    }
    ctx.set_global_style(style);
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Spector (Native wgpu)",
        options,
        Box::new(|cc| {
            setup_custom_fonts(&cc.egui_ctx);
            
            let mut style = (*cc.egui_ctx.global_style()).clone();
            style.visuals.window_fill = egui::Color32::from_black_alpha(180);
            style.visuals.panel_fill = egui::Color32::TRANSPARENT;
            style.visuals.window_corner_radius = egui::CornerRadius::ZERO;
            cc.egui_ctx.set_global_style(style);

            // Mount Shared Memory
            let shm_path = format!("/dev/shm/{}", SHM_PATH);
            let file = OpenOptions::new().read(true).open(&shm_path)
                .expect("Failed to open shared memory file. Is the spector_daemon running?");
            let mmap = unsafe { Mmap::map(&file).expect("Failed to map shared memory") };
            let shm_ptr = ShmReader(mmap.as_ptr() as *const SharedMemoryBlock);

            let wgpu_state = CustomWgpuState::new(cc, 1200);
            cc.wgpu_render_state.as_ref().unwrap().renderer.write().callback_resources.insert(wgpu_state);

            Ok(Box::new(SpectorApp {
                _mmap: mmap,
                shm_ptr,
                scale_type: ScaleType::Logarithmic,
                colormap: ColorMapType::Magma,
                dir: ScrollDirection::RTL,
                fft_idx: NUM_LAYERS - 1,
                view_len: MAX_VIEW_LEN,
                dsp_config: DspConfig::default(),
                audio_source: AudioSource::SinkMonitor,
                smooth_head_pos: 0.0,
                last_uploaded_updates: 0,
                last_uploaded_head: 0,
                current_colormap: ColorMapType::Magma,
                current_scale: ScaleType::Logarithmic,
                current_is_cqt: true,
                show_menu: false,
                drag_start_freq: None,
            }))
        }),
    )
}