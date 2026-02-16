use std::{
    f32,
    io::{BufReader, Cursor},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use image::GenericImageView;
use wgpu::util::DeviceExt;

use crate::{
    SizedWindow,
    engine::Engine,
    renderer::{
        buffers::{self, CameraDataBuffer, EnvironmentDataBuffer, ModelDataBuffer},
        loader::Voxelizer,
        quad::Quad,
    },
    ui::{Ui, UiCtx},
};

pub struct RendererCtx<'a> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface: &'a wgpu::Surface<'static>,
    pub format: &'a wgpu::TextureFormat,
    pub engine: &'a mut Engine,
    pub ui: &'a mut Ui,
}

pub struct Renderer {
    pipeline_layouts: PipelineLayouts,
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
    bind_groups: BindGroups,
    textures: Textures,
    samplers: Samplers,
    buffers: Buffers,
    timing: Option<RenderTimer>,
    quad: Quad,
    frame_id: u32,
    size: glam::UVec2,
    render_scale: f32,
    sun_direction: glam::Vec3,
    prev_camera: Option<CameraDataBuffer>,
}

struct PipelineLayouts {
    raymarch: wgpu::PipelineLayout,
    lighting_resolve: wgpu::PipelineLayout,
    deferred: wgpu::PipelineLayout,
    taa: wgpu::PipelineLayout,
    fx: wgpu::PipelineLayout,
}

struct Pipelines {
    raymarch: wgpu::ComputePipeline,
    lighting_resolve: wgpu::ComputePipeline,
    deferred: wgpu::ComputePipeline,
    taa: wgpu::ComputePipeline,
    fx: wgpu::RenderPipeline,
}

struct BindGroupLayouts {
    raymarch_gbuffer: wgpu::BindGroupLayout,
    raymarch_per_frame: wgpu::BindGroupLayout,
    raymarch_voxels: wgpu::BindGroupLayout,
    lighting_resolve_gbuffer: wgpu::BindGroupLayout,
    lighting_resolve_swap: wgpu::BindGroupLayout,
    deferred_gbuffer: wgpu::BindGroupLayout,
    deferred_swap: wgpu::BindGroupLayout,
    taa_per_frame: wgpu::BindGroupLayout,
    taa_input: wgpu::BindGroupLayout,
    taa_output: wgpu::BindGroupLayout,
    fx_input: wgpu::BindGroupLayout,
    fx_per_frame: wgpu::BindGroupLayout,
}

struct BindGroups {
    raymarch_gbuffer: Option<wgpu::BindGroup>,
    raymarch_per_frame: wgpu::BindGroup,
    raymarch_voxels: wgpu::BindGroup,
    lighting_resolve_gbuffer: Option<wgpu::BindGroup>,
    lighting_resolve_swap_a: Option<wgpu::BindGroup>,
    lighting_resolve_swap_b: Option<wgpu::BindGroup>,
    deferred_gbuffer: Option<wgpu::BindGroup>,
    deferred_swap_a: Option<wgpu::BindGroup>,
    deferred_swap_b: Option<wgpu::BindGroup>,
    taa_per_frame: wgpu::BindGroup,
    taa_input: Option<wgpu::BindGroup>,
    taa_output_a: Option<wgpu::BindGroup>,
    taa_output_b: Option<wgpu::BindGroup>,
    fx_input_a: Option<wgpu::BindGroup>,
    fx_input_b: Option<wgpu::BindGroup>,
    fx_per_frame: wgpu::BindGroup,
}

struct Textures {
    gbuffer_albedo: Option<wgpu::Texture>,
    gbuffer_normal: Option<wgpu::Texture>,
    gbuffer_depth: Option<wgpu::Texture>,
    gbuffer_velocity: Option<wgpu::Texture>,
    gbuffer_illumination: Option<wgpu::Texture>,
    gbuffer_acc_illumination_a: Option<wgpu::Texture>,
    gbuffer_acc_illumination_b: Option<wgpu::Texture>,
    deferred_output: Option<wgpu::Texture>,
    out_color_a: Option<wgpu::Texture>,
    out_color_b: Option<wgpu::Texture>,
    voxel_brickmap: wgpu::Texture,
    noise: wgpu::Texture,
}

struct Samplers {
    linear: wgpu::Sampler,
    nearest_repeat: wgpu::Sampler,
}

struct Buffers {
    voxel_scene_metadata: wgpu::Buffer,
    voxel_palette: wgpu::Buffer,
    voxel_chunk_indices: wgpu::Buffer,
    voxel_chunks: wgpu::Buffer,
    frame_metadata: wgpu::Buffer,
    environment: wgpu::Buffer,
    model: wgpu::Buffer,
}

impl Renderer {
    pub fn new(
        window: Arc<winit::window::Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        engine: &Engine,
        ui: &mut Ui,
    ) -> Self {
        let bg_layouts = BindGroupLayouts {
            raymarch_gbuffer: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_gbuffer"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }),
            raymarch_voxels: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_voxels"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            }),
            raymarch_per_frame: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raymarch_per_frame"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
            lighting_resolve_gbuffer: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("lighting_resolve_gbuffer"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::ReadOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                },
            ),
            lighting_resolve_swap: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("lighting_resolve_swap"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                },
            ),
            deferred_gbuffer: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred_gbuffer"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }),
            deferred_swap: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred_swap"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            }),
            taa_per_frame: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("taa_per_frame"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
            taa_input: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("taa_input"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            }),
            taa_output: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("taa_output"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }),
            fx_input: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fx_input"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
            fx_per_frame: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fx_per_frame"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
        };

        let pipeline_layouts = PipelineLayouts {
            raymarch: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raymarch"),
                bind_group_layouts: &[
                    &bg_layouts.raymarch_gbuffer,
                    &bg_layouts.raymarch_voxels,
                    &bg_layouts.raymarch_per_frame,
                ],
                immediate_size: 0,
            }),
            lighting_resolve: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lighting_resolve"),
                bind_group_layouts: &[
                    &bg_layouts.lighting_resolve_gbuffer,
                    &bg_layouts.lighting_resolve_swap,
                ],
                immediate_size: 0,
            }),
            deferred: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("deferred"),
                bind_group_layouts: &[&bg_layouts.deferred_gbuffer, &bg_layouts.deferred_swap],
                immediate_size: 0,
            }),
            taa: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("taa"),
                bind_group_layouts: &[
                    &bg_layouts.taa_per_frame,
                    &bg_layouts.taa_input,
                    &bg_layouts.taa_output,
                ],
                immediate_size: 0,
            }),
            fx: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fx"),
                bind_group_layouts: &[&bg_layouts.fx_input, &bg_layouts.fx_per_frame],
                immediate_size: 0,
            }),
        };

        struct Shaders {
            raymarch: wgpu::ShaderModule,
            deferred: wgpu::ShaderModule,
            lighting_resolve: wgpu::ShaderModule,
            taa: wgpu::ShaderModule,
            fx: wgpu::ShaderModule,
        }
        let shaders = Shaders {
            raymarch: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raymarch"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/raymarch.wgsl").into(),
                ),
            }),
            lighting_resolve: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lighting_resolve"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/lighting_resolve.wgsl").into(),
                ),
            }),
            deferred: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("deferred"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/deferred.wgsl").into(),
                ),
            }),
            taa: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("taa"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/taa.wgsl").into()),
            }),
            fx: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post processing shader"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/fx.wgsl").into()),
            }),
        };

        let pipelines = Pipelines {
            raymarch: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("raymarch"),
                layout: Some(&pipeline_layouts.raymarch),
                module: &shaders.raymarch,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            lighting_resolve: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lighting_resolve"),
                layout: Some(&pipeline_layouts.lighting_resolve),
                module: &shaders.lighting_resolve,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            deferred: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("deferred"),
                layout: Some(&pipeline_layouts.deferred),
                module: &shaders.deferred,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            taa: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("taa"),
                layout: Some(&pipeline_layouts.taa),
                module: &shaders.taa,
                entry_point: Some("compute_main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            fx: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post fx pipeline"),
                layout: Some(&pipeline_layouts.fx),
                vertex: wgpu::VertexState {
                    module: &shaders.fx,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: crate::renderer::quad::VERTEX_SIZE,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.fx,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                cache: None,
                multiview_mask: None,
            }),
        };

        let scene = {
            // let src = std::include_bytes!("../../assets/san_miguel.glb");
            // let src = std::include_bytes!("../../assets/bistro.glb");
            let src = std::include_bytes!("../../assets/sponza.glb");
            let mut src = BufReader::new(Cursor::new(src));
            Voxelizer::load_gltf(&mut src, device, queue).unwrap()
        };

        let textures = Textures {
            gbuffer_albedo: None,
            gbuffer_normal: None,
            gbuffer_depth: None,
            gbuffer_velocity: None,
            gbuffer_illumination: None,
            gbuffer_acc_illumination_a: None,
            gbuffer_acc_illumination_b: None,
            deferred_output: None,
            out_color_a: None,
            out_color_b: None,
            voxel_brickmap: scene.tex_brickmap,
            noise: blue_noise(device, queue).unwrap(),
        };

        let samplers = Samplers {
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            nearest_repeat: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("nearest_repeat"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
        };

        let buffers = Buffers {
            voxel_scene_metadata: {
                #[repr(C)]
                #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct BufferVoxelSceneMetadata {
                    size_chunks: glam::UVec3,
                    _pad: u32,
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxel_scene_metadata"),
                    contents: bytemuck::cast_slice(&[BufferVoxelSceneMetadata {
                        size_chunks: scene.size_chunks,
                        _pad: 0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            },
            voxel_palette: scene.buffer_palette,
            voxel_chunk_indices: scene.buffer_chunk_indices,
            voxel_chunks: scene.buffer_chunks,

            environment: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("environment"),
                size: std::mem::size_of::<buffers::EnvironmentDataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            model: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("model"),
                size: std::mem::size_of::<buffers::ModelDataBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            frame_metadata: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame_metadata"),
                size: 12,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let bind_groups = BindGroups {
            raymarch_gbuffer: None,
            raymarch_voxels: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_voxels"),
                layout: &bg_layouts.raymarch_voxels,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.voxel_scene_metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.voxel_palette.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.voxel_chunk_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffers.voxel_chunks,
                            offset: 0,
                            size: std::num::NonZeroU64::new((scene.allocated_chunks * 64) as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &textures
                                .voxel_brickmap
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("voxel_brickmap"),
                                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&textures.noise.create_view(
                            &wgpu::TextureViewDescriptor {
                                label: Some("blue_noise"),
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&samplers.nearest_repeat),
                    },
                ],
            }),
            raymarch_per_frame: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_per_frame"),
                layout: &bg_layouts.raymarch_per_frame,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.environment.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.frame_metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.model.as_entire_binding(),
                    },
                ],
            }),
            lighting_resolve_gbuffer: None,
            lighting_resolve_swap_a: None,
            lighting_resolve_swap_b: None,
            deferred_gbuffer: None,
            deferred_swap_a: None,
            deferred_swap_b: None,
            taa_per_frame: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_per_frame"),
                layout: &bg_layouts.taa_per_frame,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.environment.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.frame_metadata.as_entire_binding(),
                    },
                ],
            }),
            taa_input: None,
            taa_output_a: None,
            taa_output_b: None,
            fx_input_a: None,
            fx_input_b: None,
            fx_per_frame: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx_per_frame"),
                layout: &bg_layouts.fx_per_frame,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.frame_metadata.as_entire_binding(),
                }],
            }),
        };

        let timing = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| RenderTimer::new(device));

        let quad = Quad::new(device);

        ui.state.sun_azimuth = -2.5;
        ui.state.sun_altitude = 1.3;
        ui.state.voxel_count = scene.voxel_count;
        ui.state.scene_size = scene.size;

        let mut _self = Self {
            pipeline_layouts,
            pipelines,
            bg_layouts,
            bind_groups,
            textures,
            samplers,
            buffers,
            timing,
            quad,
            frame_id: 0,
            render_scale: 1.0,
            size: Default::default(),
            sun_direction: Default::default(),
            prev_camera: None,
        };

        _self.update_screen_resources(&window, device);

        return _self;
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.update_screen_resources(window, device);
    }

    fn update_screen_resources(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        self.size = window
            .size()
            .map(|x| ((x as f32) * self.render_scale).ceil() as u32);
        let size = wgpu::Extent3d {
            width: self.size.x,
            height: self.size.y,
            depth_or_array_layers: 1,
        };

        self.textures.gbuffer_albedo = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_albedo"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_albedo = self.textures.gbuffer_albedo.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_albedo"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_normal = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_normal"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_normal = self.textures.gbuffer_normal.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_normal"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_depth = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::R32Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_depth = self.textures.gbuffer_depth.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("gbuffer_depth"),
                ..Default::default()
            },
        );

        self.textures.gbuffer_velocity = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_velocity"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_gbuffer_velocity = self
            .textures
            .gbuffer_velocity
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_velocity"),
                ..Default::default()
            });

        self.textures.gbuffer_illumination =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_illumination"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_illumination = self
            .textures
            .gbuffer_illumination
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_illumination"),
                ..Default::default()
            });

        self.textures.gbuffer_acc_illumination_a =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_illumination_a"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_acc_illumination_read_a = self
            .textures
            .gbuffer_acc_illumination_a
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_read_a"),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
        let view_gbuffer_acc_illumination_write_a = self
            .textures
            .gbuffer_acc_illumination_a
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_write_a"),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            });

        self.textures.gbuffer_acc_illumination_b =
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gbuffer_acc_illumination_b"),
                size,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                view_formats: &[],
            }));
        let view_gbuffer_acc_illumination_read_b = self
            .textures
            .gbuffer_acc_illumination_b
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_read_b"),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });
        let view_gbuffer_acc_illumination_write_b = self
            .textures
            .gbuffer_acc_illumination_b
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("gbuffer_acc_illumination_write_b"),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            });

        self.textures.deferred_output = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deferred_output"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_deferred_output = self.textures.deferred_output.as_ref().unwrap().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("deferred_output"),
                ..Default::default()
            },
        );

        self.textures.out_color_a = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("out_color_a"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_out_color_read_a =
            self.textures
                .out_color_a
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("out_color_read_a"),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
        let view_out_color_write_a =
            self.textures
                .out_color_a
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("out_color_write_a"),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                    ..Default::default()
                });

        self.textures.out_color_b = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("out_color_b"),
            size,
            sample_count: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            view_formats: &[],
        }));
        let view_out_color_read_b =
            self.textures
                .out_color_b
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("out_color_read_b"),
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                    ..Default::default()
                });
        let view_out_color_write_b =
            self.textures
                .out_color_b
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("out_color_write_b"),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                    ..Default::default()
                });

        self.bind_groups.raymarch_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raymarch_gbuffer"),
                layout: &self.bg_layouts.raymarch_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_illumination),
                    },
                ],
            }));

        self.bind_groups.lighting_resolve_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_resolve_gbuffer"),
                layout: &self.bg_layouts.lighting_resolve_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_illumination),
                    },
                ],
            }));

        self.bind_groups.lighting_resolve_swap_a =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_resolve_swap_a"),
                layout: &self.bg_layouts.lighting_resolve_swap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_acc_illumination_read_a,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_acc_illumination_write_b,
                        ),
                    },
                ],
            }));

        self.bind_groups.lighting_resolve_swap_b =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lighting_resolve_swap_b"),
                layout: &self.bg_layouts.lighting_resolve_swap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_acc_illumination_read_b,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &view_gbuffer_acc_illumination_write_a,
                        ),
                    },
                ],
            }));

        self.bind_groups.deferred_gbuffer =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_gbuffer"),
                layout: &self.bg_layouts.deferred_gbuffer,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_deferred_output),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                    },
                ],
            }));

        self.bind_groups.deferred_swap_a =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_input_a"),
                layout: &self.bg_layouts.deferred_swap,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &view_gbuffer_acc_illumination_read_b,
                    ),
                }],
            }));

        self.bind_groups.deferred_swap_b =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("deferred_swap_b"),
                layout: &self.bg_layouts.deferred_swap,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &view_gbuffer_acc_illumination_read_a,
                    ),
                }],
            }));

        self.bind_groups.taa_input = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa_input"),
            layout: &self.bg_layouts.taa_input,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_velocity),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_gbuffer_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view_deferred_output),
                },
            ],
        }));

        self.bind_groups.taa_output_a =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_output_a"),
                layout: &self.bg_layouts.taa_output,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_out_color_read_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_out_color_write_a),
                    },
                ],
            }));

        self.bind_groups.taa_output_b =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("taa_output_a"),
                layout: &self.bg_layouts.taa_output,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_out_color_read_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_out_color_write_b),
                    },
                ],
            }));

        self.bind_groups.fx_input_a = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx_input_a"),
            layout: &self.bg_layouts.fx_input,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_out_color_read_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                },
            ],
        }));

        self.bind_groups.fx_input_b = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx_input_b"),
            layout: &self.bg_layouts.fx_input,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_out_color_read_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.samplers.linear),
                },
            ],
        }));
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut RendererCtx) {
        if ctx.ui.state.render_scale != self.render_scale {
            self.render_scale = ctx.ui.state.render_scale;
            self.update_screen_resources(ctx.window, ctx.device);
        }
        ctx.ui.state.render_resolution = self.size;

        // update uniform buffers
        {
            ctx.engine.camera.size = self.size;
            let camera = CameraDataBuffer::from_camera(&ctx.engine.camera);
            let prev_camera = match self.prev_camera {
                Some(c) => c,
                None => camera,
            };
            self.prev_camera = Some(camera);
            self.sun_direction = glam::vec3(
                ctx.ui.state.sun_altitude.cos() * ctx.ui.state.sun_azimuth.cos(),
                ctx.ui.state.sun_altitude.cos() * ctx.ui.state.sun_azimuth.sin(),
                ctx.ui.state.sun_altitude.sin(),
            )
            .normalize();
            ctx.ui.state.sun_direction = self.sun_direction;
            ctx.queue.write_buffer(
                &self.buffers.environment,
                0,
                bytemuck::cast_slice(&[EnvironmentDataBuffer {
                    sun_direction: self.sun_direction,
                    shadow_bias: ctx.ui.state.shadow_bias,
                    camera,
                    prev_camera,
                    jitter: HALTON_16[(self.frame_id as usize + 1) & 0xf],
                    prev_jitter: HALTON_16[(self.frame_id as usize) & 0xf],
                }]),
            );

            let mut model_data = ModelDataBuffer::default();
            model_data.update(&ctx.engine.model);
            ctx.queue
                .write_buffer(&self.buffers.model, 0, bytemuck::cast_slice(&[model_data]));

            ctx.queue.write_buffer(
                &self.buffers.frame_metadata,
                0,
                bytemuck::cast_slice(&[
                    self.frame_id,
                    ctx.ui.state.taa as u32,
                    ctx.ui.state.fxaa as u32,
                ]),
            );
        }

        let surface_texture = ctx.surface.get_current_texture().unwrap();
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(ctx.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        // raymarch pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("raymarch"),
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.raymarch);
            pass.set_bind_group(0, &self.bind_groups.raymarch_gbuffer, &[]);
            pass.set_bind_group(1, &self.bind_groups.raymarch_voxels, &[]);
            pass.set_bind_group(2, &self.bind_groups.raymarch_per_frame, &[]);

            pass.insert_debug_marker("raymarch");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // lighting resolve pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("lighting_resolve"),
                timestamp_writes: None,
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.lighting_resolve);
            pass.set_bind_group(0, &self.bind_groups.lighting_resolve_gbuffer, &[]);
            pass.set_bind_group(
                1,
                match self.frame_id & 1 {
                    0 => &self.bind_groups.lighting_resolve_swap_a,
                    _ => &self.bind_groups.lighting_resolve_swap_b,
                },
                &[],
            );

            pass.insert_debug_marker("lighting_resolve");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // deferred pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("deferred"),
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.deferred);
            pass.set_bind_group(0, &self.bind_groups.deferred_gbuffer, &[]);
            pass.set_bind_group(
                1,
                match self.frame_id & 1 {
                    0 => &self.bind_groups.deferred_swap_a,
                    _ => &self.bind_groups.deferred_swap_b,
                },
                &[],
            );

            pass.insert_debug_marker("deferred");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // taa pass
        {
            let descriptor = wgpu::ComputePassDescriptor {
                label: Some("taa"),
                timestamp_writes: None,
            };
            let mut pass = encoder.begin_compute_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.taa);
            pass.set_bind_group(0, &self.bind_groups.taa_per_frame, &[]);
            pass.set_bind_group(1, &self.bind_groups.taa_input, &[]);
            pass.set_bind_group(
                2,
                match self.frame_id & 1 {
                    0 => &self.bind_groups.taa_output_a,
                    _ => &self.bind_groups.taa_output_b,
                },
                &[],
            );

            pass.insert_debug_marker("taa");
            pass.dispatch_workgroups(self.size.x.div_ceil(8), self.size.y.div_ceil(8), 1);
        }

        // post fx pass
        {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("post fx"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: self.timing.as_ref().map(|timing| {
                    wgpu::RenderPassTimestampWrites {
                        query_set: &timing.query_set,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }
                }),
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.set_pipeline(&self.pipelines.fx);
            pass.set_bind_group(
                0,
                match self.frame_id & 1 {
                    0 => &self.bind_groups.fx_input_a,
                    _ => &self.bind_groups.fx_input_b,
                },
                &[],
            );
            pass.set_bind_group(1, &self.bind_groups.fx_per_frame, &[]);
            pass.set_index_buffer(self.quad.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_vertex_buffer(0, self.quad.vertex_buffer.slice(..));

            pass.insert_debug_marker("fx");
            pass.draw_indexed(0..self.quad.index_count, 0, 0..1);
        }

        if let Some(timing) = &self.timing {
            encoder.resolve_query_set(
                &timing.query_set,
                0..(2 * RenderTimer::QUERY_PASS_COUNT),
                &timing.resolve_buffer,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &timing.resolve_buffer,
                0,
                &timing.result_buffer,
                0,
                timing.result_buffer.size(),
            );
        }

        ctx.ui.frame(&mut UiCtx {
            window: ctx.window,
            device: ctx.device,
            queue: ctx.queue,
            texture_view: &texture_view,
            encoder: &mut encoder,
        });

        ctx.queue.submit([encoder.finish()]);

        ctx.window.pre_present_notify();
        surface_texture.present();
        // ctx.device
        //     .poll(wgpu::PollType::Wait {
        //         submission_index: Some(id),
        //         timeout: None,
        //     })
        //     .unwrap();

        self.frame_id += 1;

        // if let Some(timing) = &self.timing {
        //     timing
        //         .result_buffer
        //         .slice(..)
        //         .map_async(wgpu::MapMode::Read, |_| ());

        //     ctx.device
        //         .poll(wgpu::PollType::Wait {
        //             submission_index: None,
        //             timeout: Some(Duration::from_secs(5)),
        //         })
        //         .unwrap();

        //     let view = timing.result_buffer.get_mapped_range(..);
        //     let timestamps: &[u64] = bytemuck::cast_slice(&*view);

        //     let time_raymarch = Duration::from_nanos(timestamps[1] - timestamps[0]);
        //     let time_deferred = Duration::from_nanos(timestamps[3] - timestamps[2]);
        //     let time_post_fx = Duration::from_nanos(timestamps[5] - timestamps[4]);

        //     ctx.ui.state.pass_avg = vec![
        //         ("Raymarch".into(), time_raymarch),
        //         ("Deferred".into(), time_deferred),
        //         ("Post FX".into(), time_post_fx),
        //     ];

        //     drop(view);
        //     timing.result_buffer.unmap();
        // }
    }
}

struct RenderTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    result_buffer: Arc<wgpu::Buffer>,
}
impl RenderTimer {
    const QUERY_PASS_COUNT: u32 = 3;

    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp query set"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::QUERY_PASS_COUNT * 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query resolve buffer"),
            size: Self::QUERY_PASS_COUNT as u64 * 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query result buffer"),
            size: Self::QUERY_PASS_COUNT as u64 * 2 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            result_buffer: Arc::new(result_buffer),
        }
    }
}

const HALTON_16: [glam::Vec2; 16] = [
    glam::vec2(0.500000, 0.333333),
    glam::vec2(0.250000, 0.666667),
    glam::vec2(0.750000, 0.111111),
    glam::vec2(0.125000, 0.444444),
    glam::vec2(0.625000, 0.777778),
    glam::vec2(0.375000, 0.222222),
    glam::vec2(0.875000, 0.555556),
    glam::vec2(0.062500, 0.888889),
    glam::vec2(0.562500, 0.037037),
    glam::vec2(0.312500, 0.370370),
    glam::vec2(0.812500, 0.703704),
    glam::vec2(0.187500, 0.148148),
    glam::vec2(0.687500, 0.481481),
    glam::vec2(0.437500, 0.814815),
    glam::vec2(0.937500, 0.259259),
    glam::vec2(0.031250, 0.592593),
];

fn blue_noise(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    let src = std::include_bytes!("../../assets/noise.png");
    let img = image::load_from_memory_with_format(src, image::ImageFormat::Png)?.to_rgba8();
    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blue_noise"),
        size: wgpu::Extent3d {
            width: img.width(),
            height: img.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &res,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.width()),
            rows_per_image: Some(img.height()),
        },
        wgpu::Extent3d {
            width: img.width(),
            height: img.height(),
            depth_or_array_layers: 1,
        },
    );
    Ok(res)
}
