use std::{sync::Arc, time::Duration};

use glam::UVec2;
use wgpu::{
    BindGroup, Buffer, Device, RenderPipeline, Sampler, Texture, TextureView, util::DeviceExt,
};
use winit::window::Window;

use crate::{
    SizedWindow,
    engine::{CameraUniform, Engine, ModelUniform},
    renderer::mesh::{IntoMesh, Mesh, VoxelMesh},
    ui::{Ui, UiCtx},
};

pub struct RendererCtx<'a> {
    pub window: &'a winit::window::Window,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface: &'a wgpu::Surface<'static>,
    pub format: &'a wgpu::TextureFormat,
    pub engine: &'a Engine,
    pub ui: &'a mut Ui,
}

pub struct Renderer {
    depth: DepthTexture,
    pipeline: RenderPipeline,
    camera_bind_group: BindGroup,
    camera_uniform: Buffer,
    model_bind_group: BindGroup,
    model_uniform: Buffer,
    frame_avg: Duration,
    mesh: Mesh,
}

impl Renderer {
    pub fn new(
        window: Arc<Window>,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        engine: &Engine,
    ) -> Self {
        let depth = DepthTexture::new(&device, window.size());

        let (camera_uniform, camera_bind_group, camera_bind_group_layout) = {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });

            let uniform_buffer = {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("camera uniform buffer"),
                    contents: bytemuck::cast_slice(&[CameraUniform::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            (uniform_buffer, bind_group, layout)
        };

        let (model_uniform, model_bind_group, model_bind_group_layout) = {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("model bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });

            let uniform_buffer = {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("model uniform buffer"),
                    contents: bytemuck::cast_slice(&[ModelUniform::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("model bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            (uniform_buffer, bind_group, layout)
        };

        let pipeline = {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main pipeline layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &model_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("main shader"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/base.wgsl").into()),
            });

            let vertex_buffers = [wgpu::VertexBufferLayout {
                array_stride: crate::renderer::mesh::VERTEX_SIZE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 4 * 4,
                        shader_location: 1,
                    },
                ],
            }];

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DepthTexture::FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            render_pipeline
        };

        let mesh = VoxelMesh::new(&engine.scene).mesh(device);

        Self {
            depth,
            pipeline,
            camera_bind_group,
            camera_uniform,
            model_bind_group,
            model_uniform,
            mesh,
            frame_avg: Duration::ZERO,
        }
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window, device: &wgpu::Device) {
        let size = window.size();

        self.depth = DepthTexture::new(device, size);
    }

    pub fn frame<'a>(&mut self, delta_time: &Duration, ctx: &'a mut RendererCtx) {
        // update uniform buffers
        {
            ctx.queue.write_buffer(
                &self.camera_uniform,
                0,
                bytemuck::cast_slice(&[ctx.engine.camera.uniform]),
            );
            ctx.queue.write_buffer(
                &self.model_uniform,
                0,
                bytemuck::cast_slice(&[ctx.engine.model.uniform]),
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

        // main pass
        {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let mut pass = encoder.begin_render_pass(&descriptor);

            pass.push_debug_group("prepare data for draw");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.model_bind_group, &[]);
            pass.set_index_buffer(self.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(0, self.mesh.vertex_buffer.slice(..));
            pass.pop_debug_group();

            pass.insert_debug_marker("draw cube");
            pass.draw_indexed(0..self.mesh.index_count, 0, 0..1);
        }

        const FRAME_AVG_ALPHA: f64 = 0.02;
        self.frame_avg =
            self.frame_avg.mul_f64(1.0 - FRAME_AVG_ALPHA) + delta_time.mul_f64(FRAME_AVG_ALPHA);

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
    }
}

#[derive(Debug)]
struct DepthTexture {
    view: TextureView,
    _texture: Texture,
    _sampler: Sampler,
}
impl DepthTexture {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn new(device: &Device, size: UVec2) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DepthTexture::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            view,
            _texture: texture,
            _sampler: sampler,
        }
    }
}
