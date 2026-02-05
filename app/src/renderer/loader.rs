use std::{
    collections::{HashMap, VecDeque},
    io::{BufReader, Cursor},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use image::GenericImageView;
use models::Gltf;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{RendererCtx, SizedWindow, engine::Engine, renderer::buffers::CameraDataBuffer};

#[derive(Debug)]
pub struct ModelLoader {
    scene: Scene,
    pipeline: wgpu::RenderPipeline,
    bg_camera: wgpu::BindGroup,
    bg_model: wgpu::BindGroup,
    bg_scene_textures: wgpu::BindGroup,
    buffer_camera: wgpu::Buffer,
    view_depth: wgpu::TextureView,
}

impl ModelLoader {
    pub fn new(
        window: Arc<Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        engine: &Engine,
    ) -> anyhow::Result<Self> {
        let src = std::include_bytes!("../../assets/sponza.glb");
        let mut src = BufReader::new(Cursor::new(src));

        let gltf = Gltf::parse(&mut src)?;
        let scene = Scene::new(device, queue, &gltf)?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bg_layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model bg layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(256),
                },
                count: None,
            }],
        });
        let bg_layout_camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bg layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<CameraDataBuffer>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let bg_layout_scene_textures =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene textures bg layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: std::num::NonZeroU32::new(scene.views.len() as u32),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test pipeline layout"),
            bind_group_layouts: &[
                &bg_layout_camera,
                &bg_layout_model,
                &bg_layout_scene_textures,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("model test"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("../shaders/model_test.wgsl").into(),
                ),
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 1,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 2,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 16,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 3,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 28,
                                shader_location: 4,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 5,
                            }],
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 32,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 16,
                                shader_location: 6,
                            }],
                        },
                    ],
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
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };

        let buffer_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera data uniform buffer"),
            contents: bytemuck::cast_slice(&[CameraDataBuffer::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_model = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera data uniform buffer"),
            contents: bytemuck::cast_slice(
                &scene
                    .nodes
                    .iter()
                    .map(|n| ModelData {
                        matrix: n.transform.to_cols_array_2d(),
                        normal_matrix: glam::Mat3::from_mat4(n.transform.inverse())
                            .transpose()
                            .to_cols_array_2d()
                            .map(|v| [v[0], v[1], v[2], 0.0]),
                        _pad: [0.0; 36],
                    })
                    .collect::<Vec<ModelData>>(),
            ),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(
                &scene
                    .materials
                    .iter()
                    .map(|mat| MaterialData {
                        base_albedo: mat.base_albedo.to_array(),
                        metallic: mat.metallic,
                        roughness: mat.roughness,
                        normal_scale: mat.normal_scale,
                        albedo_index: mat.albedo_index,
                        normal_index: mat.normal_index,
                        _pad: [0.0, 0.0, 0.0],
                    })
                    .collect::<Vec<MaterialData>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bg_camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bg_layout_camera,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_camera.as_entire_binding(),
            }],
        });
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model bind group"),
            layout: &bg_layout_model,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_model.as_entire_binding(),
            }],
        });
        let bg_scene_textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene textures bind group"),
            layout: &bg_layout_scene_textures,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(
                        &scene.views.iter().collect::<Vec<&wgpu::TextureView>>(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_materials.as_entire_binding(),
                },
            ],
        });

        let view_depth = Self::create_screen_resource(&window, device);

        Ok(Self {
            scene,
            pipeline,
            bg_camera,
            bg_model,
            bg_scene_textures,
            buffer_camera,
            view_depth,
        })
    }

    pub fn frame<'a>(&mut self, ctx: &'a mut RendererCtx) {
        let mut camera_data = CameraDataBuffer::default();
        camera_data.update(&ctx.engine.camera);
        ctx.queue
            .write_buffer(&self.buffer_camera, 0, bytemuck::cast_slice(&[camera_data]));

        let surface_texture = ctx.surface.get_current_texture().unwrap();
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(ctx.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        {
            let descriptor = wgpu::RenderPassDescriptor {
                label: Some("post fx"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.view_depth,
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

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg_camera, &[]);
            pass.set_bind_group(2, &self.bg_scene_textures, &[]);

            for (i, node) in self.scene.nodes.iter().enumerate() {
                let Some(mesh) = self.scene.meshes.get(node.mesh_id) else {
                    continue;
                };
                pass.set_bind_group(1, &self.bg_model, &[256 * i as u32]);
                pass.set_vertex_buffer(4, mesh.primitive_buffer.slice(..));
                pass.set_vertex_buffer(5, mesh.primitive_buffer.slice(..));
                pass.set_vertex_buffer(6, mesh.primitive_buffer.slice(..));
                for (j, primitive) in mesh.primitives.iter().enumerate() {
                    pass.set_index_buffer(primitive.index_buffer.slice(..), primitive.index_format);
                    pass.set_vertex_buffer(0, primitive.position_buffer.slice(..));
                    pass.set_vertex_buffer(1, primitive.normal_buffer.slice(..));
                    pass.set_vertex_buffer(2, primitive.uv_buffer.slice(..));
                    pass.set_vertex_buffer(3, primitive.tangent_buffer.slice(..));
                    pass.draw_indexed(0..primitive.index_count, 0, (j as u32)..(j as u32 + 1));
                }
            }
        }
        ctx.queue.submit([encoder.finish()]);

        ctx.window.pre_present_notify();
        surface_texture.present();
    }

    pub fn handle_resize(&mut self, window: &Window, device: &wgpu::Device) {
        self.view_depth = Self::create_screen_resource(window, device);
    }
    fn create_screen_resource(window: &Window, device: &wgpu::Device) -> wgpu::TextureView {
        let size = window.size();

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
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelData {
    matrix: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 3],
    _pad: [f32; 36],
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrimitiveData {
    min: [f32; 3],
    _pad: f32,
    max: [f32; 3],
    material_id: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialData {
    base_albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
    _pad: [f32; 3],
}

/// Parsed scene data
#[derive(Debug)]
struct Scene {
    nodes: Vec<Node>,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    views: Vec<wgpu::TextureView>,
}

#[derive(Debug)]
struct Material {
    base_albedo: glam::Vec4,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
}

#[derive(Debug)]
struct Node {
    mesh_id: usize,
    transform: glam::Mat4,
}

#[derive(Debug)]
struct Mesh {
    primitive_buffer: wgpu::Buffer,
    primitives: Vec<Primitive>,
}

#[derive(Debug)]
struct Primitive {
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: u32,
    position_buffer: wgpu::Buffer,
    normal_buffer: wgpu::Buffer,
    tangent_buffer: wgpu::Buffer,
    uv_buffer: wgpu::Buffer,
    material_id: u32,
    min: glam::Vec3,
    max: glam::Vec3,
}

impl Scene {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, gltf: &Gltf) -> Result<Self> {
        let mut views = Vec::new();
        let mut img_index_map = HashMap::new();
        for (i, img) in gltf.meta.images.iter().enumerate() {
            let label = format!("img_{}_{}", i, img.name.as_deref().unwrap_or(""));

            match wgpu::Texture::from_gltf(device, queue, &gltf, img, &label) {
                Ok(texture) => {
                    img_index_map.insert(i as u32, views.len());
                    views.push(texture.create_view(&Default::default()));
                }
                Err(err) => {
                    eprintln!("error loading image {} - {}", &label, err);
                }
            }
        }

        let materials = (&gltf.meta.materials)
            .iter()
            .map(|m| Material::from_gltf(&gltf, &img_index_map, m))
            .collect::<Vec<Material>>();

        let mut meshes = Vec::new();
        let mut nodes = Vec::new();
        let scene = gltf
            .meta
            .scenes
            .get(gltf.meta.scene.context("no default scene")? as usize)
            .context("unable to find default scene")?;

        let mut visit_queue = VecDeque::new();
        for node in &scene.nodes {
            visit_queue.push_back((*node, models::GLTF_Y_UP_TO_Z_UP));
        }

        let mut mesh_id_map = HashMap::new();
        // visit breadth first over the scene, flattening the matrix transform
        while let Some((node_id, parent_matrix)) = visit_queue.pop_front() {
            let node = gltf
                .meta
                .nodes
                .get(node_id as usize)
                .context(format!("unable to find node with id {}", node_id))?;
            let transform = parent_matrix * node.transform.matrix;

            // add children to visit
            for child_id in &node.children {
                visit_queue.push_back((*child_id, transform));
            }

            let Some(gltf_mesh_id) = node.mesh else {
                continue;
            };
            // current node has a mesh attached
            // our final node list is flat and only has ones with meshes
            let mesh_id = match mesh_id_map.get(&gltf_mesh_id) {
                Some(id) => anyhow::Ok(*id),
                None => {
                    // now, create and push a new mesh
                    let gltf_mesh = gltf
                        .meta
                        .meshes
                        .get(gltf_mesh_id as usize)
                        .context("invalid mesh id")?;
                    let primitives = gltf_mesh
                        .primitives
                        .iter()
                        .filter_map(|p| {
                            Primitive::from_gltf(device, gltf, p)
                                .inspect_err(|err| eprintln!("error loading primitive: {:#?}", err))
                                .ok()
                        })
                        .collect::<Vec<Primitive>>();

                    let primitive_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("primitive data"),
                            contents: bytemuck::cast_slice(
                                &primitives
                                    .iter()
                                    .map(|primitive| PrimitiveData {
                                        min: primitive.min.to_array(),
                                        _pad: 0.0,
                                        max: primitive.max.to_array(),
                                        material_id: primitive.material_id,
                                    })
                                    .collect::<Vec<PrimitiveData>>(),
                            ),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    mesh_id_map.insert(gltf_mesh_id, meshes.len());
                    meshes.push(Mesh {
                        primitives,
                        primitive_buffer,
                    });
                    Ok(meshes.len() - 1)
                }
            }?;

            nodes.push(Node { mesh_id, transform });
        }

        Ok(Self {
            nodes,
            meshes,
            views,
            materials,
        })
    }
}

trait TextureExt {
    fn from_gltf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gltf: &Gltf,
        img: &models::schema::Image,
        label: &str,
    ) -> Result<wgpu::Texture>;
}
impl TextureExt for wgpu::Texture {
    fn from_gltf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gltf: &Gltf,
        img: &models::schema::Image,
        label: &str,
    ) -> Result<Self> {
        let buf_view_index = img
            .buffer_view
            .context("no buffer view for image. TODO: load image uri's")?;
        let buf_view = gltf
            .meta
            .buffer_views
            .get(buf_view_index as usize)
            .context("unable to find buffer view")?;

        let src = &gltf.bin[(buf_view.byte_offset as usize)
            ..(buf_view.byte_offset as usize + buf_view.byte_length as usize)];

        let loaded = match img.mime_type {
            Some(models::schema::MimeType::Jpeg) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Jpeg)
            }
            Some(models::schema::MimeType::Png) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Png)
            }
            _ => image::load_from_memory(src),
        }?;

        let dimensions = loaded.dimensions();
        let rgba = loaded.to_rgba8();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&label),
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            },
        );

        Ok(texture)
    }
}

impl Material {
    fn from_gltf(
        gltf: &Gltf,
        img_index_map: &HashMap<u32, usize>,
        m: &models::schema::Material,
    ) -> Self {
        let albedo_index = m
            .pbr_metallic_roughness
            .as_ref()
            .and_then(|pbr| (&pbr.base_color_texture).as_ref())
            .and_then(|info| gltf.meta.textures.get(info.index as usize))
            .and_then(|tex| tex.source)
            .and_then(|img_index| img_index_map.get(&img_index).map(|i| *i as i32))
            .unwrap_or(-1);
        let normal_index = (&m.normal_texture)
            .as_ref()
            .and_then(|info| gltf.meta.textures.get(info.index as usize))
            .and_then(|tex| tex.source)
            .and_then(|img_index| img_index_map.get(&img_index).map(|i| *i as i32))
            .unwrap_or(-1);

        let (base_albedo, roughness, metallic) = (&m.pbr_metallic_roughness)
            .as_ref()
            .map(|pbr| {
                (
                    pbr.base_color_factor,
                    pbr.roughness_factor,
                    pbr.metallic_factor,
                )
            })
            .unwrap_or((glam::Vec4::ZERO, 0.5, 0.0));

        let normal_scale = (&m.normal_texture).as_ref().map(|n| n.scale).unwrap_or(0.0);

        Self {
            base_albedo,
            roughness,
            metallic,
            normal_scale,
            albedo_index,
            normal_index,
        }
    }
}

impl Primitive {
    fn from_gltf(
        device: &wgpu::Device,
        gltf: &Gltf,
        p: &models::schema::Primitive,
    ) -> Result<Self> {
        ensure!(
            p.mode == models::schema::DrawMode::Triangles,
            "only DrawMode::Triangles is supported. TODO: add others"
        );
        let material_id = p.material.context("primitive has no material ID")?;

        struct PrimitiveBufferDescriptor<'a> {
            accessor: &'a models::schema::Accessor,
            data: &'a [u8],
        }
        // walks gltf and finds accessor, buffer view, buffer
        let get_buf_descriptor =
            |acc_index: u32,
             cmp_type: Option<models::schema::ComponentType>,
             acc_type: Option<models::schema::AccessorType>| {
                let accessor = gltf
                    .meta
                    .accessors
                    .get(acc_index as usize)
                    .context("accessor not found")?;
                let view = accessor
                    .buffer_view
                    .context("no buffer view specified")
                    .and_then(|index| {
                        gltf.meta
                            .buffer_views
                            .get(index as usize)
                            .context("no buffer view found")
                    })?;
                let buffer = gltf
                    .meta
                    .buffers
                    .get(view.buffer as usize)
                    .context("no buffer source found")?;
                ensure!(
                    buffer.uri.is_none(),
                    "buffer has external source. TODO: support this"
                );
                ensure!(
                    cmp_type
                        .as_ref()
                        .is_none_or(|ty| *ty == accessor.component_type),
                    format!(
                        "component type mismatch. expected {:?}, received {:?}",
                        cmp_type.unwrap(),
                        accessor.component_type
                    )
                );
                ensure!(
                    acc_type.as_ref().is_none_or(|ty| *ty == accessor.ty),
                    format!(
                        "accessor type mismatch. expected {:?}, received {:?}",
                        acc_type.unwrap(),
                        accessor.ty,
                    )
                );
                let mut component_length = match accessor.component_type {
                    models::schema::ComponentType::Byte
                    | models::schema::ComponentType::UnsignedByte => 1,
                    models::schema::ComponentType::Short
                    | models::schema::ComponentType::UnsignedShort => 2,
                    models::schema::ComponentType::UnsignedInt
                    | models::schema::ComponentType::Float => 4,
                    models::schema::ComponentType::Other(_) => bail!("invalid component type"),
                };
                component_length *= match accessor.ty {
                    models::schema::AccessorType::Scalar => 1,
                    models::schema::AccessorType::Vec2 => 2,
                    models::schema::AccessorType::Vec3 => 3,
                    models::schema::AccessorType::Vec4 => 4,
                    models::schema::AccessorType::Mat2 => 4,
                    models::schema::AccessorType::Mat3 => 9,
                    models::schema::AccessorType::Mat4 => 16,
                    models::schema::AccessorType::Other(_) => {
                        bail!("invalid accessor element type")
                    }
                };
                let start = (view.byte_offset + accessor.byte_offset.unwrap_or(0)) as usize;
                let end = start + (component_length * accessor.count) as usize;
                if start >= buffer.byte_length as usize || end >= buffer.byte_length as usize {
                    bail!("accessor view extends beyond buffer's bounds");
                }
                Ok(PrimitiveBufferDescriptor {
                    accessor,
                    data: &gltf.bin[start..end],
                })
            };

        let indices = p
            .indices
            .context("no index buffer. TODO: add support for direct vertex lists")
            .and_then(|i| {
                get_buf_descriptor(i, None, Some(models::schema::AccessorType::Scalar))
                    .context("index buffer")
            })?;
        let index_format = match indices.accessor.component_type {
            models::schema::ComponentType::UnsignedShort => wgpu::IndexFormat::Uint16,
            models::schema::ComponentType::UnsignedInt => wgpu::IndexFormat::Uint32,
            _ => {
                bail!("index format must be be either u16 or u32");
            }
        };

        let positions = p
            .attributes
            .get("POSITION")
            .context("no attribute")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(models::schema::ComponentType::Float),
                    Some(models::schema::AccessorType::Vec3),
                )
            })
            .context("vertex position buffer")?;

        let normals = p
            .attributes
            .get("NORMAL")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(models::schema::ComponentType::Float),
                    Some(models::schema::AccessorType::Vec3),
                )
            })
            .context("vertex normal buffer")?;

        let tangents = p
            .attributes
            .get("TANGENT")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(models::schema::ComponentType::Float),
                    Some(models::schema::AccessorType::Vec4),
                )
            })
            .context("vertex tangent buffer")?;

        let uv = p
            .attributes
            .get("TEXCOORD_0")
            .context("attribute not defined. TODO: add support for this")
            .and_then(|i| {
                get_buf_descriptor(
                    *i,
                    Some(models::schema::ComponentType::Float),
                    Some(models::schema::AccessorType::Vec2),
                )
            })
            .context("vertex coordinate buffer")?;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: indices.data,
            usage: wgpu::BufferUsages::INDEX,
        });

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex position buffer"),
            contents: positions.data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let normal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex normals buffer"),
            contents: normals.data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let tangent_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex tangent buffer"),
            contents: tangents.data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tex coord buffer"),
            contents: uv.data,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let min = &positions
            .accessor
            .min
            .as_ref()
            .context("position attribute is missing minimum bounds")?;
        let max = &positions
            .accessor
            .max
            .as_ref()
            .context("position attribute is missing maximum bounds")?;
        let [min, max] = [min, max].map(|bd| {
            glam::Vec3::from_slice(
                &bd[..3]
                    .iter()
                    .map(|x| match x {
                        models::schema::GltfNumber::Float(x) => *x as f32,
                        _ => 0.0,
                    })
                    .collect::<Vec<f32>>(),
            )
        });

        Ok(Primitive {
            index_buffer,
            index_format,
            index_count: indices.accessor.count,
            position_buffer,
            normal_buffer,
            tangent_buffer,
            uv_buffer,
            material_id,
            min,
            max,
        })
    }
}
