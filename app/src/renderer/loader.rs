use std::{
    io::{BufReader, Cursor},
    num::NonZeroU32,
    sync::Arc,
    time::Instant,
};

use anyhow::{Result, bail};
use models::{Gltf, Scene};
use vox::tree::VoxelTree;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::{RendererCtx, SizedWindow, engine::Engine, renderer::buffers::CameraDataBuffer};

// pub fn load_and_voxelize_model() -> Result<VoxelTree> {
//     const voxel_scale: f32 = 10.0;
//     let src = std::include_bytes!("../../assets/sponza.glb");
//     let mut src = bufreader::new(cursor::new(src));

//     let mut timer = Instant::now();

//     let gltf = Gltf::parse(&mut src)?;
//     println!("gltf parse: {:?}", timer.elapsed());
//     timer = Instant::now();

//     let scene = Scene::from_gltf(&gltf)?;
//     println!("scene parse: {:?}", timer.elapsed());
//     timer = Instant::now();

//     let mut tri_count = 0;
//     let scene_base: glam::Vec3A = (scene.min.floor() * VOXEL_SCALE).into();
//     let scene_size = ((scene.max - scene.min).ceil() * VOXEL_SCALE).as_uvec3();

//     enum Indices<'a> {
//         U16(&'a [[u16; 3]]),
//         U32(&'a [[u32; 3]]),
//     }

//     let mut tree = VoxelTree::new(scene_size);

//     for node in &scene.nodes {
//         let Some(mesh) = scene.meshes.get(node.mesh_id) else {
//             continue;
//         };
//         for primitive in &mesh.primitives {
//             tri_count += primitive.indices.count / 3;

//             let indices = match primitive.indices.component_type {
//                 models::schema::ComponentType::UnsignedShort => Indices::U16(
//                     bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end])
//                         .as_chunks()
//                         .0,
//                 ),
//                 models::schema::ComponentType::UnsignedInt => Indices::U32(
//                     bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end])
//                         .as_chunks()
//                         .0,
//                 ),
//                 _ => bail!("invalid index format"),
//             };
//             let positions: &[glam::Vec3] =
//                 bytemuck::cast_slice(&gltf.bin[primitive.positions.start..primitive.positions.end]);

//             const EXTENT: glam::Vec3A = glam::vec3a(0.5, 0.5, 0.5);

//             for tri in 0..(primitive.indices.count as usize / 3) {
//                 let i = match indices {
//                     Indices::U16(v) => v[tri].map(|v| v as usize),
//                     Indices::U32(v) => v[tri].map(|v| v as usize),
//                 };
//                 let [v0, v1, v2] =
//                     i.map(|j| node.transform.transform_point3a(positions[j].into()) * VOXEL_SCALE);

//                 let intersects = |center: glam::Vec3A| {
//                     let v0 = v0 - center;
//                     let v1 = v1 - center;
//                     let v2 = v2 - center;

//                     // Test AABB face normals (X, Y, Z axes)
//                     if v0.x.min(v1.x).min(v2.x) > EXTENT.x || v0.x.max(v1.x).max(v2.x) < -EXTENT.x {
//                         return false;
//                     }
//                     if v0.y.min(v1.y).min(v2.y) > EXTENT.y || v0.y.max(v1.y).max(v2.y) < -EXTENT.y {
//                         return false;
//                     }
//                     if v0.z.min(v1.z).min(v2.z) > EXTENT.z || v0.z.max(v1.z).max(v2.z) < -EXTENT.z {
//                         return false;
//                     }

//                     // Test triangle normal
//                     let e0 = v1 - v0;
//                     let e1 = v2 - v1;
//                     let e2 = v0 - v2;
//                     let normal = e0.cross(e1);
//                     let d = normal.dot(v0);
//                     let r = EXTENT.x * normal.x.abs()
//                         + EXTENT.y * normal.y.abs()
//                         + EXTENT.z * normal.z.abs();
//                     if d.abs() > r {
//                         return false;
//                     }

//                     // Test 9 edge cross-product axes (3 edges x 3 AABB axes)
//                     // Axis: X x e0
//                     let p0 = v0.z * v1.y - v0.y * v1.z;
//                     let p2 = v2.z * (v1.y - v0.y) - v2.y * (v1.z - v0.z);
//                     let rad = EXTENT.y * e0.z.abs() + EXTENT.z * e0.y.abs();
//                     if p0.min(p2) > rad || p0.max(p2) < -rad {
//                         return false;
//                     }

//                     // Axis: Y x e0
//                     let p0 = v0.x * v1.z - v0.z * v1.x;
//                     let p2 = v2.x * (v1.z - v0.z) - v2.z * (v1.x - v0.x);
//                     let rad = EXTENT.x * e0.z.abs() + EXTENT.z * e0.x.abs();
//                     if p0.min(p2) > rad || p0.max(p2) < -rad {
//                         return false;
//                     }

//                     // Axis: Z x e0
//                     let p0 = v0.y * v1.x - v0.x * v1.y;
//                     let p2 = v2.y * (v1.x - v0.x) - v2.x * (v1.y - v0.y);
//                     let rad = EXTENT.x * e0.y.abs() + EXTENT.y * e0.x.abs();
//                     if p0.min(p2) > rad || p0.max(p2) < -rad {
//                         return false;
//                     }

//                     // Axis: X x e1
//                     let p1 = v1.z * v2.y - v1.y * v2.z;
//                     let p0 = v0.z * (v2.y - v1.y) - v0.y * (v2.z - v1.z);
//                     let rad = EXTENT.y * e1.z.abs() + EXTENT.z * e1.y.abs();
//                     if p0.min(p1) > rad || p0.max(p1) < -rad {
//                         return false;
//                     }

//                     // Axis: Y x e1
//                     let p1 = v1.x * v2.z - v1.z * v2.x;
//                     let p0 = v0.x * (v2.z - v1.z) - v0.z * (v2.x - v1.x);
//                     let rad = EXTENT.x * e1.z.abs() + EXTENT.z * e1.x.abs();
//                     if p0.min(p1) > rad || p0.max(p1) < -rad {
//                         return false;
//                     }

//                     // Axis: Z x e1
//                     let p1 = v1.y * v2.x - v1.x * v2.y;
//                     let p0 = v0.y * (v2.x - v1.x) - v0.x * (v2.y - v1.y);
//                     let rad = EXTENT.x * e1.y.abs() + EXTENT.y * e1.x.abs();
//                     if p0.min(p1) > rad || p0.max(p1) < -rad {
//                         return false;
//                     }

//                     // Axis: X x e2
//                     let p2 = v2.z * v0.y - v2.y * v0.z;
//                     let p1 = v1.z * (v0.y - v2.y) - v1.y * (v0.z - v2.z);
//                     let rad = EXTENT.y * e2.z.abs() + EXTENT.z * e2.y.abs();
//                     if p1.min(p2) > rad || p1.max(p2) < -rad {
//                         return false;
//                     }

//                     // Axis: Y x e2
//                     let p2 = v2.x * v0.z - v2.z * v0.x;
//                     let p1 = v1.x * (v0.z - v2.z) - v1.z * (v0.x - v2.x);
//                     let rad = EXTENT.x * e2.z.abs() + EXTENT.z * e2.x.abs();
//                     if p1.min(p2) > rad || p1.max(p2) < -rad {
//                         return false;
//                     }

//                     // Axis: Z x e2
//                     let p2 = v2.y * v0.x - v2.x * v0.y;
//                     let p1 = v1.y * (v0.x - v2.x) - v1.x * (v0.y - v2.y);
//                     let rad = EXTENT.x * e2.y.abs() + EXTENT.y * e2.x.abs();
//                     if p1.min(p2) > rad || p1.max(p2) < -rad {
//                         return false;
//                     }

//                     true
//                 };

//                 let min = v0.min(v1).min(v2);
//                 let max = v0.max(v1).max(v2);

//                 let start = (min - scene_base).floor().as_uvec3().max(glam::UVec3::ZERO);
//                 let end = (max - scene_base).ceil().as_uvec3().min(scene_size);

//                 for x in start.x..end.x {
//                     for y in start.y..end.y {
//                         for z in start.z..end.z {
//                             if tree.get(glam::uvec3(x, y, z)) != 0 {
//                                 continue;
//                             }
//                             let center =
//                                 scene_base + glam::vec3a(x as f32, y as f32, z as f32) + EXTENT;
//                             if intersects(center) {
//                                 tree.insert(glam::uvec3(x, y, z), 1);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     println!("voxelize: {:?}", timer.elapsed());
//     println!(" -- triangles: {}", tri_count);

//     Ok(tree)
// }

const VOXEL_SCALE: f32 = 10.0;

pub fn voxelize(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<wgpu::Texture> {
    let src = std::include_bytes!("../../assets/sponza.glb");
    let mut src = BufReader::new(Cursor::new(src));

    let gltf = Gltf::parse(&mut src)?;
    let scene = Scene::from_gltf(&gltf)?;
    let base = scene.min.floor().as_uvec3();
    let size = ((scene.max.ceil() - scene.min.floor()) * VOXEL_SCALE).as_uvec3();

    let bg_layout_shared = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shared voxelizer bind group layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
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
                count: std::num::NonZeroU32::new(scene.textures.len() as u32),
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            },
        ],
    });
    let bg_layout_per_primitive =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("per-primitive voxelizer bind group layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxelizer pipeline layout"),
        bind_group_layouts: &[&bg_layout_shared, &bg_layout_per_primitive],
        push_constant_ranges: &[],
    });
    let pipeline = {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxelize"),
            source: wgpu::ShaderSource::Wgsl(std::include_str!("../shaders/voxelize.wgsl").into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("voxelize pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    // result 3d voxel texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel texture"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: size.z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // shared bind group
    let bg_shared = {
        #[repr(C)]
        #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SceneBufferEntry {
            base: glam::Vec4,
            size: glam::Vec3,
            scale: f32,
        }
        // binding 0
        let buffer_scene = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(&[SceneBufferEntry {
                base: base.as_vec3().extend(0.0),
                size: size.as_vec3(),
                scale: VOXEL_SCALE,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        #[repr(C)]
        #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MaterialBufferEntry {
            base_albedo: glam::Vec4,
            metallic: f32,
            roughness: f32,
            normal_scale: f32,
            albedo_index: i32,
            normal_index: i32,
            _pad: [f32; 3],
        }
        // binding 1
        let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(
                &scene
                    .materials
                    .iter()
                    .map(|mat| MaterialBufferEntry {
                        base_albedo: mat.base_albedo,
                        metallic: mat.metallic,
                        roughness: mat.roughness,
                        normal_scale: mat.normal_scale,
                        albedo_index: mat.albedo_index,
                        normal_index: mat.normal_index,
                        _pad: [0.0; 3],
                    })
                    .collect::<Vec<MaterialBufferEntry>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // binding 2
        let texture_views = scene
            .textures
            .iter()
            .map(|tex| {
                let (width, height) = tex.dimensions();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
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
                    &tex,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    ..Default::default()
                });

                view
            })
            .collect::<Vec<wgpu::TextureView>>();
        // binding 3
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // binding 4
        // out texture (already made)

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared voxelizer bind group"),
            layout: &bg_layout_shared,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_scene.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureViewArray(
                        &texture_views.iter().collect::<Vec<&wgpu::TextureView>>(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&texture.create_view(
                        &wgpu::TextureViewDescriptor {
                            label: Some("voxelized result texture output view"),
                            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                            ..Default::default()
                        },
                    )),
                },
            ],
        })
    };

    struct PrimitiveGroup {
        bg: wgpu::BindGroup,
        index_count: u32,
    }
    let mut bg_per_primitive = Vec::new();
    for node in &scene.nodes {
        let Some(mesh) = scene.meshes.get(node.mesh_id) else {
            continue;
        };
        for primitive in &mesh.primitives {
            // each (object, primitive) pair in the scene gets its own bind group
            // inefficient but idc its just a generation step for now

            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PrimitiveBufferEntry {
                matrix: glam::Mat4,
                normal_matrix: [[f32; 4]; 3],
                material_id: u32,
                index_count: u32,
                _pad: [f32; 2],
            }
            // binding 0
            let buffer_primitive = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("primitive data uniform buffer"),
                contents: bytemuck::cast_slice(&[PrimitiveBufferEntry {
                    matrix: node.transform,
                    normal_matrix: glam::Mat3::from_mat4(node.transform.inverse())
                        .transpose()
                        .to_cols_array_2d()
                        .map(|v| [v[0], v[1], v[2], 0.0]),
                    material_id: primitive.material_id,
                    index_count: primitive.indices.count,
                    _pad: [0.0; 2],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // binding 1
            let indices_u32 =
                bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end])
                    .iter()
                    .map(|idx: &u16| *idx as u32)
                    .collect::<Vec<u32>>();
            let buffer_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("primitive indices data"),
                contents: match primitive.indices.component_type {
                    models::schema::ComponentType::UnsignedShort => {
                        &bytemuck::cast_slice(&indices_u32)
                    }
                    _ => &gltf.bin[primitive.indices.start..primitive.indices.end],
                },
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            // binding 2
            let buffer_positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("primitive vertex position data"),
                contents: &gltf.bin[primitive.positions.start..primitive.positions.end],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            bg_per_primitive.push(PrimitiveGroup {
                bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("per primitive voxelize bind group"),
                    layout: &bg_layout_per_primitive,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer_primitive.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffer_indices.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer_positions.as_entire_binding(),
                        },
                    ],
                }),
                index_count: primitive.indices.count,
            });
        }
    }

    // now execute
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let descriptor = wgpu::ComputePassDescriptor {
            label: Some("voxelization pass"),
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_compute_pass(&descriptor);

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg_shared), &[]);

        for primitive in bg_per_primitive {
            pass.set_bind_group(1, Some(&primitive.bg), &[]);

            let tris = primitive.index_count / 3;
            pass.dispatch_workgroups(tris.div_ceil(64), 1, 1);
        }
    }
    queue.submit([encoder.finish()]);

    Ok(texture)
}

#[derive(Debug)]
pub struct ModelViewer {
    scene: Scene,
    pipeline: wgpu::RenderPipeline,
    bg_camera: wgpu::BindGroup,
    bg_model: wgpu::BindGroup,
    bg_scene_textures: wgpu::BindGroup,
    mesh_buffers: Vec<MeshBuffers>,
    buffer_camera: wgpu::Buffer,
    view_depth: wgpu::TextureView,
}

impl ModelViewer {
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
        let scene = Scene::from_gltf(&gltf)?;

        let views: Vec<wgpu::TextureView> = scene
            .textures
            .iter()
            .map(|tex| {
                let (width, height) = tex.dimensions();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
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
                    &tex,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    ..Default::default()
                });

                view
            })
            .collect();

        let mesh_buffers: Vec<MeshBuffers> = scene
            .meshes
            .iter()
            .map(|mesh| {
                let primitive_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive data"),
                        contents: bytemuck::cast_slice(
                            &mesh
                                .primitives
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
                let primitives = mesh
                    .primitives
                    .iter()
                    .map(|p| {
                        let index_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("index buffer"),
                                contents: &gltf.bin[p.indices.start..p.indices.end],
                                usage: wgpu::BufferUsages::INDEX,
                            });

                        let position_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex position buffer"),
                                contents: &gltf.bin[p.positions.start..p.positions.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let normal_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex normals buffer"),
                                contents: &gltf.bin[p.normals.start..p.normals.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let tangent_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("vertex tangent buffer"),
                                contents: &gltf.bin[p.tangents.start..p.tangents.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        let uv_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("tex coord buffer"),
                                contents: &gltf.bin[p.uv.start..p.uv.end],
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                        PrimitiveBuffers {
                            index_buffer,
                            position_buffer,
                            normal_buffer,
                            tangent_buffer,
                            uv_buffer,
                            index_format: match p.indices.component_type {
                                models::schema::ComponentType::UnsignedShort => {
                                    wgpu::IndexFormat::Uint16
                                }
                                _ => wgpu::IndexFormat::Uint32,
                            },
                            index_count: p.indices.count,
                        }
                    })
                    .collect();

                MeshBuffers {
                    primitive_buffer,
                    primitives,
                }
            })
            .collect();

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
                        count: std::num::NonZeroU32::new(views.len() as u32),
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
                        &views.iter().collect::<Vec<&wgpu::TextureView>>(),
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
            mesh_buffers,
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
                let Some(mesh_buffers) = self.mesh_buffers.get(node.mesh_id) else {
                    continue;
                };
                pass.set_bind_group(1, &self.bg_model, &[256 * i as u32]);
                pass.set_vertex_buffer(4, mesh_buffers.primitive_buffer.slice(..));
                pass.set_vertex_buffer(5, mesh_buffers.primitive_buffer.slice(..));
                pass.set_vertex_buffer(6, mesh_buffers.primitive_buffer.slice(..));
                for (j, primitive) in mesh_buffers.primitives.iter().enumerate() {
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

#[derive(Debug)]
struct MeshBuffers {
    primitive_buffer: wgpu::Buffer,
    primitives: Vec<PrimitiveBuffers>,
}
#[derive(Debug)]
struct PrimitiveBuffers {
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: u32,
    position_buffer: wgpu::Buffer,
    normal_buffer: wgpu::Buffer,
    tangent_buffer: wgpu::Buffer,
    uv_buffer: wgpu::Buffer,
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
