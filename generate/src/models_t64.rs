use std::io::{BufRead, Read, Seek};

use crate::{
    MAX_STORAGE_BUFFER_BINDING_SIZE,
    gltf::{self, Gltf, Scene},
    utils::{
        layout::{DeviceUtils, sampler, storage_buffer, storage_texture, texture, uniform_buffer},
        pipeline::PipelineUtils,
    },
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

const VOXELS_PER_UNIT: u32 = 16;

pub fn voxelize<R: BufRead + Seek>(
    src: &mut R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: Option<String>,
) -> Result<VoxelModel> {
    let gltf = Gltf::parse(src)?;
    let scene = Scene::from_gltf(&gltf)?;
    return voxelize_gltf(device, queue, name, gltf, scene);
    // let mut voxelizer = Voxelizer::new(device, scene.textures.len() as u32);
    // voxelizer.load_gltf(device, queue, gltf, scene, name)
}

pub fn load_voxel_header<R: Read>(data: &mut R) -> Result<VoxelMetadata> {
    let mut buf = [0; 4];
    data.read_exact(&mut buf)?;
    let header_length = bytemuck::cast_slice::<u8, u32>(&buf)[0] as usize;
    let mut buf = vec![0; header_length as usize];
    data.read_exact(&mut buf)?;
    let header: VoxelFileHeader = serde_json::from_slice(&buf)?;
    Ok(header.meta)
}

pub struct VoxelModel {
    pub meta: VoxelMetadata,
    pub data: VoxelBufferData,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VoxelMetadata {
    pub name: String,
    pub bounding_size: u32,
    pub index_levels: u32,
    pub voxel_count: u32,
    pub allocated_index_chunks: u32,
}
#[derive(Debug)]
pub struct VoxelBufferData {
    pub buffer_palette: wgpu::Buffer,
    pub buffer_index_chunks: wgpu::Buffer,
    pub buffer_leaf_chunks: wgpu::Buffer,
}

impl VoxelModel {
    pub fn serialize(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>> {
        let buffer_palette = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_palette"),
            size: self.data.buffer_palette.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_index_chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_index_chunks"),
            size: self.data.buffer_index_chunks.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_leaf_chunks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_leaf_chunks"),
            size: self.data.buffer_leaf_chunks.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_palette,
            0,
            &buffer_palette,
            0,
            buffer_palette.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_index_chunks,
            0,
            &buffer_index_chunks,
            0,
            buffer_index_chunks.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.data.buffer_leaf_chunks,
            0,
            &buffer_leaf_chunks,
            0,
            buffer_leaf_chunks.size(),
        );
        queue.submit([encoder.finish()]);

        let buffer_palette = buffer_palette.slice(..);
        buffer_palette.map_async(wgpu::MapMode::Read, |_| {});

        let buffer_index_chunks = buffer_index_chunks.slice(..);
        buffer_index_chunks.map_async(wgpu::MapMode::Read, |_| {});

        let buffer_leaf_chunks = buffer_leaf_chunks.slice(..);
        buffer_leaf_chunks.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let mut body = Vec::new();

        let data_palette = buffer_palette.get_mapped_range();
        let buffer_palette = BufferView {
            start: body.len(),
            end: body.len() + data_palette.len(),
        };
        body.extend_from_slice(&data_palette);

        let data_index_chunks = buffer_index_chunks.get_mapped_range();
        let buffer_index_chunks = BufferView {
            start: body.len(),
            end: body.len() + data_index_chunks.len(),
        };
        body.extend_from_slice(&data_index_chunks);

        let data_leaf_chunks = buffer_leaf_chunks.get_mapped_range();
        let buffer_leaf_chunks = BufferView {
            start: body.len(),
            end: body.len() + data_leaf_chunks.len(),
        };
        body.extend_from_slice(&data_leaf_chunks);

        let header = VoxelFileHeader {
            meta: self.meta.clone(),
            file: VoxelFileInfo {
                buffer_palette,
                buffer_index_chunks,
                buffer_leaf_chunks,
            },
        };
        let header = serde_json::to_vec(&header)?;
        let mut data: Vec<u8> = bytemuck::cast_slice(&[header.len() as u32]).to_vec();
        data.extend_from_slice(&header);
        data.extend_from_slice(&body);

        println!(
            "serialized data size (pre-compression): {:.2} MB",
            data.len() as f64 / (1024.0 * 1024.0)
        );
        println!(
            "   -palette size: {:.2} MB",
            (data_palette.len() as f64) / (1024.0 * 1024.0)
        );
        println!(
            "   -index chunks size: {:.2} MB",
            (data_index_chunks.len() as f64) / (1024.0 * 1024.0)
        );
        println!(
            "   -leaf chunks size: {:.2} MB",
            (data_leaf_chunks.len() as f64) / (1024.0 * 1024.0)
        );
        println!("  -bounding size: {}", self.meta.bounding_size);
        println!("   -voxel count: {}", self.meta.voxel_count);
        println!(
            "   -index chunk count: {}",
            self.meta.allocated_index_chunks
        );
        println!(
            "   -bytes per voxel: {:.2}",
            (data.len() as f64) / (self.meta.voxel_count as f64)
        );

        Ok(data)
    }

    pub fn deserialize(device: &wgpu::Device, _queue: &wgpu::Queue, data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            bail!("Invalid model")
        }
        let header_length = bytemuck::cast_slice::<u8, u32>(&data[0..4])[0] as usize;

        let header: VoxelFileHeader = serde_json::from_slice(&data[4..(4 + header_length)])?;

        let buf = &data[4 + header_length..];

        let buffer_palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_palette", header.meta.name)),
            contents: &buf[header.file.buffer_palette.start..header.file.buffer_palette.end],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_index_chunks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_index_chunks", header.meta.name)),
            contents: &buf
                [header.file.buffer_index_chunks.start..header.file.buffer_index_chunks.end],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_leaf_chunks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_buffer_leaf_chunks", header.meta.name)),
            contents: &buf
                [header.file.buffer_leaf_chunks.start..header.file.buffer_leaf_chunks.end],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            meta: header.meta,
            data: VoxelBufferData {
                buffer_palette,
                buffer_index_chunks,
                buffer_leaf_chunks,
            },
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoxelFileHeader {
    meta: VoxelMetadata,
    file: VoxelFileInfo,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct VoxelFileInfo {
    buffer_palette: BufferView,
    buffer_index_chunks: BufferView,
    buffer_leaf_chunks: BufferView,
}
#[derive(Serialize, Deserialize, Debug)]
struct BufferView {
    start: usize,
    end: usize,
}

struct Voxelizer {
    pipelines: Pipelines,
    bg_layouts: BindGroupLayouts,
}

struct VoxelizeCtx<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    bg_layouts: &'a BindGroupLayouts,
    pipelines: &'a Pipelines,
    scene: &'a Scene,
    gltf: &'a Gltf,
}
struct Pipelines {
    voxelize: wgpu::ComputePipeline,
    tree_leaf: wgpu::ComputePipeline,
    tree_index: wgpu::ComputePipeline,
}
struct BindGroupLayouts {
    voxelize_shared: wgpu::BindGroupLayout,
    voxelize_scene_textures: wgpu::BindGroupLayout,
    voxelize_per_primitive: wgpu::BindGroupLayout,
    tree_data: wgpu::BindGroupLayout,
    tree_alloc_texture: wgpu::BindGroupLayout,
}

fn layouts(device: &wgpu::Device, scene_texture_count: u32) -> (Pipelines, BindGroupLayouts) {
    let bg_layouts = BindGroupLayouts {
        voxelize_shared: device.layout(
            "voxelize_shared",
            wgpu::ShaderStages::COMPUTE,
            (
                uniform_buffer(),
                storage_buffer().read_only(),
                sampler().filtering(),
                storage_texture().r32uint().dimension_3d().read_only(),
                storage_texture().r32uint().dimension_3d().write_only(),
            ),
        ),
        voxelize_scene_textures: device.layout(
            "voxelize_scene_textures",
            wgpu::ShaderStages::COMPUTE,
            texture()
                .float()
                .dimension_2d()
                .count(std::num::NonZeroU32::new(scene_texture_count).unwrap()),
        ),
        voxelize_per_primitive: device.layout(
            "voxelize_per_primitive",
            wgpu::ShaderStages::COMPUTE,
            (
                uniform_buffer(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
                storage_buffer().read_only(),
            ),
        ),
        tree_data: device.layout(
            "tree_data",
            wgpu::ShaderStages::COMPUTE,
            (
                storage_buffer().read_write(),
                storage_buffer().read_write(),
                storage_texture().rgba32uint().dimension_3d().read_only(),
                storage_texture().rgba32uint().dimension_3d().write_only(),
            ),
        ),
        tree_alloc_texture: device.layout(
            "tree_alloc_texture",
            wgpu::ShaderStages::COMPUTE,
            (
                storage_buffer().read_write(),
                storage_texture().r32uint().dimension_3d().read_only(),
                storage_texture().r32uint().dimension_3d().read_only(),
                uniform_buffer(),
            ),
        ),
    };
    let pipelines = Pipelines {
        voxelize: device
            .compute_pipeline(
                "voxelize",
                &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxelize"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("shaders/voxelize.wgsl").into(),
                    ),
                }),
            )
            .layout(&[
                &bg_layouts.voxelize_shared,
                &bg_layouts.voxelize_scene_textures,
                &bg_layouts.voxelize_per_primitive,
            ]),
        tree_leaf: device
            .compute_pipeline(
                "tree_leaf",
                &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxel brickmap packing"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("shaders/tree_64.wgsl").into(),
                    ),
                }),
            )
            .entry_point("compute_leaf")
            .layout(&[&bg_layouts.tree_data, &bg_layouts.tree_alloc_texture]),
        tree_index: device
            .compute_pipeline(
                "tree_index",
                &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("voxel brickmap packing"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::include_str!("shaders/tree_64.wgsl").into(),
                    ),
                }),
            )
            .entry_point("compute_index")
            .layout(&[&bg_layouts.tree_data, &bg_layouts.tree_alloc_texture]),
    };
    (pipelines, bg_layouts)
}

fn voxelize_gltf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: Option<String>,
    gltf: Gltf,
    scene: Scene,
) -> Result<VoxelModel> {
    let name = name.unwrap_or(String::from("model"));

    let (pipelines, bg_layouts) = layouts(device, scene.textures.len() as u32);

    let ctx = VoxelizeCtx {
        device,
        queue,
        bg_layouts: &bg_layouts,
        pipelines: &pipelines,
        scene: &scene,
        gltf: &gltf,
    };

    let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_uvec3();
    let base_unscaled = ctx.scene.min.floor();
    let size = size_unscaled * VOXELS_PER_UNIT;

    let raw_chunk_size = size.map(|x| x.div_ceil(64));
    let mut raw_chunk_indices = vec![0u32; raw_chunk_size.element_product() as usize];
    let mut raw_chunk_count = 0;

    for node in &ctx.scene.nodes {
        let Some(mesh) = ctx.scene.meshes.get(node.mesh_id) else {
            continue;
        };
        for primitive in &mesh.primitives {
            let indices: &[[u16; 3]] =
                bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end]);
            let positions: &[[f32; 3]] =
                bytemuck::cast_slice(&gltf.bin[primitive.positions.start..primitive.positions.end]);

            for tri in indices {
                let pos = tri
                    .map(|i| glam::Vec3::from_array(positions[i as usize]))
                    .map(|p| {
                        (node.transform.transform_point3(p) - base_unscaled)
                            * (VOXELS_PER_UNIT as f32)
                    });
                let min = pos[0].min(pos[1]).min(pos[2]);
                let max = pos[0].max(pos[1]).max(pos[2]);

                let min = min.floor().max(glam::Vec3::ZERO).as_uvec3();
                let max = max.ceil().as_uvec3().min(size);

                let min_chunk = min / 64;
                let max_chunk = (max / 64 + 1).min(raw_chunk_size);

                for z in min_chunk.z..max_chunk.z {
                    for y in min_chunk.y..max_chunk.y {
                        for x in min_chunk.x..max_chunk.x {
                            let i =
                                x + y * raw_chunk_size.x + z * raw_chunk_size.x * raw_chunk_size.y;
                            if raw_chunk_indices[i as usize] == 0 {
                                raw_chunk_count += 1;
                                raw_chunk_indices[i as usize] = raw_chunk_count;
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "raw size: {:.2} MiB",
        (size.as_u64vec3().element_product() as f64 * 8.0) / (1024.0 * 1024.0)
    );
    println!(
        "chunked size: {:.2} MiB",
        ((raw_chunk_count as u64 * 64 * 64 * 64) as f64 * 8.0) / (1024.0 * 1024.0)
    );

    let tex_raw_chunk_indices = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raw_chunk_indices"),
        size: wgpu::Extent3d {
            width: raw_chunk_size.x,
            height: raw_chunk_size.y,
            depth_or_array_layers: raw_chunk_size.z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_raw_chunk_indices,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&raw_chunk_indices),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * raw_chunk_size.x),
            rows_per_image: Some(raw_chunk_size.y),
        },
        wgpu::Extent3d {
            width: raw_chunk_size.x,
            height: raw_chunk_size.y,
            depth_or_array_layers: raw_chunk_size.z,
        },
    );

    let raw_voxel_count = raw_chunk_count * 64 * 64 * 64;
    let raw_minor_size = ((raw_voxel_count as f64).powf(1.0 / 3.0).ceil() as u32)
        .max(1)
        .next_multiple_of(64);
    let raw_major_size = raw_voxel_count
        .div_ceil(raw_minor_size * raw_minor_size)
        .next_multiple_of(64);

    dbg!(&raw_chunk_size);
    dbg!(size);
    dbg!(glam::uvec3(raw_minor_size, raw_minor_size, raw_major_size));

    let tex_raw_voxels = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raw_voxels"),
        size: wgpu::Extent3d {
            width: raw_minor_size,
            height: raw_minor_size,
            depth_or_array_layers: raw_major_size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let bg_voxelize_shared =
        create_bg_voxelize_shared(&ctx, &tex_raw_voxels, &tex_raw_chunk_indices);
    let bg_voxelize_scene_textures = create_bg_voxelize_scene_textures(&ctx);
    let bg_voxelize_per_primitive = create_bg_voxelize_per_primitive(&ctx);

    let palette = build_palette(&ctx);

    let bounding_size = next_pow_4(size.max_element()).max(16);
    let index_levels = bounding_size.ilog(4);

    let max_voxels = (bounding_size as u64).pow(3);
    let max_leaf_chunks_bytes = u64::min(MAX_STORAGE_BUFFER_BINDING_SIZE as u64, max_voxels * 4);

    let mut max_index_chunks = 0;
    for k in 1..=index_levels {
        max_index_chunks += ((bounding_size >> (2 * k)) as u64).pow(3);
    }
    let max_index_chunks_bytes = u64::min(
        MAX_STORAGE_BUFFER_BINDING_SIZE as u64,
        max_index_chunks * 12,
    );

    // println!(
    //     "index map size: {:.2} MiB",
    //     (((bounding_size >> 4) as u64).pow(3) as f64 * 32.0) / (1024.0 * 1024.0)
    // );

    let buffer_index_chunks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("index_chunks"),
        size: max_index_chunks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_leaf_chunks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("leaf_chunks"),
        size: max_leaf_chunks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tex_index_map_a = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("index_map_a"),
        size: wgpu::Extent3d {
            width: bounding_size >> 6,
            height: bounding_size >> 6,
            depth_or_array_layers: bounding_size >> 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let tex_index_map_b = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("index_map_b"),
        size: wgpu::Extent3d {
            width: bounding_size >> 4,
            height: bounding_size >> 4,
            depth_or_array_layers: bounding_size >> 4,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let bg_tree_data_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tree_data_a"),
        layout: &bg_layouts.tree_data,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_index_chunks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_leaf_chunks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &tex_index_map_a.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_index_map_b.create_view(&Default::default()),
                ),
            },
        ],
    });
    let bg_tree_data_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tree_data_b"),
        layout: &bg_layouts.tree_data,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_index_chunks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_leaf_chunks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &tex_index_map_b.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_index_map_a.create_view(&Default::default()),
                ),
            },
        ],
    });

    // atomic counters for building the tree
    let buffer_alloc_data = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree_alloc_data"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // i read from the alloc metadata buffer by copying it here
    let buffer_alloc_results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree_alloc_results"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bg_tree_alloc_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tree_alloc_texture"),
        layout: &bg_layouts.tree_alloc_texture,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_alloc_data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_chunk_indices.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_voxels.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: palette.buffer_palette.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    // voxelize
    {
        let descriptor = wgpu::ComputePassDescriptor {
            label: Some("voxelize"),
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_compute_pass(&descriptor);

        pass.set_pipeline(&pipelines.voxelize);
        pass.set_bind_group(0, Some(&bg_voxelize_shared), &[]);
        pass.set_bind_group(1, Some(&bg_voxelize_scene_textures), &[]);

        for primitive in &bg_voxelize_per_primitive {
            pass.set_bind_group(2, Some(&primitive.bg), &[]);

            let tris = primitive.index_count / 3;
            pass.dispatch_workgroups(tris.div_ceil(64), 1, 1);
        }
    }

    // leaf chunks
    {
        let descriptor = wgpu::ComputePassDescriptor {
            label: Some("tree_leaf"),
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_compute_pass(&descriptor);

        pass.set_pipeline(&pipelines.tree_leaf);
        pass.set_bind_group(0, Some(&bg_tree_data_a), &[]);
        pass.set_bind_group(1, Some(&bg_tree_alloc_texture), &[]);

        let size = bounding_size >> 4;
        pass.dispatch_workgroups(size, size, size);
    }

    // index passes
    for k in 2..index_levels {
        let descriptor = wgpu::ComputePassDescriptor {
            label: Some("tree_index"),
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_compute_pass(&descriptor);

        pass.set_pipeline(&pipelines.tree_index);
        pass.set_bind_group(
            0,
            Some(match k & 1 {
                0 => &bg_tree_data_b,
                _ => &bg_tree_data_a,
            }),
            &[],
        );
        pass.set_bind_group(1, Some(&bg_tree_alloc_texture), &[]);

        let size = bounding_size >> (2 + 2 * k);
        pass.dispatch_workgroups(size, size, size);
    }

    // we read from the results buffer
    encoder.copy_buffer_to_buffer(
        &buffer_alloc_data,
        0,
        &buffer_alloc_results,
        0,
        buffer_alloc_data.size(),
    );

    queue.submit([encoder.finish()]);

    let mut tree = TreeData {
        bounding_size,
        index_levels,
        max_index_chunks: max_index_chunks_bytes.div_ceil(12),
        max_voxels: max_leaf_chunks_bytes.div_ceil(4),
        buffer_index_chunks,
        buffer_leaf_chunks,
        buffer_alloc_results,
    };
    let allocated = pack_tree(device, queue, &mut tree).context("error packing voxel tree")?;

    Ok(VoxelModel {
        meta: VoxelMetadata {
            name,
            bounding_size: tree.bounding_size,
            index_levels: tree.index_levels,
            voxel_count: allocated.voxel_count,
            allocated_index_chunks: allocated.index_chunk_count,
        },
        data: VoxelBufferData {
            buffer_palette: palette.buffer_palette,
            buffer_index_chunks: tree.buffer_index_chunks,
            buffer_leaf_chunks: tree.buffer_leaf_chunks,
        },
    })
}

fn create_bg_voxelize_shared(
    ctx: &'_ VoxelizeCtx<'_>,
    tex_raw_voxels: &wgpu::Texture,
    tex_raw_chunk_indices: &wgpu::Texture,
) -> wgpu::BindGroup {
    let base_unscaled = ctx.scene.min.floor().as_ivec3();
    let size_unscaled = (ctx.scene.max.ceil() - ctx.scene.min.floor())
        .ceil()
        .as_ivec3();

    #[repr(C)]
    #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SceneBufferEntry {
        base: glam::Vec4,
        size: glam::Vec3,
        scale: f32,
    }
    let buffer_scene = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer_scene"),
            contents: bytemuck::cast_slice(&[SceneBufferEntry {
                base: base_unscaled.as_vec3().extend(0.0),
                size: size_unscaled.as_vec3(),
                scale: VOXELS_PER_UNIT as f32,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    #[repr(C)]
    #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct MaterialBufferEntry {
        base_albedo: glam::Vec4,
        base_metallic: f32,
        base_roughness: f32,
        normal_scale: f32,
        albedo_index: i32,
        normal_index: i32,
        metallic_roughness_index: i32,
        double_sided: u32,
        _pad: f32,
    }
    let buffer_materials = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material data storage buffer"),
            contents: bytemuck::cast_slice(
                &ctx.scene
                    .materials
                    .iter()
                    .map(|mat| MaterialBufferEntry {
                        base_albedo: mat.base_albedo,
                        base_metallic: mat.base_metallic,
                        base_roughness: mat.base_roughness,
                        normal_scale: mat.normal_scale,
                        albedo_index: mat.albedo_index,
                        normal_index: mat.normal_index,
                        metallic_roughness_index: mat.metallic_roughness_index,
                        double_sided: mat.double_sided as u32,
                        _pad: 0.0,
                    })
                    .collect::<Vec<MaterialBufferEntry>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxelize_shared"),
        layout: &ctx.bg_layouts.voxelize_shared,
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
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_chunk_indices.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_voxels.create_view(&Default::default()),
                ),
            },
        ],
    })
}

fn create_bg_voxelize_scene_textures(ctx: &'_ VoxelizeCtx<'_>) -> wgpu::BindGroup {
    let texture_views = ctx
        .scene
        .textures
        .iter()
        .map(|tex| {
            let (width, height) = tex.data.dimensions();
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: match tex.encoding {
                    gltf::scene::TextureEncoding::Linear => wgpu::TextureFormat::Rgba8Unorm,
                    gltf::scene::TextureEncoding::Srgb => wgpu::TextureFormat::Rgba8Unorm,
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &tex.data,
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

    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene textures bind group"),
        layout: &ctx.bg_layouts.voxelize_scene_textures,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureViewArray(
                &texture_views.iter().collect::<Vec<&wgpu::TextureView>>(),
            ),
        }],
    })
}

struct PrimitiveGroup {
    bg: wgpu::BindGroup,
    index_count: u32,
}

fn create_bg_voxelize_per_primitive(ctx: &'_ VoxelizeCtx<'_>) -> Vec<PrimitiveGroup> {
    let mut res = Vec::new();
    for node in &ctx.scene.nodes {
        let Some(mesh) = ctx.scene.meshes.get(node.mesh_id) else {
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
            let buffer_primitive =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

            let indices_u32 =
                bytemuck::cast_slice(&ctx.gltf.bin[primitive.indices.start..primitive.indices.end])
                    .iter()
                    .map(|idx: &u16| *idx as u32)
                    .collect::<Vec<u32>>();
            let buffer_indices = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive indices data"),
                    contents: match primitive.indices.component_type {
                        gltf::schema::ComponentType::UnsignedShort => {
                            &bytemuck::cast_slice(&indices_u32)
                        }
                        _ => &ctx.gltf.bin[primitive.indices.start..primitive.indices.end],
                    },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            let buffer_positions =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex position data"),
                        contents: &ctx.gltf.bin[primitive.positions.start..primitive.positions.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
            let buffer_normals = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex normal data"),
                    contents: &ctx.gltf.bin[primitive.normals.start..primitive.normals.end],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            let buffer_tangents =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("primitive vertex tangent data"),
                        contents: &ctx.gltf.bin[primitive.tangents.start..primitive.tangents.end],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
            let buffer_uv = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive vertex uv data"),
                    contents: &ctx.gltf.bin[primitive.uv.start..primitive.uv.end],
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            res.push(PrimitiveGroup {
                bg: ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("per primitive voxelize bind group"),
                    layout: &ctx.bg_layouts.voxelize_per_primitive,
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
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buffer_normals.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buffer_tangents.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: buffer_uv.as_entire_binding(),
                        },
                    ],
                }),
                index_count: primitive.indices.count,
            });
        }
    }
    res
}

const fn next_pow_4(x: u32) -> u32 {
    let mut k = 1;
    while k < x {
        k <<= 2;
    }
    k
}

// impl Voxelizer {
//     pub fn new(device: &wgpu::Device, scene_tex_count: u32) -> Self {
//         let bg_layouts = BindGroupLayouts {
//             voxelize_shared: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                 label: Some("shared voxelizer bind group layout"),
//                 entries: &[
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 0,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Uniform,
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 1,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Storage { read_only: true },
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 2,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 3,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::StorageTexture {
//                             access: wgpu::StorageTextureAccess::WriteOnly,
//                             format: wgpu::TextureFormat::Rg32Uint,
//                             view_dimension: wgpu::TextureViewDimension::D3,
//                         },
//                         count: None,
//                     },
//                 ],
//             }),
//             voxelize_per_sector: device.layout(
//                 "label",
//                 wgpu::ShaderStages::all(),
//                 uniform_buffer(),
//             ),
//             // have to separate this from the shared layout due to wgpu restriction
//             voxelize_scene_textures: device.create_bind_group_layout(
//                 &wgpu::BindGroupLayoutDescriptor {
//                     label: Some("scene textures bind group layout"),
//                     entries: &[wgpu::BindGroupLayoutEntry {
//                         binding: 0,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Texture {
//                             sample_type: wgpu::TextureSampleType::Float { filterable: true },
//                             view_dimension: wgpu::TextureViewDimension::D2,
//                             multisampled: false,
//                         },
//                         count: std::num::NonZeroU32::new(scene_tex_count),
//                     }],
//                 },
//             ),
//             voxelize_per_primitive: device.create_bind_group_layout(
//                 &wgpu::BindGroupLayoutDescriptor {
//                     label: Some("per-primitive voxelizer bind group layout"),
//                     entries: &[
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 0,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Uniform,
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 1,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Storage { read_only: true },
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 2,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Storage { read_only: true },
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 3,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Storage { read_only: true },
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 4,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Storage { read_only: true },
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                         wgpu::BindGroupLayoutEntry {
//                             binding: 5,
//                             visibility: wgpu::ShaderStages::COMPUTE,
//                             ty: wgpu::BindingType::Buffer {
//                                 ty: wgpu::BufferBindingType::Storage { read_only: true },
//                                 has_dynamic_offset: false,
//                                 min_binding_size: None,
//                             },
//                             count: None,
//                         },
//                     ],
//                 },
//             ),
//             // palette: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//             //     label: Some("palette data layout"),
//             //     entries: &[wgpu::BindGroupLayoutEntry {
//             //         binding: 0,
//             //         visibility: wgpu::ShaderStages::COMPUTE,
//             //         ty: wgpu::BindingType::Buffer {
//             //             ty: wgpu::BufferBindingType::Uniform,
//             //             has_dynamic_offset: false,
//             //             min_binding_size: None,
//             //         },
//             //         count: None,
//             //     }],
//             // }),
//             tree_data: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                 label: Some("voxel brickmap data layout"),
//                 entries: &[
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 0,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Storage { read_only: false },
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 1,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Storage { read_only: false },
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 2,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::StorageTexture {
//                             access: wgpu::StorageTextureAccess::ReadOnly,
//                             format: wgpu::TextureFormat::Rgba32Uint,
//                             view_dimension: wgpu::TextureViewDimension::D3,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 3,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::StorageTexture {
//                             access: wgpu::StorageTextureAccess::WriteOnly,
//                             format: wgpu::TextureFormat::Rgba32Uint,
//                             view_dimension: wgpu::TextureViewDimension::D3,
//                         },
//                         count: None,
//                     },
//                 ],
//             }),
//             tree_alloc_texture: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//                 label: Some("voxel brickmap input alloc and texture"),
//                 entries: &[
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 0,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Storage { read_only: false },
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 1,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::StorageTexture {
//                             access: wgpu::StorageTextureAccess::ReadOnly,
//                             format: wgpu::TextureFormat::Rg32Uint,
//                             view_dimension: wgpu::TextureViewDimension::D3,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 2,
//                         visibility: wgpu::ShaderStages::COMPUTE,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Uniform,
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                 ],
//             }),
//         };
//         let pipelines = Pipelines {
//             voxelize: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//                 label: Some("voxelize pipeline"),
//                 layout: Some(
//                     &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                         label: Some("voxelizer pipeline layout"),
//                         bind_group_layouts: &[
//                             &bg_layouts.voxelize_shared,
//                             &bg_layouts.voxelize_scene_textures,
//                             &bg_layouts.voxelize_per_primitive,
//                         ],
//                         immediate_size: 0,
//                     }),
//                 ),
//                 module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
//                     label: Some("voxelize"),
//                     source: wgpu::ShaderSource::Wgsl(
//                         std::include_str!("shaders/voxelize.wgsl").into(),
//                     ),
//                 }),
//                 entry_point: Some("compute_main"),
//                 compilation_options: Default::default(),
//                 cache: None,
//             }),
//             tree_leaf: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//                 label: Some("tree_leaf"),
//                 layout: Some(
//                     &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                         label: Some("voxel brickmap packing"),
//                         bind_group_layouts: &[
//                             &bg_layouts.tree_data,
//                             &bg_layouts.tree_alloc_texture,
//                         ],
//                         immediate_size: 0,
//                     }),
//                 ),
//                 module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
//                     label: Some("voxel brickmap packing"),
//                     source: wgpu::ShaderSource::Wgsl(
//                         std::include_str!("shaders/tree_64.wgsl").into(),
//                     ),
//                 }),
//                 entry_point: Some("compute_leaf"),
//                 compilation_options: Default::default(),
//                 cache: None,
//             }),
//             tree_index: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//                 label: Some("tree_index"),
//                 layout: Some(
//                     &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                         label: Some("voxel brickmap packing"),
//                         bind_group_layouts: &[
//                             &bg_layouts.tree_data,
//                             &bg_layouts.tree_alloc_texture,
//                         ],
//                         immediate_size: 0,
//                     }),
//                 ),
//                 module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
//                     label: Some("voxel brickmap packing"),
//                     source: wgpu::ShaderSource::Wgsl(
//                         std::include_str!("shaders/tree_64.wgsl").into(),
//                     ),
//                 }),
//                 entry_point: Some("compute_index"),
//                 compilation_options: Default::default(),
//                 cache: None,
//             }),
//         };

//         Self {
//             pipelines,
//             bg_layouts,
//         }
//     }

//     pub fn load_gltf(
//         &mut self,
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//         gltf: Gltf,
//         scene: Scene,
//         name: Option<String>,
//     ) -> Result<VoxelModel> {
//         let voxelizer = Self::new(device, scene.textures.len() as u32);

//         let name = name.unwrap_or(String::from("model"));

//         dbg!(&scene.min, &scene.max);

//         let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_ivec3();
//         let size = size_unscaled.as_uvec3() * VOXELS_PER_UNIT;

//         println!(
//             "raw size: {}MB",
//             (size.element_product() as f32) / (1024.0 * 256.0)
//         );

//         // texture of raw 3d voxels
//         let raw_voxels = device.create_texture(&wgpu::TextureDescriptor {
//             label: Some("voxel texture"),
//             size: wgpu::Extent3d {
//                 width: size.x,
//                 height: size.y,
//                 depth_or_array_layers: size.z,
//             },
//             mip_level_count: 1,
//             sample_count: 1,
//             dimension: wgpu::TextureDimension::D3,
//             format: wgpu::TextureFormat::Rg32Uint,
//             usage: wgpu::TextureUsages::STORAGE_BINDING,
//             view_formats: &[],
//         });

//         let mut encoder = device.create_command_encoder(&Default::default());

//         let palette = voxelizer.build_palette(device, queue, &mut encoder);

//         voxelizer.voxelize(device, queue, &mut encoder, &raw_voxels, &scene, &gltf);

//         let mut tree = voxelizer.build_tree(&palette, device, &mut encoder, &raw_voxels, size);

//         queue.submit([encoder.finish()]);

//         let allocated = pack_tree(device, queue, &mut tree).context("error building voxel tree")?;

//         Ok(VoxelModel {
//             meta: VoxelMetadata {
//                 name,
//                 bounding_size: tree.bounding_size,
//                 index_levels: tree.index_levels,
//                 voxel_count: allocated.voxel_count,
//                 allocated_index_chunks: allocated.index_chunk_count,
//             },
//             data: VoxelBufferData {
//                 buffer_palette: palette.buffer_palette,
//                 buffer_index_chunks: tree.buffer_index_chunks,
//                 buffer_leaf_chunks: tree.buffer_leaf_chunks,
//             },
//         })
//     }
// }

struct TreeData {
    bounding_size: u32,
    index_levels: u32,
    max_index_chunks: u64,
    max_voxels: u64,
    buffer_index_chunks: wgpu::Buffer,
    buffer_leaf_chunks: wgpu::Buffer,
    buffer_alloc_results: wgpu::Buffer,
}
struct PaletteData {
    buffer_palette: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AllocatorResults {
    index_chunk_count: u32,
    voxel_count: u32,
}

// impl Voxelizer {
//     fn voxelize(
//         &self,
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//         encoder: &mut wgpu::CommandEncoder,
//         output: &wgpu::Texture,
//         scene: &Scene,
//         gltf: &Gltf,
//     ) {
//         let base_unscaled = scene.min.floor().as_ivec3();
//         let size_unscaled = (scene.max.ceil() - scene.min.floor()).ceil().as_ivec3();

//         let bg_voxelize_shared = {
//             #[repr(C)]
//             #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
//             struct SceneBufferEntry {
//                 base: glam::Vec4,
//                 size: glam::Vec3,
//                 scale: f32,
//             }
//             // binding 0
//             let buffer_scene = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some("material data storage buffer"),
//                 contents: bytemuck::cast_slice(&[SceneBufferEntry {
//                     base: base_unscaled.as_vec3().extend(0.0),
//                     size: size_unscaled.as_vec3(),
//                     scale: VOXELS_PER_UNIT as f32,
//                 }]),
//                 usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
//             });

//             #[repr(C)]
//             #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
//             struct MaterialBufferEntry {
//                 base_albedo: glam::Vec4,
//                 base_metallic: f32,
//                 base_roughness: f32,
//                 normal_scale: f32,
//                 albedo_index: i32,
//                 normal_index: i32,
//                 metallic_roughness_index: i32,
//                 double_sided: u32,
//                 _pad: f32,
//             }
//             // binding 1
//             let buffer_materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some("material data storage buffer"),
//                 contents: bytemuck::cast_slice(
//                     &scene
//                         .materials
//                         .iter()
//                         .map(|mat| MaterialBufferEntry {
//                             base_albedo: mat.base_albedo,
//                             base_metallic: mat.base_metallic,
//                             base_roughness: mat.base_roughness,
//                             normal_scale: mat.normal_scale,
//                             albedo_index: mat.albedo_index,
//                             normal_index: mat.normal_index,
//                             metallic_roughness_index: mat.metallic_roughness_index,
//                             double_sided: mat.double_sided as u32,
//                             _pad: 0.0,
//                         })
//                         .collect::<Vec<MaterialBufferEntry>>(),
//                 ),
//                 usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//             });
//             // binding 2
//             let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
//                 address_mode_u: wgpu::AddressMode::Repeat,
//                 address_mode_v: wgpu::AddressMode::Repeat,
//                 address_mode_w: wgpu::AddressMode::ClampToEdge,
//                 mag_filter: wgpu::FilterMode::Linear,
//                 min_filter: wgpu::FilterMode::Linear,
//                 mipmap_filter: wgpu::MipmapFilterMode::Nearest,
//                 ..Default::default()
//             });
//             device.create_bind_group(&wgpu::BindGroupDescriptor {
//                 label: Some("shared voxelizer bind group"),
//                 layout: &self.bg_layouts.voxelize_shared,
//                 entries: &[
//                     wgpu::BindGroupEntry {
//                         binding: 0,
//                         resource: buffer_scene.as_entire_binding(),
//                     },
//                     wgpu::BindGroupEntry {
//                         binding: 1,
//                         resource: buffer_materials.as_entire_binding(),
//                     },
//                     wgpu::BindGroupEntry {
//                         binding: 2,
//                         resource: wgpu::BindingResource::Sampler(&sampler),
//                     },
//                     wgpu::BindGroupEntry {
//                         binding: 3,
//                         resource: wgpu::BindingResource::TextureView(&output.create_view(
//                             &wgpu::TextureViewDescriptor {
//                                 label: Some("voxelized result texture output view"),
//                                 usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
//                                 ..Default::default()
//                             },
//                         )),
//                     },
//                 ],
//             })
//         };

//         dbg!(&scene.textures.len());
//         dbg!(&scene.materials.len());
//         dbg!(
//             &scene
//                 .meshes
//                 .iter()
//                 .map(|mesh| mesh.primitives.len())
//                 .sum::<usize>()
//         );
//         println!(
//             "texture data size (total): {:.2} MB",
//             &scene
//                 .textures
//                 .iter()
//                 .map(|tex| tex.data.len() as f64 / (1024.0 * 1024.0))
//                 .sum::<f64>()
//         );
//         dbg!(&gltf.bin.len());

//         let bg_voxelize_scene_textures = {
//             let texture_views = scene
//                 .textures
//                 .iter()
//                 .map(|tex| {
//                     let (width, height) = tex.data.dimensions();
//                     let texture = device.create_texture(&wgpu::TextureDescriptor {
//                         label: None,
//                         size: wgpu::Extent3d {
//                             width,
//                             height,
//                             depth_or_array_layers: 1,
//                         },
//                         mip_level_count: 1,
//                         sample_count: 1,
//                         dimension: wgpu::TextureDimension::D2,
//                         format: match tex.encoding {
//                             gltf::scene::TextureEncoding::Linear => wgpu::TextureFormat::Rgba8Unorm,
//                             gltf::scene::TextureEncoding::Srgb => wgpu::TextureFormat::Rgba8Unorm,
//                         },
//                         usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
//                         view_formats: &[],
//                     });
//                     queue.write_texture(
//                         wgpu::TexelCopyTextureInfo {
//                             texture: &texture,
//                             mip_level: 0,
//                             origin: wgpu::Origin3d::ZERO,
//                             aspect: wgpu::TextureAspect::All,
//                         },
//                         &tex.data,
//                         wgpu::TexelCopyBufferLayout {
//                             offset: 0,
//                             bytes_per_row: Some(4 * width),
//                             rows_per_image: Some(height),
//                         },
//                         wgpu::Extent3d {
//                             width,
//                             height,
//                             depth_or_array_layers: 1,
//                         },
//                     );
//                     let view = texture.create_view(&wgpu::TextureViewDescriptor {
//                         ..Default::default()
//                     });

//                     view
//                 })
//                 .collect::<Vec<wgpu::TextureView>>();

//             device.create_bind_group(&wgpu::BindGroupDescriptor {
//                 label: Some("scene textures bind group"),
//                 layout: &self.bg_layouts.voxelize_scene_textures,
//                 entries: &[wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: wgpu::BindingResource::TextureViewArray(
//                         &texture_views.iter().collect::<Vec<&wgpu::TextureView>>(),
//                     ),
//                 }],
//             })
//         };

//         struct PrimitiveGroup {
//             bg: wgpu::BindGroup,
//             index_count: u32,
//         }
//         let mut bg_voxelize_per_primitive = Vec::new();
//         for node in &scene.nodes {
//             let Some(mesh) = scene.meshes.get(node.mesh_id) else {
//                 continue;
//             };
//             for primitive in &mesh.primitives {
//                 // each (object, primitive) pair in the scene gets its own bind group
//                 // inefficient but idc its just a generation step for now

//                 #[repr(C)]
//                 #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
//                 struct PrimitiveBufferEntry {
//                     matrix: glam::Mat4,
//                     normal_matrix: [[f32; 4]; 3],
//                     material_id: u32,
//                     index_count: u32,
//                     _pad: [f32; 2],
//                 }
//                 // binding 0
//                 let buffer_primitive =
//                     device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                         label: Some("primitive data uniform buffer"),
//                         contents: bytemuck::cast_slice(&[PrimitiveBufferEntry {
//                             matrix: node.transform,
//                             normal_matrix: glam::Mat3::from_mat4(node.transform.inverse())
//                                 .transpose()
//                                 .to_cols_array_2d()
//                                 .map(|v| [v[0], v[1], v[2], 0.0]),
//                             material_id: primitive.material_id,
//                             index_count: primitive.indices.count,
//                             _pad: [0.0; 2],
//                         }]),
//                         usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
//                     });

//                 // binding 1
//                 let indices_u32 =
//                     bytemuck::cast_slice(&gltf.bin[primitive.indices.start..primitive.indices.end])
//                         .iter()
//                         .map(|idx: &u16| *idx as u32)
//                         .collect::<Vec<u32>>();
//                 let buffer_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                     label: Some("primitive indices data"),
//                     contents: match primitive.indices.component_type {
//                         gltf::schema::ComponentType::UnsignedShort => {
//                             &bytemuck::cast_slice(&indices_u32)
//                         }
//                         _ => &gltf.bin[primitive.indices.start..primitive.indices.end],
//                     },
//                     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//                 });
//                 // binding 2
//                 let buffer_positions =
//                     device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                         label: Some("primitive vertex position data"),
//                         contents: &gltf.bin[primitive.positions.start..primitive.positions.end],
//                         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//                     });
//                 // binding 3
//                 let buffer_normals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                     label: Some("primitive vertex normal data"),
//                     contents: &gltf.bin[primitive.normals.start..primitive.normals.end],
//                     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//                 });
//                 // binding 4
//                 let buffer_tangents =
//                     device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                         label: Some("primitive vertex tangent data"),
//                         contents: &gltf.bin[primitive.tangents.start..primitive.tangents.end],
//                         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//                     });
//                 // binding 5
//                 let buffer_uv = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                     label: Some("primitive vertex uv data"),
//                     contents: &gltf.bin[primitive.uv.start..primitive.uv.end],
//                     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//                 });

//                 bg_voxelize_per_primitive.push(PrimitiveGroup {
//                     bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
//                         label: Some("per primitive voxelize bind group"),
//                         layout: &self.bg_layouts.voxelize_per_primitive,
//                         entries: &[
//                             wgpu::BindGroupEntry {
//                                 binding: 0,
//                                 resource: buffer_primitive.as_entire_binding(),
//                             },
//                             wgpu::BindGroupEntry {
//                                 binding: 1,
//                                 resource: buffer_indices.as_entire_binding(),
//                             },
//                             wgpu::BindGroupEntry {
//                                 binding: 2,
//                                 resource: buffer_positions.as_entire_binding(),
//                             },
//                             wgpu::BindGroupEntry {
//                                 binding: 3,
//                                 resource: buffer_normals.as_entire_binding(),
//                             },
//                             wgpu::BindGroupEntry {
//                                 binding: 4,
//                                 resource: buffer_tangents.as_entire_binding(),
//                             },
//                             wgpu::BindGroupEntry {
//                                 binding: 5,
//                                 resource: buffer_uv.as_entire_binding(),
//                             },
//                         ],
//                     }),
//                     index_count: primitive.indices.count,
//                 });
//             }
//         }
//         {
//             let descriptor = wgpu::ComputePassDescriptor {
//                 label: Some("voxelization pass"),
//                 timestamp_writes: None,
//             };
//             let mut pass = encoder.begin_compute_pass(&descriptor);

//             pass.set_pipeline(&self.pipelines.voxelize);
//             pass.set_bind_group(0, Some(&bg_voxelize_shared), &[]);
//             pass.set_bind_group(1, Some(&bg_voxelize_scene_textures), &[]);

//             for primitive in &bg_voxelize_per_primitive {
//                 pass.set_bind_group(2, Some(&primitive.bg), &[]);

//                 let tris = primitive.index_count / 3;
//                 pass.dispatch_workgroups(tris.div_ceil(64), 1, 1);
//             }
//         }
//     }

//     fn build_palette(
//         &self,
//         device: &wgpu::Device,
//         _queue: &wgpu::Queue,
//         _encoder: &mut wgpu::CommandEncoder,
//     ) -> PaletteData {
//         // create the palette bind group
//         // palette is just hand-generated for now, should do some cool generation in a previous compute stage in the future
//         // that's based on the actual albedo distribution in the gltf
//         let mut palette = vec![glam::Vec4::ZERO];
//         for r in 0..9u32 {
//             for g in 0..9u32 {
//                 for b in 0..9u32 {
//                     let rgba = (glam::vec3(r as f32, g as f32, b as f32) / 8.0).extend(1.0);
//                     palette.push(rgba);
//                 }
//             }
//         }
//         for i in 0..69u32 {
//             let v = (i * 255 / 68) & 0xff;
//             let rgb = (glam::uvec3(v, v, v).as_vec3() / 255.0).extend(1.0);
//             palette.push(rgb);
//         }
//         while palette.len() < 1024 {
//             palette.push(glam::Vec4::ZERO);
//         }

//         let buffer_palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//             label: Some("palette colors"),
//             contents: bytemuck::cast_slice(&palette),
//             usage: wgpu::BufferUsages::UNIFORM
//                 | wgpu::BufferUsages::COPY_DST
//                 | wgpu::BufferUsages::COPY_SRC,
//         });

//         // let tex_palette_lut = device.create_texture(&wgpu::TextureDescriptor {
//         //     label: Some("palette LUT"),
//         //     size: wgpu::Extent3d {
//         //         width: 32,
//         //         height: 32,
//         //         depth_or_array_layers: 32,
//         //     },
//         //     mip_level_count: 1,
//         //     sample_count: 1,
//         //     dimension: wgpu::TextureDimension::D3,
//         //     format: wgpu::TextureFormat::R32Uint,
//         //     usage: wgpu::TextureUsages::STORAGE_BINDING,
//         //     view_formats: &[],
//         // });

//         // let bg_palette = device.create_bind_group(&wgpu::BindGroupDescriptor {
//         //     label: Some("palette bin group"),
//         //     layout: &self.bg_layouts.palette,
//         //     entries: &[
//         //         wgpu::BindGroupEntry {
//         //             binding: 0,
//         //             resource: buffer_palette.as_entire_binding(),
//         //         },
//         //         // wgpu::BindGroupEntry {
//         //         //     binding: 1,
//         //         //     resource: wgpu::BindingResource::TextureView(&tex_palette_lut.create_view(
//         //         //         &wgpu::TextureViewDescriptor {
//         //         //             label: Some("palette LUT view"),
//         //         //             usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
//         //         //             ..Default::default()
//         //         //         },
//         //         //     )),
//         //         // },
//         //     ],
//         // });
//         // {
//         //     let descriptor = wgpu::ComputePassDescriptor {
//         //         label: Some("palette LUT generation pass"),
//         //         timestamp_writes: None,
//         //     };
//         //     let mut pass = encoder.begin_compute_pass(&descriptor);

//         //     pass.set_pipeline(&self.pipelines.palette);
//         //     pass.set_bind_group(0, Some(&bg_palette), &[]);

//         //     pass.dispatch_workgroups(8, 8, 8);
//         // }

//         PaletteData {
//             buffer_palette,
//             // tex_palette_lut,
//         }
//     }

//     fn build_tree(
//         &self,
//         palette: &PaletteData,
//         device: &wgpu::Device,
//         encoder: &mut wgpu::CommandEncoder,
//         voxels_raw: &wgpu::Texture,
//         size: glam::UVec3,
//     ) -> TreeData {
//         let next_pow_4 = |x: u32| {
//             let mut k = 1;
//             while k < x {
//                 k <<= 2;
//             }
//             k
//         };
//         let bounding_size = next_pow_4(size.max_element()).max(16);
//         let index_levels = bounding_size.ilog(4);

//         let max_voxels = (bounding_size as u64).pow(3);
//         let max_leaf_chunks_bytes =
//             u64::min(MAX_STORAGE_BUFFER_BINDING_SIZE as u64, max_voxels * 4);

//         let mut max_index_chunks = 0;
//         for k in 1..=index_levels {
//             max_index_chunks += ((bounding_size >> (2 * k)) as u64).pow(3);
//         }
//         let max_index_chunks_bytes = u64::min(
//             MAX_STORAGE_BUFFER_BINDING_SIZE as u64,
//             max_index_chunks * 12,
//         );

//         dbg!(
//             bounding_size,
//             index_levels,
//             max_voxels,
//             max_leaf_chunks_bytes,
//             max_index_chunks,
//             max_index_chunks_bytes
//         );

//         // create tree data bind group
//         // this contains the resulting storage buffers/texture that make up the voxel tree
//         let buffer_index_chunks = device.create_buffer(&wgpu::BufferDescriptor {
//             label: Some("index_chunks"),
//             size: max_index_chunks_bytes,
//             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//             mapped_at_creation: false,
//         });
//         let buffer_leaf_chunks = device.create_buffer(&wgpu::BufferDescriptor {
//             label: Some("leaf_chunks"),
//             size: max_leaf_chunks_bytes,
//             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//             mapped_at_creation: false,
//         });

//         let index_map_desc = wgpu::TextureDescriptor {
//             label: Some("index_map"),
//             size: wgpu::Extent3d {
//                 width: bounding_size >> 4,
//                 height: bounding_size >> 4,
//                 depth_or_array_layers: bounding_size >> 4,
//             },
//             mip_level_count: 1,
//             sample_count: 1,
//             dimension: wgpu::TextureDimension::D3,
//             format: wgpu::TextureFormat::Rgba32Uint,
//             usage: wgpu::TextureUsages::STORAGE_BINDING,
//             view_formats: &[],
//         };
//         let index_map_view_desc = wgpu::TextureViewDescriptor {
//             label: Some("index_map view"),
//             usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
//             ..Default::default()
//         };
//         let index_map_a = device.create_texture(&wgpu::TextureDescriptor {
//             // the size can be reduced by a factor of 4 here, since the earliest pass
//             // that writes to this is the first index pass, wheras index_map_b is written to in the leaf pass
//             size: wgpu::Extent3d {
//                 width: bounding_size >> 6,
//                 height: bounding_size >> 6,
//                 depth_or_array_layers: bounding_size >> 6,
//             },
//             ..index_map_desc
//         });
//         let index_map_b = device.create_texture(&index_map_desc);
//         let index_map_a_view = index_map_a.create_view(&index_map_view_desc);
//         let index_map_b_view = index_map_b.create_view(&index_map_view_desc);

//         let bg_tree_data_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label: Some("tree_data_a"),
//             layout: &self.bg_layouts.tree_data,
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: buffer_index_chunks.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: buffer_leaf_chunks.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 2,
//                     resource: wgpu::BindingResource::TextureView(&index_map_a_view),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 3,
//                     resource: wgpu::BindingResource::TextureView(&index_map_b_view),
//                 },
//             ],
//         });
//         let bg_tree_data_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label: Some("tree_data_b"),
//             layout: &self.bg_layouts.tree_data,
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: buffer_index_chunks.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: buffer_leaf_chunks.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 2,
//                     resource: wgpu::BindingResource::TextureView(&index_map_b_view),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 3,
//                     resource: wgpu::BindingResource::TextureView(&index_map_a_view),
//                 },
//             ],
//         });

//         // build tree allocator atomic buffer bind group
//         // contains both the raw voxel texture, and atomic counters for building the tree
//         let buffer_alloc_data = device.create_buffer(&wgpu::BufferDescriptor {
//             label: Some("tree_alloc_data"),
//             size: 8,
//             usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//             mapped_at_creation: false,
//         });
//         // i read from the alloc metadata buffer by copying it here
//         let buffer_alloc_results = device.create_buffer(&wgpu::BufferDescriptor {
//             label: Some("tree_alloc_results"),
//             size: 8,
//             usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
//             mapped_at_creation: false,
//         });
//         let bg_tree_alloc_texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label: Some("tree_alloc_data_and_raw_texture"),
//             layout: &self.bg_layouts.tree_alloc_texture,
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: buffer_alloc_data.as_entire_binding(),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: wgpu::BindingResource::TextureView(&voxels_raw.create_view(
//                         &wgpu::TextureViewDescriptor {
//                             label: Some("raw voxels view"),
//                             dimension: Some(wgpu::TextureViewDimension::D3),
//                             usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
//                             ..Default::default()
//                         },
//                     )),
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 2,
//                     resource: palette.buffer_palette.as_entire_binding(),
//                 },
//             ],
//         });

//         // now execute

//         // leaf pass
//         {
//             let descriptor = wgpu::ComputePassDescriptor {
//                 label: Some("tree_leaf"),
//                 timestamp_writes: None,
//             };
//             let mut pass = encoder.begin_compute_pass(&descriptor);

//             pass.set_pipeline(&self.pipelines.tree_leaf);
//             pass.set_bind_group(0, Some(&bg_tree_data_a), &[]);
//             pass.set_bind_group(1, Some(&bg_tree_alloc_texture), &[]);

//             let size = bounding_size >> 4;
//             pass.dispatch_workgroups(size, size, size);
//         }

//         // index passes
//         for k in 2..index_levels {
//             let descriptor = wgpu::ComputePassDescriptor {
//                 label: Some("tree_index"),
//                 timestamp_writes: None,
//             };
//             let mut pass = encoder.begin_compute_pass(&descriptor);

//             pass.set_pipeline(&self.pipelines.tree_index);
//             pass.set_bind_group(
//                 0,
//                 Some(match k & 1 {
//                     0 => &bg_tree_data_b,
//                     _ => &bg_tree_data_a,
//                 }),
//                 &[],
//             );
//             pass.set_bind_group(1, Some(&bg_tree_alloc_texture), &[]);

//             let size = bounding_size >> (2 + 2 * k);
//             pass.dispatch_workgroups(size, size, size);
//         }

//         // we read from the results buffer
//         encoder.copy_buffer_to_buffer(
//             &buffer_alloc_data,
//             0,
//             &buffer_alloc_results,
//             0,
//             buffer_alloc_data.size(),
//         );

//         TreeData {
//             bounding_size,
//             index_levels,
//             max_index_chunks: max_index_chunks_bytes.div_ceil(12),
//             max_voxels: max_leaf_chunks_bytes.div_ceil(4),
//             buffer_index_chunks,
//             buffer_leaf_chunks,
//             buffer_alloc_results,
//         }
//     }
// }

fn build_palette(ctx: &'_ VoxelizeCtx<'_>) -> PaletteData {
    // create the palette bind group
    // palette is just hand-generated for now, should do some cool generation in a previous compute stage in the future
    // that's based on the actual albedo distribution in the gltf
    let mut palette = vec![glam::Vec4::ZERO];
    for r in 0..9u32 {
        for g in 0..9u32 {
            for b in 0..9u32 {
                let rgba = (glam::vec3(r as f32, g as f32, b as f32) / 8.0).extend(1.0);
                palette.push(rgba);
            }
        }
    }
    for i in 0..69u32 {
        let v = (i * 255 / 68) & 0xff;
        let rgb = (glam::uvec3(v, v, v).as_vec3() / 255.0).extend(1.0);
        palette.push(rgb);
    }
    while palette.len() < 1024 {
        palette.push(glam::Vec4::ZERO);
    }

    let buffer_palette = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette colors"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

    PaletteData { buffer_palette }
}

fn pack_tree(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tree: &mut TreeData,
) -> Result<AllocatorResults> {
    let alloc_results = {
        tree.buffer_alloc_results
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = tree.buffer_alloc_results.get_mapped_range(..);
        bytemuck::cast_slice::<_, AllocatorResults>(&*view)[0]
    };

    if (alloc_results.voxel_count as u64) >= tree.max_voxels {
        bail!(
            "too many voxels were generated to fit in the output buffer. generated: {}, max: {}",
            alloc_results.voxel_count,
            tree.max_voxels
        );
    }
    if (alloc_results.index_chunk_count as u64) >= tree.max_index_chunks {
        bail!(
            "too many index chunks were generated to fit in the output buffer. generated: {}, max: {}",
            alloc_results.index_chunk_count,
            tree.max_index_chunks
        );
    }

    // these ops rely on reading back the atomic counter data, hence the new encoder
    let mut encoder = device.create_command_encoder(&Default::default());

    // shrink the leaf and index buffers
    // in the future, once I'm adding edits this probably can't happen this way, i'll have to make the allocator
    // request different sectors dynamically and do hella tracking

    let allocated_index_chunk_bytes = alloc_results.index_chunk_count * 12;
    let buffer_index_chunks_cpct = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree_index_chunks"),
        size: allocated_index_chunk_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &tree.buffer_index_chunks,
        0,
        &buffer_index_chunks_cpct,
        0,
        Some(allocated_index_chunk_bytes as u64),
    );

    let allocated_leaf_chunk_bytes = alloc_results.voxel_count * 4;
    let buffer_leaf_chunks_cpct = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree_leaf_chunks"),
        size: allocated_leaf_chunk_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &tree.buffer_leaf_chunks,
        0,
        &buffer_leaf_chunks_cpct,
        0,
        Some(allocated_leaf_chunk_bytes as u64),
    );

    queue.submit([encoder.finish()]);

    tree.buffer_index_chunks.destroy();
    tree.buffer_leaf_chunks.destroy();

    tree.buffer_leaf_chunks = buffer_leaf_chunks_cpct;
    tree.buffer_index_chunks = buffer_index_chunks_cpct;

    Ok(alloc_results)
}
