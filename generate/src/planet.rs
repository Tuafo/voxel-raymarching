//! Procedural planet voxelizer.
//!
//! Generates a single sparse-tree voxel model from a sphere SDF + 3D fbm
//! noise. Uses the existing `tree_64.wgsl` tree-build passes unchanged, so
//! the output `.voxel` file is bit-identical in format to those produced by
//! `voxelize_gltf` and is loaded by the runtime engine without any changes.
//!
//! High-level pipeline:
//!   1. Run `classify_chunks` (planet_fill.wgsl): decides which 64^3 world
//!      chunks intersect the planet surface band, and atomically allocates
//!      a slot for each in `raw_voxels`.
//!   2. Read back the chunk count.
//!   3. Allocate `raw_voxels` (size = ceil(chunk_count^(1/3)) cubes), and
//!      run `fill_voxels` to evaluate the SDF and write packed voxels.
//!   4. Run `tree_64.wgsl` `compute_leaf` then repeated `compute_index`
//!      passes to assemble the sparse 4^3 tree. (Reused unchanged.)
//!   5. Pack-shrink output buffers; return a `VoxelModel`.
//!
//! Voxel format mirrors `generate/src/shaders/voxelize.wgsl::pack_voxel`.

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use utils::layout::{
    DeviceUtils, storage_buffer, storage_texture, uniform_buffer,
};
use utils::pipeline::PipelineUtils;
use wgpu::util::DeviceExt;

use crate::MAX_STORAGE_BUFFER_BINDING_SIZE;
use crate::models::{VoxelBufferData, VoxelMetadata, VoxelModel};

/// All parameters that control planet generation.
#[derive(Debug, Clone)]
pub struct PlanetConfig {
    pub name: String,
    /// Voxels per world unit (matches `voxels_per_unit` in the gltf path).
    /// Affects metadata only; planet sizing is in voxels via `radius`.
    pub voxels_per_unit: u32,
    /// Planet radius, in voxels.
    pub radius: f32,
    /// Surface noise amplitude, in voxels (peak-to-peak terrain height).
    pub amplitude: f32,
    /// Frequency of the lowest noise octave (per unit on the unit sphere).
    pub base_freq: f32,
    /// Number of fbm octaves.
    pub octaves: u32,
    /// Crust thickness used for the chunk-classify SDF band, in voxels.
    /// A chunk is allocated if any sampled point falls within ±this of the
    /// surface, so make this >= the largest voxel filled-shell you want.
    pub crust_thickness: f32,
    /// Hash seed for the noise.
    pub seed: u32,
    /// Altitude (relative to base sphere) thresholds for biome bands.
    pub sea_level: f32,
    pub snow_height: f32,
    pub sand_band: f32,
}

impl PlanetConfig {
    pub fn default_for(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            // Match sponza's scale (engine sets ModelTransform.scale = 1/voxels_per_unit,
            // so this controls the world-space size of one voxel).
            voxels_per_unit: 16,
            // Almost fills a 1024^3 bounding cube: 2*(440+24+16) = 960 < 1024.
            radius: 440.0,
            amplitude: 24.0,
            base_freq: 4.5,
            octaves: 5,
            crust_thickness: 8.0,
            seed: 0xC0FFEE,
            sea_level: -2.0,
            snow_height: 16.0,
            sand_band: 2.0,
        }
    }
}

/// Generates a planet `VoxelModel`. The result can be `.serialize()`-d to
/// the same `.voxel` file format used by glTF voxelization.
pub fn voxelize_planet(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cfg: &PlanetConfig,
) -> Result<VoxelModel> {
    // ---- sizing ----
    // We need a cube big enough to hold the planet plus one crust thickness.
    let half_extent = (cfg.radius + cfg.amplitude + cfg.crust_thickness * 2.0).ceil() as u32;
    let unpadded = (half_extent * 2).max(64);
    let bounding_size = next_pow_4(unpadded).max(64);
    let index_levels = bounding_size.ilog(4);
    if bounding_size > (1 << 21) {
        bail!(
            "bounding_size {} exceeds the 2^21 limit baked into the runtime tree (radius too large)",
            bounding_size
        );
    }

    let chunks_per_axis = bounding_size / 64;
    let center = glam::Vec3::splat(bounding_size as f32 * 0.5);
    let world_size = glam::UVec3::splat(bounding_size); // metadata: cube

    println!(
        "planet '{}': bounding_size={}, chunks_per_axis={}, index_levels={}, radius={:.1}, amplitude={:.1}",
        cfg.name, bounding_size, chunks_per_axis, index_levels, cfg.radius, cfg.amplitude
    );

    // ---- raw_chunk_indices texture (size = chunks_per_axis^3, R32Uint) ----
    let tex_raw_chunk_indices = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_raw_chunk_indices"),
        size: wgpu::Extent3d {
            width: chunks_per_axis,
            height: chunks_per_axis,
            depth_or_array_layers: chunks_per_axis,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Zero-init: a fresh 3D texture is undefined in wgpu, so we write zeros.
    let zero_chunk_data = vec![0u32; (chunks_per_axis * chunks_per_axis * chunks_per_axis) as usize];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_raw_chunk_indices,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&zero_chunk_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * chunks_per_axis),
            rows_per_image: Some(chunks_per_axis),
        },
        wgpu::Extent3d {
            width: chunks_per_axis,
            height: chunks_per_axis,
            depth_or_array_layers: chunks_per_axis,
        },
    );

    // ---- atomic chunk counter ----
    let buffer_chunk_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("planet_chunk_counter"),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        contents: bytemuck::cast_slice(&[0u32]),
    });
    let buffer_chunk_counter_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_chunk_counter_readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // ---- planet params uniform (initially with raw_minor_chunks=0 — set later) ----
    let mut params = PlanetParams::from_config(cfg, bounding_size, chunks_per_axis, center, 0);
    let buffer_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("planet_params"),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        contents: bytemuck::bytes_of(&params),
    });

    // ---- planet_fill pipeline + bind group layout ----
    // Single bind group, two entry points. We can reuse the same bind group
    // for both passes by giving it both read+write views of raw_chunk_indices.
    // For raw_voxels we also use a write view that's swapped between passes.
    let bg_layout_planet = device.layout(
        "planet_fill_bg_layout",
        wgpu::ShaderStages::COMPUTE,
        (
            uniform_buffer(),
            storage_texture().r32uint().dimension_3d().write_only(),
            storage_texture().r32uint().dimension_3d().read_only(),
            storage_texture().r32uint().dimension_3d().write_only(),
            storage_buffer().read_write(),
        ),
    );
    let shader_planet = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("planet_fill"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/planet_fill.wgsl").into()),
    });
    let pipeline_classify = device
        .compute_pipeline("planet_classify", &shader_planet)
        .entry_point("classify_chunks")
        .layout(&[&bg_layout_planet]);
    let pipeline_fill = device
        .compute_pipeline("planet_fill", &shader_planet)
        .entry_point("fill_voxels")
        .layout(&[&bg_layout_planet]);

    // We need a placeholder raw_voxels texture for pass 1 (the layout requires
    // a bound view even though classify_chunks doesn't write to it).
    let tex_raw_voxels_placeholder = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_raw_voxels_placeholder"),
        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 64 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // Likewise a placeholder for raw_chunk_indices on the OPPOSITE side per pass:
    // classify writes to the real one, so it needs a dummy for the read slot;
    // fill reads from the real one, so it needs a dummy for the write slot.
    // We can't bind the same storage texture as both read_only and write_only
    // within a single dispatch (wgpu validation rejects it).
    let tex_chunk_indices_placeholder = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_chunk_indices_placeholder"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let bg_classify = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_classify_bg"),
        layout: &bg_layout_planet,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_params.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_chunk_indices.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                // dummy on the read slot — classify only writes
                resource: wgpu::BindingResource::TextureView(
                    &tex_chunk_indices_placeholder.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_voxels_placeholder.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry { binding: 4, resource: buffer_chunk_counter.as_entire_binding() },
        ],
    });

    // ---- pass 1: classify ----
    let timer = std::time::Instant::now();
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet_classify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_classify);
            pass.set_bind_group(0, Some(&bg_classify), &[]);
            // workgroup_size is (1,1,1), so dispatch = chunks_per_axis^3.
            pass.dispatch_workgroups(chunks_per_axis, chunks_per_axis, chunks_per_axis);
        }
        encoder.copy_buffer_to_buffer(
            &buffer_chunk_counter,
            0,
            &buffer_chunk_counter_readback,
            0,
            4,
        );
        queue.submit([encoder.finish()]);
    }

    // ---- read back chunk count ----
    let raw_chunk_count = {
        buffer_chunk_counter_readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = buffer_chunk_counter_readback.get_mapped_range(..);
        let n = bytemuck::cast_slice::<_, u32>(&view)[0];
        drop(view);
        buffer_chunk_counter_readback.unmap();
        n
    };
    println!(
        "planet classify: {} of {} chunks active ({:.1}%) in {:?}",
        raw_chunk_count,
        chunks_per_axis.pow(3),
        100.0 * raw_chunk_count as f32 / chunks_per_axis.pow(3) as f32,
        timer.elapsed()
    );
    if raw_chunk_count == 0 {
        bail!("planet classify produced 0 active chunks (planet entirely outside bounds?)");
    }

    // ---- allocate raw_voxels with the same layout convention as voxelize_gltf ----
    let raw_voxel_count = (raw_chunk_count as u64) * 64 * 64 * 64;
    let raw_minor_size = ((raw_voxel_count as f64).powf(1.0 / 3.0).ceil() as u32)
        .max(1)
        .next_multiple_of(64);
    let raw_major_size = (raw_voxel_count as u32)
        .div_ceil(raw_minor_size * raw_minor_size)
        .next_multiple_of(64);
    println!(
        "planet raw_voxels: {}x{}x{} ({:.1} MiB)",
        raw_minor_size,
        raw_minor_size,
        raw_major_size,
        (raw_minor_size as f64 * raw_minor_size as f64 * raw_major_size as f64 * 4.0)
            / (1024.0 * 1024.0)
    );

    let tex_raw_voxels = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_raw_voxels"),
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

    // Update params with the raw_voxels chunk-grid stride.
    params.raw_minor_chunks = raw_minor_size / 64;
    queue.write_buffer(&buffer_params, 0, bytemuck::bytes_of(&params));

    // Bind group for the fill pass: real raw_voxels view.
    let bg_fill = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_fill_bg"),
        layout: &bg_layout_planet,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_params.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                // dummy on the write slot — fill only reads
                resource: wgpu::BindingResource::TextureView(
                    &tex_chunk_indices_placeholder.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_chunk_indices.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &tex_raw_voxels.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry { binding: 4, resource: buffer_chunk_counter.as_entire_binding() },
        ],
    });

    // ---- pass 2: fill voxels ----
    let timer = std::time::Instant::now();
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet_fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_fill);
            pass.set_bind_group(0, Some(&bg_fill), &[]);
            // workgroup_size (4,4,4) covers a 64^3 chunk via 16x16x16 inner loop per thread.
            pass.dispatch_workgroups(chunks_per_axis, chunks_per_axis, chunks_per_axis);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    println!("planet fill_voxels in {:?}", timer.elapsed());

    // ---- now run the standard tree-build passes on the populated raw textures ----
    let palette = build_planet_palette(device, queue);

    let model = build_tree(
        device,
        queue,
        cfg.name.clone(),
        cfg.voxels_per_unit,
        bounding_size,
        index_levels,
        world_size,
        &tex_raw_chunk_indices,
        &tex_raw_voxels,
        palette,
    )?;

    Ok(model)
}

// ---------------------------------------------------------------------------
// Tree-build orchestration. This mirrors the second half of `voxelize_gltf`
// in models.rs (which builds the sparse 4^3 tree from raw_chunk_indices +
// raw_voxels via tree_64.wgsl), but is duplicated here to keep this module
// self-contained and avoid touching the working glTF pipeline.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
struct AllocatorResults {
    index_chunk_count: u32,
    index_leaf_count: u32,
    voxel_count: u32,
}

fn build_tree(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    name: String,
    voxels_per_unit: u32,
    bounding_size: u32,
    index_levels: u32,
    size: glam::UVec3,
    tex_raw_chunk_indices: &wgpu::Texture,
    tex_raw_voxels: &wgpu::Texture,
    palette: PlanetPalette,
) -> Result<VoxelModel> {
    // ---- bind group layouts (must match generate/src/shaders/tree_64.wgsl) ----
    let bg_layout_tree_data = device.layout(
        "planet_tree_data",
        wgpu::ShaderStages::COMPUTE,
        (
            storage_buffer().read_write(),
            storage_buffer().read_write(),
            storage_buffer().read_write(),
            storage_texture().rgba32uint().dimension_3d().read_only(),
            storage_texture().rgba32uint().dimension_3d().write_only(),
        ),
    );
    let bg_layout_tree_alloc = device.layout(
        "planet_tree_alloc",
        wgpu::ShaderStages::COMPUTE,
        (
            storage_buffer().read_write(),
            storage_texture().r32uint().dimension_3d().read_only(),
            storage_texture().r32uint().dimension_3d().read_only(),
            uniform_buffer(),
        ),
    );
    let shader_tree = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tree_64"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tree_64.wgsl").into()),
    });
    let pipeline_tree_leaf = device
        .compute_pipeline("planet_tree_leaf", &shader_tree)
        .entry_point("compute_leaf")
        .layout(&[&bg_layout_tree_data, &bg_layout_tree_alloc]);
    let pipeline_tree_index = device
        .compute_pipeline("planet_tree_index", &shader_tree)
        .entry_point("compute_index")
        .layout(&[&bg_layout_tree_data, &bg_layout_tree_alloc]);

    // ---- output buffer sizes (mirror voxelize_gltf) ----
    let raw_chunk_size = size.map(|x| x.div_ceil(64));
    let raw_chunks_total = raw_chunk_size.element_product() as u64;
    let max_voxels = raw_chunks_total * 64 * 64 * 64; // upper bound
    let max_leaf_chunks_bytes =
        u64::min(MAX_STORAGE_BUFFER_BINDING_SIZE as u64, max_voxels * 4);

    let mut max_index_chunks: u64 = 0;
    for k in 1..=index_levels {
        let s = (bounding_size >> (2 * k)) as u64;
        max_index_chunks += s * s * s;
    }
    let max_index_chunks_bytes = u64::min(
        (MAX_STORAGE_BUFFER_BINDING_SIZE as u64 - 11).next_multiple_of(12),
        max_index_chunks * 12,
    );
    let max_index_chunks_eff = max_index_chunks_bytes / 12;
    let max_index_leaf_positions_bytes = u64::min(
        (MAX_STORAGE_BUFFER_BINDING_SIZE as u64 - 7).next_multiple_of(8),
        max_index_chunks_eff * 8,
    );

    let buffer_index_chunks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_index_chunks"),
        size: max_index_chunks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_leaf_chunks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_leaf_chunks"),
        size: max_leaf_chunks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer_index_leaf_positions = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_index_leaf_positions"),
        size: max_index_leaf_positions_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let tex_index_map_a = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet_index_map_a"),
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
        label: Some("planet_index_map_b"),
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

    let bg_tree_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_tree_data_a"),
        layout: &bg_layout_tree_data,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_index_chunks.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer_leaf_chunks.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_index_leaf_positions.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&tex_index_map_a.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&tex_index_map_b.create_view(&Default::default())),
            },
        ],
    });
    let bg_tree_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_tree_data_b"),
        layout: &bg_layout_tree_data,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_index_chunks.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer_leaf_chunks.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_index_leaf_positions.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&tex_index_map_b.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&tex_index_map_a.create_view(&Default::default())),
            },
        ],
    });

    let buffer_alloc_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("planet_alloc_data"),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        contents: bytemuck::cast_slice(&[AllocatorResults::default()]),
    });
    let buffer_alloc_results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_alloc_results"),
        size: buffer_alloc_data.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bg_tree_alloc = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("planet_tree_alloc_bg"),
        layout: &bg_layout_tree_alloc,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_alloc_data.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tex_raw_chunk_indices.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&tex_raw_voxels.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry { binding: 3, resource: palette.buffer_palette.as_entire_binding() },
        ],
    });

    // ---- run leaf + index passes ----
    let timer = std::time::Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet_tree_leaf"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_tree_leaf);
        pass.set_bind_group(0, Some(&bg_tree_a), &[]);
        pass.set_bind_group(1, Some(&bg_tree_alloc), &[]);
        let s = bounding_size >> 4;
        pass.dispatch_workgroups(s, s, s);
    }
    for k in 2..index_levels {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet_tree_index"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_tree_index);
        pass.set_bind_group(
            0,
            Some(match k & 1 {
                0 => &bg_tree_b,
                _ => &bg_tree_a,
            }),
            &[],
        );
        pass.set_bind_group(1, Some(&bg_tree_alloc), &[]);
        let s = bounding_size >> (2 + 2 * k);
        pass.dispatch_workgroups(s, s, s);
    }
    encoder.copy_buffer_to_buffer(&buffer_alloc_data, 0, &buffer_alloc_results, 0, buffer_alloc_data.size());
    queue.submit([encoder.finish()]);
    println!("planet tree-build encoder submitted in {:?}", timer.elapsed());

    // ---- read alloc results ----
    let alloc_results = {
        buffer_alloc_results.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = buffer_alloc_results.get_mapped_range(..);
        let r = bytemuck::cast_slice::<_, AllocatorResults>(&view)[0];
        drop(view);
        buffer_alloc_results.unmap();
        r
    };
    println!("planet alloc_results: {:?}", alloc_results);

    if (alloc_results.voxel_count as u64) >= max_leaf_chunks_bytes / 4 {
        bail!("planet voxel_count {} exceeded leaf buffer", alloc_results.voxel_count);
    }
    if (alloc_results.index_chunk_count as u64) >= max_index_chunks_eff {
        bail!("planet index_chunk_count {} exceeded index buffer", alloc_results.index_chunk_count);
    }

    // ---- compact-copy output buffers (mirrors pack_tree) ----
    let allocated_index_chunk_bytes = (alloc_results.index_chunk_count as u64) * 12;
    let buf_index_chunks_cpct = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_index_chunks_cpct"),
        size: allocated_index_chunk_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let allocated_index_leaf_positions_bytes = (alloc_results.index_leaf_count as u64 + 1) * 8;
    let buf_index_leaf_positions_cpct = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_index_leaf_positions_cpct"),
        size: allocated_index_leaf_positions_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let allocated_leaf_chunk_bytes = (alloc_results.voxel_count as u64) * 4;
    let buf_leaf_chunks_cpct = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet_leaf_chunks_cpct"),
        size: allocated_leaf_chunk_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&buffer_index_chunks, 0, &buf_index_chunks_cpct, 0, Some(allocated_index_chunk_bytes));
    encoder.copy_buffer_to_buffer(&buffer_index_leaf_positions, 0, &buf_index_leaf_positions_cpct, 0, Some(allocated_index_leaf_positions_bytes));
    encoder.copy_buffer_to_buffer(&buffer_leaf_chunks, 0, &buf_leaf_chunks_cpct, 0, Some(allocated_leaf_chunk_bytes));
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    buffer_index_chunks.destroy();
    buffer_leaf_chunks.destroy();
    buffer_index_leaf_positions.destroy();

    Ok(VoxelModel {
        meta: VoxelMetadata {
            name,
            voxels_per_unit,
            bounding_size,
            size,
            index_levels,
            voxel_count: alloc_results.voxel_count,
            allocated_index_chunks: alloc_results.index_chunk_count,
        },
        data: VoxelBufferData {
            buffer_palette: palette.buffer_palette,
            buffer_index_chunks: buf_index_chunks_cpct,
            buffer_leaf_chunks: buf_leaf_chunks_cpct,
            buffer_index_leaf_positions: buf_index_leaf_positions_cpct,
        },
    })
}

// ---------------------------------------------------------------------------
// Procedural palette (16 hand-picked terrain colors padded to 1024).
// Indices are referenced by planet_fill.wgsl via PlanetParams.pal_*.
// ---------------------------------------------------------------------------

struct PlanetPalette {
    buffer_palette: wgpu::Buffer,
}

const PAL_ROCK: u32 = 1;
const PAL_GRASS: u32 = 2;
const PAL_SAND: u32 = 3;
const PAL_SNOW: u32 = 4;
const PAL_DIRT: u32 = 5;
const PAL_WATER: u32 = 6;
const PAL_LAVA: u32 = 7;

fn build_planet_palette(device: &wgpu::Device, _queue: &wgpu::Queue) -> PlanetPalette {
    // 1024-entry palette, vec4<f32> each. Index 0 reserved for "empty"
    // (never sampled at runtime since palette_index is only read for
    // non-empty voxels, but kept zero for safety).
    let mut rgba = vec![glam::Vec4::ZERO; 1024];
    let set = |rgba: &mut Vec<glam::Vec4>, i: u32, srgb: [f32; 3]| {
        // Engine palette stores linear RGB.
        let lin = srgb_to_linear(glam::Vec3::from(srgb));
        rgba[i as usize] = lin.extend(1.0);
    };
    set(&mut rgba, PAL_ROCK, [0.45, 0.42, 0.38]);
    set(&mut rgba, PAL_GRASS, [0.20, 0.45, 0.18]);
    set(&mut rgba, PAL_SAND, [0.85, 0.78, 0.55]);
    set(&mut rgba, PAL_SNOW, [0.95, 0.96, 0.98]);
    set(&mut rgba, PAL_DIRT, [0.42, 0.30, 0.20]);
    set(&mut rgba, PAL_WATER, [0.10, 0.30, 0.55]);
    set(&mut rgba, PAL_LAVA, [1.00, 0.40, 0.10]);

    let buffer_palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("planet_palette"),
        contents: bytemuck::cast_slice(&rgba),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });
    PlanetPalette { buffer_palette }
}

fn srgb_to_linear(srgb: glam::Vec3) -> glam::Vec3 {
    let f = |c: f32| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    glam::vec3(f(srgb.x), f(srgb.y), f(srgb.z))
}

// ---------------------------------------------------------------------------
// Uniform layout shared with planet_fill.wgsl.
// std140 alignment: vec3 followed by f32 packs naturally.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct PlanetParams {
    bounding_size: u32,
    chunks_per_axis: u32,
    raw_minor_chunks: u32,
    seed: u32,

    center: [f32; 3],
    radius: f32,

    amplitude: f32,
    base_freq: f32,
    octaves: u32,
    crust_thickness: f32,

    sea_level: f32,
    snow_height: f32,
    sand_band: f32,
    _pad0: f32,

    pal_rock: u32,
    pal_grass: u32,
    pal_sand: u32,
    pal_snow: u32,
    pal_dirt: u32,
    pal_water: u32,
    pal_lava: u32,
    _pad1: u32,
}

impl PlanetParams {
    fn from_config(
        cfg: &PlanetConfig,
        bounding_size: u32,
        chunks_per_axis: u32,
        center: glam::Vec3,
        raw_minor_chunks: u32,
    ) -> Self {
        Self {
            bounding_size,
            chunks_per_axis,
            raw_minor_chunks,
            seed: cfg.seed,
            center: [center.x, center.y, center.z],
            radius: cfg.radius,
            amplitude: cfg.amplitude,
            base_freq: cfg.base_freq,
            octaves: cfg.octaves,
            crust_thickness: cfg.crust_thickness,
            sea_level: cfg.sea_level,
            snow_height: cfg.snow_height,
            sand_band: cfg.sand_band,
            _pad0: 0.0,
            pal_rock: PAL_ROCK,
            pal_grass: PAL_GRASS,
            pal_sand: PAL_SAND,
            pal_snow: PAL_SNOW,
            pal_dirt: PAL_DIRT,
            pal_water: PAL_WATER,
            pal_lava: PAL_LAVA,
            _pad1: 0,
        }
    }
}

const fn next_pow_4(x: u32) -> u32 {
    let mut k = 1u32;
    while k < x {
        k <<= 2;
    }
    k
}
