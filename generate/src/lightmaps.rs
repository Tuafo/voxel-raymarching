use anyhow::{Context, Result, bail};
use image::EncodableLayout;
use serde::{Deserialize, Serialize};
use std::{
    array,
    f64::consts::PI,
    io::{BufRead, Cursor, Read, Seek},
};
use wgpu::{BufferSlice, util::DeviceExt};

#[derive(Debug)]
pub struct LightmapResult {
    pub cubemap: wgpu::Texture,
    pub downsampled: wgpu::Texture,
    pub prefilter: wgpu::Texture,
    pub brdf: wgpu::Texture,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LightmapHeader {
    pub name: String,
    hdr: ExternalTextureView,
    downsampled: SerializedTextureView,
    prefilter: Vec<SerializedTextureView>,
    brdf: SerializedTextureView,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ExternalTextureView {
    start: usize,
    end: usize,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerializedTextureView {
    start: usize,
    end: usize,
    mip_level: u32,
    size: glam::UVec3,
    bytes_per_row_padded: u32,
}
impl SerializedTextureView {
    /// Pads and copies a texture into `buf`. Each mip of the texture returns an individual view.
    ///
    /// Polls the device here, so not super efficient for lots of calls. Just used in build step rn though.
    fn from_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        buf: &mut Vec<u8>,
    ) -> Result<Vec<Self>> {
        let bytes_per_texel = texture
            .format()
            .block_copy_size(None)
            .context("couldn't determine block copy size")?;

        let mut buffers = Vec::new();
        let mut res = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());

        for i in 0..texture.mip_level_count() {
            let size = glam::uvec3(
                (texture.width() >> i).max(1),
                (texture.height() >> i).max(1),
                texture.depth_or_array_layers(),
            );
            let bytes_per_row_padded = (size.x * bytes_per_texel + 255) & !255;
            let bytes_padded = (bytes_per_row_padded * size.y * size.z) as u64;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("texture copy buffer"),
                size: bytes_padded,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: i,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row_padded),
                        rows_per_image: Some(size.y),
                    },
                },
                wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: size.z,
                },
            );
            buffers.push(buffer);
            res.push(Self {
                start: 0,
                end: 0,
                mip_level: i,
                size,
                bytes_per_row_padded,
            })
        }

        queue.submit([encoder.finish()]);

        let views = buffers
            .iter()
            .map(|buffer| {
                let view = buffer.slice(..);
                view.map_async(wgpu::MapMode::Read, |_| {});
                view
            })
            .collect::<Vec<BufferSlice<'_>>>();

        device.poll(wgpu::PollType::wait_indefinitely())?;

        for i in 0..texture.mip_level_count() {
            let view = views[i as usize];

            let data = view.get_mapped_range();
            res[i as usize].start = buf.len();
            res[i as usize].end = buf.len() + data.len();
            buf.extend_from_slice(&data);
        }

        Ok(res)
    }

    fn deserialize(
        &self,
        src: &[u8],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("staging_buffer"),
            contents: &src[self.start..self.end],
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let size = wgpu::Extent3d {
            width: self.size.x,
            height: self.size.y,
            depth_or_array_layers: self.size.z,
        };

        let mut encoder = device.create_command_encoder(&Default::default());

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row_padded),
                    rows_per_image: Some(self.size.y),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: self.mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );

        queue.submit([encoder.finish()]);
    }
}

pub fn load_lightmap_header<R: Read>(data: &mut R) -> Result<LightmapHeader> {
    let mut buf = [0; 4];
    data.read_exact(&mut buf)?;
    let header_length = bytemuck::cast_slice::<u8, u32>(&buf)[0] as usize;
    let mut buf = vec![0; header_length as usize];
    data.read_exact(&mut buf)?;
    let header: LightmapHeader = serde_json::from_slice(&buf)?;
    Ok(header)
}

impl LightmapResult {
    pub fn serialize<R: BufRead + Seek>(
        &self,
        name: &str,
        hdr: &mut R,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>> {
        let mut buf = Vec::new();

        hdr.read_to_end(&mut buf)?;
        let hdr = ExternalTextureView {
            start: 0,
            end: buf.len(),
        };

        let downsampled =
            SerializedTextureView::from_texture(device, queue, &self.downsampled, &mut buf)?[0]
                .clone();
        let prefilter =
            SerializedTextureView::from_texture(device, queue, &self.prefilter, &mut buf)?;
        let brdf =
            SerializedTextureView::from_texture(device, queue, &self.brdf, &mut buf)?[0].clone();

        let header = LightmapHeader {
            name: String::from(name),
            hdr,
            downsampled,
            prefilter,
            brdf,
        };

        let header = serde_json::to_vec(&header)?;
        let mut res: Vec<u8> = bytemuck::cast_slice(&[header.len() as u32]).to_vec();
        res.extend_from_slice(&header);
        res.extend_from_slice(&buf);

        Ok(res)
    }

    pub fn deserialize(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            bail!("Invalid lightmap")
        }

        let header_length = bytemuck::cast_slice::<u8, u32>(&data[0..4])[0] as usize;
        let header: LightmapHeader = serde_json::from_slice(&data[4..(4 + header_length)])?;

        let buf = &data[4 + header_length..];

        let hdr_src = Cursor::new(&buf[header.hdr.start..header.hdr.end]);

        let res = Self {
            cubemap: generate_skybox(hdr_src, device, queue)?,
            downsampled: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("downsampled"),
                size: wgpu::Extent3d {
                    width: header.downsampled.size.x,
                    height: header.downsampled.size.y,
                    depth_or_array_layers: header.downsampled.size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
            prefilter: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("prefilter"),
                size: wgpu::Extent3d {
                    width: header.prefilter[0].size.x,
                    height: header.prefilter[0].size.y,
                    depth_or_array_layers: header.prefilter[0].size.z,
                },
                mip_level_count: header.prefilter.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
            brdf: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("brdf"),
                size: wgpu::Extent3d {
                    width: header.brdf.size.x,
                    height: header.brdf.size.y,
                    depth_or_array_layers: header.brdf.size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        };

        header
            .downsampled
            .deserialize(&buf, device, queue, &res.downsampled);
        for mip in header.prefilter {
            mip.deserialize(&buf, device, queue, &res.prefilter);
        }
        header.brdf.deserialize(&buf, device, queue, &res.brdf);

        Ok(res)
    }
}

struct BindGroupLayouts {
    cubemap: wgpu::BindGroupLayout,
    downsampled: wgpu::BindGroupLayout,
    prefilter: wgpu::BindGroupLayout,
    brdf: wgpu::BindGroupLayout,
}

struct BindGroups {
    cubemap: wgpu::BindGroup,
    downsampled: wgpu::BindGroup,
    prefilter: [wgpu::BindGroup; PREFILTER_LEVELS as usize],
    brdf: wgpu::BindGroup,
}

struct Pipelines {
    cubemap: wgpu::ComputePipeline,
    downsampled: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
    brdf: wgpu::ComputePipeline,
}

struct Resources {
    sampler: wgpu::Sampler,
    tex_hdr: wgpu::Texture,
    tex_cubemap: wgpu::Texture,
    tex_downsampled: wgpu::Texture,
    tex_prefilter: wgpu::Texture,
    tex_brdf_lut: wgpu::Texture,
}

const SKYBOX_RESOLUTION: u32 = 2048;
const SKYBOX_MAX_COMPONENT: f32 = 50.0;
const DOWNSAMPLED_RESOLUTION: u32 = 256;
const DOWNSAMPLE_STEP: f64 = 0.001;
const DOWNSAMPLE_RADIUS: f64 = 0.05;
const PREFILTER_RESOLUTION: u32 = 256;
const PREFILTER_LEVELS: u32 = 5;
const PREFILTER_SAMPLES: u32 = 4096;
const BRDF_LUT_RESOLUTION: u32 = 1024;
const BRDF_LUT_SAMPLES: u32 = 2048;

pub fn generate_skybox<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
    let tex_hdr = load_hdr(src, device, queue)?;

    let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cubemap"),
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
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cubemap"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cubemap"),
                bind_group_layouts: &[&bg_layout],
                immediate_size: 0,
            }),
        ),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cubemap"),
            source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/cubemap.wgsl").into()),
        }),
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let res = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cubemap"),
        size: wgpu::Extent3d {
            width: SKYBOX_RESOLUTION,
            height: SKYBOX_RESOLUTION,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cubemap"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &tex_hdr.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&res.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cubemap"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.insert_debug_marker("cubemap");
        pass.dispatch_workgroups(
            SKYBOX_RESOLUTION.div_ceil(8),
            SKYBOX_RESOLUTION.div_ceil(8),
            6,
        );
    }
    queue.submit([encoder.finish()]);

    Ok(res)
}

pub fn generate_lighting<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<LightmapResult> {
    let tex_hdr = load_hdr(src, device, queue)?;

    let bg_layouts = BindGroupLayouts {
        cubemap: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cubemap"),
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }),
        downsampled: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsampled"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }),
        prefilter: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefilter"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        brdf: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brdf"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        }),
    };

    let pipelines = Pipelines {
        cubemap: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cubemap"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cubemap"),
                    bind_group_layouts: &[&bg_layouts.cubemap],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cubemap"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/cubemap.wgsl").into()),
            }),
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        }),
        downsampled: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("downsampled"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("downsampled"),
                    bind_group_layouts: &[&bg_layouts.downsampled],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("downsampled"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("shaders/downsample.wgsl").into(),
                ),
            }),
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("delta", DOWNSAMPLE_STEP), ("radius", DOWNSAMPLE_RADIUS)],
                ..Default::default()
            },
            cache: None,
        }),
        prefilter: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefilter"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("prefilter"),
                    bind_group_layouts: &[&bg_layouts.prefilter],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prefilter"),
                source: wgpu::ShaderSource::Wgsl(
                    std::include_str!("shaders/prefilter.wgsl").into(),
                ),
            }),
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        }),
        brdf: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("brdf"),
                    bind_group_layouts: &[&bg_layouts.brdf],
                    immediate_size: 0,
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("brdf"),
                source: wgpu::ShaderSource::Wgsl(std::include_str!("shaders/brdf.wgsl").into()),
            }),
            entry_point: Some("compute_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("sample_count", BRDF_LUT_SAMPLES as f64)],
                ..Default::default()
            },
            cache: None,
        }),
    };

    let resources = Resources {
        sampler: device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
        tex_hdr: tex_hdr,
        tex_cubemap: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cubemap"),
            size: wgpu::Extent3d {
                width: SKYBOX_RESOLUTION,
                height: SKYBOX_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_downsampled: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("downsampled"),
            size: wgpu::Extent3d {
                width: DOWNSAMPLED_RESOLUTION,
                height: DOWNSAMPLED_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_prefilter: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefilter"),
            size: wgpu::Extent3d {
                width: PREFILTER_RESOLUTION,
                height: PREFILTER_RESOLUTION,
                depth_or_array_layers: 6,
            },
            mip_level_count: PREFILTER_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        tex_brdf_lut: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_RESOLUTION,
                height: BRDF_LUT_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
    };

    let bind_groups = BindGroups {
        cubemap: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cubemap"),
            layout: &bg_layouts.cubemap,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_hdr.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_cubemap.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        }),
        downsampled: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("downsampled"),
            layout: &bg_layouts.downsampled,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &resources
                            .tex_cubemap
                            .create_view(&wgpu::TextureViewDescriptor {
                                label: Some("skybox"),
                                dimension: Some(wgpu::TextureViewDimension::Cube),
                                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                ..Default::default()
                            }),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &resources.tex_downsampled.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        }),
        prefilter: array::from_fn(|i| {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PrefilterData {
                roughness: f32,
                sample_count: u32,
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefilter"),
                contents: bytemuck::cast_slice(&[PrefilterData {
                    roughness: (i as f32) / (PREFILTER_LEVELS as f32 - 1.0),
                    sample_count: PREFILTER_SAMPLES,
                }]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("prefilter_{}", i)),
                layout: &bg_layouts.prefilter,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &resources
                                .tex_cubemap
                                .create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("skybox"),
                                    dimension: Some(wgpu::TextureViewDimension::Cube),
                                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &resources
                                .tex_prefilter
                                .create_view(&wgpu::TextureViewDescriptor {
                                    base_mip_level: i as u32,
                                    mip_level_count: Some(1),
                                    ..Default::default()
                                }),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&resources.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            })
        }),
        brdf: device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brdf"),
            layout: &bg_layouts.brdf,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &resources.tex_brdf_lut.create_view(&Default::default()),
                ),
            }],
        }),
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cubemap"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cubemap"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.cubemap);
        pass.set_bind_group(0, &bind_groups.cubemap, &[]);
        pass.insert_debug_marker("cubemap");
        pass.dispatch_workgroups(
            SKYBOX_RESOLUTION.div_ceil(8),
            SKYBOX_RESOLUTION.div_ceil(8),
            6,
        );
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("irradiance"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.downsampled);
        pass.set_bind_group(0, &bind_groups.downsampled, &[]);
        pass.insert_debug_marker("irradiance");
        pass.dispatch_workgroups(
            DOWNSAMPLED_RESOLUTION.div_ceil(8),
            DOWNSAMPLED_RESOLUTION.div_ceil(8),
            6,
        );
    }

    for i in 0..PREFILTER_LEVELS {
        let mip_resolution = PREFILTER_RESOLUTION >> i;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("prefilter_{}", i)),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.prefilter);
        pass.set_bind_group(0, &bind_groups.prefilter[i as usize], &[]);
        pass.insert_debug_marker("prefilter");
        pass.dispatch_workgroups(mip_resolution.div_ceil(8), mip_resolution.div_ceil(8), 6);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brdf"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.brdf);
        pass.set_bind_group(0, &bind_groups.brdf, &[]);
        pass.insert_debug_marker("brdf");
        pass.dispatch_workgroups(
            BRDF_LUT_RESOLUTION.div_ceil(8),
            BRDF_LUT_RESOLUTION.div_ceil(8),
            1,
        );
    }

    queue.submit([encoder.finish()]);

    Ok(LightmapResult {
        cubemap: resources.tex_cubemap,
        downsampled: resources.tex_downsampled,
        prefilter: resources.tex_prefilter,
        brdf: resources.tex_brdf_lut,
    })
}

fn load_hdr<R: BufRead + Seek>(
    src: R,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
    let img = image::load(src, image::ImageFormat::Hdr)?;
    let mut src = img.into_rgba32f();
    src.pixels_mut().for_each(|pixel| {
        pixel.0[0] = pixel.0[0].min(SKYBOX_MAX_COMPONENT);
        pixel.0[1] = pixel.0[1].min(SKYBOX_MAX_COMPONENT);
        pixel.0[2] = pixel.0[2].min(SKYBOX_MAX_COMPONENT);
    });

    let tex_hdr = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr"),
        size: wgpu::Extent3d {
            width: src.width(),
            height: src.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_hdr,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &src.as_bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16 * src.width()),
            rows_per_image: Some(src.height()),
        },
        wgpu::Extent3d {
            width: src.width(),
            height: src.height(),
            depth_or_array_layers: 1,
        },
    );

    Ok(tex_hdr)
}
