use std::num::NonZero;

use wgpu::TextureFormat;

pub trait DeviceUtils {
    fn layout(&self, label: &str, entries: impl IntoEntries) -> wgpu::BindGroupLayout;
}
impl DeviceUtils for wgpu::Device {
    fn layout(&self, label: &str, entries: impl IntoEntries) -> wgpu::BindGroupLayout {
        self.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries.into_entries(),
        })
    }
}

trait BindingType {
    fn into_base(&self) -> wgpu::BindingType;
}

pub struct LayoutEntry<T: BindingType> {
    visibility: wgpu::ShaderStages,
    ty: T,
    count: Option<NonZero<u32>>,
}
impl<T: BindingType> LayoutEntry<T> {
    pub fn visible_vertex(mut self) -> Self {
        self.visibility |= wgpu::ShaderStages::VERTEX;
        self
    }
    pub fn visible_fragment(mut self) -> Self {
        self.visibility |= wgpu::ShaderStages::FRAGMENT;
        self
    }
    pub fn visible_compute(mut self) -> Self {
        self.visibility |= wgpu::ShaderStages::COMPUTE;
        self
    }
    pub fn count(mut self, count: NonZero<u32>) -> Self {
        self.count = Some(count);
        self
    }
}

pub struct Texture {
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
    multisampled: bool,
}

pub fn texture() -> LayoutEntry<Texture> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl BindingType for Texture {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Texture {
            sample_type: self.sample_type,
            view_dimension: self.view_dimension,
            multisampled: self.multisampled,
        }
    }
}

impl LayoutEntry<Texture> {
    pub fn sample_float(mut self, filterable: bool) -> Self {
        self.ty.sample_type = wgpu::TextureSampleType::Float { filterable };
        self
    }
    pub fn sample_depth(mut self) -> Self {
        self.ty.sample_type = wgpu::TextureSampleType::Depth;
        self
    }
    pub fn sample_uint(mut self) -> Self {
        self.ty.sample_type = wgpu::TextureSampleType::Uint;
        self
    }
    pub fn sample_sint(mut self) -> Self {
        self.ty.sample_type = wgpu::TextureSampleType::Sint;
        self
    }

    pub fn dimension_1d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D1;
        self
    }
    pub fn dimension_2d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D2;
        self
    }
    pub fn dimension_3d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D3;
        self
    }
    pub fn dimension_2d_array(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D2Array;
        self
    }
    pub fn dimension_cube(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::Cube;
        self
    }
    pub fn dimension_cube_array(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::CubeArray;
        self
    }

    pub fn multisampled(mut self, value: bool) -> Self {
        self.ty.multisampled = value;
        self
    }
}

pub struct StorageTexture {
    access: wgpu::StorageTextureAccess,
    format: wgpu::TextureFormat,
    view_dimension: wgpu::TextureViewDimension,
}

pub fn storage_texture(format: TextureFormat) -> LayoutEntry<StorageTexture> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

impl BindingType for StorageTexture {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::StorageTexture {
            access: self.access,
            format: self.format,
            view_dimension: self.view_dimension,
        }
    }
}

impl LayoutEntry<StorageTexture> {
    pub fn read_only(mut self) -> Self {
        self.ty.access = wgpu::StorageTextureAccess::ReadOnly;
        self
    }
    pub fn write_only(mut self) -> Self {
        self.ty.access = wgpu::StorageTextureAccess::WriteOnly;
        self
    }
    pub fn read_write(mut self) -> Self {
        self.ty.access = wgpu::StorageTextureAccess::ReadWrite;
        self
    }
    pub fn atomic(mut self) -> Self {
        self.ty.access = wgpu::StorageTextureAccess::Atomic;
        self
    }

    pub fn dimension_1d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D1;
        self
    }
    pub fn dimension_2d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D2;
        self
    }
    pub fn dimension_3d(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D3;
        self
    }
    pub fn dimension_2d_array(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::D2Array;
        self
    }
    pub fn dimension_cube(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::Cube;
        self
    }
    pub fn dimension_cube_array(mut self) -> Self {
        self.ty.view_dimension = wgpu::TextureViewDimension::CubeArray;
        self
    }
}

pub struct UniformBuffer {
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
}

pub fn uniform_buffer() -> LayoutEntry<UniformBuffer> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: UniformBuffer {
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl BindingType for UniformBuffer {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: self.has_dynamic_offset,
            min_binding_size: self.min_binding_size,
        }
    }
}

impl LayoutEntry<UniformBuffer> {
    pub fn dynamic_offset(mut self, value: bool) -> Self {
        self.ty.has_dynamic_offset = value;
        self
    }
    pub fn min_binding_size(mut self, value: NonZero<u64>) -> Self {
        self.ty.min_binding_size = Some(value);
        self
    }
}

pub struct StorageBuffer {
    read_only: bool,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
}

pub fn storage_buffer() -> LayoutEntry<StorageBuffer> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: StorageBuffer {
            read_only: false,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl BindingType for StorageBuffer {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: self.read_only,
            },
            has_dynamic_offset: self.has_dynamic_offset,
            min_binding_size: self.min_binding_size,
        }
    }
}

impl LayoutEntry<StorageBuffer> {
    pub fn read_only(mut self, value: bool) -> Self {
        self.ty.read_only = value;
        self
    }
    pub fn dynamic_offset(mut self, value: bool) -> Self {
        self.ty.has_dynamic_offset = value;
        self
    }
    pub fn min_binding_size(mut self, value: NonZero<u64>) -> Self {
        self.ty.min_binding_size = Some(value);
        self
    }
}

pub struct Sampler(wgpu::SamplerBindingType);

pub fn sampler() -> LayoutEntry<Sampler> {
    LayoutEntry {
        visibility: wgpu::ShaderStages::NONE,
        ty: Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl BindingType for Sampler {
    fn into_base(&self) -> wgpu::BindingType {
        wgpu::BindingType::Sampler(self.0)
    }
}

impl LayoutEntry<Sampler> {
    pub fn filtering(mut self) -> Self {
        self.ty.0 = wgpu::SamplerBindingType::Filtering;
        self
    }
    pub fn non_filtering(mut self) -> Self {
        self.ty.0 = wgpu::SamplerBindingType::NonFiltering;
        self
    }
    pub fn comparison(mut self) -> Self {
        self.ty.0 = wgpu::SamplerBindingType::Comparison;
        self
    }
}

pub trait IntoEntries {
    fn into_entries(self) -> Vec<wgpu::BindGroupLayoutEntry>;
}
impl<T: BindingType> IntoEntries for LayoutEntry<T> {
    fn into_entries(self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: self.visibility,
            ty: self.ty.into_base(),
            count: self.count,
        }]
    }
}
macro_rules! impl_into_entries {
    ($($name:ident),* ) => {
        #[allow(non_snake_case)]
        impl<$($name: BindingType),*> IntoEntries for ($(LayoutEntry<$name>,)*) {
            fn into_entries(self) -> Vec<wgpu::BindGroupLayoutEntry> {
                let ($($name,)*) = self;
                let mut _i = 0;
                vec![$(
                    wgpu::BindGroupLayoutEntry {
                        binding: {
                            let cur = _i;
                            _i += 1;
                            cur
                        },
                        visibility: $name.visibility,
                        ty: $name.ty.into_base(),
                        count: $name.count,
                    }
                ),*]
            }
        }
    }
}
impl_into_entries!(T1);
impl_into_entries!(T1, T2);
impl_into_entries!(T1, T2, T3);
impl_into_entries!(T1, T2, T3, T4);
impl_into_entries!(T1, T2, T3, T4, T5);
impl_into_entries!(T1, T2, T3, T4, T5, T6);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
impl_into_entries!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
impl_into_entries!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15
);
impl_into_entries!(
    T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
);
