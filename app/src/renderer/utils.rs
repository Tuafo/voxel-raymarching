#[derive(Debug)]
pub struct SwapchainTexture {
    pub a: wgpu::Texture,
    pub b: wgpu::Texture,
}

#[derive(Debug)]
pub struct SwapchainBindGroup {
    pub a: wgpu::BindGroup,
    pub b: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct SwapchainBindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub layout: &'a wgpu::BindGroupLayout,
    pub entries: &'a [SwapchainBindGroupEntry<'a>],
}

#[derive(Debug)]
pub struct SwapchainBindGroupEntry<'a> {
    pub binding: u32,
    pub resource: SwapchainBindingResource<'a>,
}

#[derive(Debug)]
pub struct SwapchainTextureView {
    pub a: wgpu::TextureView,
    pub b: wgpu::TextureView,
}

#[derive(Debug)]
pub enum SwapchainBindingResource<'a> {
    Single(wgpu::BindingResource<'a>),
    Swap(wgpu::BindingResource<'a>, wgpu::BindingResource<'a>),
}

pub trait DeviceSwapExt {
    fn create_texture_swap(&self, desc: &wgpu::TextureDescriptor<'_>) -> SwapchainTexture;
    fn create_bind_group_swap(&self, desc: &SwapchainBindGroupDescriptor<'_>)
    -> SwapchainBindGroup;
}

impl DeviceSwapExt for wgpu::Device {
    fn create_texture_swap(&self, desc: &wgpu::TextureDescriptor<'_>) -> SwapchainTexture {
        SwapchainTexture {
            a: self.create_texture(desc),
            b: self.create_texture(desc),
        }
    }

    fn create_bind_group_swap(&self, desc: &SwapchainBindGroupDescriptor) -> SwapchainBindGroup {
        let entries_a = desc
            .entries
            .iter()
            .map(|e| wgpu::BindGroupEntry {
                binding: e.binding,
                resource: match &e.resource {
                    SwapchainBindingResource::Single(r) => r.clone(),
                    SwapchainBindingResource::Swap(r, _) => r.clone(),
                },
            })
            .collect::<Vec<wgpu::BindGroupEntry>>();
        let a = wgpu::BindGroupDescriptor {
            label: desc.label.clone(),
            layout: desc.layout,
            entries: &entries_a,
        };
        let entries_b = desc
            .entries
            .iter()
            .map(|e| wgpu::BindGroupEntry {
                binding: e.binding,
                resource: match &e.resource {
                    SwapchainBindingResource::Single(r) => r.clone(),
                    SwapchainBindingResource::Swap(_, r) => r.clone(),
                },
            })
            .collect::<Vec<wgpu::BindGroupEntry>>();
        let b = wgpu::BindGroupDescriptor {
            label: desc.label.clone(),
            layout: desc.layout,
            entries: &entries_b,
        };
        SwapchainBindGroup {
            a: self.create_bind_group(&a),
            b: self.create_bind_group(&b),
        }
    }
}

impl SwapchainTexture {
    pub fn create_view(&self, desc: &wgpu::TextureViewDescriptor) -> SwapchainTextureView {
        SwapchainTextureView {
            a: self.a.create_view(desc),
            b: self.b.create_view(desc),
        }
    }
}

impl SwapchainTextureView {
    // Set's this texture's primary swap on even frames, and its secondary swap on odd frames
    pub fn both(&self) -> SwapchainBindingResource {
        SwapchainBindingResource::Swap(
            wgpu::BindingResource::TextureView(&self.a),
            wgpu::BindingResource::TextureView(&self.b),
        )
    }

    // Set's this texture's secondary swap on even frames, and its primary swap on odd frames
    pub fn both_reversed(&self) -> SwapchainBindingResource {
        SwapchainBindingResource::Swap(
            wgpu::BindingResource::TextureView(&self.b),
            wgpu::BindingResource::TextureView(&self.a),
        )
    }
}

pub trait PassSwapExt {
    fn set_bind_group_swap<'a>(
        &mut self,
        index: u32,
        bind_group: &'a Option<SwapchainBindGroup>,
        offsets: &[wgpu::DynamicOffset],
        frame_id: u32,
    );
}

impl PassSwapExt for wgpu::ComputePass<'_> {
    fn set_bind_group_swap<'a>(
        &mut self,
        index: u32,
        bind_group: &'a Option<SwapchainBindGroup>,
        offsets: &[wgpu::DynamicOffset],
        frame_id: u32,
    ) {
        self.set_bind_group(
            index,
            match frame_id & 1 {
                0 => Some(&bind_group.as_ref().unwrap().a),
                _ => Some(&bind_group.as_ref().unwrap().b),
            },
            offsets,
        );
    }
}

impl PassSwapExt for wgpu::RenderPass<'_> {
    fn set_bind_group_swap<'a>(
        &mut self,
        index: u32,
        bind_group: &'a Option<SwapchainBindGroup>,
        offsets: &[wgpu::DynamicOffset],
        frame_id: u32,
    ) {
        self.set_bind_group(
            index,
            match frame_id & 1 {
                0 => Some(&bind_group.as_ref().unwrap().a),
                _ => Some(&bind_group.as_ref().unwrap().b),
            },
            offsets,
        );
    }
}
