#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneDataBuffer {
    size: [u32; 3],
    _pad: u32,
    palette: [u32; 256],
}

impl SceneDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
        let mut palette = [0; 256];
        for (i, mat) in scene.palette.iter().enumerate() {
            let rgba = mat.rgba;
            palette[i] = (rgba[0] as u32) << 24
                | (rgba[1] as u32) << 16
                | (rgba[2] as u32) << 8
                | rgba[3] as u32;
        }
        Self {
            size: [
                scene.size.x as u32,
                scene.size.y as u32,
                scene.size.z as u32,
            ],
            _pad: 0,
            palette,
        }
    }
}

pub struct VoxelDataBuffer(pub Vec<u32>);

impl VoxelDataBuffer {
    pub fn new(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut voxels = vec![0; (scene.size.element_product() as usize + 3) >> 2];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;

                voxels[index >> 2] |=
                    (palette_index as u32) << (3 - (index as i64) & 0x3) as usize * 8;
            }
        }

        println!("voxel data load took {:#?}", timer.elapsed());
        Self(voxels)
    }
}
