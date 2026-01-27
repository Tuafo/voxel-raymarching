pub struct BrickMap {
    index_bounds: glam::UVec3,
    index: Vec<IndexChunk>,
    data: Vec<DataChunk>,
}

#[derive(Clone)]
struct IndexChunk {
    base_addr: [u32; 64],
    mask: u64,
}
impl Default for IndexChunk {
    fn default() -> Self {
        Self {
            base_addr: [0; 64],
            mask: 0,
        }
    }
}

struct DataChunk {
    data: [u8; 64],
}
impl Default for DataChunk {
    fn default() -> Self {
        Self { data: [0; 64] }
    }
}

fn root_index(bounds: glam::UVec3, pos: glam::UVec3) -> u32 {
    (pos.x >> 4) * bounds.y * bounds.z + (pos.y >> 4) * bounds.z + (pos.z >> 4)
}
fn brick_index(pos: glam::UVec3) -> u32 {
    ((pos.x >> 2) & 3) << 4 | ((pos.y >> 2) & 3) << 2 | ((pos.z >> 2) & 3)
}
fn base_index(pos: glam::UVec3) -> u32 {
    (pos.x & 3) << 4 | (pos.y & 3) << 2 | pos.z & 3
}

impl BrickMap {
    // pub const fn bitmask_lut() -> [u64; 64 * 8] {
    //     // for ray_dir in 0..8 {
    //     //     let dir = glam::
    //     // }
    // }
    //

    pub fn new(size: glam::UVec3) -> Self {
        let index_bounds = size.map(|x| (x + 15) >> 4);
        Self {
            index_bounds,
            index: vec![IndexChunk::default(); index_bounds.element_product() as usize],
            data: Vec::new(),
        }
    }

    pub fn get(&self, pos: glam::UVec3) -> u8 {
        let index_chunk = &self.index[root_index(self.index_bounds, pos) as usize];
        let brick_index = brick_index(pos);
        if index_chunk.mask & (1 << brick_index) == 0 {
            return 0;
        }
        self.data[index_chunk.base_addr[brick_index as usize] as usize].data
            [base_index(pos) as usize]
    }

    pub fn insert(&mut self, pos: glam::UVec3, value: u8) {
        let index_chunk = &mut self.index[root_index(self.index_bounds, pos) as usize];
        let brick_index = brick_index(pos);

        let brick_occupied = index_chunk.mask & (1 << brick_index) != 0;
        if !brick_occupied {
            if value == 0 {
                return;
            }
            index_chunk.base_addr[brick_index as usize] = self.data.len() as u32;
            index_chunk.mask |= 1 << brick_index;
            self.data.push(DataChunk::default());
        }
        self.data[index_chunk.base_addr[brick_index as usize] as usize].data
            [base_index(pos) as usize] = value;
    }

    pub fn from_scene(scene: &crate::vox::Scene) -> Self {
        let timer = std::time::Instant::now();

        let x_run = scene.size.y as usize * scene.size.z as usize;
        let y_run = scene.size.z as usize;

        let mut _self = Self::new(scene.size.as_uvec3());
        let mut voxels = vec![0; scene.size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                let index = pos.x * x_run + pos.y * y_run + pos.z;
                voxels[index] = palette_index;
            }
        }

        for x in 0..scene.size.x {
            for y in 0..scene.size.y {
                for z in 0..scene.size.z {
                    let pos = glam::ivec3(x, y, z).as_uvec3();
                    let index = pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                    let voxel = voxels[index];
                    _self.insert(pos, voxel);
                }
            }
        }
        // let _t = std::time::Instant::now();
        // _self.merge_optimize();
        // println!("merge optmization took {:#?}", _t.elapsed());

        {
            let mut errors = 0;
            for x in 0..scene.size.x {
                for y in 0..scene.size.y {
                    for z in 0..scene.size.z {
                        let pos = glam::ivec3(x, y, z).as_uvec3();
                        let index =
                            pos.x as usize * x_run + pos.y as usize * y_run + pos.z as usize;
                        let voxel = voxels[index];
                        // assert_eq!(voxel, tree.get(pos));
                        let tval = _self.get(pos);
                        if tval != voxel {
                            println!(
                                "pos {} failed. actual: {}, tree: {}",
                                pos,
                                voxel,
                                _self.get(pos)
                            );
                            errors += 1;
                        }
                    }
                }
            }
            println!("finished with {} errors", errors);
        }

        println!("built tree in {:#?}", timer.elapsed());
        println!(
            "tree length: {} mb",
            (_self.data.len() * 64 + _self.index.len() * 264) as f64 / 1000000.0
        );
        println!(
            "original length: {} mb",
            (scene.size.element_product()) as f64 / 1000000.0
        );
        _self
    }
}
