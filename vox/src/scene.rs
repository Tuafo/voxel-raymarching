#![allow(dead_code, unused_variables)]

use std::{
    collections::HashMap,
    error::Error,
    i32,
    io::{BufRead, Seek},
};

use glam::{IVec3, Mat4, U8Vec3, Vec4Swizzles, ivec3, ivec4, vec4};

use crate::chunks::{Chunk, GroupChunk, ShapeChunk, SizeChunk, TransformChunk, VoxelChunk};
use crate::extensions::ReadExt;

#[derive(Debug)]
pub struct Scene {
    pub version: i32,
    pub base: IVec3,
    pub size: IVec3,
    pub voxel_count: u32,
    pub palette: Box<[Material; 256]>,
    transforms: HashMap<TransformID, Transform>,
    models: HashMap<ModelID, Model>,
    instances: Vec<Instance>,
}
pub struct Object<'a> {
    pub transform: &'a Transform,
    pub model: &'a Model,
}
#[derive(Debug)]
pub struct Transform {
    id: TransformID,
    children: Vec<TransformID>,
    pub matrix: Mat4,
}
#[derive(Debug)]
pub struct Model {
    id: ModelID,
    pub size: IVec3,
    pub data: Vec<u8>,
    pub voxel_count: u32,
}

impl Scene {
    pub fn new<R: BufRead + Seek>(src: &mut R) -> Result<Self, Box<dyn Error>> {
        let header_id = src.read_bytes::<4>()?;
        assert_eq!(&header_id, b"VOX ");

        let version = src.read_i32()?;

        let chunks = read_chunks(src)?;

        macro_rules! filter_by_type {
            ($chunk_type:path) => {
                chunks
                    .iter()
                    .filter_map(|c| match c {
                        $chunk_type(g) => Some(g),
                        _ => None,
                    })
                    .collect()
            };
        }
        let size_chunks: Vec<&SizeChunk> = filter_by_type!(Chunk::Size);
        let voxel_chunks: Vec<&VoxelChunk> = filter_by_type!(Chunk::Voxels);
        let trf_chunks: Vec<&TransformChunk> = filter_by_type!(Chunk::Transform);
        let group_chunks: Vec<&GroupChunk> = filter_by_type!(Chunk::Group);
        let shape_chunks: Vec<&ShapeChunk> = filter_by_type!(Chunk::Shape);

        let palette = {
            let mut palette = Box::new([Material::default(); 256]);

            if let Some(colors) = chunks.iter().find_map(|c| match c {
                Chunk::Palette(colors) => Some(colors),
                _ => None,
            }) {
                for i in 1..256 {
                    palette[i].rgba = colors.palette[i - 1];
                }
            }

            for matl in chunks.iter().filter_map(|c| match c {
                Chunk::Material(matl) => Some(matl),
                _ => None,
            }) {
                let material = &mut palette[matl.id as usize - 1];
                material.surface = matl.surface;
                material.roughness = matl.roughness;
                material.specular = matl.specular;
                material.ior = matl.ior;
                material.attenuation = matl.attenuation;
                material.flux = matl.flux;
                material.plastic = matl.plastic;
            }

            palette
        };

        let models = {
            if size_chunks.len() != voxel_chunks.len() {
                return Err(format!(
                    "mismatch between bounds chunks ({}) and voxel data chunks ({})",
                    size_chunks.len(),
                    voxel_chunks.len()
                )
                .into());
            }

            let mut models = HashMap::new();
            for i in 0..size_chunks.len() {
                let id = i as ModelID;
                let size = IVec3::from_slice(&size_chunks[i].size);

                let model = Model::new(id, size, &voxel_chunks[i]);

                models.insert(id, model);
            }

            models
        };

        let mut node_trf_parent_map: HashMap<i32, Vec<u32>> = HashMap::new();

        let transforms = {
            let mut transforms = HashMap::new();
            for trf in trf_chunks {
                let id = trf.id as TransformID;

                let frame = trf.frames.first();
                let matrix = frame.and_then(|f| Some(f.matrix)).unwrap_or(Mat4::IDENTITY);

                let children: Vec<TransformID> = group_chunks
                    .iter()
                    .find(|&g| g.id == trf.child_id)
                    .map(|&g| g.children.iter().map(|c| *c as TransformID).collect())
                    .unwrap_or(Vec::new());

                match node_trf_parent_map.get_mut(&trf.child_id) {
                    Some(bucket) => {
                        bucket.push(id);
                    }
                    None => {
                        node_trf_parent_map.insert(trf.child_id, vec![id]);
                    }
                };

                transforms.insert(
                    id,
                    Transform {
                        id,
                        children,
                        matrix,
                    },
                );
            }

            transforms
        };

        let mut voxel_count = 0;

        let instances = {
            let mut instances = vec![];
            for shape in shape_chunks {
                let Some(transforms) = node_trf_parent_map.get(&shape.id) else {
                    continue;
                };

                for &transform in transforms {
                    for model in &shape.models {
                        voxel_count += models.get(&model.id).map(|m| m.voxel_count).unwrap_or(0);

                        instances.push(Instance {
                            model: model.id,
                            transform,
                        });
                    }
                }
            }

            instances
        };

        let (base, size) = {
            let mut min = IVec3::MAX;
            let mut max = IVec3::MIN;

            for instance in &instances {
                let (transform, model) = (
                    transforms.get(&instance.transform).unwrap(),
                    models.get(&instance.model).unwrap(),
                );

                const CORNERS: [IVec3; 8] = [
                    ivec3(0, 0, 0),
                    ivec3(0, 0, 1),
                    ivec3(0, 1, 0),
                    ivec3(0, 1, 1),
                    ivec3(1, 0, 0),
                    ivec3(1, 0, 1),
                    ivec3(1, 1, 0),
                    ivec3(1, 1, 1),
                ];
                for corner in CORNERS {
                    let point = corner * model.size - model.size / 2;
                    let point = vec4(point.x as f32, point.y as f32, point.z as f32, 1.0);
                    let point = (transform.matrix * point).xyz();
                    let point = ivec3(
                        point.x.round() as i32,
                        point.y.round() as i32,
                        point.z.round() as i32,
                    );
                    min = min.min(point);
                    max = max.max(point);
                }
            }

            let base = min;
            let size = max - min;
            (base, size)
        };

        Ok(Self {
            version,
            palette,
            transforms,
            models,
            instances,
            base,
            size,
            voxel_count,
        })
    }

    pub fn load(src: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut reader = std::io::Cursor::new(src);
        Self::new(&mut reader)
    }

    pub fn instances(&self) -> impl Iterator<Item = Object<'_>> {
        self.instances
            .iter()
            .map(|instance| self.get_instance(instance))
    }

    pub fn palette(&self) -> Box<[u32; 256]> {
        let mut res = [0u32; 256];
        for i in 0..256 {
            let rgba = self.palette[i].rgba;
            res[i] = (rgba[0] as u32) << 16 | (rgba[1] as u32) << 8 | (rgba[2] as u32) << 0
        }
        return Box::new(res);
    }
}

impl Object<'_> {
    /// Returns iterator over `(world (x,y,z), palette index)` for non-empty voxels of this object, column-major.
    pub fn voxels(&self) -> impl Iterator<Item = (IVec3, u8)> {
        let [sx, sy, sz] = [
            self.model.size.x as usize,
            self.model.size.y as usize,
            self.model.size.z as usize,
        ];
        let local_offset = -self.model.size / 2;
        let local_offset = vec4(
            local_offset.x as f32,
            local_offset.y as f32,
            local_offset.z as f32,
            0.0,
        );
        let matrix = self.transform.matrix;

        (0..sx).flat_map(move |x| {
            (0..sy).flat_map(move |y| {
                (0..sz).flat_map(move |z| {
                    let index = x * sy * sz + y * sz + z;
                    match self.model.data[index] {
                        0 => None,
                        val => {
                            let mut point = ivec4(x as i32, y as i32, z as i32, 1).as_vec4();
                            point = matrix * (point + local_offset);

                            Some((point.xyz().round().as_ivec3(), val))
                        }
                    }
                })
            })
        })
    }
}

impl Scene {
    fn get_instance(&self, instance: &Instance) -> Object<'_> {
        Object {
            transform: self.transforms.get(&instance.transform).unwrap(),
            model: self.models.get(&instance.model).unwrap(),
        }
    }
}

impl Model {
    fn new(id: ModelID, size: IVec3, chunk: &VoxelChunk) -> Self {
        let voxels = &chunk.voxels;

        let mut data = vec![0u8; size.element_product() as usize];
        for voxel in voxels {
            let pos = U8Vec3::from_array(voxel.position).as_usizevec3();

            data[pos.x * size.y as usize * size.z as usize + pos.y * size.z as usize + pos.z] =
                voxel.color_index;
        }

        Self {
            id,
            size,
            data,
            voxel_count: voxels.len() as u32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub rgba: [u8; 4],
    surface: MaterialSurface,
    roughness: f32,
    specular: f32,
    ior: f32,
    attenuation: f32,
    flux: f32,
    plastic: bool,
}
#[derive(Debug, Clone, Copy)]
pub enum MaterialSurface {
    Diffuse,
    Metal,
    Glass,
    Emissive,
}
#[derive(Debug)]
struct Instance {
    model: ModelID,
    transform: TransformID,
}

type TransformID = u32;
pub type ModelID = u32;

impl Default for Material {
    fn default() -> Self {
        Self {
            rgba: [0, 0, 0, 255],
            surface: MaterialSurface::Diffuse,
            roughness: 0.5,
            specular: 0.5,
            ior: 1.0,
            attenuation: 0.0,
            flux: 0.0,
            plastic: false,
        }
    }
}
// impl Encode for Transform {
//     fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {

//     }
// }

fn read_chunks<R: BufRead + Seek>(src: &mut R) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let id = src.read_bytes::<4>()?;
    let content_bytes = src.read_i32()? as u64;
    let child_bytes = src.read_i32()? as u64;

    let mut start_pos = src.stream_position()?;
    let chunk = Chunk::parse(&id, src)?;

    if chunk.is_none() {
        src.seek_relative(content_bytes as i64)?;
    }
    let mut end_pos = src.stream_position()?;
    if end_pos - start_pos != content_bytes {
        return Err(format!(
            "while parsing chunk body, expected {} bytes, but found {} bytes. id: {}",
            content_bytes,
            end_pos - start_pos,
            String::from_utf8(id.to_vec())?,
        )
        .into());
    }
    let mut result = vec![];
    if let Some(c) = chunk {
        result.push(c);
    }

    // recursively read children
    start_pos = end_pos;
    while end_pos - start_pos < child_bytes {
        result.append(&mut read_chunks(src)?);
        end_pos = src.stream_position()?;
    }
    if end_pos - start_pos != child_bytes {
        return Err(format!(
            "while parsing chunk children, expected {} bytes, but found {} bytes.",
            child_bytes,
            end_pos - start_pos
        )
        .into());
    }

    Ok(result)
}
