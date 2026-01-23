#![allow(dead_code, unused_variables)]
use std::{
    collections::HashMap,
    error::Error,
    io::{BufRead, Seek},
};

use glam::{IVec3, IVec4, Mat4, Vec4};

use crate::vox::lib::ModelID;
use crate::vox::{
    extensions::{PropertiesExt, ReadExt},
    lib::MaterialSurface,
};

pub trait ChunkBody {
    fn parse<R>(reader: &mut R) -> Result<Self, Box<dyn Error>>
    where
        R: BufRead + Seek,
        Self: Sized;
}

#[derive(Debug)]
pub enum Chunk {
    Main(MainChunk),
    Pack(PackChunk),
    Size(SizeChunk),
    Voxels(VoxelChunk),
    Palette(PaletteChunk),
    Transform(TransformChunk),
    Group(GroupChunk),
    Shape(ShapeChunk),
    Material(MaterialChunk),
    Layer(LayerChunk),
}
impl Chunk {
    pub fn parse<R: BufRead + Seek>(
        id: &[u8; 4],
        reader: &mut R,
    ) -> Result<Option<Self>, Box<dyn Error>> {
        let chunk = match id {
            b"MAIN" => Some(Chunk::Main(MainChunk::parse(reader)?)),
            b"PACK" => Some(Chunk::Pack(PackChunk::parse(reader)?)),
            b"SIZE" => Some(Chunk::Size(SizeChunk::parse(reader)?)),
            b"XYZI" => Some(Chunk::Voxels(VoxelChunk::parse(reader)?)),
            b"RGBA" => Some(Chunk::Palette(PaletteChunk::parse(reader)?)),
            b"nTRN" => Some(Chunk::Transform(TransformChunk::parse(reader)?)),
            b"nGRP" => Some(Chunk::Group(GroupChunk::parse(reader)?)),
            b"nSHP" => Some(Chunk::Shape(ShapeChunk::parse(reader)?)),
            b"MATL" => Some(Chunk::Material(MaterialChunk::parse(reader)?)),
            b"LAYR" => Some(Chunk::Layer(LayerChunk::parse(reader)?)),
            _ => None,
        };
        Ok(chunk)
    }
}

type NodeID = i32;
type LayerID = i32;
type MaterialID = i32;
type FrameIndex = i32;

#[derive(Debug)]
pub struct MainChunk {}
impl ChunkBody for MainChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        Ok(Self {})
    }
}

#[derive(Debug)]
pub struct PackChunk {
    pub model_count: i32,
}
impl ChunkBody for PackChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let model_count = reader.read_i32()?;
        Ok(Self { model_count })
    }
}

#[derive(Debug)]
pub struct SizeChunk {
    pub size: [i32; 3],
}
impl ChunkBody for SizeChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let size_x = reader.read_i32()?;
        let size_y = reader.read_i32()?;
        let size_z = reader.read_i32()?;
        Ok(Self {
            size: [size_x, size_y, size_z],
        })
    }
}

#[derive(Debug)]
pub struct VoxelChunk {
    pub voxels: Vec<RawVoxel>,
}
#[derive(Debug)]
pub struct RawVoxel {
    pub position: [u8; 3],
    pub color_index: u8,
}
impl ChunkBody for VoxelChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let count = reader.read_i32()? as usize;
        let mut voxels = Vec::with_capacity(count);

        let mut raw = vec![0u8; count * 4];
        reader.read_exact(&mut raw)?;
        for i in 0..count {
            voxels.push(RawVoxel {
                position: [raw[i * 4 + 0], raw[i * 4 + 1], raw[i * 4 + 2]],
                color_index: raw[i * 4 + 3],
            })
        }

        Ok(Self { voxels })
    }
}

#[derive(Debug)]
pub struct PaletteChunk {
    pub palette: Box<[[u8; 4]; 256]>,
}
impl ChunkBody for PaletteChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let mut palette = [[0u8; 4]; 256];

        let mut raw = vec![0u8; 256 * 4];
        reader.read_exact(&mut raw)?;

        for i in 0..256 {
            palette[i][0] = raw[i * 4 + 0];
            palette[i][1] = raw[i * 4 + 1];
            palette[i][2] = raw[i * 4 + 2];
            palette[i][3] = raw[i * 4 + 3];
        }

        Ok(Self {
            palette: Box::new(palette),
        })
    }
}

#[derive(Debug)]
pub struct TransformChunk {
    pub id: NodeID,
    pub name: String,
    pub hidden: bool,
    pub child_id: NodeID,
    pub layer_id: LayerID,
    pub frames: Vec<Frame>,
}
#[derive(Debug)]
pub struct Frame {
    pub index: FrameIndex,
    pub matrix: Mat4,
}
impl ChunkBody for TransformChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let id: NodeID = reader.read_i32()?;

        let node_attributes = reader.read_vox_dict()?;
        let name = match node_attributes.get("_name") {
            Some(name) => name.clone(),
            None => format!("node_{}", id),
        };
        let hidden = node_attributes.get("_hidden").is_some_and(|v| v == "1");

        let child_id: NodeID = reader.read_i32()?;
        let reserved_id = reader.read_i32()?;
        assert_eq!(reserved_id, -1);
        let layer_id = reader.read_i32()?;

        let frame_count = reader.read_i32()?;
        let mut frames = Vec::with_capacity(frame_count as usize);
        for i in 0..frame_count {
            let trf_dict = reader.read_vox_dict()?;

            let matrix = {
                let position = trf_dict
                    .get("_t")
                    .and_then(|s| {
                        let val = s
                            .split_whitespace()
                            .flat_map(|c| c.parse::<i32>())
                            .take(3)
                            .collect::<Vec<i32>>();
                        if val.len() == 3 {
                            Some(IVec3::from_slice(&val[..]))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(IVec3::ZERO);
                let mut rows = trf_dict
                    .get("_r")
                    .and_then(|s| s.parse::<i32>().ok())
                    .map(|enc| {
                        let ri_0 = enc >> 0 & 0x3;
                        let ri_1 = enc >> 2 & 0x3;
                        let ri_2 = match (ri_0, ri_1) {
                            (1, 2) | (2, 1) => 0,
                            (0, 2) | (2, 0) => 1,
                            (0, 1) | (1, 0) => 2,
                            _ => panic!("invalid rotation row index"),
                        };
                        let mut rows: [IVec4; 3] = [ri_0, ri_1, ri_2].map(|ri| match ri {
                            0 => IVec4::X,
                            1 => IVec4::Y,
                            _ => IVec4::Z,
                        });
                        if enc >> 4 & 0x1 == 1 {
                            rows[0] *= -1;
                        }
                        if enc >> 5 & 0x1 == 1 {
                            rows[1] *= -1;
                        }
                        if enc >> 6 & 0x1 == 1 {
                            rows[2] *= -1;
                        }
                        rows
                    })
                    .unwrap_or([IVec4::X, IVec4::Y, IVec4::Z]);

                rows[0].w = position.x;
                rows[1].w = position.y;
                rows[2].w = position.z;

                Mat4::from_cols(
                    rows[0].as_vec4(),
                    rows[1].as_vec4(),
                    rows[2].as_vec4(),
                    Vec4::W,
                )
                .transpose()
            };

            frames.push(Frame { index: i, matrix });
        }

        let res = Self {
            id,
            name,
            hidden,
            child_id,
            layer_id,
            frames,
        };
        Ok(res)
    }
}

#[derive(Debug)]
pub struct GroupChunk {
    pub id: NodeID,
    pub attributes: HashMap<String, String>,
    pub children: Vec<NodeID>,
}
impl ChunkBody for GroupChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let id: NodeID = reader.read_i32()?;
        let attributes = reader.read_vox_dict()?;

        let child_count = reader.read_i32()?;
        let mut children = Vec::with_capacity(child_count as usize);
        for _ in 0..child_count {
            children.push(reader.read_i32()?);
        }

        let res = Self {
            id,
            attributes,
            children,
        };
        Ok(res)
    }
}

#[derive(Debug)]
pub struct ShapeChunk {
    pub id: NodeID,
    pub attributes: HashMap<String, String>,
    pub models: Vec<RawModel>,
}
#[derive(Debug)]
pub struct RawModel {
    pub id: ModelID,
    pub frame: FrameIndex,
}
impl ChunkBody for ShapeChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let id = reader.read_i32()?;
        let attributes = reader.read_vox_dict()?;

        let model_count = reader.read_i32()?;
        let mut models = Vec::with_capacity(model_count as usize);
        for _ in 0..model_count {
            let model_id = reader.read_i32()? as ModelID;
            let model_attributes = reader.read_vox_dict()?;
            let frame = model_attributes
                .get("_f")
                .and_then(|f| str::parse::<i32>(f).ok())
                .unwrap_or(0);
            models.push(RawModel {
                id: model_id,
                frame,
            });
        }

        let res = Self {
            id,
            attributes,
            models,
        };
        Ok(res)
    }
}

#[derive(Debug)]
pub struct MaterialChunk {
    pub id: MaterialID,
    pub surface: MaterialSurface,
    pub weight: f32,
    pub roughness: f32,
    pub specular: f32,
    pub ior: f32,
    pub attenuation: f32,
    pub flux: f32,
    pub plastic: bool,
}
impl ChunkBody for MaterialChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let id = reader.read_i32()?;

        let props = reader.read_vox_dict()?;
        let surface = match props.get("_type").map(|s| s.as_str()) {
            Some("_diffuse") => MaterialSurface::Diffuse,
            Some("_metal") => MaterialSurface::Metal,
            Some("_glass") => MaterialSurface::Glass,
            Some("_emit") => MaterialSurface::Emissive,
            _ => MaterialSurface::Diffuse,
        };

        let weight = props.parse_or("_weight", 1.0);
        let roughness = props.parse_or("_rough", 0.5);
        let specular = props.parse_or("_spec", 0.5);
        let ior = props.parse_or("_ior", 1.5);
        let attenuation = props.parse_or("_att", 0.0);
        let flux = props.parse_or("_flux", 1.0);
        let plastic = props.get("_plastic").is_some();

        let res = Self {
            id,
            surface,
            weight,
            roughness,
            specular,
            ior,
            attenuation,
            flux,
            plastic,
        };
        Ok(res)
    }
}

#[derive(Debug)]
pub struct LayerChunk {
    pub id: LayerID,
    pub name: String,
    pub hidden: bool,
}
impl ChunkBody for LayerChunk {
    fn parse<R: BufRead + Seek>(reader: &mut R) -> Result<Self, Box<dyn Error>> {
        let id = reader.read_i32()?;
        let attributes = reader.read_vox_dict()?;
        let name = attributes.parse_or("_name", format!("layer_{}", id));
        let hidden = attributes.get("_hidden").is_some_and(|h| h == "1");
        let reserved_id = reader.read_i32()?;
        assert_eq!(reserved_id, -1);

        let res = Self { id, name, hidden };
        Ok(res)
    }
}
