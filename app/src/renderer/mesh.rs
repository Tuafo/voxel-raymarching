use std::time::Instant;

use glam::{IVec3, Vec3, usizevec3};
use wgpu::{Buffer, util::DeviceExt};

use vox::Scene;

pub trait IntoMesh {
    fn mesh(&mut self, device: &wgpu::Device) -> Mesh;
}

/// Mesh object that is to be used with the `base.wgsl` shader.
#[derive(Debug)]
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u32], index_count: u32) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 3],
}
pub const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;

#[inline]
fn vertex(position: [f32; 3], color: [f32; 3]) -> Vertex {
    Vertex {
        position: [position[0], position[1], position[2], 1.0],
        color,
    }
}
#[inline]
fn vert(position: IVec3, color: [f32; 3]) -> Vertex {
    let position = position.as_vec3();
    Vertex {
        position: [position.x, position.y, position.z, 1.0],
        color,
    }
}

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const CYAN: [f32; 3] = [0.0, 1.0, 1.0];
const PINK: [f32; 3] = [1.0, 0.0, 1.0];

/// `Mesh` variant of a 3D cube
#[derive(Debug)]
pub struct Cube {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Cube {
    pub fn new() -> Self {
        let vertices = vec![
            // top
            vertex([-1.0, -1.0, 1.0], RED),
            vertex([1.0, -1.0, 1.0], RED),
            vertex([1.0, 1.0, 1.0], RED),
            vertex([-1.0, 1.0, 1.0], RED),
            // bottom
            vertex([-1.0, 1.0, -1.0], GREEN),
            vertex([1.0, 1.0, -1.0], GREEN),
            vertex([1.0, -1.0, -1.0], GREEN),
            vertex([-1.0, -1.0, -1.0], GREEN),
            // right
            vertex([1.0, -1.0, -1.0], BLUE),
            vertex([1.0, 1.0, -1.0], BLUE),
            vertex([1.0, 1.0, 1.0], BLUE),
            vertex([1.0, -1.0, 1.0], BLUE),
            // left
            vertex([-1.0, -1.0, 1.0], YELLOW),
            vertex([-1.0, 1.0, 1.0], YELLOW),
            vertex([-1.0, 1.0, -1.0], YELLOW),
            vertex([-1.0, -1.0, -1.0], YELLOW),
            // front
            vertex([1.0, 1.0, -1.0], CYAN),
            vertex([-1.0, 1.0, -1.0], CYAN),
            vertex([-1.0, 1.0, 1.0], CYAN),
            vertex([1.0, 1.0, 1.0], CYAN),
            // back
            vertex([1.0, -1.0, 1.0], PINK),
            vertex([-1.0, -1.0, 1.0], PINK),
            vertex([-1.0, -1.0, -1.0], PINK),
            vertex([1.0, -1.0, -1.0], PINK),
        ];

        let indices: Vec<u32> = vec![
            0, 1, 2, 2, 3, 0, // top
            4, 5, 6, 6, 7, 4, // bottom
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // front
            20, 21, 22, 22, 23, 20, // back
        ];

        Self { vertices, indices }
    }
}

impl IntoMesh for Cube {
    fn mesh(&mut self, device: &wgpu::Device) -> Mesh {
        Mesh::new(
            device,
            &self.vertices,
            &self.indices,
            self.indices.len() as u32,
        )
    }
}

#[derive(Debug)]
pub struct VoxelMesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl VoxelMesh {
    pub fn new(scene: &Scene) -> Self {
        let timer = Instant::now();

        let size = scene.size.as_usizevec3();
        let mut voxels = vec![0u8; size.element_product() as usize];
        for instance in scene.instances() {
            for (pos, palette_index) in instance.voxels() {
                let pos = (pos - scene.base).as_usizevec3();
                voxels[pos.x * size.y * size.z + pos.y * size.z + pos.z] = palette_index;
            }
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let mut emit_voxel = |pos: Vec3, palette_index: u8| {
            let color = scene.palette[palette_index as usize].rgba;
            let color = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ];
            let vi = vertices.len() as u32;
            vertices.extend(
                [
                    // top
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    // bottom
                    [-1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    // right
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    // left
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    // front
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    // back
                    [1.0, -1.0, 1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                ]
                .iter()
                .map(|p| vertex((Vec3::from_slice(p) + pos).to_array(), color)),
            );
            indices.extend(
                [
                    0, 1, 2, 2, 3, 0, // top
                    4, 5, 6, 6, 7, 4, // bottom
                    8, 9, 10, 10, 11, 8, // right
                    12, 13, 14, 14, 15, 12, // left
                    16, 17, 18, 18, 19, 16, // front
                    20, 21, 22, 22, 23, 20, // back
                ]
                .iter()
                .map(|i| i + vi),
            );
        };

        for x in 0..size.x {
            for y in 0..size.y {
                for z in 0..size.z {
                    match voxels[x * size.y * size.z + y * size.z + z] {
                        0 => {}
                        i => {
                            emit_voxel(usizevec3(x, y, z).as_vec3(), i);
                        }
                    }
                }
            }
        }

        println!("load took {:#?}", timer.elapsed());

        Self { vertices, indices }
    }
}

impl IntoMesh for VoxelMesh {
    fn mesh(&mut self, device: &wgpu::Device) -> Mesh {
        Mesh::new(
            device,
            &self.vertices,
            &self.indices,
            self.indices.len() as u32,
        )
    }
}
