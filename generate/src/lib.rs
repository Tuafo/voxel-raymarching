pub mod gltf;
mod lightmaps;
mod models;
pub mod models_new;

pub use lightmaps::*;
pub use models::*;

pub const MODEL_FILE_EXT: &'static str = "voxel";
pub const LIGHTMAP_FILE_EXT: &'static str = "lightmap";
