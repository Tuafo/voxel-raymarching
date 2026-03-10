pub mod gltf;
mod lightmaps;
mod models;
mod models_t64;

pub use models_t64::*;
// pub use models::*;
pub use lightmaps::*;

pub const MODEL_FILE_EXT: &'static str = "voxel";
pub const LIGHTMAP_FILE_EXT: &'static str = "lightmap";
