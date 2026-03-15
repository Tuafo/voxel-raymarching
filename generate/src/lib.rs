pub mod gltf;
mod lightmaps;
mod models;
mod models_t64;
mod palette;
mod utils;

pub use lightmaps::*;
pub use models_t64::*;

pub const MODEL_FILE_EXT: &'static str = "voxel";
pub const LIGHTMAP_FILE_EXT: &'static str = "lightmap";

pub const MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1024 * 1024 * 1024;
