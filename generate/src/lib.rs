pub mod gltf;
mod lightmaps;
mod models;
mod palette;
pub mod planet;

pub use lightmaps::*;
pub use models::*;
pub use planet::{PlanetConfig, voxelize_planet};

pub const MODEL_FILE_EXT: &'static str = "voxel";
pub const LIGHTMAP_FILE_EXT: &'static str = "lightmap";

pub const MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1024 * 1024 * 1024;
