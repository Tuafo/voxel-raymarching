#[derive(Debug)]
pub struct Config {
    pub render_scale: f32,
    pub sun_altitude: f32,
    pub sun_azimuth: f32,
    pub shadow_bias: f32,
    pub shadow_spread: f32,
    pub filter_shadows: bool,
    pub shadow_filter_radius: f32,
    pub voxel_normal_factor: f32,
    pub indirect_sky_intensity: f32,
    pub ambient_ray_max_distance: u32,
    pub view: DebugView,
    pub fxaa: bool,
    pub taa: bool,
    pub max_fps: Option<u32>,
    pub print_debug_info: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            render_scale: 0.5,
            sun_azimuth: -2.5,
            sun_altitude: 1.3,
            shadow_bias: 0.0005,
            shadow_spread: 0.05,
            filter_shadows: true,
            shadow_filter_radius: 7.0,
            voxel_normal_factor: 0.5,
            indirect_sky_intensity: 0.5,
            ambient_ray_max_distance: 10,
            view: DebugView::Composite,
            fxaa: false,
            taa: true,
            max_fps: None,
            print_debug_info: false,
        }
    }
}

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub enum DebugView {
    #[default]
    Composite = 0,
    Albedo = 1,
    Depth = 2,
    HitNormal = 3,
    SurfaceNormal = 4,
    Roughness = 5,
    Metallic = 6,
    Shadow = 7,
    Ambient = 8,
    Specular = 9,
    Velocity = 10,
    SkyAlbedo = 11,
    SkyIrradiance = 12,
    SkyPrefiler = 13,
}

pub const DEBUG_VIEWS: &'static [(&'static str, DebugView)] = &[
    ("Composite", DebugView::Composite),
    ("Albedo", DebugView::Albedo),
    ("Depth", DebugView::Depth),
    ("Hit Normal", DebugView::HitNormal),
    ("Surface Normal", DebugView::SurfaceNormal),
    ("Roughness", DebugView::Roughness),
    ("Metallic", DebugView::Metallic),
    ("Shadow", DebugView::Shadow),
    ("Ambient", DebugView::Ambient),
    ("Specular", DebugView::Specular),
    ("Velocity", DebugView::Velocity),
    ("Sky Albedo", DebugView::SkyAlbedo),
    ("Sky Irradiance", DebugView::SkyIrradiance),
    ("Sky Prefiler", DebugView::SkyPrefiler),
];
