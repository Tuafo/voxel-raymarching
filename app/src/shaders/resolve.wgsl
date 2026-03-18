@group(0) @binding(0) var<storage, read> cur_voxel_map: array<u32>; // voxel hashmap, two words (key, value) per entry
@group(0) @binding(1) var<storage, read> prev_voxel_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> cur_voxel_lighting: array<u32>;
@group(0) @binding(3) var<storage, read> prev_voxel_lighting: array<f32>;

struct Environment {
    sun_direction: vec3<f32>,
    shadow_bias: f32,
    camera: Camera,
    prev_camera: Camera,
    shadow_spread: f32,
    filter_shadows: u32,
    shadow_filter_radius: f32,
    max_ambient_distance: u32,
    smooth_normal_factor: f32,
    indirect_sky_intensity: f32,
    debug_view: u32,
}
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    jitter: vec2<f32>,
    far: f32,
    fov: f32,
}
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
    fxaa_enabled: u32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
    inv_normal_transform: mat3x3<f32>,
}
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;
@group(3) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

var<workgroup> stack: array<array<u32, 11>, 64>;

const ACC_ALPHA: f32 = 0.25;

@compute @workgroup_size(32, 1, 1)
fn compute_main(in: ComputeIn) {
    let key = cur_voxel_map[in.id.x << 1u];
    if key == 0u {
        return;
    }

    let cur_index = cur_voxel_map[(in.id.x << 1u) | 1u];
    if cur_index == 0u {
        return;
    }

    let cur_shadow = cur_voxel_lighting[cur_index];
    let cur_visible_count = cur_shadow & 0xFFFFu;
    let cur_shadow_count = cur_shadow >> 16u;

    var shadow = f32(cur_shadow_count) / max(1.0, f32(cur_visible_count));

    let prev = map_get(key);
    if prev.exists {
        let acc_shadow = prev_voxel_lighting[prev.value];
        shadow = shadow * ACC_ALPHA + acc_shadow * (1.0 - ACC_ALPHA);
    }

    cur_voxel_lighting[cur_index] = bitcast<u32>(shadow);
}

/// ------------------------------------------------------
/// -------------------- map utils -----------------------

struct MapResult {
    exists: bool,
    value: u32,
}

fn map_get(id: u32) -> MapResult {
    let n = arrayLength(&prev_voxel_map) >> 1u;

    var key = hash_murmur3(id) % n;
    for (var i = 0u; i < 4u; i++) {
        if prev_voxel_map[key << 1u] == id {
            var res: MapResult;
            res.exists = true;
            res.value = prev_voxel_map[(key << 1u) + 1u];
            return res;
        }
        key += 1u;
        if key >= n {
            key = 0u;
        }
    }

    var res: MapResult;
    res.exists = false;
    return res;
}

// from https://github.com/aappleby/smhasher
fn hash_murmur3(seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51u;
    const C2: u32 = 0x1b873593u;

    var h = 0u;
    var k = seed;

    k *= C1;
    k = (k << 15u) | (k >> 17u);
    k *= C2;

    h ^= k;
    h = (h << 13u) | (h >> 19u);
    h = h * 5u + 0xe6546b64u;
    h ^= 4u;

    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
