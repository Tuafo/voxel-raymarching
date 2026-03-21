struct VisibleVoxel {
    data: u32,
    leaf_index: u32,
    pos: array<u32, 2>,
}
@group(0) @binding(0) var<storage, read> visible_voxels: array<VisibleVoxel>;
@group(0) @binding(1) var<storage, read> cur_voxel_lighting: array<u32>;
@group(0) @binding(2) var<storage, read_write> acc_voxel_lighting: array<u32>;

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

const MAX_HISTORY_LENGTH: u32 = 255u;

@compute @workgroup_size(256, 1, 1)
fn compute_main(in: ComputeIn) {
    let visible = visible_voxels[in.id.x];
    if visible.data == 0u {
        return;
    }

    let cur = cur_voxel_lighting[in.id.x];
    // let cur_visible = cur & 0xFFFFu;
    // if cur_visible == 0u {
    //     return;
    // }
    // let cur_shadow_count = min(cur);
    let cur_shadow = f32(cur & 1u);
    let cur_ao = f32((cur >> 1u) & 0xFFu) / 255.0;

    let acc = acc_voxel_lighting[visible.leaf_index];
    let acc_shadow = f32((acc >> 8u) & 0xFFFu) / 4095.0;
    let acc_ao = f32(acc >> 20u) / 4095.0;
    let history_len = min(MAX_HISTORY_LENGTH, (acc & 0xFFu) + 1u);

    let alpha = 1.0 / f32(history_len);
    let res_shadow = mix(acc_shadow, cur_shadow, alpha);
    let res_ao = mix(acc_ao, cur_ao, alpha);

    let res = ((u32(res_ao * 4095.0) & 0xFFFu) << 20u) | ((u32(res_shadow * 4095.0) & 0xFFFu) << 8u) | history_len;
    acc_voxel_lighting[visible.leaf_index] = res;
}
