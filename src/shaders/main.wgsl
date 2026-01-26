struct SceneData {
    size: vec3<u32>,
    palette: array<vec4<u32>, 64>
}
@group(0) @binding(0) var out_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> scene: SceneData;
@group(0) @binding(2) var<storage, read> voxels: array<u32>;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
}
@group(1) @binding(0) var<uniform> camera: Camera;

struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
}
@group(2) @binding(0) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const EPSILON: f32 = 1.0 / 65536.0;
const DDA_MAX_STEPS: u32 = 1000u;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(out_texture).xy;

    let uv = vec2<f32>(in.id.xy) / vec2<f32>(dimensions);

    let ray = start_ray(uv);

    if !ray.in_bounds {
        textureStore(out_texture, vec2<i32>(in.id.xy), vec4<f32>(0.0));
        return;
    }

    let size = vec3<i32>(scene.size);

    // DDA ray marching
    // see https://cse.yorku.ca/~amana/research/grid.pdf
    let pos_step = vec3<i32>(step(-ray.direction, vec3(0.0)) - step(ray.direction, vec3(0.0)));
    let t_delta = abs(vec3(1.0) / (ray.direction + EPSILON));
    var pos = vec3<i32>(floor(ray.origin));
    var t_max = vec3<f32>(0.0);
    if ray.direction.x > 0.0 {
        t_max.x = 1.0 - fract(ray.origin.x);
    } else {
        t_max.x = fract(ray.origin.x);
    }
    if ray.direction.y > 0.0 {
        t_max.y = 1.0 - fract(ray.origin.y);
    } else {
        t_max.y = fract(ray.origin.y);
    }
    if ray.direction.z > 0.0 {
        t_max.z = 1.0 - fract(ray.origin.z);
    } else {
        t_max.z = fract(ray.origin.z);
    }
    t_max *= t_delta;

    var res = voxel(pos);
    for (var i = 0u; i < DDA_MAX_STEPS && res == 0u; i++) {
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                pos.x += pos_step.x;
                if pos.x < 0 || pos.x >= size.x {
                    break;
                }
                t_max.x += t_delta.x;
            } else {
                pos.z += pos_step.z;
                if pos.z < 0 || pos.z >= size.z {
                    break;
                }
                t_max.z += t_delta.z;
            }
        } else {
            if t_max.y < t_max.z {
                pos.y += pos_step.y;
                if pos.y < 0 || pos.y >= size.y {
                    break;
                }
                t_max.y += t_delta.y;
            } else {
                pos.z += pos_step.z;
                if pos.z < 0 || pos.z >= size.z {
                    break;
                }
                t_max.z += t_delta.z;
            }
        }
        res = voxel(pos);
    }
    if res == 0u {
        textureStore(out_texture, vec2<i32>(in.id.xy), vec4<f32>(0.0));
        return;
    }

    var color = palette_color(res);

    // color = color * 0.001 + vec4<f32>(ray.origin / 256.0, 1.0);

    textureStore(out_texture, vec2<i32>(in.id.xy), color);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    in_bounds: bool,
}

fn start_ray(uv: vec2<f32>) -> Ray {
    let hs_far = vec2<f32>(uv.x, 1.0 - uv.y) * 2.0 - 1.0;
    let cs_far = vec4<f32>(hs_far, 1.0, 1.0);

    let ws_far_sc = camera.inv_view_proj * cs_far;
    let ws_far = ws_far_sc.xyz / ws_far_sc.w;

    let ws_origin = camera.ws_position;
    let ws_direction = ws_far - camera.ws_position;

    let ls_origin = (model.inv_transform * vec4<f32>(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4<f32>(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = vec3<f32>(0.0);
    let bd_max = vec3<f32>(scene.size);

    let inv_dir = 1.0 / safe_vec3(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    var ray: Ray;
    ray.origin = ls_origin + (max(0.0, t_near) + 1e-3) * ls_direction;
    ray.direction = ls_direction;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    return ray;
}

fn safe_vec3(v: vec3<f32>) -> vec3<f32> {
    return sign(v) * max(vec3(EPSILON), abs(v));
}

/// Gets palette index at the given local space position
fn voxel(position: vec3<i32>) -> u32 {
    let index = position.x * i32(scene.size.y) * i32(scene.size.z) + position.y * i32(scene.size.z) + position.z;
    return (voxels[index >> 0x2u] >> (8u * (u32(index) & 0x3u))) & 0xFFu;
}

/// Palette color lookup
fn palette_color(index: u32) -> vec4<f32> {
    let rgba = scene.palette[index >> 2u][index & 3u];
    return vec4<f32>(
        f32((rgba >> 24u) & 0xFFu) / 255.0,
        f32((rgba >> 16u) & 0xFFu) / 255.0,
        f32((rgba >> 8u) & 0xFFu) / 255.0,
        f32(rgba & 0xFFu) / 255.0
    );
}
