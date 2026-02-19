@group(0) @binding(0) var tex_out_albedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_out_normal: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var tex_out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var tex_out_velocity: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var tex_out_illum: texture_storage_2d<rgba8unorm, write>;

struct VoxelSceneMetadata {
    size: vec3<u32>,
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
struct Chunk {
    mask: array<u32, 16>,
}
@group(1) @binding(0) var<uniform> scene: VoxelSceneMetadata;
@group(1) @binding(1) var<uniform> palette: Palette;
@group(1) @binding(2) var<storage, read> chunk_indices: array<u32>;
@group(1) @binding(3) var<storage, read> chunks: array<Chunk>;
@group(1) @binding(4) var brickmap: texture_storage_3d<r32uint, read>;
@group(1) @binding(5) var tex_noise: texture_3d<f32>;
@group(1) @binding(6) var sampler_noise: sampler;

struct Environment {
    sun_direction: vec3<f32>,
    shadow_bias: f32,
    camera: Camera,
    prev_camera: Camera,
    jitter: vec2<f32>,
    prev_jitter: vec2<f32>,
    shadow_spread: f32,
    filter_shadows: u32,
    shadow_filter_radius: f32,
}
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    ws_position: vec3<f32>,
    forward: vec3<f32>,
    near: f32,
    far: f32,
    fov: f32,
}
struct Model {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    normal_transform: mat3x3<f32>,
}
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
    fxaa_enabled: u32,
}
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const DDA_MAX_STEPS: u32 = 300u;
const SKY_COLOR: vec3<f32> = vec3(0.5, 0.9, 1.5);

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let res = trace_scene(in);

    textureStore(tex_out_albedo, pos, vec4(res.albedo, 1.0));
    textureStore(tex_out_normal, pos, vec4(res.normal, 1.0));
    textureStore(tex_out_depth, pos, vec4(res.depth, 0.0, 0.0, 1.0));
    textureStore(tex_out_velocity, pos, vec4(res.velocity, 0.0, 1.0));
    textureStore(tex_out_illum, pos, vec4(res.illumination, 1.0));
}

struct SceneResult {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    velocity: vec2<f32>,
    illumination: vec3<f32>,
}

fn trace_scene(in: ComputeIn) -> SceneResult {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_out_albedo).xy);
    let texel_size = 1.0 / vec2<f32>(dimensions);

    let ray = raymarch(start_ray(vec2<u32>(pos)));

    if !ray.hit {
        var res: SceneResult;
        res.albedo = SKY_COLOR;
        res.normal = vec3(0.0);
        res.depth = 1.0;
        res.velocity = vec2(0.0);
        res.illumination = vec3(1.0);
        return res;
    }

    let ws_pos_h = model.transform * vec4<f32>(ray.local_pos, 1.0);
    let ws_pos = ws_pos_h.xyz;
    let depth = normalized_depth(ws_pos);

    let cs_pos = environment.camera.view_proj * ws_pos_h;
    let ndc = cs_pos.xyz / cs_pos.w;
    let uv = ndc.xy * vec2(0.5, -0.5) + 0.5;

    let prev_cs_pos = environment.prev_camera.view_proj * ws_pos_h;
    let prev_ndc = prev_cs_pos.xy / prev_cs_pos.w;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

    let velocity = uv - prev_uv;

    let albedo = palette_color(ray.palette_index);
    let ws_surface_normal = normalize(model.normal_transform * ray.surface_normal);
    let ws_hit_normal = normalize(model.normal_transform * ray.hit_normal);

    // var shadow_ray_dir = normalize(environment.sun_direction);
    // let shadow_ray_origin = res.local_pos + environment.shadow_bias * res.hit_normal;
    // let shadow_factor = 0.0;
    // let in_shadow = raymarch_shadow(Ray(shadow_ray_origin, shadow_ray_dir, 0.0, true));
    // let in_shadow = false

    let noise = blue_noise(in.id.xy);

    let illumination = trace_illumination(pos, noise, ray);
    let shadow = trace_shadow(pos, noise, ray);
    // let illumination = vec3(1.0);
    // let acc_illum = textureLoad(tex_acc_illum, pos, 0).rgb;
    // let illumination = ILLUM_ACC_ALPHA * cur_illum + (1.0 - ILLUM_ACC_ALPHA) * acc_illum;


    // let noise_dimensions = textureDimensions(noise).xy;
    // var noise = textureLoad(noise, in.id.xy % noise_dimensions, 0).rgb;

    var res: SceneResult;
    res.albedo = albedo;
    res.normal = ws_surface_normal;
    res.depth = depth;
    res.illumination = vec3(shadow, illumination, 1.0);
    res.velocity = velocity;
    // res.illumination = noise;
    return res;
}

fn normalized_depth(ws_pos: vec3<f32>) -> f32 {
    let depth = dot(ws_pos.xyz - environment.camera.ws_position, environment.camera.forward);
    return (depth - environment.camera.near) / (environment.camera.far - environment.camera.near);
}

fn trace_illumination(pos: vec2<i32>, noise: vec3<f32>, hit: RaymarchResult) -> f32 {
     var ray: SparseRay;
     ray.origin = hit.local_pos + environment.shadow_bias * hit.hit_normal;

     let dir = rand_hemisphere_direction(noise.xy);
     let n = hit.hit_normal;
     var b1 = vec3<f32>(0.0);
     var b2 = vec3<f32>(0.0);
     if (n.z < 0.0) {
         let a = 1.0 / (1.0 - n.z);
         let b = n.x * n.y * a;
         b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
         b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
     }
     else{
         let a = 1.0 / (1.0 + n.z);
         let b = -n.x * n.y * a;
         b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
         b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
     }
     ray.direction = normalize(b1 * dir.x + b2 * dir.y + n * dir.z);

     // let dir = normalize(noise.xyz) * 2.0 - 1.0;
     // ray.direction = normalize(dir * sign(dot(hit.hit_normal, dir)));
     // ray.direction = dir;

     const MAX_DISTANCE: f32 = 1000.0;
     const MAX_DISTANCE_OCCLUSION: f32 = 0.15;
     var res = raymarch_sparse(ray, MAX_DISTANCE);
     if res.hit {
         res.distance *= MAX_DISTANCE_OCCLUSION;
     }

     return res.distance / MAX_DISTANCE;
     // ray.direction = normalize(dir);
     // ray.direction = normalize(noise * 2.0 - 1.0);
}

fn trace_shadow(pos: vec2<i32>, noise: vec3<f32>, hit: RaymarchResult) -> f32 {
    // let fragment_id = pos.x + pos.y * dimensions.x;
    // var seed = u32(fragment_id) * 1973u + frame.frame_id * 927u + 26699u;
    // var dir_offset = normalize(rand_direction(seed));
    // let sun_right = normalize(cross(vec3(0.0, 0.0, 1.0), -environment.sun_direction));
    // let sun_up = normalize(cross(-environment.sun_direction, sun_right));

    // let basis_offset = concentric_disk_sample(noise.x, noise.y);
    // let dir_offset = (sun_right * basis_offset.x + sun_up * basis_offset.y) * 0.02;


    // let u = noise.x;
    // let v = noise.y;

    // let z = u * 2.0 - 1.0;
    // let phi = v * 2.0 * 3.14159265359;
    // let r = sqrt(max(0.0, 1.0 - z * z));

    // let dir_offset= normalize(vec3<f32>(r * cos(phi), r * sin(phi), z)) * 0.02;
    // let dir_offset = normalize(noise * 2.0 - 1.0) * environment.shadow_spread;
    let dir_offset = rand_hemisphere_direction(noise.xy) * environment.shadow_spread;

    var ray: SparseRay;
    ray.origin = hit.local_pos + environment.shadow_bias * hit.hit_normal;
    ray.direction = normalize(environment.sun_direction + dir_offset);

    let occluded = raymarch_shadow(ray);
    // let occluded = false;
    return select(1.0, 0.0, occluded);
}

fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    const FRACT_PHI: f32 = 0.61803398875;
    const FRACT_SQRT_2: f32 = 0.41421356237;
    const OFFSET: vec2<f32> = vec2<f32>(FRACT_PHI, FRACT_SQRT_2);

    let frame_offset_seed = (frame.frame_id >> 5u) & 0xffu;
    let frame_offset = vec2<u32>(OFFSET * 128.0 * f32(frame_offset_seed));

    let id = pos + frame_offset;
    let sample_pos = vec3<u32>(
        id.x & 0x7fu,
        id.y & 0x7fu,
        frame.frame_id & 0x1fu,
    );
    let noise = textureLoad(tex_noise, sample_pos, 0).rgb;
    return noise;

    // let ctr = f32(frame.frame_id % 500);
    // let noise_uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(textureDimensions(tex_noise).xy);
    // var noise = textureSampleLevel(tex_noise, sampler_noise, noise_uv, 0.0).rgb;
    // noise = (noise + vec3(PHI, SQRT_2, SQRT_3) * ctr) % 1.0;
    // return noise;
}

fn rand_hemisphere_direction(noise: vec2<f32>) -> vec3<f32> {
    let xy = noise * 2.0 - 1.0;
    let z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
    return vec3(xy, z);
}

struct SparseRay {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct SparseRaymarchResult {
    hit: bool,
    distance: f32,
}

fn raymarch_sparse(ray: SparseRay, max_distance: f32) -> SparseRaymarchResult {
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    if all(step == vec3(0)) {
        return SparseRaymarchResult(false, max_distance);
    }

    for (var i = 0u; i < 256u && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {
        let chunk_pos_index = u32(pos.z) * scene.size.x * scene.size.y + u32(pos.y) * scene.size.x + u32(pos.x);
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // if i > 6u {
            //     return true;
            // }
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            for (var j = 0u; j < 100u && all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)); j++) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
                    let t_total = t_entry * 8.0 + t_brick_entry;
                    return SparseRaymarchResult(true, min(max_distance, t_total));
                }

                prev_ray_length = brick_ray_length;

                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * step;
            }
        }

        prev_ray_length = ray_length;

        mask = step_mask(ray_length);
        ray_length += vec3<f32>(mask) * ray_delta;
        pos += vec3<i32>(mask) * step;
    }
    return SparseRaymarchResult(false, max_distance);
}

fn raymarch_shadow(ray: SparseRay) -> bool {
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3<f32>(0.0);
    var mask = vec3(false);

    if all(step == vec3(0)) {
        return false;
    }

    for (var i = 0u; i < 256u && all(pos < vec3<i32>(scene.size)) && all(pos >= vec3(0)); i++) {

        let chunk_pos_index = u32(pos.z) * scene.size.x * scene.size.y + u32(pos.y) * scene.size.x + u32(pos.x);
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // if i > 6u {
            //     return true;
            // }
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            for (var j = 0u; j < 100u && all(brick_pos < vec3(8)) && all(brick_pos >= vec3(0)); j++) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    return true;
                }

                prev_ray_length = brick_ray_length;

                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * step;
            }
        }

        prev_ray_length = ray_length;

        mask = step_mask(ray_length);
        ray_length += vec3<f32>(mask) * ray_delta;
        pos += vec3<i32>(mask) * step;
    }
    return false;
}

struct Ray {
    ls_origin: vec3<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_start: f32,
    in_bounds: bool,
}

struct RaymarchResult {
    hit: bool,
    palette_index: u32,
    surface_normal: vec3<f32>,
    hit_normal: vec3<f32>,
    local_pos: vec3<f32>,
}

fn raymarch(ray: Ray) -> RaymarchResult {
    if !ray.in_bounds {
        return RaymarchResult();
    }

    let size_chunks = vec3<i32>(scene.size);
    let origin = ray.origin / 8.0;
    let dir = ray.direction;

    let ray_step = vec3<i32>(sign(dir));
    let ray_delta = vec3(1.0) / max(vec3(1e-7), abs(dir));

    var pos = vec3<i32>(floor(origin));
    var ray_length = ray_delta * (sign(dir) * (vec3<f32>(pos) - origin) + (sign(dir) * 0.5) + 0.5);
    var prev_ray_length = vec3(0.0);

    if any(pos >= size_chunks) || any(pos < vec3(0)) {
        return RaymarchResult();
    }

    for (var i = 0u; i < DDA_MAX_STEPS; i++) {
        let chunk_pos_index = pos.z * size_chunks.x * size_chunks.y + pos.y * size_chunks.x + pos.x;
        let chunk_index = chunk_indices[chunk_pos_index];

        if chunk_index != 0u {
            // now we do dda within the brick
            var chunk = chunks[chunk_index - 1u];

            var mask = step_mask(prev_ray_length);

            let t_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
            let brick_origin = clamp((origin - vec3<f32>(pos) + dir * (t_entry + 1e-6)) * 8.0, vec3(1e-6), vec3(8.0 - 1e-6));

            var brick_pos = vec3<i32>(floor(brick_origin));
            var brick_ray_length = ray_delta * (sign(dir) * (floor(brick_origin) - brick_origin) + (sign(dir) * 0.5) + 0.5);

            prev_ray_length = vec3<f32>(0.0);

            while all(brick_pos >= vec3(0)) && all(brick_pos < vec3(8)) {
                let voxel_index = (brick_pos.z << 6u) | (brick_pos.y << 3u) | brick_pos.x;
                if (chunk.mask[u32(voxel_index) >> 5u] & (1u << (u32(voxel_index) & 31u))) != 0u {
                    let brick_index = i32(chunk_index - 1u);
                    let base_index = vec3<i32>(
                        (brick_index % size_chunks.x) << 3u,
                        ((brick_index / size_chunks.x) % size_chunks.y) << 3u,
                        (brick_index / (size_chunks.x * size_chunks.y)) << 3u
                    );

                    let packed = textureLoad(brickmap, vec3<i32>(base_index) + brick_pos).r;

                    let palette_index = packed & 0x3ffu;
                    let normal_packed = packed >> 11u;

                    // this is the smooth per-voxel normal
                    let surface_normal = decode_normal_octahedral(normal_packed);

                    // this is the flat normal from the sign of the ray entry
                    let hit_normal = normalize(-vec3<f32>(sign(dir)) * vec3<f32>(mask));

                    // t_total is the total t-value traveled from the camera to the hit voxel
                    // ray.t_start refers to how far we had to project forward to get into the volume
                    let t_brick_entry = min(min(prev_ray_length.x, prev_ray_length.y), prev_ray_length.z);
                    let t_total = ray.t_start + t_entry * 8.0 + t_brick_entry;
                    let local_pos = ray.ls_origin + dir * t_total;

                    return RaymarchResult(true, palette_index, surface_normal, hit_normal, local_pos, );
                }

                prev_ray_length = brick_ray_length;

                // for some reason the branchless approach is faster here,
                // just some weird register optimization with naga,
                // likely doesn't happen on the outer loop since the scene size is non-constant,
                // worth further investigation as it's non-negligable at least on my machine
                mask = step_mask(brick_ray_length);
                brick_ray_length += vec3<f32>(mask) * ray_delta;
                brick_pos += vec3<i32>(mask) * ray_step;
            }
        }

        prev_ray_length = ray_length;

        // simple DDA traversal http://cse.yorku.ca/~amana/research/grid.pdf
        // trying clean "branchless" versions ate up ALU cycles on my nvidia card
        // simple is fast
        if ray_length.x < ray_length.y {
            if ray_length.x < ray_length.z {
                pos.x += ray_step.x;
                if pos.x < 0 || pos.x >= size_chunks.x {
                    break;
                }
                ray_length.x += ray_delta.x;
            } else {
                pos.z += ray_step.z;
                if pos.z < 0 || pos.z >= size_chunks.z {
                    break;
                }
                ray_length.z += ray_delta.z;
            }
        } else {
            if ray_length.y < ray_length.z {
                pos.y += ray_step.y;
                if pos.y < 0 || pos.y >= size_chunks.y {
                    break;
                }
                ray_length.y += ray_delta.y;
            } else {
                pos.z += ray_step.z;
                if pos.z < 0 || pos.z >= size_chunks.z {
                    break;
                }
                ray_length.z += ray_delta.z;
            }
        }
    }

    return RaymarchResult();
}

fn step_mask(ray_length: vec3<f32>) -> vec3<bool> {
    var res = vec3(false);

    res.x = ray_length.x < ray_length.y && ray_length.x < ray_length.z;
    res.y = !res.x && ray_length.y < ray_length.z;
    res.z = !res.x && !res.y;

    return res;
}

fn start_ray(pos: vec2<u32>) -> Ray {
    let camera = environment.camera;
    let dimensions = textureDimensions(tex_out_albedo).xy;

    var pixel_pos = vec2<f32>(pos);
    if frame.taa_enabled == 0u {
        pixel_pos += 0.5;
    } else {
        pixel_pos += environment.jitter;
    }
    let uv = pixel_pos / vec2<f32>(dimensions);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    // aabb simple test and project on the scene volume
    let bd_min = (model.inv_transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let bd_max = (model.inv_transform * vec4(vec3<f32>(scene.size * 8u), 1.0)).xyz;

    // let inv_dir = 1.0 / safe_vec3(ls_direction);
    let inv_dir = safe_inverse(ls_direction);
    let t0 = (bd_min - ls_origin) * inv_dir;
    let t1 = (bd_max - ls_origin) * inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    let t_start = max(0.0, t_near + 1e-7);

    var ray: Ray;
    ray.ls_origin = ls_origin;
    ray.origin = ls_origin + t_start * ls_direction;
    ray.direction = ls_direction;
    ray.t_start = t_start;
    ray.in_bounds = t_near < t_far && t_far > 0.0;
    return ray;
}

fn safe_inverse(v: vec3<f32>) -> vec3<f32> {
    return vec3(
        select(1.0 / v.x, 1e10, v.x == 0.0),
        select(1.0 / v.y, 1e10, v.y == 0.0),
        select(1.0 / v.z, 1e10, v.z == 0.0),
    );
}

fn palette_color(index: u32) -> vec3<f32> {
    return palette.data[index].rgb;
}

fn pcg_hash(seed: ptr<function, u32>) -> u32 {
    let state = *seed;
    *seed = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: ptr<function, u32>) -> f32 {
    return f32(pcg_hash(seed)) * 2.3283064365386963e-10;
}

fn rand_vec3(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(
        rand_float(seed),
        rand_float(seed),
        rand_float(seed)
    ) * 2.0 - 1.0;
}

fn rand_gaussian(seed: ptr<function, u32>) -> f32 {
    let theta = 2 * 3.1415926 * rand_float(seed);
    let rho = sqrt(-2.0 * log(max(1.0 - rand_float(seed), 1e-7)));
    return rho * cos(theta);
}

fn rand_direction(seed: u32) -> vec3<f32> {
    var state = seed;

    state = state * 747796405u + 2891336453u;
    var wx = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    wx = (wx >> 22u) ^ wx;
    let u = f32(wx) * 2.3283064365386963e-10;

    state = state * 747796405u + 2891336453u;
    var wy = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    wy = (wy >> 22u) ^ wy;
    let v = f32(wy) * 2.3283064365386963e-10;

    // let u = rand_float(seed);
    // let v = rand_float(seed);

    let z = u * 2.0 - 1.0;
    let phi = v * 2.0 * 3.14159265359;
    let r = sqrt(max(0.0, 1.0 - z * z));

    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

/// decodes world space normal from lower 21 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
fn decode_normal_octahedral(packed: u32) -> vec3<f32> {
    let x = f32((packed >> 11u) & 0x3ffu) / 1023.0;
    let y = f32((packed >> 1u) & 0x3ffu) / 1023.0;
    let sgn = f32(packed & 1u) * 2.0 - 1.0;
    var res = vec3<f32>(0.);
    res.x = x - y;
    res.y = x + y - 1.0;
    res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
    return normalize(res);
}
