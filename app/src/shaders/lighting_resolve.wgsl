@group(0) @binding(0) var main_sampler: sampler;
@group(0) @binding(1) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_cur_illum: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_acc_illum: texture_storage_2d<rgba16float, read>;
@group(1) @binding(1) var tex_out_illum: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var tex_normal: texture_storage_2d<r32uint, read>;
@group(1) @binding(3) var tex_depth: texture_storage_2d<r32float, read>;

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
@group(2) @binding(0) var<uniform> environment: Environment;
@group(2) @binding(1) var<uniform> frame: FrameMetadata;
@group(2) @binding(2) var<uniform> model: Model;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    resolve_illum(in);
}

fn resolve_illum(in: ComputeIn) {
    let dimensions = vec2<i32>(textureDimensions(tex_cur_illum).xy);
    let pos = vec2<i32>(in.id.xy);
    
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
    
    let velocity = textureLoad(tex_velocity, pos).rg;
    let acc_uv = uv - velocity;
    let acc_pos = vec2<i32>(acc_uv * vec2<f32>(dimensions));
    
    let cur = textureLoad(tex_cur_illum, pos).rgb;
    var history_length = 1u;

    var res = cur;
    
    if is_reprojection_valid(pos, acc_pos, dimensions) {
        let acc_pos = vec2<i32>(acc_uv * vec2<f32>(dimensions));
        
        // let acc = sample_catmull_rom_5(acc_uv, vec2<f32>(dimensions));
        let acc = textureLoad(tex_acc_illum, acc_pos).rgb;
        
        let alpha = 0.01;

        res = mix(acc, cur, alpha);
        res.g = 0.0;
    } else {
        res.g = 1.0;
    }
 
    textureStore(tex_out_illum, pos, vec4<f32>(res, 1.));
}

fn is_reprojection_valid(pos: vec2<i32>, acc_pos: vec2<i32>, dimensions: vec2<i32>) -> bool {
    if any(acc_pos < vec2(0)) || any(acc_pos >= dimensions) {
        return false;
    }

    // let texel_size = 1.0 / vec2<f32>(dimensions);

    // let cur = gather_surface(pos, texel_size);
    // let acc = gather_surface(acc_pos, texel_size);

    // let plane_distance = abs(dot(cur.ws_pos - acc.ws_pos, cur.ws_normal));
    // if plane_distance > 0.1 {
    //     return false;
    // }
    
    return true;
}

// 5-tap approximation of of Catmull-Rom filter
// very similar results to 9-tap
// from https://advances.realtimerendering.com/s2016/Filmic%20SMAA%20v7.pptx
// fn sample_catmull_rom_5(uv: vec2<f32>, dimensions: vec2<f32>) -> vec3<f32> {
//     let texel_size = 1.0 / dimensions;
//     let pos = uv * dimensions;
//     let center_pos = floor(pos - 0.5) + 0.5;
//     let f = pos - center_pos;
//     let f2 = f * f;
//     let f3 = f * f2;

//     const SHARPNESS: f32 = 0.4;
//     let c = SHARPNESS;
//     let w0 = -c * f3 + 2.0 * c * f2 - c * f;
//     let w1 = (2.0 - c) * f3 - (3.0 - c) * f2 + 1.0;
//     let w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c) * f2 + c * f;
//     let w3 = c * f3 - c * f2;

//     let w12 = w1 + w2;
//     let tc12 = texel_size * (center_pos + w2 / w12);
//     let center_color = textureSampleLevel(tex_acc_illum, main_sampler, tc12.xy, 0.0).rgb;

//     let tc0 = texel_size * (center_pos - 1.0);
//     let tc3 = texel_size * (center_pos + 2.0);

//     var color = vec4(0.0);
//     color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc12.x, tc0.y), 0.0).rgb, 1.0) * (w12.x * w0.y);
//     color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc0.x, tc12.y), 0.0).rgb, 1.0) * (w0.x * w12.y);
//     color += vec4(center_color, 1.0) * (w12.x * w12.y);
//     color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc3.x, tc12.y), 0.0).rgb, 1.0) * (w3.x * w12.y);
//     color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc12.x, tc3.y), 0.0).rgb, 1.0) * (w12.x * w3.y);

//     let res = color.rgb / color.a;
//     return max(res, vec3(0.0));
// }


struct SurfaceData {
    ws_pos: vec3<f32>,
    ws_normal: vec3<f32>,
}

fn gather_surface(pos: vec2<i32>, texel_size: vec2<f32>) -> SurfaceData {
    let depth = textureLoad(tex_depth, pos).r;
    let packed = textureLoad(tex_normal, pos).r;

    let uv = (vec2<f32>(pos) + 0.5) * texel_size;
	let uv_jittered  = (vec2<f32>(pos) + environment.camera.jitter) * texel_size;

    let ray = primary_ray(select(uv_jittered, uv, frame.taa_enabled == 0u));
    
    let voxel = unpack_voxel(packed);

    let ls_pos = ray.ls_origin + ray.direction * depth;
    let ws_pos = (model.transform * vec4(ls_pos, 1.0)).xyz;

    var res: SurfaceData;
    res.ws_pos = ws_pos;
    res.ws_normal = voxel.ws_normal;
    return res;
}

struct Ray {
    ls_origin: vec3<f32>,
    direction: vec3<f32>,
};

fn primary_ray(uv: vec2<f32>) -> Ray {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let ts_near = environment.camera.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let ws_near = ts_near.xyz / ts_near.w;

    let ts_far = environment.camera.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ws_far = ts_far.xyz / ts_far.w;

    let ws_direction = normalize(ws_far - ws_near);
    let ws_origin = ws_near;

    let ls_origin = (model.inv_transform * vec4(ws_origin, 1.0)).xyz;
    let ls_direction = normalize((model.inv_transform * vec4(ws_direction, 0.0)).xyz);

    var ray: Ray;
    ray.ls_origin = ls_origin;
    ray.direction = ls_direction;
    return ray;
}

struct Voxel {
    ws_normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    hit_mask: vec3<bool>,
}
fn unpack_voxel(packed: u32) -> Voxel {
    var res: Voxel;
    res.ws_normal = decode_normal_octahedral(packed >> 11u);
    res.metallic = f32((packed >> 10u) & 1u);
    res.roughness = f32((packed >> 6u) & 15u) / 16.0;
    res.hit_mask = decode_hit_mask((packed >> 3u) & 7u);
    return res;
}

fn decode_hit_mask(packed: u32) -> vec3<bool> {
    let mask = vec3<u32>(
        (packed >> 2u) & 1u,
        (packed >> 1u) & 1u,
        packed & 1u,
    );
    return vec3<bool>(mask);
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