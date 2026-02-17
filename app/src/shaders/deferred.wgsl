@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_normal: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_depth: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var tex_velocity: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_illumination: texture_2d<f32>;

@group(2) @binding(0) var sampler_linear: sampler;
@group(2) @binding(1) var sampler_noise: sampler;
@group(2) @binding(2) var tex_noise: texture_2d<f32>;

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
struct FrameMetadata {
    frame_id: u32,
    taa_enabled: u32,
    fxaa_enabled: u32,
}
@group(3) @binding(0) var<uniform> environment: Environment;
@group(3) @binding(1) var<uniform> frame: FrameMetadata;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const AMBIENT_INTENSITY: f32 = 0.15;
const DIRECTIONAL_INTENSITY: f32 = 0.85;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);
    let dimensions = vec2<i32>(textureDimensions(tex_albedo).xy);

    let noise = blue_noise(in.id.xy);

    let albedo_sample = textureLoad(tex_albedo, pos);
    let albedo = albedo_sample.rgb;
    let illumination = gather_illumination(pos, noise);
    let shadow = illumination.r;
    let radiance = illumination.g;

    let normal = normalize(textureLoad(tex_normal, pos).rgb);
    let depth = textureLoad(tex_depth, pos).r;
    let velocity = textureLoad(tex_velocity, pos).rg;

    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

    let diff = max(dot(normal, ws_light_dir), 0.0);

    var color = albedo * radiance * (diff * DIRECTIONAL_INTENSITY * shadow + AMBIENT_INTENSITY);

    // color *= 0.5;
    // color += vec3(abs(velocity) * 50.0, 0.0);

    // color *= 0.0001;
    // color += shadow_factor;
    // color *= 0.0001;
    // color += radiance;

    // let noise = blue_noise(pos, dimensions);
    // color *= 0.0001;
    // color += vec3(noise.r);

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}

fn gather_illumination(pos: vec2<i32>, noise: vec3<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(tex_illumination).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;

    if environment.filter_shadows == 0u {
        return textureLoad(tex_illumination, pos, 0).rgb;
    } else {
        // let lighting = textureLoad(tex_illumination, pos, 0).rgb;
        // let shadow = filter_spatial(uv, texel_size, noise).value.r;
        // return vec3(shadow, lighting.gb);
        return filter_spatial(uv, texel_size, noise).value;
    }
}

const FILTER_KERNEL: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    vec2<f32>(-0.326212, -0.405805),
    vec2<f32>(-0.840144, -0.073580),
    vec2<f32>(-0.695914,  0.457137),
    vec2<f32>(-0.203345,  0.620716),
    vec2<f32>( 0.962340, -0.194983),
    vec2<f32>( 0.473434, -0.480026),
    vec2<f32>( 0.519456,  0.767022),
    vec2<f32>( 0.185461, -0.893124),
    vec2<f32>( 0.507431,  0.064425),
    vec2<f32>( 0.896420,  0.412458),
    vec2<f32>(-0.321940, -0.932615),
    vec2<f32>(-0.791559, -0.597705)
);
struct FilterResult {
    value: vec3<f32>,
    min: vec3<f32>,
    max: vec3<f32>,
}
fn filter_spatial(uv: vec2<f32>, texel_size: vec2<f32>, noise: vec3<f32>) -> FilterResult {
    var weight = 0.0;
    var cur = vec3(0.0);
    var min_val = vec3(1.0);
    var max_val = vec3(0.0);

    let radius = environment.shadow_filter_radius * texel_size;
    let t = noise.r * 6.2831853;
    let s_t = sin(t);
    let c_t = cos(t);
    let rotation = mat2x2<f32>(c_t, s_t, -s_t, c_t);

    for (var i = 0u; i < 12u; i++) {
        let offset = rotation * FILTER_KERNEL[i];
        let sample_uv = uv + offset * radius;

        let val = textureSampleLevel(tex_illumination, sampler_linear, sample_uv, 0).rgb;
        min_val = min(min_val, val);
        max_val = max(max_val, val);
        cur += val;
        weight += 1.0;
    }
    cur /= weight;

    var res: FilterResult;
    res.value = cur;
    res.min = min_val;
    res.max = max_val;
    return res;
}

fn blue_noise(pos: vec2<u32>) -> vec3<f32> {
    let noise_pos = vec2<u32>(pos.x & 0x7fu, pos.y & 0x7fu);
    let noise = textureLoad(tex_noise, vec2<i32>(noise_pos), 0).rgb;
    return noise;
}
