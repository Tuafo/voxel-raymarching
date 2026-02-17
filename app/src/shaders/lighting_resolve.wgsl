@group(0) @binding(0) var main_sampler: sampler;
@group(0) @binding(1) var tex_velocity: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_cur_illum: texture_2d<f32>;

@group(1) @binding(0) var tex_acc_illum: texture_2d<f32>;
@group(1) @binding(1) var tex_out_illum: texture_storage_2d<rgba16float, write>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const SHADOW_ACC_ALPHA: f32 = 0.1;
const RADIANCE_ACC_ALPHA: f32 = 0.1;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let illum = resolve_illum(in);
    // let illum = textureLoad(tex_cur_illum, in.id.xy, 0).rgb;
    textureStore(tex_out_illum, vec2<i32>(in.id.xy), vec4(illum, 1.0));
}

fn resolve_illum(in: ComputeIn) -> vec3<f32> {
    let dimensions = vec2<i32>(textureDimensions(tex_cur_illum).xy);
    let pos = vec2<i32>(in.id.xy);

    let texel_size = 1.0 / vec2<f32>(dimensions);
    let uv = (vec2<f32>(pos) + 0.5) * texel_size;

    let velocity = textureLoad(tex_velocity, pos).rg;

    let cur = textureLoad(tex_cur_illum, pos, 0).rgb;
    let acc_uv = uv - velocity;
    if any(acc_uv < vec2(0.0)) || any(acc_uv >= vec2(1.0)) {
        return cur;
    }
    let acc = sample_catmull_rom_5(acc_uv, vec2<f32>(dimensions));

    let cur_shadow = cur.r;
    let acc_shadow = acc.r;
    let shadow = mix(acc_shadow, cur_shadow, SHADOW_ACC_ALPHA);

    let cur_radiance = cur.g;
    let acc_radiance = acc.g;
    let radiance = mix(acc_radiance, cur_radiance, RADIANCE_ACC_ALPHA);
    // let radiance = cur_radiance;

    return vec3(shadow, radiance, 0.0);
}

struct FilterResult {
    value: f32,
    min: f32,
    max: f32,
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
fn filter_spatial(uv: vec2<f32>) -> FilterResult {
    let dimensions = textureDimensions(tex_cur_illum).xy;
    let texel_size = 1.0 / vec2<f32>(dimensions);

    var weight = 0.0;
    var cur = 0.0;
    var min_val = 1.0;
    var max_val = 0.0;

    let radius = 4.0 * texel_size;

    for (var i = 0u; i < 12u; i++) {
        let sample_uv = uv + FILTER_KERNEL[i] * radius;

        let val = textureSampleLevel(tex_cur_illum, main_sampler, sample_uv, 0).r;
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

// 5-tap approximation of of Catmull-Rom filter
// very similar results to 9-tap
// from https://advances.realtimerendering.com/s2016/Filmic%20SMAA%20v7.pptx
fn sample_catmull_rom_5(uv: vec2<f32>, dimensions: vec2<f32>) -> vec3<f32> {
    let texel_size = 1.0 / dimensions;
    let pos = uv * dimensions;
    let center_pos = floor(pos - 0.5) + 0.5;
    let f = pos - center_pos;
    let f2 = f * f;
    let f3 = f * f2;

    const SHARPNESS: f32 = 0.4;
    let c = SHARPNESS;
    let w0 = -c * f3 + 2.0 * c * f2 - c * f;
    let w1 = (2.0 - c) * f3 - (3.0 - c) * f2 + 1.0;
    let w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c) * f2 + c * f;
    let w3 = c * f3 - c * f2;

    let w12 = w1 + w2;
    let tc12 = texel_size * (center_pos + w2 / w12);
    let center_color = textureSampleLevel(tex_acc_illum, main_sampler, tc12.xy, 0.0).rgb;

    let tc0 = texel_size * (center_pos - 1.0);
    let tc3 = texel_size * (center_pos + 2.0);

    var color = vec4(0.0);
    color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc12.x, tc0.y), 0.0).rgb, 1.0) * (w12.x * w0.y);
    color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc0.x, tc12.y), 0.0).rgb, 1.0) * (w0.x * w12.y);
    color += vec4(center_color, 1.0) * (w12.x * w12.y);
    color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc3.x, tc12.y), 0.0).rgb, 1.0) * (w3.x * w12.y);
    color += vec4(textureSampleLevel(tex_acc_illum, main_sampler, vec2(tc12.x, tc3.y), 0.0).rgb, 1.0) * (w12.x * w3.y);

    let res = color.rgb / color.a;
    return max(res, vec3(0.0));
}
