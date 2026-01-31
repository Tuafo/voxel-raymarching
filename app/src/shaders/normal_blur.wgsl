@group(0) @binding(0) var out_normal: texture_storage_2d<rgba8snorm, write>;
@group(0) @binding(1) var tex_normal: texture_2d<f32>;
@group(0) @binding(2) var main_sampler: sampler;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(tex_normal).xy;

    let texel_size = 1.0 / vec2<f32>(dimensions);

    let uv = vec2<f32>(in.id.xy) * texel_size;

    let x = uv.x;
    let y = uv.y;
    let dx = texel_size.x;
    let dy = texel_size.y;


    let a = textureSampleLevel(tex_normal, main_sampler, vec2(x - 2.0 * dx, y + 2.0 * dy), 0.0).rgb;
    let b = textureSampleLevel(tex_normal, main_sampler, vec2(x + 0.0 * dx, y + 2.0 * dy), 0.0).rgb;
    let c = textureSampleLevel(tex_normal, main_sampler, vec2(x + 2.0 * dx, y + 2.0 * dy), 0.0).rgb;

    let d = textureSampleLevel(tex_normal, main_sampler, vec2(x - 2.0 * dx, y + 0.0 * dy), 0.0).rgb;
    let e = textureSampleLevel(tex_normal, main_sampler, vec2(x + 0.0 * dx, y + 0.0 * dy), 0.0).rgb;
    let f = textureSampleLevel(tex_normal, main_sampler, vec2(x + 2.0 * dx, y + 0.0 * dy), 0.0).rgb;

    let g = textureSampleLevel(tex_normal, main_sampler, vec2(x - 2.0 * dx, y - 2.0 * dy), 0.0).rgb;
    let h = textureSampleLevel(tex_normal, main_sampler, vec2(x + 0.0 * dx, y - 2.0 * dy), 0.0).rgb;
    let i = textureSampleLevel(tex_normal, main_sampler, vec2(x + 2.0 * dx, y - 2.0 * dy), 0.0).rgb;

    let j = textureSampleLevel(tex_normal, main_sampler, vec2(x - 1.0 * dx, y + 1.0 * dy), 0.0).rgb;
    let k = textureSampleLevel(tex_normal, main_sampler, vec2(x + 1.0 * dx, y + 1.0 * dy), 0.0).rgb;
    let l = textureSampleLevel(tex_normal, main_sampler, vec2(x - 1.0 * dx, y - 1.0 * dy), 0.0).rgb;
    let m = textureSampleLevel(tex_normal, main_sampler, vec2(x + 1.0 * dx, y - 1.0 * dy), 0.0).rgb;

    var res = vec3<f32>(0.0);
    res += e * 0.125;
    res += (a + c + g + i) * 0.03125;
    res += (b + d + f + h) * 0.0625;
    res += (j + k + l + m) * 0.125;
    res = normalize(res);

    textureStore(out_normal, vec2<i32>(in.id.xy), vec4(res, 1.0));
}
