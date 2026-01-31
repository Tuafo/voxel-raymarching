@group(0) @binding(0) var tex_albedo: texture_2d<f32>;
@group(0) @binding(1) var tex_normal: texture_2d<f32>;
@group(0) @binding(2) var main_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = vec2(position.x + 1.0, 1.0 - position.y) * 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo = textureSample(tex_albedo, main_sampler, in.uv).rgb;
    var normal = textureSample(tex_normal, main_sampler, in.uv).rgb;

    let ws_light_dir = normalize(vec3(-4.0, 2.0, 10.0));

    let diff = max(dot(normal, ws_light_dir), 0.0);

    let color = albedo * (diff + 0.2);
    // color *= 0.000001;
    // color += res.normal;

    return vec4<f32>(color, 1.0);
}
