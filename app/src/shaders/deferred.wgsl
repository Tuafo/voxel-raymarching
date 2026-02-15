@group(0) @binding(0) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var tex_albedo: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var tex_normal: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var tex_depth: texture_storage_2d<r32float, read>;
@group(0) @binding(4) var tex_velocity: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0) var tex_illumination: texture_storage_2d<rgba16float, read>;

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

const AMBIENT_INTENSITY: f32 = 0.15;
const DIRECTIONAL_INTENSITY: f32 = 0.85;

@compute @workgroup_size(8, 8, 1)
fn compute_main(in: ComputeIn) {
    let pos = vec2<i32>(in.id.xy);

    let albedo_sample = textureLoad(tex_albedo, pos);
    let albedo = albedo_sample.rgb;
    let illumination = textureLoad(tex_illumination, pos).rgb;

    let normal = normalize(textureLoad(tex_normal, pos).rgb);
    let depth = textureLoad(tex_depth, pos).r;
    let velocity = textureLoad(tex_velocity, pos).rg;

    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));

    let diff = max(dot(normal, ws_light_dir), 0.0);

    var color = albedo * (diff * DIRECTIONAL_INTENSITY + AMBIENT_INTENSITY) * illumination;

    // color *= 0.5;
    // color += vec3(abs(velocity) * 50.0, 0.0);

    // color *= 0.0001;
    // color += illumination;

    textureStore(out_color, vec2<i32>(in.id.xy), vec4(color, 1.0));
}