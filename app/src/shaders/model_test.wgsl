struct Camera {
    view_proj_matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
}
@group(1) @binding(0) var<uniform> model: Model;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) color: vec3<f32>,
}


@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) material_id: u32,
) -> VertexOutput {
    // out.Position = camera.view_proj_matrix *  vec4(position, 1.0);

    // let albedo = vec3(1.0);
    let albedo = rand_vec3(material_id);
    let ws_normal = normalize(model.normal_matrix * normal);


    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));
    let diff = max(dot(normal, ws_light_dir), 0.0);
    let color = albedo * (diff * 0.8 + 0.2);

    var out: VertexOutput;
    out.Position = camera.view_proj_matrix * model.matrix * vec4(position, 1.0);
    // out.color = ws_normal;
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}

// 1. The Hash Function (PCG style)
// Takes a seed, returns a random u32.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 2. The Converter
// returns vec3<f32> in range [0.0, 1.0]
fn rand_vec3(seed: u32) -> vec3<f32> {
    // Chain the hashes to get 3 unique numbers from 1 seed
    let h1 = pcg_hash(seed);
    let h2 = pcg_hash(h1);
    let h3 = pcg_hash(h2);

    // Convert u32 to f32 [0, 1]
    // (1.0 / 4294967295.0 = 2.3283064365386963e-10)
    let inv_max = 2.3283064365386963e-10;

    return vec3<f32>(
        f32(h1) * inv_max,
        f32(h2) * inv_max,
        f32(h3) * inv_max
    );
}
