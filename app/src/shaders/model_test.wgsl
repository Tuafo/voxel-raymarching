struct Camera {
    view_proj_matrix: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

struct Model {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
}
@group(1) @binding(0) var<uniform> model: Model;

@group(2) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(2) @binding(1) var tex_sampler: sampler;
struct Material {
    base_albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
    double_sided: u32,
}
@group(2) @binding(2) var<storage, read> materials: array<Material>;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) material_id: u32,
    @location(1) vertex_color: vec3<f32>,
    @location(2) vertex_normal: vec3<f32>,
    @location(3) vertex_tangent: vec3<f32>,
    @location(4) vertex_bitangent: vec3<f32>,
    @location(5) uv: vec2<f32>,
}


@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) material_id: u32,
    @location(5) ls_min: vec3<f32>,
    @location(6) ls_max: vec3<f32>,
) -> VertexOutput {
    let v_color = vec3(0.0);
    let v_normal = normalize(model.normal_matrix * normal);
    let v_tangent = normalize((model.matrix * vec4(tangent.xyz, 0.0)).xyz);
    let v_bitangent = cross(v_normal, v_tangent) * tangent.w;

    let ws_min = (model.matrix * vec4(ls_min, 1.0)).xyz;
    let ws_max = (model.matrix * vec4(ls_max, 1.0)).xyz;
    let ctr = (ws_min + ws_max) * 0.5;

    var out: VertexOutput;
    out.Position = camera.view_proj_matrix * model.matrix * vec4(position, 1.0);
    out.material_id = material_id;
    // out.vertex_color = v_color;
    out.vertex_color = mix(v_color, vec3(1.0, 0.0, 0.0), length((model.matrix * vec4(position, 1.0)).xyz - ctr) * 0.05);
    out.vertex_normal = v_normal;
    out.vertex_tangent = v_tangent;
    out.vertex_bitangent = v_bitangent;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[in.material_id];

    var albedo = in.vertex_color * 0.000001;
    if material.albedo_index >= 0 {
        albedo += textureSample(textures[material.albedo_index], tex_sampler, in.uv).rgb;
    }

    var ws_normal = normalize(in.vertex_normal);
    if material.normal_index >= 0 {
        ws_normal = textureSample(textures[material.normal_index], tex_sampler, in.uv).rgb * 2.0 - 1.0;
        ws_normal.x = ws_normal.x * material.normal_scale;
        ws_normal.y = ws_normal.y * material.normal_scale;
        ws_normal = ws_normal.x * in.vertex_tangent + ws_normal.y * in.vertex_bitangent + ws_normal.z * in.vertex_normal;
        ws_normal = normalize(ws_normal);
        // var ws_tangent = normalize(model.in.vertex_tangent.xyz);
        // let ws_bitangent = cross(ws_normal, ws_tangent) * in.vertex_tangent.w;
        // let tbn = mat3x3<f32>(
        //     ws_tangent,
        //     ws_bitangent,
        //     ws_normal
        // );

        // tangent_normal = textureSample(textures[material.normal_index], tex_sampler, in.uv).rgb * 2.0 - 1.0;
        // ws_normal = normalize(tbn * tangent_normal);
    }
    // ws_normal = ws_normal * 0.000001 + normalize(in.vertex_normal);

    let ws_light_dir = normalize(vec3(3.0, -1.0, 10.0));
    let diff = max(dot(ws_normal, ws_light_dir), 0.0);
    let color = albedo * (diff * 0.8 + 0.2);

    return vec4<f32>(color, 1.0);
    // return vec4<f32>(color * 0.0000001 + ws_normal * 0.25, 1.0);
    // return vec4<f32>(color * 0.0000001 + normalize(tangent_normal) * 0.5 + 0.5, 1.0);
    // return vec4<f32>(color * 0.0000001 + vec3(in.vertex_tangent.w * 0.5 + 0.5), 1.0);
}
