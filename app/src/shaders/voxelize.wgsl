// struct Camera {
//     view_proj_matrix: mat4x4<f32>,
// }
// @group(0) @binding(0) var<uniform> camera: Camera;

// struct Node {
//     matrix: mat4x4<f32>,
//     normal_matrix: mat3x3<f32>,
// }
// @group(0) @binding(1) var<storage, read> nodes: array<Node>;

// struct Primitive {
//     min_bounds: vec3<f32>,
//     node_id: u32,
//     max_bounds: vec3<f32>,
//     material_id: u32,
//     index_offset: u32,
//     index_count: u32,
//     vertex_pos_offset: u32,
//     vertex_normal_offset: u32,
// }
// @group(0) @binding(2) var<storage, read> primitives: array<Primitive>;

// struct Material {
//     base_albedo: vec4<f32>,
//     metallic: f32,
//     roughness: f32,
//     normal_scale: f32,
//     albedo_index: i32,
//     normal_index: i32,
// }
// @group(0) @binding(3) var<storage, read> materials: array<Material>;

// @group(0) @binding(4) var textures: binding_array<texture_2d<f32>>;
// @group(0) @binding(5) var tex_sampler: sampler;
// @group(0) @binding(6) var out_voxels: texture_storage_3d<rgba8unorm, write>;

// @group(1) @binding(0) var<storage, read> indices: array<u32>;
// @group(1) @binding(1) var<storage, read> positions: array<f32>;

// struct PrimitiveId {
    //     // making this a struct as the release notes cite an issue on DX12
    //     id: u32,
    // }
    // var<immediate> primitive_id: PrimitiveId;
    
struct Scene {
    base: vec3<f32>, // before scaling by voxel scale
    size: vec3<f32>, // before scaling by voxel scale
    scale: f32, // number of voxels per unit
}
struct Material {
    base_albedo: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    albedo_index: i32,
    normal_index: i32,
}
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var out_voxels: texture_storage_3d<rgba8unorm, write>;

@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;

struct Primitive {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    material_id: u32,
    index_count: u32,
}
@group(2) @binding(0) var<uniform> primitive: Primitive;
@group(2) @binding(1) var<storage, read> indices: array<u32>;
@group(2) @binding(2) var<storage, read> positions: array<f32>;
@group(2) @binding(3) var<storage, read> normals: array<f32>;
@group(2) @binding(4) var<storage, read> tangents: array<f32>;
@group(2) @binding(5) var<storage, read> uvs: array<f32>;
    
struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(64, 1, 1)
fn compute_main(in: ComputeIn) {
    if in.id.x * 3u >= primitive.index_count {
        return;
    }

    let index_base = in.id.x * 3u;
    let i0 = indices[index_base + 0u];
    let i1 = indices[index_base + 1u];
    let i2 = indices[index_base + 2u];

    let l_p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
    let l_p1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]);
    let l_p2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]);

    // triangle ws vertex positions
    let v0 = ((primitive.matrix * vec4(l_p0, 1.0)).xyz - scene.base) * scene.scale;
    let v1 = ((primitive.matrix * vec4(l_p1, 1.0)).xyz - scene.base) * scene.scale;
    let v2 = ((primitive.matrix * vec4(l_p2, 1.0)).xyz - scene.base) * scene.scale;

    var min_bd = min(min(v0, v1), v2);
    var max_bd = max(max(v0, v1), v2);

    min_bd = max(floor(min_bd), vec3(0.0));
    max_bd = min(ceil(max_bd), scene.size * scene.scale - 1.0);

    let min_bd_i = vec3<u32>(min_bd);
    let max_bd_i = vec3<u32>(max_bd);

    for (var x = min_bd_i.x; x < max_bd_i.x; x++) {
        for (var y = min_bd_i.y; y < max_bd_i.y; y++) {
            for (var z = min_bd_i.z; z < max_bd_i.z; z++) {
                let center = vec3(f32(x), f32(y), f32(z)) + 0.5;
                if test_voxel(center, v0, v1, v2) {
                    // intersection passed
                    let point = project_onto_plane(center, v0, v1, v2);
                    let weights = tri_weights(point, v0, v1, v2);

                    let uv_0 = vec2(uvs[i0 * 2u], uvs[i0 * 2u + 1u]);
                    let uv_1 = vec2(uvs[i1 * 2u], uvs[i1 * 2u + 1u]);
                    let uv_2 = vec2(uvs[i2 * 2u], uvs[i2 * 2u + 1u]);
                    let uv = weights.x * uv_0 + weights.y * uv_1 + weights.z * uv_2;

                    let material = materials[primitive.material_id];

                    var albedo = vec3(0.0);
                    if material.albedo_index >= 0 {
                        albedo = textureSampleLevel(textures[material.albedo_index], tex_sampler, uv, 0.0).rgb;
                    }

                    textureStore(out_voxels, vec3<i32>(i32(x), i32(y), i32(z)), vec4(albedo, 1.0));
                }
            }
        }
    }
}

const EXTENT: f32 = 0.6;

fn test_voxel(center: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> bool {
    let v0 = p0 - center;
    let v1 = p1 - center;
    let v2 = p2 - center;

    if any(min(min(v0, v1), v2) > vec3(EXTENT)) || any(max(max(v0, v1), v2) < vec3(-EXTENT)) {
        return false;
    }

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let normal = cross(e0, e1);
    let d = dot(normal, v0);
    let r = 0.5 * (abs(normal.x) + abs(normal.y) + abs(normal.z));
    if abs(d) > r {
        return false;
    }

    let axes = array<vec3<f32>, 9>(
        vec3(0.0, e0.z, -e0.y),
        vec3(0.0, e1.z, -e1.y),
        vec3(0.0, e2.z, -e2.y),
        vec3(-e0.z, 0.0, e0.x),
        vec3(-e1.z, 0.0, e1.x),
        vec3(-e2.z, 0.0, e2.x),
        vec3(e0.y, -e0.x, 0.0),
        vec3(e1.y, -e1.x, 0.0),
        vec3(e2.y, -e2.x, 0.0),
    );

    for (var i = 0; i < 9; i++) {
        let axis = axes[i];
        let p0 = dot(v0, axis);
        let p1 = dot(v1, axis);
        let p2 = dot(v2, axis);
        let r = 0.5 * (abs(axis.x) + abs(axis.y) + abs(axis.z));
        if max(max(p0, p1), p2) < -r || min(min(p0, p1), p2) > r {
            return false;
        }
    }
    return true;
}

fn tri_weights(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    return vec3(u, v, w);
}

fn project_onto_plane(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let normal = normalize(cross(b - a, c - a));
    let dist = dot(normal, p - a);
    return p - dist * normal;
}

fn hash(n: u32) -> u32 {
    var x = n;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

fn rand_vec3(seed: u32) -> vec3<f32> {
    let h1 = hash(seed);
    let h2 = hash(h1);
    let h3 = hash(h2);

    return vec3<f32>(
        f32(h1) / 4294967295.0,
        f32(h2) / 4294967295.0,
        f32(h3) / 4294967295.0
    );
}