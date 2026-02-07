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
    size: vec3<f32>, // after scaling by voxel scale
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
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var out_voxels: texture_storage_3d<rgba8unorm, write>;
    
struct Primitive {
    matrix: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    material_id: u32,
    index_count: u32,
}
@group(1) @binding(0) var<uniform> primitive: Primitive;
@group(1) @binding(1) var<storage, read> indices: array<u32>;
@group(1) @binding(2) var<storage, read> positions: array<f32>;
    
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

    var min_bd = (min(min(v0, v1), v2) - scene.base) * scene.scale;
    var max_bd = (max(max(v0, v1), v2) - scene.base) * scene.scale;

    min_bd = max(floor(min_bd), vec3(0.0));
    max_bd = min(ceil(max_bd), scene.size);

    let min_bd_i = vec3<u32>(min_bd);
    let max_bd_i = vec3<u32>(max_bd);

    for (var x = min_bd_i.x; x < max_bd_i.x; x++) {
        for (var y = min_bd_i.y; y < max_bd_i.y; y++) {
            for (var z = min_bd_i.z; z < max_bd_i.z; z++) {
                // now, test for intersection
                let center = vec3(f32(x), f32(y), f32(z)) + 0.5;

                let v0 = v0 - center;
                let v1 = v1 - center;
                let v2 = v2 - center;

                if any(min(min(v0, v1), v2) > vec3(0.5)) || any(max(max(v0, v1), v2) < vec3(-0.5)) {
                    continue;
                }

                let e0 = v1 - v0;
                let e1 = v2 - v1;
                let e2 = v0 - v2;
                let normal = cross(e0, e1);
                let d = dot(normal, v0);
                let r = 0.5 * (abs(normal.x) + abs(normal.y) + abs(normal.z));
                if abs(d) > r {
                    continue;
                }

                // test 9 axes
                let axes = array<vec3<f32>, 9>(
                    vec3(0.0, -e0.z, e0.y),
                    vec3(0.0, -e1.z, e1.y),
                    vec3(0.0, -e2.z, e2.y),
                    vec3(e0.z, 0.0, -e0.x),
                    vec3(e1.z, 0.0, -e1.x),
                    vec3(e2.z, 0.0, -e2.x),
                    vec3(-e0.y, e0.x, 0.0),
                    vec3(-e1.y, e1.x, 0.0),
                    vec3(-e2.y, e2.x, 0.0),
                );

                var separated = false;
                for (var i = 0; i < 9; i++) {
                    let axis = axes[i];
                    let p0 = dot(v0, axis);
                    let p1 = dot(v1, axis);
                    let p2 = dot(v2, axis);
                    let r = 0.5 * (abs(axis.x) + abs(axis.y) + abs(axis.z));
                    if max(max(p0, p1), p2) < -r || min(min(p0, p1), p2) > r {
                        separated = true;
                        break;
                    }
                }

                if separated {
                    continue;
                }

                // intersection passed
                textureStore(out_voxels, vec3<u32>(x, y, z), vec4(1.0));
            }
        }


        let material = materials[primitive.material_id];
    }