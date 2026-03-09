@group(0) @binding(0) var<storage, read_write> chunks: array<u32>;
@group(0) @binding(1) var<storage, read_write> voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write> child_index_map: array<u32>;
// @group(0) @binding(2) var tex_child_index_map: texture_storage_3d<r32uint, read_write>;
// @group(0) @binding(1) var voxels: texture_storage_3d<r32uint, write>;

struct Allocator {
    chunk_count: atomic<u32>, // the total number of index chunks allocated
    leaf_count: atomic<u32>, // the total number of leaf chunks allocated
    voxel_count: atomic<u32>, // total number of voxels in the scene
}
struct Palette {
    data: array<vec4<f32>, 1024>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var raw_voxels: texture_storage_3d<rg32uint, read>;
@group(1) @binding(2) var<uniform> palette: Palette;

struct ComputeIn {
    @builtin(global_invocation_id) pos: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_pos: vec3<u32>,
    @builtin(local_invocation_id) local_pos: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
}

struct Shared {
    mask: array<atomic<u32>, 2>,
    is_empty: bool,
    chunk_index: u32,
}
var<workgroup> group: Shared;

const LEAF_CHUNK_WORDS: u32 = 64u;
const INDEX_ENTRY_WORDS: u32 = 3u;
const INDEX_CHUNK_WORDS: u32 = 64u * INDEX_ENTRY_WORDS;

@compute @workgroup_size(4, 4, 4)
fn compute_leaf(in: ComputeIn) {
    let raw = textureLoad(raw_voxels, vec3<i32>(in.pos)).rgb;
    let albedo_packed = raw.r;
    let normal_metallic_roughness = raw.g;
    let material_id = raw.b;

    let albedo_srgb = vec3<f32>(
        f32(albedo_packed >> 24u) / 255.0,
        f32((albedo_packed >> 16u) & 0xffu) / 255.0,
        f32((albedo_packed >> 8u) & 0xffu) / 255.0,
    );
    let albedo_linear = srgb_to_linear(albedo_srgb);
    let albedo_oklab = linear_rgb_to_oklab(albedo_linear);

    var palette_index = 0u;
    if albedo_packed > 0u {
        var min_distance = 1e10;
        for (var i = 1u; i < 1024u; i++) {
            let palette_rgb = palette.data[i].rgb;
            let palette_oklab = linear_rgb_to_oklab(palette_rgb);

            let d = distance(palette_oklab, albedo_oklab);
            if d < min_distance {
                palette_index = i;
                min_distance = d;
            }
        }

        atomicOr(&group.mask[in.local_index >> 5u], 1u << (in.local_index & 31u));
    }

    let packed = normal_metallic_roughness | palette_index;

    workgroupBarrier();

    if in.local_index == 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );
        
        group.is_empty = (mask[0] == 0u) && (mask[1] == 0u);
        if !group.is_empty {
            group.chunk_index = atomicAdd(&alloc.leaf_count, 1u);
        }

        // used by next pass
        let parent_index = in.workgroup_pos.x
        + in.workgroup_pos.y * in.num_workgroups.x
        + in.workgroup_pos.z * in.num_workgroups.x * in.num_workgroups.y;
        let parent_base = parent_index * 3u;

        child_index_map[parent_base + 0u] = (group.chunk_index << 1u) | 1u; // lowest bit is 1, as it's a leaf chunk
        child_index_map[parent_base + 1u] = mask[0];
        child_index_map[parent_base + 2u] = mask[1];
    }

    workgroupBarrier();

    if !group.is_empty {
        let chunk_base = group.chunk_index * LEAF_CHUNK_WORDS;
        voxels[chunk_base + in.local_index] = packed;
    }
}

@compute @workgroup_size(4, 4, 4)
fn compute_index_base(in: ComputeIn) {
    let child_offset = in.pos.x
    + in.pos.y * (in.num_workgroups.x << 2u)
    + in.pos.z * (in.num_workgroups.x << 2u) * (in.num_workgroups.y << 2u); 
    let child_base = child_offset * 3u;

    let child_index = child_index_map[child_base + 0u];
    let child_mask = array<u32, 2>(
        child_index_map[child_base + 1u],
        child_index_map[child_base + 2u],
    );

    if child_mask[0] == 0u && child_mask[1] == 0u {
        atomicOr(&group.mask[in.local_index >> 5u], 1u << (in.local_index & 31u));
    }

    workgroupBarrier();

    if in.local_index == 0u {
        let mask = array<u32, 2>(
            atomicLoad(&group.mask[0]),
            atomicLoad(&group.mask[1]),
        );
        
        group.is_empty = (mask[0] == 0u) && (mask[1] == 0u);
        if !group.is_empty {
            group.chunk_index = atomicAdd(&alloc.chunk_count, 1u);
        }

        // used by next pass
        let parent_index = in.workgroup_pos.x
        + in.workgroup_pos.y * in.num_workgroups.x
        + in.workgroup_pos.z * in.num_workgroups.x * in.num_workgroups.y;
        let parent_base = parent_index * 3u;

        child_index_map[parent_base + 0u] = (group.chunk_index << 1u); // lowest bit is 0, as it's not a leaf chunk
        child_index_map[parent_base + 1u] = mask[0];
        child_index_map[parent_base + 2u] = mask[1];
    }

    workgroupBarrier();

    if !group.is_empty {
        let chunk_base = (group.chunk_index - 1u) * INDEX_CHUNK_WORDS;
        let entry_base = chunk_base + in.local_index * INDEX_ENTRY_WORDS;
        chunks[entry_base + 0u] = child_index;
        chunks[entry_base + 1u] = child_mask[0];
        chunks[entry_base + 2u] = child_mask[1];
    }
}

// fn update_index(in: ComputeIn) {
//     var brick_voxel_count = 0u;
//     var mask: array<u32, 2>;
//     for (var i = 0u; i < 64u; i++) {
//         if brick.data[i] != 0u {
//             brick_voxel_count += 1u;
//             mask[i >> 5u] |= 1u << (i & 31u);
//         }
//     }
//     if brick_voxel_count == 0u {
//         brick.is_empty = true;
//         return;
//     } 

//     atomicAdd(&alloc.voxel_count, brick_voxel_count);

//     var chunk_index = 0u;

//     for (var d = CHUNK_LEVELS; d > 0u; d--) {
//         let base = CHUNK_WORDS * chunk_index;
        
//         if d == 1u {
//             // bottom index level, allocate new and set mask directly
//             brick.base_index = atomicAdd(&alloc.leaf_count, 1u);

//             chunks[base] = mask[0];
//             chunks[base + 1u] = mask[1];
//             chunks[base + 2u] = brick.base_index;
//         } else {
//             let shift = (d - 2u) << 1u;
//             let offset_pos = vec3<u32>(
//                 (pos.x >> shift) & 3u,
//                 (pos.y >> shift) & 3u,
//                 (pos.z >> shift) & 3u,
//             );
//             let offset = (offset_pos.z << 4u) | (offset_pos.y << 2u) | offset_pos.x;

//         }


//         chunks[base + (offset >> 5u)] |= 1u << (offset & 31u);

//         if d == 0u {
//             // next is leaf, set mask & index
//             chunks
//         }

//         chunk_index = chunks[base + 2u];
        
//         if d > 0u && chunk_index == 0u {
//             // allocate next child
//             chunk_index = atomicAdd(&alloc.chunk_count, 1u);
//             chunks[base + 2u] = chunk_index;
//         }

//         chunks[CHUNK_WORDS * chunk_index]
//     }

//     // pointer to the chunk
//     chunk_indices[in.chunk_pos.z * in.size_chunks.y * in.size_chunks.x + in.chunk_pos.y * in.size_chunks.x + in.chunk_pos.x] = chunk_index + 1u;

//     // build chunk
//     // chunks[chunk_index].brick_index = brick_index + 1u;
//     chunks[chunk_index].mask = mask;

//     var size_bricks = textureDimensions(voxels);
//         size_bricks.x = (size_bricks.x + 7u) >> 3u;
//         size_bricks.y = (size_bricks.y + 7u) >> 3u;
//         size_bricks.z = (size_bricks.z + 7u) >> 3u;
//     brick.base_pos = vec3<u32>(
//         (chunk_index % size_bricks.x) << 3u,
//         ((chunk_index / size_bricks.x) % size_bricks.y) << 3u,
//         (chunk_index / (size_bricks.x * size_bricks.y)) << 3u
//     );
//     brick.is_empty = false;
// }

fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    return pow(srgb, vec3(2.2));
}

fn linear_rgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
    const im1: mat3x3<f32> = mat3x3<f32>(0.4121656120, 0.2118591070, 0.0883097947,
                              0.5362752080, 0.6807189584, 0.2818474174,
                              0.0514575653, 0.1074065790, 0.6302613616);
    const im2: mat3x3<f32> = mat3x3<f32>(0.2104542553, 1.9779984951, 0.0259040371,
                              0.7936177850, -2.4285922050, 0.7827717662,
                              -0.0040720468, 0.4505937099, -0.8086757660);
    let lms = im1 * rgb;
    return im2 * (sign(lms) * pow(abs(lms), vec3(1.0/3.0)));
}
