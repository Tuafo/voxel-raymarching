struct Chunk {
    brick_index: u32,
    mask: array<u32, 16>,
}
@group(0) @binding(0) var<storage, read_write> chunk_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> chunks: array<Chunk>;
@group(0) @binding(2) var voxels: texture_storage_3d<r32uint, write>;

struct Allocator {
    chunk_count: atomic<u32>,
    brick_count: atomic<u32>,
}
struct Palette {
    data: array<vec4<f32>, 256>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var raw_voxels: texture_storage_3d<rg32uint, read>;
@group(1) @binding(2) var<uniform> palette: Palette;
// @group(1) @binding(3) var palette_lut: texture_storage_3d<r32uint, read>;

struct ComputeIn {
    @builtin(workgroup_id) chunk_pos: vec3<u32>,
    @builtin(num_workgroups) chunk_count: vec3<u32>,
    @builtin(local_invocation_index) brick_index: u32,
    @builtin(local_invocation_id) brick_pos: vec3<u32>,
    @builtin(global_invocation_id) voxel_pos: vec3<u32>,
}

struct BrickGroup {
    is_empty: bool,
    base_pos: vec3<u32>,
    data: array<u32, 512>,
}
// var<workgroup> brick: array<u32, 512>;
var<workgroup> brick: BrickGroup;

@compute @workgroup_size(8, 8, 8)
fn compute_main(in: ComputeIn) {

    let raw = textureLoad(raw_voxels, vec3<i32>(in.voxel_pos)).rg;
    let albedo_packed = raw.r;
    let normal_packed = raw.g;

    let albedo = vec3<f32>(
        f32(albedo_packed >> 24u) / 255.0,
        f32((albedo_packed >> 16u) & 0xffu) / 255.0,
        f32((albedo_packed >> 8u) & 0xffu) / 255.0,
    );
    // let lut_pos = vec3<u32>(
    //     (albedo_u.r & 0xffu) >> 3u,
    //     (albedo_u.g & 0xffu) >> 3u,
    //     (albedo_u.b & 0xffu) >> 3u,
    // );
    var palette_index = 0u;
    if albedo_packed > 0u {
        // palette_index = textureLoad(palette_lut, vec3<i32>(lut_pos)).r & 0xffu;
        // palette_index = 1u + in.brick_index % 255u;
        var min_distance = 1e10;
        for (var i = 1u; i < 256u; i++) {
            let palette_rgb = palette.data[i].rgb;
            let d = distance(palette_rgb, albedo);
            if d < min_distance {
                palette_index = i;
                min_distance = d;
            }
        }
    }

    let packed = (normal_packed << 11u) | palette_index;

    brick.data[in.brick_index] = packed;

    workgroupBarrier();

    if in.brick_index == 0u {
        // build mask here
        var is_empty = true;
        var mask = array<u32, 16>();
        for (var i = 0u; i < 512u; i++) {
            if brick.data[i] != 0u {
                is_empty = false;
                mask[i >> 5u] |= 1u << (i & 31u);
            }
        }
        if is_empty {
            brick.is_empty = true;
        } else {
            let chunk_index = atomicAdd(&alloc.chunk_count, 1u);
            let brick_index = atomicAdd(&alloc.brick_count, 1u);

            // pointer to the chunk
            chunk_indices[in.chunk_pos.z * in.chunk_count.y * in.chunk_count.x + in.chunk_pos.y * in.chunk_count.x + in.chunk_pos.x] = chunk_index + 1u;

            // build chunk
            chunks[chunk_index].brick_index = brick_index + 1u;
            chunks[chunk_index].mask = mask;

            var size_bricks = textureDimensions(voxels);
                size_bricks.x = (size_bricks.x + 7u) >> 3u;
                size_bricks.y = (size_bricks.y + 7u) >> 3u;
                size_bricks.z = (size_bricks.z + 7u) >> 3u;
            brick.base_pos = vec3<u32>(
                (brick_index % size_bricks.x) << 3u,
                ((brick_index / size_bricks.x) % size_bricks.y) << 3u,
                (brick_index / (size_bricks.x * size_bricks.y)) << 3u
            );
            brick.is_empty = false;
        }
    }

    workgroupBarrier();

    if brick.is_empty {
        return;
    }
    textureStore(voxels, vec3<i32>(brick.base_pos + in.brick_pos), vec4<u32>(packed, 0u, 0u, 0u));
}
