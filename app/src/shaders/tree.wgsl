struct Chunk {
    brick_index: u32,
    mask: array<u32, 16>,
}
struct Brick {
    data: array<u32, 512>,
}
@group(0) @binding(0) var<storage, read_write> chunk_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> chunks: array<Chunk>;
@group(0) @binding(2) var<storage, read_write> bricks: array<Brick>;

struct Allocator {
    chunk_count: atomic<u32>,
    brick_count: atomic<u32>,
}
@group(1) @binding(0) var<storage, read_write> alloc: Allocator;
@group(1) @binding(1) var voxels: texture_storage_3d<rgba8unorm, read>;

struct ComputeIn {
    @builtin(workgroup_id) chunk_pos: vec3<u32>,
    @builtin(num_workgroups) chunk_count: vec3<u32>,
    @builtin(local_invocation_index) brick_index: u32,
    @builtin(global_invocation_id) voxel_pos: vec3<u32>,
}

var<workgroup> brick: array<u32, 512>;

@compute @workgroup_size(8, 8, 8)
fn compute_main(in: ComputeIn) {
    let albedo = vec4<u32>(textureLoad(voxels, vec3<i32>(in.voxel_pos)) * 255.0);
    let albedo_packed = ((albedo.r & 0xffu) << 24u) | ((albedo.g & 0xffu) << 16u) | ((albedo.b & 0xffu) << 8u) | (albedo.a & 0xffu);

    brick[in.brick_index] = albedo_packed;

    workgroupBarrier();

    if in.brick_index == 0u {
        // build mask here
        var is_empty = true;
        var mask = array<u32, 16>();
        for (var i = 0u; i < 512u; i++) {
            if brick[i] != 0u {
                is_empty = false;
                mask[i >> 5u] |= 1u << (i & 31u);
            }
        }
        if is_empty {
            return;
        }
        let chunk_index = atomicAdd(&alloc.chunk_count, 1u);
        let brick_index = atomicAdd(&alloc.brick_count, 1u);

        // pointer to the chunk
        chunk_indices[in.chunk_pos.z * in.chunk_count.y * in.chunk_count.x + in.chunk_pos.y * in.chunk_count.x + in.chunk_pos.x] = chunk_index + 1u;
        
        // build chunk
        chunks[chunk_index].brick_index = brick_index + 1u;
        chunks[chunk_index].mask = mask;

        // copy over actual brick data
        bricks[brick_index].data = brick;
    }
}