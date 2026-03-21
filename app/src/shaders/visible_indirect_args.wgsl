struct VoxelInfo {
    visible_count: u32,
    failed_to_add: u32,
}
struct IndirectArgs {
    workgroups: array<u32, 3>,
}
@group(0) @binding(0) var<storage, read> voxel_info: VoxelInfo;
@group(0) @binding(1) var<storage, read_write> args: IndirectArgs;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(1, 1, 1)
fn compute_main() {
    args.workgroups[0] = min(65535u, (voxel_info.visible_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE);
    args.workgroups[1] = 1u;
    args.workgroups[2] = 1u;
}
