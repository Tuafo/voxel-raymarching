@group(0) @binding(0) var out_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> scene: SceneData;
@group(0) @binding(2) var<storage, read> voxels: array<u32>;

struct SceneData {
    size: vec3<u32>,
    palette: array<vec4<u32>, 64>
}

struct ComputeIn {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@compute @workgroup_size(8, 8)
fn compute_main(in: ComputeIn) {
    let dimensions = textureDimensions(out_texture).xy;

    let pixel_pos = in.id.xy;
    let uv = vec2<f32>(pixel_pos) / vec2<f32>(dimensions);

    // let palette_index = u32(i32(in.id.x) % 256);
    let ws_pos = vec3<u32>(u32(i32(in.id.x) % i32(scene.size.x)), u32(i32(in.id.y) % i32(scene.size.y)), 50u);
    let palette_index = voxel(ws_pos);

    let color = palette_color(palette_index);

    textureStore(out_texture, vec2<i32>(in.id.xy), color);
}

fn voxel(position: vec3<u32>) -> u32 {
    let index = position.x * scene.size.x * scene.size.y + position.y * scene.size.y + position.z;
    return (voxels[index >> 2u] >> u32(3 - i32(index & 0x3u))) & 0xFFu;
}

fn palette_color(index: u32) -> vec4<f32> {
    let rgba = scene.palette[index >> 2u][index & 3u];
    return vec4<f32>(
        f32((rgba >> 24u) & 0xFFu) / 255.0,
        f32((rgba >> 16u) & 0xFFu) / 255.0,
        f32((rgba >> 8u) & 0xFFu) / 255.0,
        f32(rgba & 0xFFu) / 255.0
    );
}
