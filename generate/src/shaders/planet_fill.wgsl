// Procedural planet voxelizer.
//
// Two compute entry points:
//   classify_chunks: dispatched once per 64^3 world chunk. Tests the chunk
//     against the planet SDF; if any sample lies in the surface band, the
//     chunk is allocated by atomically incrementing chunk_counter and
//     writing (idx<<1)|1 into raw_chunk_indices.
//   fill_voxels: dispatched once per world chunk. For chunks that were
//     allocated, evaluates the SDF at every voxel and packs the result
//     into raw_voxels at the slot indicated by raw_chunk_indices.
//
// Output texture layouts and voxel encoding match generate/src/shaders/voxelize.wgsl
// and generate/src/shaders/tree_64.wgsl exactly (so the existing tree-build
// passes can be reused unmodified).

struct PlanetParams {
    bounding_size: u32,
    chunks_per_axis: u32,
    raw_minor_chunks: u32, // raw_minor_size / 64
    seed: u32,

    center: vec3<f32>,
    radius: f32,

    amplitude: f32,
    base_freq: f32,
    octaves: u32,
    crust_thickness: f32,

    sea_level: f32, // SDF offset where "water" starts (in voxel units, relative to surface)
    snow_height: f32, // SDF offset where snow starts
    sand_band: f32, // band width near sea level mapped to sand
    _pad0: f32,

    // Palette indices for terrain materials. We keep the palette compact
    // and fully procedural: see planet.rs for the layout.
    pal_rock: u32,
    pal_grass: u32,
    pal_sand: u32,
    pal_snow: u32,
    pal_dirt: u32,
    pal_water: u32,
    pal_lava: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PlanetParams;
@group(0) @binding(1) var raw_chunk_indices_w: texture_storage_3d<r32uint, write>;
@group(0) @binding(2) var raw_chunk_indices_r: texture_storage_3d<r32uint, read>;
@group(0) @binding(3) var raw_voxels: texture_storage_3d<r32uint, write>;
@group(0) @binding(4) var<storage, read_write> chunk_counter: atomic<u32>;

const RAW_CHUNK_SIZE: u32 = 64u;

// ---------------------------------------------------------------
// classify pass
// dispatch: (chunks_per_axis, chunks_per_axis, chunks_per_axis), workgroup (1,1,1)
// One thread per world chunk.
// ---------------------------------------------------------------

@compute @workgroup_size(1, 1, 1)
fn classify_chunks(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3(params.chunks_per_axis)) {
        return;
    }

    let chunk_origin = vec3<f32>(gid * RAW_CHUNK_SIZE);
    let chunk_extent = f32(RAW_CHUNK_SIZE);

    // Sample 4^3 points spaced through the chunk; accept if the chunk
    // straddles the surface band (any sample within the crust thickness
    // OR samples have mixed sign).
    var has_solid = false;
    var has_empty = false;
    var has_band = false;

    for (var i = 0u; i < 64u; i++) {
        let lp = vec3<f32>(
            f32(i & 3u),
            f32((i >> 2u) & 3u),
            f32((i >> 4u) & 3u),
        );
        let pos = chunk_origin + (lp + 0.5) * (chunk_extent / 4.0);
        let d = sdf_terrain(pos);
        if d < 0.0 { has_solid = true; } else { has_empty = true; }
        if abs(d) < params.crust_thickness { has_band = true; }
    }

    let is_active = has_band || (has_solid && has_empty);
    if !is_active {
        textureStore(raw_chunk_indices_w, vec3<i32>(gid), vec4(0u));
        return;
    }

    let idx = atomicAdd(&chunk_counter, 1u) + 1u; // 1-based; tree_64 ignores idx==0
    // Encoding matches voxelize_gltf: (chunk_index << 1) | 1
    textureStore(raw_chunk_indices_w, vec3<i32>(gid), vec4((idx << 1u) | 1u, 0u, 0u, 0u));
}

// ---------------------------------------------------------------
// fill pass
// dispatch: (chunks_per_axis, chunks_per_axis, chunks_per_axis), workgroup (4,4,4)
// One workgroup per world chunk; each thread writes a 16^3 voxel block (handled
// in a 4x4x4 inner loop) so that the workgroup covers all 64^3 voxels.
// ---------------------------------------------------------------

@compute @workgroup_size(4, 4, 4)
fn fill_voxels(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if any(wg >= vec3(params.chunks_per_axis)) {
        return;
    }

    let raw_ci_packed = textureLoad(raw_chunk_indices_r, vec3<i32>(wg)).r;
    if (raw_ci_packed & 1u) == 0u {
        return; // empty chunk
    }
    let raw_ci = raw_ci_packed >> 1u;

    // Mirrors tree_64.wgsl: linear chunk index -> 3D offset into raw_voxels.
    let rmc = params.raw_minor_chunks;
    let raw_chunk_offset = vec3<u32>(
        raw_ci % rmc,
        (raw_ci / rmc) % rmc,
        raw_ci / (rmc * rmc),
    ) * RAW_CHUNK_SIZE;

    let world_chunk_origin = wg * RAW_CHUNK_SIZE;

    // Each thread covers a 16x16x16 sub-block of the 64^3 chunk.
    let block_origin = lid * 16u;

    for (var z = 0u; z < 16u; z++) {
        for (var y = 0u; y < 16u; y++) {
            for (var x = 0u; x < 16u; x++) {
                let local = block_origin + vec3(x, y, z);
                let world_pos = vec3<f32>(world_chunk_origin + local);

                let voxel = sample_voxel(world_pos);

                let dst = vec3<i32>(raw_chunk_offset + local);
                textureStore(raw_voxels, dst, vec4(voxel, 0u, 0u, 0u));
            }
        }
    }
}

// ---------------------------------------------------------------
// SDF + voxel material logic
// ---------------------------------------------------------------

fn sdf_sphere(pos: vec3<f32>) -> f32 {
    return length(pos - params.center) - params.radius;
}

fn sdf_terrain(pos: vec3<f32>) -> f32 {
    let base = sdf_sphere(pos);
    // Domain-warp the noise lookup with the surface direction so detail
    // wraps around the planet rather than being a slab.
    let dir = normalize(pos - params.center);
    let h = fbm(dir * params.base_freq, params.octaves) * params.amplitude;
    return base - h;
}

fn sample_voxel(pos: vec3<f32>) -> u32 {
    let d = sdf_terrain(pos);
    if d >= 0.0 {
        return 0u; // empty
    }

    // Estimate gradient (=> outward normal) via finite differences.
    let e = 0.75;
    let gx = sdf_terrain(pos + vec3(e, 0.0, 0.0)) - sdf_terrain(pos - vec3(e, 0.0, 0.0));
    let gy = sdf_terrain(pos + vec3(0.0, e, 0.0)) - sdf_terrain(pos - vec3(0.0, e, 0.0));
    let gz = sdf_terrain(pos + vec3(0.0, 0.0, e)) - sdf_terrain(pos - vec3(0.0, 0.0, e));
    var grad = vec3(gx, gy, gz);
    let grad_len = length(grad);
    var normal: vec3<f32>;
    if grad_len < 1e-6 {
        normal = normalize(pos - params.center);
    } else {
        normal = grad / grad_len;
    }

    // Material selection by altitude relative to the un-bumped sphere
    // (so we get coherent biomes regardless of local noise).
    let altitude = length(pos - params.center) - params.radius;
    let up = normalize(pos - params.center);
    let slope = clamp(dot(normal, up), 0.0, 1.0);

    var palette_index: u32 = params.pal_rock;
    var roughness: f32 = 0.85;
    var metallic: f32 = 0.0;
    var emissive: bool = false;
    var emissive_intensity: f32 = 0.0;

    if altitude < params.sea_level - params.sand_band {
        // Deep submerged rock
        palette_index = params.pal_rock;
        roughness = 0.6;
    } else if altitude < params.sea_level + params.sand_band {
        // Beach band
        palette_index = params.pal_sand;
        roughness = 0.9;
    } else if altitude > params.snow_height {
        palette_index = params.pal_snow;
        roughness = 0.7;
    } else if slope < 0.55 {
        // Steep faces show rock regardless of altitude.
        palette_index = params.pal_rock;
        roughness = 0.85;
    } else if slope < 0.8 {
        palette_index = params.pal_dirt;
        roughness = 0.95;
    } else {
        palette_index = params.pal_grass;
        roughness = 0.9;
    }

    return pack_voxel(normal, metallic, roughness, palette_index, emissive, emissive_intensity);
}

fn pack_voxel(
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    palette_index: u32,
    is_emissive: bool,
    emissive_intensity: f32,
) -> u32 {
    let n = encode_normal_octahedral_leaf(normal);
    var m: u32;
    var r: u32;
    if is_emissive {
        m = 1u;
        r = (u32(saturate(emissive_intensity) * 7.0 + 0.5) << 1u) | 1u;
    } else if metallic > 0.5 {
        m = 1u;
        r = u32(saturate(roughness) * 7.0 + 0.5) << 1u;
    } else {
        m = 0u;
        r = u32(saturate(roughness) * 15.0 + 0.5);
    }
    return (n << 15u) | (m << 14u) | (r << 10u) | (palette_index & 0x3ffu);
}

fn encode_normal_octahedral_leaf(normal: vec3<f32>) -> u32 {
    var n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var nrm = vec2<f32>(0.0);
    nrm.y = n.y * 0.5 + 0.5;
    nrm.x = n.x * 0.5 + nrm.y;
    nrm.y = n.x * -0.5 + nrm.y;
    let sgn = select(0u, 1u, n.z >= 0.0);
    return (u32(nrm.x * 255.0 + 0.5) << 9u) | (u32(nrm.y * 255.0 + 0.5) << 1u) | sgn;
}

// ---------------------------------------------------------------
// 3D value-noise based fbm. No texture reads required.
// ---------------------------------------------------------------

fn hash3(p: vec3<i32>) -> f32 {
    var h: u32 = u32(p.x) * 0x8da6b343u
              ^ u32(p.y) * 0xd8163841u
              ^ u32(p.z) * 0xcb1ab31fu
              ^ params.seed * 0x165667b1u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash3(i + vec3(0, 0, 0));
    let n100 = hash3(i + vec3(1, 0, 0));
    let n010 = hash3(i + vec3(0, 1, 0));
    let n110 = hash3(i + vec3(1, 1, 0));
    let n001 = hash3(i + vec3(0, 0, 1));
    let n101 = hash3(i + vec3(1, 0, 1));
    let n011 = hash3(i + vec3(0, 1, 1));
    let n111 = hash3(i + vec3(1, 1, 1));

    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z) * 2.0 - 1.0; // [-1, 1]
}

fn fbm(p_in: vec3<f32>, octaves: u32) -> f32 {
    var p = p_in;
    var amp: f32 = 1.0;
    var sum: f32 = 0.0;
    var norm: f32 = 0.0;
    for (var i = 0u; i < octaves; i++) {
        sum += amp * value_noise(p);
        norm += amp;
        amp *= 0.5;
        p *= 2.03;
    }
    return sum / max(norm, 1e-6);
}
