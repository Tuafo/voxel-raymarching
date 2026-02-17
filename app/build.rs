use anyhow::Result;
use flate2::{read::ZlibDecoder, write::ZlibEncoder};
use heck::ToSnakeCase;
use loader::voxelize;
use std::{
    env, fs,
    io::{self, Write},
    path,
};

fn main() -> Result<()> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = path::Path::new(&out_dir);

    struct ModelSource {
        name: String,
        file: fs::File,
    }
    let mut sources = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut name_fallback_counter = 0;
    for entry in fs::read_dir("assets/models")? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .is_none_or(|e| !e.eq_ignore_ascii_case("glb"))
        {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|stem| {
                let mut name = stem.to_string_lossy().to_string();
                name = name.to_snake_case().to_ascii_lowercase();
                name = name.replace(|c: char| !c.is_alphanumeric(), "_");
                while name.contains("__") {
                    name = name.replace("__", "_");
                }
                if name.chars().next().map_or(false, |c| c.is_numeric()) {
                    name.insert(0, '_');
                }
                if names.contains(&name) {
                    None
                } else {
                    Some(name)
                }
            })
            .unwrap_or_else(|| {
                loop {
                    let name = format!("model_{}", name_fallback_counter);
                    name_fallback_counter += 1;
                    if !names.contains(&name) {
                        names.insert(name.clone());
                        return name;
                    }
                }
            });

        let file = fs::File::open(path)?;
        sources.push(ModelSource { name, file });
    }

    let (device, queue) = {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

        let mut features = wgpu::Features::default();
        features |= wgpu::Features::TEXTURE_BINDING_ARRAY;
        features |= wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;

        let mut limits = wgpu::Limits::default();
        limits.max_sampled_textures_per_shader_stage = 128;
        limits.max_buffer_size = 2 * 1024 * 1024 * 1024;
        // limits.max_binding_array_elements_per_shader_stage = 406;
        limits.max_binding_array_elements_per_shader_stage = 128;
        limits.max_storage_textures_per_shader_stage = 6;
        limits.max_compute_invocations_per_workgroup = 512;

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))
    }?;

    for src in &sources {
        let mut reader = io::BufReader::new(&src.file);
        let data = voxelize(&mut reader, &device, &queue, Some(src.name.clone()))?
            .serialize(&device, &queue)?;

        let path = out_dir.join(format!("{}.bin", &src.name));
        let file = fs::File::create(&path)?;
        let mut enc = ZlibEncoder::new(file, flate2::Compression::best());
        enc.write_all(&data)?;
        enc.finish()?;
    }

    let model_idents = sources
        .iter()
        .map(|src| quote::format_ident!("{}", &src.name));

    let model_meta_defs = sources.iter().map(|src| {
        let name_ident = quote::format_ident!("{}", &src.name);
        let name = &src.name;
        let path = format!("{}.bin", src.name);
        quote::quote! {
            #name_ident: ModelEntry {
                name: #name,
                path: concat!(env!("OUT_DIR"), "/", #path),
            }
        }
    });

    let code = quote::quote! {
        mod models {
            use wgpu::util::DeviceExt;
            use loader::VoxelModel;
            use std::io::Read;

            /// Atlas of all the available models in the runtime.
            ///
            /// `model.load(device, queue)` on any of the members loads the binary file into memory and populates GPU resources for rendering.
            pub const MODELS: ModelAtlas = ModelAtlas {
                #(#model_meta_defs,)*
            };

            #[derive(Debug)]
            pub struct ModelAtlas {
                #(pub #model_idents: ModelEntry,)*
            }

            /// Listing of a model available to the runtime.
            ///
            /// Running `self.load(device, queue)` will load the associated `.bin` file and popuate the GPU.
            #[derive(Debug)]
            pub struct ModelEntry {
                pub name: &'static str,
                pub path: &'static str,
            }

            impl ModelEntry {
                pub fn load(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<VoxelModel> {
                    let timer = std::time::Instant::now();

                    let file = std::fs::File::open(&self.path)?;
                    let reader = std::io::BufReader::new(file);
                    let mut decoder = flate2::read::ZlibDecoder::new(reader);
                    let mut buf = Vec::new();
                    decoder.read_to_end(&mut buf)?;

                    println!("file read: {:#?}", timer.elapsed());
                    let timer = std::time::Instant::now();

                    let model = VoxelModel::deserialize(device, queue, &buf)?;

                    println!("data deserialize: {:#?}", timer.elapsed());

                    Ok(model)
                }
            }
        }
    };

    let ast = syn::parse2(code).unwrap();
    let formatted = prettyplease::unparse(&ast);

    let dest_path = out_dir.join("assets.rs");
    fs::write(&dest_path, formatted)?;

    println!("cargo::rerun-if-changed=assets/models");

    Ok(())
}
