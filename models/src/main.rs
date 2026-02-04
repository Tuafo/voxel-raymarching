use std::{
    fs::File,
    io::{BufReader, Cursor},
    path::PathBuf,
};

use anyhow::Result;

use clap::Parser;
use models::schema;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to .glb file
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(args.input)?;
    let mut src = BufReader::new(file);

    let model = models::Gltf::parse(&mut src)?;

    // println!("{:#?}", model.header);
    // println!("{:#?}", model.meta);
    for (i, im) in model.meta.images.iter().enumerate() {
        let Some(buf_view_index) = im.buffer_view else {
            continue;
        };
        let Some(buf_view) = model.meta.buffer_views.get(buf_view_index as usize) else {
            continue;
        };
        let src = &model.bin[(buf_view.byte_offset as usize)
            ..(buf_view.byte_offset as usize + buf_view.byte_length as usize)];
        let res = match im.mime_type {
            Some(schema::MimeType::Jpeg) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Jpeg)
            }
            Some(schema::MimeType::Png) => {
                image::load_from_memory_with_format(src, image::ImageFormat::Png)
            }
            _ => image::load_from_memory(src),
        };
        let filename = format!("img_{}_{}", i, im.name.as_deref().unwrap_or(""));
        let Ok(res) = res else {
            println!("failed to load image {}", filename);
            continue;
        };

        // let mut path = PathBuf::new();
        // path.set_file_name(filename);
        // path.set_extension("png");
        // res.save(path).expect("Failed to save image");
    }
    Ok(())
}
