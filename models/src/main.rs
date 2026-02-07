use std::{
    fs::File,
    io::{BufReader, Cursor},
    path::PathBuf,
};

use anyhow::Result;

use clap::Parser;
use models::{scene, schema};

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

    let gltf = schema::Gltf::parse(&mut src)?;
    let scene = scene::Scene::from_gltf(&gltf);
    Ok(())
}
