# wgpu-hello-cube

A basic rotating cube example using WebGPU in Rust.

### Overview

---

This is a simple 3D rotating cube with vertex colors rendered using WebGPU with `wgpu`. It's an updated and self-contained version of the official cube example from `wgpu`, and uses `glam` for linear algebra rather than `cgmath`.

## Resources

This example is based on:

- The official [wgpu-rs cube example](https://github.com/gfx-rs/wgpu-rs/tree/master/examples/cube)
- The excellent [Learn WGPU tutorial](https://sotrh.github.io/learn-wgpu/)

### Main Dependencies

- **wgpu**: ([crates.io](https://crates.io/crates/wgpu/))
- **glam**: Really nice linear algebra crate for these purposes ([crates.io](https://crates.io/crates/glam/))
- **winit**: Window handling ([crates.io](https://crates.io/crates/winit/))
- **bytemuck**: Good for serializing uniform buffers ([crates.io](https://crates.io/crates/bytemuck/))

### To Run

```bash
cargo run
```

## From Here

Some reasonable next steps to take as an exercise might be:

- Add in instancing
- Add user input to control the camera in `camera.rs`
- Whatever really, its graphics - you can do anything
