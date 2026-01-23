use std::time::Duration;

use glam::UVec2;
use winit::keyboard::KeyCode;

use crate::input::Input;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj_matrix: [[f32; 4]; 4],
}

#[derive(Debug)]
pub struct Camera {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub forward: glam::Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub size: glam::UVec2,
    pub uniform: CameraUniform,
}

impl Camera {
    const UP: glam::Vec3 = glam::Vec3::Z;

    pub fn new(size: UVec2) -> Self {
        let mut _self = Self {
            position: glam::Vec3::NEG_Y * 5.0,
            velocity: glam::Vec3::ZERO,
            forward: glam::Vec3::Y,
            fov: 45.0,
            near: 0.01,
            far: 100.0,
            size,
            uniform: CameraUniform::default(),
        };

        return _self;
    }

    pub fn update(&mut self, delta_time: &Duration, input: &Input) {
        let mut in_vec = glam::ivec2(
            input.key_down(KeyCode::KeyD) as i32 - input.key_down(KeyCode::KeyA) as i32,
            input.key_down(KeyCode::KeyW) as i32 - input.key_down(KeyCode::KeyS) as i32,
        )
        .as_vec2();
        in_vec /= in_vec.length().max(1.0);

        let right = -Self::UP.cross(self.forward);

        let targ_velocity = in_vec.x * right + in_vec.y * self.forward;

        let delta_ms = (delta_time.as_secs_f64() * 1000.0).clamp(0.1, 1000.0);
        self.velocity = self.velocity.lerp(targ_velocity, (delta_ms * 0.01) as f32);

        self.position += self.velocity * (delta_ms * 0.005) as f32;

        let view = glam::Mat4::look_at_rh(self.position, self.position + self.forward, Self::UP);
        let proj = glam::Mat4::perspective_rh(
            self.fov,
            self.size.x as f32 / self.size.y.max(1) as f32,
            self.near,
            self.far,
        );
        let view_proj = proj * view;

        self.uniform.view_proj_matrix = view_proj.to_cols_array_2d();
    }
}
