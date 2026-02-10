#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraDataBuffer {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub ws_position: [f32; 3],
    pub _pad_0: f32,
    pub forward: [f32; 3],
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub _pad_1: f32,
    pub _pad_2: f32,
}

impl CameraDataBuffer {
    pub fn update(&mut self, camera: &crate::engine::Camera) {
        self.view_proj = camera.view_proj.as_mat4().to_cols_array_2d();
        self.inv_view_proj = camera.inv_view_proj.as_mat4().to_cols_array_2d();
        self.ws_position = camera.position.as_vec3().to_array();
        self.forward = camera.forward.as_vec3().to_array();
        self.near = camera.near as f32;
        self.far = camera.far as f32;
        self.fov = camera.fov as f32;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDataBuffer {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],
    pub normal_transform: [[f32; 4]; 3],
}

impl ModelDataBuffer {
    pub fn update(&mut self, model: &crate::engine::Model) {
        self.transform = model.transform.as_mat4().to_cols_array_2d();
        self.inv_transform = model.inv_transform.as_mat4().to_cols_array_2d();
        self.normal_transform = glam::DMat3::from_mat4(model.inv_transform)
            .transpose()
            .as_mat3()
            .to_cols_array_2d()
            .map(|v| [v[0], v[1], v[2], 0.0]);
    }
}
