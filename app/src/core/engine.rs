use std::time::Duration;

use winit::keyboard::KeyCode;

use crate::{
    SizedWindow,
    config::Config,
    core::{camera::Camera, input::Input, transform::ModelTransform},
    ui::Ui,
};

pub struct Engine {
    pub camera: Camera,
    pub model: ModelTransform,
    cursor_locked: bool,
}

pub struct EngineCtx<'a> {
    pub window: &'a winit::window::Window,
    pub config: &'a Config,
    pub input: &'a Input,
    pub ui: &'a mut Ui,
}

const FRAME_AVG_DECAY_ALPHA: f64 = 0.02;

impl Engine {
    pub fn new(window: &winit::window::Window, config: &Config) -> Self {
        let camera = Camera::new(window.size(), config);

        let model = ModelTransform::new();

        Self {
            camera,
            model,
            cursor_locked: false,
        }
    }

    pub fn handle_resize(&mut self, window: &winit::window::Window) {
        self.camera.size = window.size();
    }

    pub fn frame(&mut self, delta_time: &Duration, ctx: EngineCtx<'_>) {
        let EngineCtx {
            window,
            config,
            input,
            ui,
        } = ctx;

        if !self.cursor_locked && input.mouse.left.clicked {
            self.cursor_locked = true;
            window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined))
                .unwrap();
            window.set_cursor_visible(false);
        } else if self.cursor_locked && input.key_down(KeyCode::Space) {
            self.cursor_locked = false;

            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            window.set_cursor_visible(true);
        }

        self.camera.update(delta_time, &input, self.cursor_locked);

        self.model.scale = glam::DVec3::ONE / (config.voxel_scale as f64);
        self.model.update();

        ui.debug.frame_avg = ui.debug.frame_avg.mul_f64(1.0 - FRAME_AVG_DECAY_ALPHA)
            + delta_time.mul_f64(FRAME_AVG_DECAY_ALPHA);

        ui.debug.camera_pos = self.camera.position;
        ui.debug.camera_rotation = self.camera.rotation;
        ui.debug.camera_forward = self.camera.forward;
        ui.debug.camera_near = self.camera.near;
        ui.debug.camera_far = self.camera.far;
    }
}
