mod app;
mod config;
mod core;
mod ui;

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

use std::{sync::Arc, time::Instant};

use pollster::block_on;
use winit::{
    application::ApplicationHandler, event::DeviceEvent, event::WindowEvent, window::Window,
};

use crate::app::App;

struct Program {
    app: Option<App>,
}

impl ApplicationHandler for Program {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let cfg = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
            .with_title("wgpu demo");

        let window = Arc::new(event_loop.create_window(cfg).unwrap());
        self.app = Some(block_on(App::new(window.clone())).unwrap());

        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = self.app.as_mut().unwrap();
        state.handle_input(event_loop, &event);
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let state = self.app.as_mut().unwrap();

        let Some(max_fps) = state.config.max_fps else {
            return;
        };

        let now = Instant::now();
        let prev_time = state.prev_time.unwrap_or(now);
        let elapsed = now - prev_time;
        let targ_frame_time = std::time::Duration::from_secs_f64(1.0 / (max_fps as f64));

        if elapsed >= targ_frame_time {
            state.window.request_redraw();
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                now + targ_frame_time - elapsed,
            ));
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let state = self.app.as_mut().unwrap();
        state.handle_device_input(event_loop, &event);
    }
}

pub trait SizedWindow {
    fn size(&self) -> glam::UVec2;
}
impl SizedWindow for Window {
    fn size(&self) -> glam::UVec2 {
        let size = self.inner_size();
        glam::uvec2(size.width, size.height)
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop.run_app(&mut Program { app: None }).unwrap();
}
