use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, StoreOp, TextureFormat, TextureView};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
use egui_winit::State;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct UIRenderer {
    state: State,
    renderer: Renderer,
    frame_started: bool,
}

impl UIRenderer {
    pub fn context(&self) -> &egui::Context {
        self.state.egui_ctx()
    }

    pub fn new(device: &Device, out_format: TextureFormat, window: &Window) -> Self {
        let ctx = egui::Context::default();

        let state = egui_winit::State::new(
            ctx,
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024),
        );
        let renderer = Renderer::new(device, out_format, RendererOptions::default());

        Self {
            state,
            renderer,
            frame_started: false,
        }
    }

    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self.state.on_window_event(window, event);
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.state.egui_ctx().begin_pass(raw_input);
        self.frame_started = true;
    }

    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        window: &Window,
        texture_view: &TextureView,
        descriptor: ScreenDescriptor,
    ) {
        if !self.frame_started {
            return;
        }

        self.context()
            .set_pixels_per_point(descriptor.pixels_per_point);

        let full_output = self.state.egui_ctx().end_pass();

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .state
            .egui_ctx()
            .tessellate(full_output.shapes, self.state.egui_ctx().pixels_per_point());
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &tris, &descriptor);
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: egui_wgpu::wgpu::Operations {
                    load: egui_wgpu::wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(&mut pass.forget_lifetime(), &tris, &descriptor);

        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }

        self.frame_started = false;
    }
}
