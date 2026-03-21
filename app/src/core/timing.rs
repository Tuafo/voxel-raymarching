use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

pub trait TimedEncoder<'a> {
    fn begin_compute_pass_timed(
        &'a mut self,
        label: &'a str,
        timer: &'a mut Option<RenderTimer>,
    ) -> wgpu::ComputePass<'a>;
    #[allow(dead_code)]
    fn begin_render_pass_timed(
        &'a mut self,
        desc: RenderPassDescriptor<'a>,
        timer: &'a mut Option<RenderTimer>,
    ) -> wgpu::RenderPass<'a>;
}

/// Describes the attachments of a render pass with timing queries automatically added.
///
/// For use with [`CommandEncoder::begin_render_pass_timed`].
///
/// Corresponds to [WebGPU `GPURenderPassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdescriptor).
#[derive(Clone, Debug, Default)]
#[allow(unused)]
pub struct RenderPassDescriptor<'a> {
    pub label: &'a str,
    /// The color attachments of the render pass.
    pub color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>],
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment<'a>>,
    pub occlusion_query_set: Option<&'a wgpu::QuerySet>,
    /// The mask of multiview image layers to use for this render pass. For example, if you wish
    /// to render to the first 2 layers, you would use 3=0b11. If you wanted ro render to only the
    /// 2nd layer, you would use 2=0b10. If you aren't using multiview this should be `None`.
    ///
    /// Note that setting bits higher than the number of texture layers is a validation error.
    ///
    /// This doesn't influence load/store/clear/etc operations, as those are defined for attachments,
    /// therefore affecting all attachments. Meaning, this affects only any shaders executed on the `RenderPass`.
    pub multiview_mask: Option<std::num::NonZeroU32>,
}

impl<'a> TimedEncoder<'a> for wgpu::CommandEncoder {
    fn begin_render_pass_timed(
        &'a mut self,
        desc: RenderPassDescriptor<'a>,
        timer: &'a mut Option<RenderTimer>,
    ) -> wgpu::RenderPass<'a> {
        let timestamp_writes = if let Some(timer) = timer
            && !timer.waiting_results
        {
            let pass_index = match timer.indices.get(desc.label) {
                Some(i) => *i,
                None => {
                    timer
                        .indices
                        .insert(String::from(desc.label), timer.results.len() as u32);
                    timer.results.push(PassTimerResults {
                        label: Some(String::from(desc.label)),
                        time: Duration::ZERO,
                    });
                    (timer.results.len() - 1) as u32
                }
            };
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &timer.query_set,
                beginning_of_pass_write_index: Some(pass_index * 2),
                end_of_pass_write_index: Some(pass_index * 2 + 1),
            })
        } else {
            None
        };
        self.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(desc.label),
            timestamp_writes,
            color_attachments: desc.color_attachments,
            depth_stencil_attachment: desc.depth_stencil_attachment,
            occlusion_query_set: desc.occlusion_query_set,
            multiview_mask: desc.multiview_mask,
        })
    }
    fn begin_compute_pass_timed(
        &'a mut self,
        label: &'a str,
        timer: &'a mut Option<RenderTimer>,
    ) -> wgpu::ComputePass<'a> {
        let timestamp_writes = if let Some(timer) = timer
            && !timer.waiting_results
        {
            let pass_index = match timer.indices.get(label) {
                Some(i) => *i,
                None => {
                    timer
                        .indices
                        .insert(String::from(label), timer.results.len() as u32);
                    timer.results.push(PassTimerResults {
                        label: Some(String::from(label)),
                        time: Duration::ZERO,
                    });
                    (timer.results.len() - 1) as u32
                }
            };
            Some(wgpu::ComputePassTimestampWrites {
                query_set: &timer.query_set,
                beginning_of_pass_write_index: Some(pass_index * 2),
                end_of_pass_write_index: Some(pass_index * 2 + 1),
            })
        } else {
            None
        };
        self.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        })
    }
}

#[allow(dead_code)]
pub struct RenderTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    mapped: Arc<AtomicBool>,
    waiting_results: bool,
    max_query_count: u32,
    results: Vec<PassTimerResults>,
    indices: HashMap<String, u32>,
}

#[derive(Debug, Default, Clone)]
struct PassTimerResults {
    label: Option<String>,
    time: Duration,
}

const RENDERTIME_ACC_ALPHA: f64 = 0.025;

impl RenderTimer {
    pub fn new(device: &wgpu::Device, max_query_count: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp query set"),
            ty: wgpu::QueryType::Timestamp,
            count: max_query_count * 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query resolve buffer"),
            size: (max_query_count as u64) * 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result_buffer"),
            size: (max_query_count as u64) * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            result_buffer,
            max_query_count,
            mapped: Arc::new(AtomicBool::new(false)),
            waiting_results: false,
            indices: HashMap::new(),
            results: Vec::new(),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.waiting_results {
            return;
        }

        encoder.resolve_query_set(
            &self.query_set,
            0..self.results.len() as u32 * 2,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.result_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    pub fn gather_results<'a>(&'a mut self) -> Option<Vec<(String, Duration)>> {
        if !self.waiting_results {
            let mapped = self.mapped.clone();
            self.result_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    if res.is_ok() {
                        mapped.store(true, Ordering::Release);
                    }
                });
            self.waiting_results = true;
            return None;
        }

        if self.mapped.load(Ordering::Acquire) {
            let mut res = Vec::new();

            let view = self.result_buffer.get_mapped_range(..);
            let timestamps: &[u64] = bytemuck::cast_slice(&*view);

            for i in 0..self.results.len() {
                let prev = &mut self.results[i];
                let Some(label) = &prev.label else {
                    continue;
                };
                let time = Duration::from_nanos(timestamps[i * 2 + 1] - timestamps[i * 2]);
                prev.time = time.mul_f64(RENDERTIME_ACC_ALPHA)
                    + prev.time.mul_f64(1.0 - RENDERTIME_ACC_ALPHA);
                res.push((label.clone(), prev.time));
            }
            drop(view);

            self.result_buffer.unmap();
            self.mapped.store(false, Ordering::Release);
            self.waiting_results = false;

            return Some(res);
        }
        None
    }
}
