use std::collections::HashMap;

use anyhow::Result;
use wgpu::*;

/// Manages the GPU device and buffers for compute dispatch.
pub struct GpuContext {
    device: Device,
    queue: Queue,
    buffers: HashMap<i32, Buffer>,
    next_handle: i32,
    adapter_name: String,
}

impl GpuContext {
    /// Initialize the GPU context. Uses the primary backend (Metal on macOS).
    pub fn new() -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let adapter_name = adapter.get_info().name.clone();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("qt45_gpu"),
                ..Default::default()
            },
        ))?;

        Ok(Self {
            device,
            queue,
            buffers: HashMap::new(),
            next_handle: 1,
            adapter_name,
        })
    }

    /// Get the GPU adapter name for diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Allocate a GPU buffer. Returns a handle for later reference.
    pub fn alloc(&mut self, size: usize) -> i32 {
        let handle = self.next_handle;
        self.next_handle += 1;
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("gpu_buf_{handle}")),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffers.insert(handle, buffer);
        handle
    }

    /// Write data from the host into a GPU buffer.
    pub fn write_buffer(&self, handle: i32, data: &[u8]) -> Result<()> {
        let buf = self
            .buffers
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("Unknown GPU buffer handle {handle}"))?;
        self.queue.write_buffer(buf, 0, data);
        Ok(())
    }

    /// Compile a WGSL shader, bind to a buffer, and dispatch compute workgroups.
    pub fn dispatch_shader(
        &self,
        wgsl: &str,
        buf_handle: i32,
        workgroups: u32,
    ) -> Result<()> {
        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("compute_pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let buf = self
            .buffers
            .get(&buf_handle)
            .ok_or_else(|| anyhow::anyhow!("Unknown GPU buffer handle {buf_handle}"))?;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("compute_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read data back from a GPU buffer to host memory.
    pub fn read_buffer(&self, handle: i32, size: usize) -> Result<Vec<u8>> {
        let buf = self
            .buffers
            .get(&handle)
            .ok_or_else(|| anyhow::anyhow!("Unknown GPU buffer handle {handle}"))?;

        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging_read"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });

        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv()??;

        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }

    /// Free a GPU buffer by handle.
    #[allow(dead_code)]
    pub fn free(&mut self, handle: i32) {
        self.buffers.remove(&handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_double_shader() {
        let mut gpu = GpuContext::new().expect("GPU should be available");
        println!("GPU adapter: {}", gpu.adapter_name());

        // Input: 8 floats
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let byte_len = input.len() * 4;
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Allocate, write, dispatch, read
        let handle = gpu.alloc(byte_len);
        gpu.write_buffer(handle, &input_bytes).unwrap();

        let wgsl = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) {
        data[id.x] = data[id.x] * 2.0;
    }
}
"#;

        gpu.dispatch_shader(wgsl, handle, 1).unwrap();

        let result_bytes = gpu.read_buffer(handle, byte_len).unwrap();
        let result: Vec<f32> = result_bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        gpu.free(handle);
    }
}
