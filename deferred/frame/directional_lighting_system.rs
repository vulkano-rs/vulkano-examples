// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Queue;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::ImageViewAccess;
use vulkano::pipeline::blend::AttachmentBlend;
use vulkano::pipeline::blend::BlendFactor;
use vulkano::pipeline::blend::BlendOp;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use cgmath::Vector3;

use std::sync::Arc;

pub struct DirectionalLightingSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
}

impl DirectionalLightingSystem {
    /// Initializes the directional lighting system.
    pub fn new<R>(gfx_queue: Arc<Queue>, subpass: Subpass<R>) -> DirectionalLightingSystem
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        // TODO: vulkano doesn't allow us to draw without a vertex buffer, otherwise we could
        //       hard-code these values in the shader
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(gfx_queue.device().clone(), BufferUsage::all(), Some(gfx_queue.family()), [
                Vertex { position: [-1.0, -1.0] },
                Vertex { position: [-1.0, 3.0] },
                Vertex { position: [3.0, -1.0] }
            ].iter().cloned()).expect("failed to create buffer")
        };

        let pipeline = {
            let vs = vs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(gfx_queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .render_pass(subpass)
                .build(gfx_queue.device().clone())
                .unwrap()) as Arc<_>
        };

        DirectionalLightingSystem {
            gfx_queue: gfx_queue,
            vertex_buffer: vertex_buffer,
            pipeline: pipeline,
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw<C, N>(&self, viewport_dimensions: [u32; 2], color_input: C, normals_input: N,
                      direction: Vector3<f32>, color: [f32; 3]) -> AutoCommandBuffer
        where C: ImageViewAccess + Send + Sync + 'static,
              N: ImageViewAccess + Send + Sync + 'static,
    {
        let push_constants = fs::ty::PushConstants {
            color: [color[0], color[1], color[2], 1.0],
            direction: direction.extend(0.0).into(),
        };

        let descriptor_set = PersistentDescriptorSet::start(self.pipeline.clone(), 0)
            .add_image(color_input)
            .unwrap()
            .add_image(normals_input)
            .unwrap()
            .build()
            .unwrap();

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [viewport_dimensions[0] as f32,
                            viewport_dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        AutoCommandBufferBuilder::secondary_graphics(self.gfx_queue.device().clone(),
                                                     self.gfx_queue.family(),
                                                     self.pipeline.clone().subpass())
            .unwrap()
            .draw(self.pipeline.clone(),
                  dynamic_state,
                  vec![self.vertex_buffer.clone()],
                  descriptor_set,
                  push_constants)
            .unwrap()
            .build()
            .unwrap()
    }
}

#[derive(Debug, Clone)]
struct Vertex {
    position: [f32; 2]
}
impl_vertex!(Vertex, position);

mod vs {
    #[derive(VulkanoShader)]
    #[allow(dead_code)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[allow(dead_code)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;

layout(push_constant) uniform PushConstants {
    vec4 color;
    vec4 direction;
} push_constants;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 in_normal = normalize(subpassLoad(u_normals).rgb);
    float light_percent = max(-dot(push_constants.direction.xyz, in_normal), 0.0);

    vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
    f_color.rgb = light_percent * push_constants.color.rgb * in_diffuse;
    f_color.a = 1.0;
}
"]
    struct Dummy;
}
