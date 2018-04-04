// Copyright (c) 2018 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;
extern crate cgmath;

use std::sync::Arc;
use std::mem;
use std::time::Instant;

use cgmath::{Rad, Point3, Vector3, Matrix3, Matrix4};

use winit::{EventsLoop, WindowBuilder, Event, WindowEvent};

use vulkano_win::VkSurfaceBuild;

use vulkano::{
    instance::{
        Instance,
        PhysicalDevice,
    },
    device::{
        Device,
        DeviceExtensions,
    },
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
        CpuBufferPool,
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        DynamicState,
    },
    descriptor::descriptor_set::PersistentDescriptorSet,
    image::{
        ImageUsage,
        AttachmentImage,
        ImageLayout,
    },
    format::{
        Format,
        ClearValue,
    },
    framebuffer::{
        Framebuffer,
        Subpass,
    },
    pipeline::{
        GraphicsPipeline,
        viewport::Viewport,
    },
    swapchain::{
        self,
        PresentMode,
        SurfaceTransform,
        Swapchain,
        AcquireError,
        SwapchainCreationError,
    },
    sync::{
        self as vk_sync,
        GpuFuture,
    }
};

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None)
            .expect("failed to create Vulkan instance")
    };

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .expect("failed to create window");

    let queue_family = physical.queue_families()
        .find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            .. DeviceExtensions::none()
        };

        Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        ).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let mut dimensions;
    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical)
            .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            caps.supported_usage_flags,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None,
        ).expect("failed to create swapchain")
    };

    // let attachment_format = Format::B8G8R8A8Unorm;
    let attachment_format = swapchain.format();
    let attachment_image_newframe = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        attachment_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            transfer_destination: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let attachment_image_ping = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        attachment_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            transfer_destination: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let attachment_image_pong = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        attachment_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            transfer_destination: true,
            transfer_source: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let scene_vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { a_position: [f32; 2] }
        impl_vertex!(Vertex, a_position);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { a_position: [-0.5, -0.25] },
            Vertex { a_position: [0.0, 0.5] },
            Vertex { a_position: [0.25, -0.1] },
        ].iter().cloned()).expect("failed to create buffer")
    };

    let postprocess_vertex_buffer = {
        #[derive(Debug, Clone)]
        struct VertexUv { a_position: [f32; 2], a_texcoord: [f32; 2] }
        impl_vertex!(VertexUv, a_position, a_texcoord);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            VertexUv { a_position: [-1.0, 3.0], a_texcoord: [0.0, 2.0] },
            VertexUv { a_position: [-1.0, -1.0], a_texcoord: [0.0, 0.0] },
            VertexUv { a_position: [3.0, -1.0], a_texcoord: [2.0, 0.0] },
        ].iter().cloned()).expect("failed to create buffer")
    };

    let uniform_buffer = CpuBufferPool::<scene_vs_mod::ty::Matrices>::new(
        device.clone(),
        BufferUsage::all(),
    );

    let proj = cgmath::perspective(
        Rad(std::f32::consts::FRAC_PI_2),
        { dimensions[0] as f32 / dimensions[1] as f32 },
        0.01,
        100.0,
    );

    let view = Matrix4::look_at(
        Point3::new(0.3, 0.3, 1.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
    );

    let scene_vs = scene_vs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");
    let scene_fs = scene_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");

    let postprocess_vs = postprocess_vs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");
    let postprocess_fs = postprocess_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");

    let copy_fs = copy_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");

    let scene_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                newframe_color: {
                    load: Clear,
                    store: Store,
                    format: attachment_format,
                    samples: 1,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                }
            },
            pass: {
                color: [newframe_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let postprocess_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                pong_color: {
                    load: DontCare,
                    store: Store,
                    format: attachment_format,
                    samples: 1,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                }
            },
            pass: {
                color: [pong_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let copy_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                output_color: {
                    load: DontCare,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                    initial_layout: ImageLayout::PresentSrc,
                    final_layout: ImageLayout::PresentSrc,
                }
            },
            pass: {
                color: [output_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let newframe_framebuffer = Arc::new({
        Framebuffer::start(scene_renderpass.clone())
            .add(attachment_image_newframe.clone()).unwrap()
            .build().unwrap()
    });

    let pong_framebuffer = Arc::new({
        Framebuffer::start(postprocess_renderpass.clone())
            .add(attachment_image_pong.clone()).unwrap()
            .build().unwrap()
    });

    let scene_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(scene_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(scene_fs.main_entry_point(), ())
            .render_pass(Subpass::from(scene_renderpass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
    });

    let postprocess_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(postprocess_fs.main_entry_point(), ())
            .render_pass(Subpass::from(
                postprocess_renderpass.clone(),
                0,
            )
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    let copy_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(copy_fs.main_entry_point(), ())
            .render_pass(Subpass::from(copy_renderpass.clone(), 0)
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    println!("Building clear command buffer");
    let clear_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
        device.clone(),
        queue.family(),
        )
        .unwrap()
        .clear_color_image(
            attachment_image_newframe.clone(),
            ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
        )
        .unwrap()
        .clear_color_image(
            attachment_image_ping.clone(),
            ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
        )
        .unwrap()
        .clear_color_image(
            attachment_image_pong.clone(),
            ClearValue::Float([0.0, 0.0, 1.0, 1.0]),
        )
        .unwrap()
        .build().unwrap();

    let mut previous_frame_end: Box<GpuFuture> = Box::new({
        vk_sync::now(device.clone())
            .then_execute(queue.clone(), clear_command_buffer)
            .unwrap()
    });

    // let mut previous_frame_end: Box<GpuFuture> = Box::new(vk_sync::now(device.clone()));

    let mut framebuffers: Option<Vec<Arc<Framebuffer<_,_>>>> = None;
    let mut recreate_swapchain = false;


    let time_start = Instant::now();
    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            dimensions = surface.capabilities(physical)
                .expect("failed to get surface capabilities")
                .current_extent.unwrap();

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);

            framebuffers = None;
            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            let new_framebuffers = Some({
                images.iter()
                    .map(|image| Arc::new({
                        Framebuffer::start(postprocess_renderpass.clone())
                            .add(image.clone()).unwrap()
                            .build().unwrap()
                    }))
                    .collect::<Vec<_>>()
            });
            mem::replace(&mut framebuffers, new_framebuffers);
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err),
        };

        let uniform_buffer_subbuffer = {
            let elapsed = time_start.elapsed();
            let rotation_factor = elapsed.as_secs() as f64
                + elapsed.subsec_nanos() as f64 * 1e-9;
            let rotation = Matrix3::from_angle_z(Rad(rotation_factor as f32));

            let uniform_data = scene_vs_mod::ty::Matrices {
                model: Matrix4::from(rotation).into(),
                view: view.into(),
                proj: proj.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let scene_set = Arc::new({
            PersistentDescriptorSet::start(scene_pipeline.clone(), 0)
                .add_buffer(uniform_buffer_subbuffer).unwrap()
                .build().unwrap()
        });

        let postprocess_set = Arc::new({
            PersistentDescriptorSet::start(postprocess_pipeline.clone(), 0)
                .add_image(attachment_image_newframe.clone()).unwrap()
                .add_image(attachment_image_ping.clone()).unwrap()
                .build().unwrap()
        });

        let copy_set = Arc::new({
            PersistentDescriptorSet::start(copy_pipeline.clone(), 0)
                .add_image(attachment_image_pong.clone()).unwrap()
                .build().unwrap()
        });

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family()
            )
            .unwrap()
            .begin_render_pass(
                newframe_framebuffer.clone(),
                false,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            )
            .unwrap()
            .draw(
                scene_pipeline.clone(),
                dynamic_state.clone(),
                scene_vertex_buffer.clone(),
                scene_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .copy_image(
                attachment_image_pong.clone(), [0, 0, 0], 0, 0,
                attachment_image_ping.clone(), [0, 0, 0], 0, 0,
                [dimensions[0], dimensions[1], 1], 0,
            )
            .unwrap()
            .begin_render_pass(
                pong_framebuffer.clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                postprocess_pipeline.clone(),
                dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                postprocess_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .begin_render_pass(
                framebuffers.as_ref().unwrap()[image_num].clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                copy_pipeline.clone(),
                dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                copy_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            // .copy_image(
            //     attachment_image_pong.clone(), [0, 0, 0], 0, 0,
            //     images[image_num].clone(), [0, 0, 0], 0, 0,
            //     [dimensions[0], dimensions[1], 1], 0,
            // )
            // .unwrap()
            .build()
            .unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => previous_frame_end = Box::new(future),
            Err(vk_sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vk_sync::now(device.clone()));
            },
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vk_sync::now(device.clone()));
            },
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event: WindowEvent::Closed, .. } => done = true,
                _ => (),
            }
        });
        if done { return; }
    }
}

#[allow(dead_code)]
mod scene_vs_mod {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(set = 0, binding = 0) uniform Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
} matrices;

layout(location = 0) in vec2 a_position;

void main() {
    gl_Position = matrices.proj
        * matrices.view
        * matrices.model
        * vec4(a_position, 0.0, 1.0);
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod scene_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod postprocess_vs_mod {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;

layout(location = 0) out vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod postprocess_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(set = 0, binding = 0, rgba8) uniform readonly image2D u_newframe;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D u_prevframe;

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

vec4 blend_with_factor(vec4 src_color, vec4 dst_color, float factor) {
    return (src_color * factor) + (dst_color * (1. - factor));
}

void main() {
    ivec2 index = ivec2(gl_FragCoord.xy);
    vec4 c1 = imageLoad(u_newframe, index);
    vec4 c2 = imageLoad(u_prevframe, index);
    f_color = blend_with_factor(c2, c1, 0.8);
    // f_color = c1;
}
"]
    struct Dummy;
}

mod copy_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(set = 0, binding = 0, rgba8) uniform readonly image2D u_image;

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

void main() {
    ivec2 index = ivec2(gl_FragCoord.xy);
    f_color = imageLoad(u_image, index);
}
"]
    struct Dummy;
}

