// Copyright (c) 2018 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/*

TODO:
- Comments!
- Add a more visually plasing scene
- Refactor (this file is a bit too long)
- HDR image formats
- Optimizations
    - some things can maybe be done in subpasses
    - can we reuse some images? (vulkano currently protests very much against this)
        * reusing would also make it possible to repeat the blurring process
*/

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
    },
    sampler::Sampler,
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
    let postprocess_dimensions = [dimensions[0] / 2, dimensions[0] / 2];

    let sampler = Sampler::simple_repeat_linear_no_mipmap(device.clone());

    // let color_format = Format::B8G8R8A8Unorm;
    let color_format = swapchain.format();
    let depth_format = Format::D16Unorm;
    let scene_color_attachment = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        color_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            sampled: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let scene_depth_attachment = AttachmentImage::with_usage(
        device.clone(),
        dimensions,
        depth_format,
        ImageUsage {
            storage: true,
            depth_stencil_attachment: true,
            sampled: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let sep_attachment = AttachmentImage::with_usage(
        device.clone(),
        postprocess_dimensions,
        color_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            sampled: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let ping_attachment = AttachmentImage::with_usage(
        device.clone(),
        postprocess_dimensions,
        color_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            sampled: true,
            .. ImageUsage::none()
        },
    ).expect("failed to create attachment image");

    let pong_attachment = AttachmentImage::with_usage(
        device.clone(),
        postprocess_dimensions,
        color_format,
        ImageUsage {
            storage: true,
            color_attachment: true,
            sampled: true,
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

    let matrix_uniform_buffer = CpuBufferPool::<scene_vs_mod::ty::Matrices>::new(
        device.clone(),
        BufferUsage::all(),
    );

    let blur_direction_uniform_buffer_horizontal = {
        CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::all(),
            postprocess_blur_fs_mod::ty::BlurDirection {
                direction: [1, 0],
            }
        ).expect("failed to create buffer")
    };

    let blur_direction_uniform_buffer_vertical = {
        CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::all(),
            postprocess_blur_fs_mod::ty::BlurDirection {
                direction: [0, 1],
            }
        ).expect("failed to create buffer")
    };

    let blur_kernel_uniform_buffer = {
        CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::all(),
            postprocess_blur_fs_mod::ty::BlurKernel {
                kernel: [
	                0.382925,
                    0.24173,
                    0.060598,
                    0.005977,
                    0.000229,
                    0.000003,
                ],
            }
        ).expect("failed to create buffer")
    };

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
    let postprocess_sep_fs = postprocess_sep_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");
    let postprocess_blur_fs = postprocess_blur_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");
    let postprocess_tonemap_fs = postprocess_tonemap_fs_mod::Shader::load(device.clone())
        .expect("failed to create shader module");

    let scene_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                scene_color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                scene_depth: {
                    load: Clear,
                    store: Store,
                    format: depth_format,
                    samples: 1,
                }
            },
            pass: {
                color: [scene_color],
                depth_stencil: {scene_depth}
            }
        ).unwrap()
    });

    let postprocess_sep_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                sep_color: {
                    load: DontCare,
                    store: Store,
                    format: color_format,
                    samples: 1,
                }
            },
            pass: {
                color: [sep_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let postprocess_blur_ping_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                ping_color: {
                    load: DontCare,
                    store: Store,
                    format: color_format,
                    samples: 1,
                }
            },
            pass: {
                color: [ping_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let postprocess_blur_pong_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                pong_color: {
                    load: DontCare,
                    store: Store,
                    format: color_format,
                    samples: 1,
                }
            },
            pass: {
                color: [pong_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let postprocess_tonemap_renderpass = Arc::new({
        single_pass_renderpass!(
            device.clone(),
            attachments: {
                output_color: {
                    load: DontCare,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [output_color],
                depth_stencil: {}
            }
        ).unwrap()
    });

    let scene_framebuffer = Arc::new({
        Framebuffer::start(scene_renderpass.clone())
            .add(scene_color_attachment.clone()).unwrap()
            .add(scene_depth_attachment.clone()).unwrap()
            .build().unwrap()
    });

    let sep_framebuffer = Arc::new({
        Framebuffer::start(postprocess_sep_renderpass.clone())
            .add(sep_attachment.clone()).unwrap()
            .build().unwrap()
    });

    let ping_framebuffer = Arc::new({
        Framebuffer::start(postprocess_blur_ping_renderpass.clone())
            .add(ping_attachment.clone()).unwrap()
            .build().unwrap()
    });

    let pong_framebuffer = Arc::new({
        Framebuffer::start(postprocess_blur_pong_renderpass.clone())
            .add(pong_attachment.clone()).unwrap()
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

    let postprocess_sep_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(postprocess_sep_fs.main_entry_point(), ())
            .render_pass(Subpass::from(postprocess_sep_renderpass.clone(), 0)
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    let postprocess_blur_ping_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(postprocess_blur_fs.main_entry_point(), ())
            .render_pass(Subpass::from(
                postprocess_blur_ping_renderpass.clone(),
                0,
            )
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    let postprocess_blur_pong_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(postprocess_blur_fs.main_entry_point(), ())
            .render_pass(Subpass::from(
                postprocess_blur_pong_renderpass.clone(),
                0,
            )
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    let postprocess_tonemap_pipeline = Arc::new({
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(postprocess_vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(postprocess_tonemap_fs.main_entry_point(), ())
            .render_pass(Subpass::from(postprocess_tonemap_renderpass.clone(), 0)
            .unwrap())
            .build(device.clone())
            .unwrap()
    });

    let postprocess_sep_set = Arc::new({
        PersistentDescriptorSet::start(postprocess_sep_pipeline.clone(), 0)
            .add_sampled_image(scene_color_attachment.clone(), sampler.clone())
            .unwrap()
            .build()
            .unwrap()
    });

    // Ping set is used to render to ping, therefore has to use pong attachment
    let postprocess_blur_ping_set = Arc::new({
        PersistentDescriptorSet::start(postprocess_blur_ping_pipeline.clone(), 0)
            .add_sampled_image(sep_attachment.clone(), sampler.clone())
            .unwrap()
            .add_buffer(blur_direction_uniform_buffer_horizontal.clone())
            .unwrap()
            .add_buffer(blur_kernel_uniform_buffer.clone())
            .unwrap()
            .build()
            .unwrap()
    });

    // Pong set is used to render to pong, therefore has to use ping attachment
    let postprocess_blur_pong_set = Arc::new({
        PersistentDescriptorSet::start(postprocess_blur_pong_pipeline.clone(), 0)
            .add_sampled_image(ping_attachment.clone(), sampler.clone())
            .unwrap()
            .add_buffer(blur_direction_uniform_buffer_vertical.clone())
            .unwrap()
            .add_buffer(blur_kernel_uniform_buffer.clone())
            .unwrap()
            .build()
            .unwrap()
    });

    let postprocess_tonemap_set = Arc::new({
        PersistentDescriptorSet::start(postprocess_tonemap_pipeline.clone(), 0)
            .add_sampled_image(scene_color_attachment.clone(), sampler.clone())
            .unwrap()
            .add_sampled_image(pong_attachment.clone(), sampler.clone())
            .unwrap()
            .build()
            .unwrap()
    });

    let mut previous_frame_end: Box<GpuFuture> = Box::new(vk_sync::now(device.clone()));

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
                        Framebuffer::start(postprocess_tonemap_renderpass.clone())
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

        let matrix_uniform_buffer_subbuffer = {
            let elapsed = time_start.elapsed();
            let rotation_factor = elapsed.as_secs() as f64
                + elapsed.subsec_nanos() as f64 * 1e-9;
            let rotation = Matrix3::from_angle_z(Rad(rotation_factor as f32));

            let uniform_data = scene_vs_mod::ty::Matrices {
                model: Matrix4::from(rotation).into(),
                view: view.into(),
                proj: proj.into(),
            };

            matrix_uniform_buffer.next(uniform_data).unwrap()
        };

        let scene_set = Arc::new({
            PersistentDescriptorSet::start(scene_pipeline.clone(), 0)
                .add_buffer(matrix_uniform_buffer_subbuffer).unwrap()
                .build().unwrap()
        });

        let scene_dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let postprocess_dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [
                    postprocess_dimensions[0] as f32,
                    postprocess_dimensions[1] as f32,
                ],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family()
            )
            .unwrap()
            // BEGIN SCENE
            .begin_render_pass(
                scene_framebuffer.clone(),
                false,
                vec![
                    ClearValue::Float([0.0, 0.0, 0.0, 1.0]),
                    ClearValue::Depth(1.0),
                ],
            )
            .unwrap()
            .draw(
                scene_pipeline.clone(),
                scene_dynamic_state.clone(),
                scene_vertex_buffer.clone(),
                scene_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            // END SCENE
            // BEGIN SEP
            .begin_render_pass(
                sep_framebuffer.clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                postprocess_sep_pipeline.clone(),
                postprocess_dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                postprocess_sep_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            // END SEP
            // BEGIN BLUR
            .begin_render_pass(
                ping_framebuffer.clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                postprocess_blur_ping_pipeline.clone(),
                postprocess_dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                postprocess_blur_ping_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .begin_render_pass(
                pong_framebuffer.clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                postprocess_blur_pong_pipeline.clone(),
                postprocess_dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                postprocess_blur_pong_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            // END BLUR
            // BEGIN TONEMAP
            .begin_render_pass(
                framebuffers.as_ref().unwrap()[image_num].clone(),
                false,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                postprocess_tonemap_pipeline.clone(),
                scene_dynamic_state.clone(),
                postprocess_vertex_buffer.clone(),
                postprocess_tonemap_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            // END TONEMAP
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
} u_matrices;

layout(location = 0) in vec2 a_position;

void main() {
    gl_Position = u_matrices.proj
        * u_matrices.view
        * u_matrices.model
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
    f_color = vec4(0.2, 0.9, 0.9, 1.0);
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
mod postprocess_sep_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(set = 0, binding = 0) uniform sampler2D u_image;

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 color = texture(u_image, v_texcoord);

    // Convert to grayscale and compute brightness
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    f_color = brightness > 0.7 ? color : vec4(0.0, 0.0, 0.0, 1.0);
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod postprocess_blur_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

#define KERNEL_LENGTH 6

layout(set = 0, binding = 0) uniform sampler2D u_image;
layout(set = 0, binding = 1) uniform BlurDirection {
    ivec2 direction;
} u_blur_direction;
layout(set = 0, binding = 2) uniform BlurKernel {
    float[KERNEL_LENGTH] kernel;
} u_blur_kernel;

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

void main() {
    vec2 blur_direction = vec2(u_blur_direction.direction);
    vec2 px_direction = vec2(1) / vec2(textureSize(u_image, 0)) * blur_direction;
    vec4 color_sum = u_blur_kernel.kernel[0] * texture(u_image, v_texcoord);
    for (int i = 1; i <= KERNEL_LENGTH; i++) {
        float k = u_blur_kernel.kernel[i];
        color_sum += k * texture(u_image,  px_direction * vec2(i) + v_texcoord);
        color_sum += k * texture(u_image, -px_direction * vec2(i) + v_texcoord);
    }
    f_color = color_sum;
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod postprocess_tonemap_fs_mod {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(set = 0, binding = 0) uniform sampler2D u_image;
layout(set = 0, binding = 1) uniform sampler2D u_image_blur;

layout(location = 0) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

void main() {
    vec3 color = texture(u_image, v_texcoord).rgb;
    vec3 bloom = texture(u_image_blur, v_texcoord).rgb;

    const float gamma = 2.2;

    // Additive blending
    color += bloom;

    // Reinhard tone mapping
    vec3 mapped = color / (color + vec3(1.0));

    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / gamma));

    f_color = vec4(mapped, 1.0);
}
"]
    struct Dummy;
}
