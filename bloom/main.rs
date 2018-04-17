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
- Code Comments!
- Add a more visually plasing scene
- Refactor (this file is a bit too long)
- HDR image formats
- Do not call the last step "tonemap", as we do not use HDR attachments
- Optimizations
    - subpixel sampling with stride 2
    - some things can maybe be done in subpasses
    - can we reuse some images? (vulkano currently protests very much against this)
        * reusing would also make it possible to repeat the blurring process
*/

//! Bloom example, using multiple renderpasses
//!
//! # Introduction to Bloom
//!
//! Bloom (also called glow) is a postprocessing technique used to convey that
//! some objects in the rendered scene are really bright. Imagine a scene with
//! light sources. In classical rendering, the lights can be rendered using
//! its color or a brighter variant, but there is only so much that can be
//! done, as we cannot exceed white color. Bloom makes the color of bright
//! objects bleed out of their frame, adding a psychological effect of
//! brightness.
//!
//! # Implementing Bloom
//!
//! Bloom happens in the screenspace, and usually requires at lest 3-4
//! rendering passes. Conceptually, the following happens:
//!     1.  Scene is rendered to an image
//!     2.  Bright colors are separated from the rendered scene to its own image
//!     3.  Separated highlights are blurred
//!     4.  Blurred highlights are merged back to the original image.
//!
//! Note that steps 1. and 2. can be merged in one using multiple render targets
//! for optimization.
//!
//! ## Separation Pass
//!
//! During separation, we copy bright pixels from one image to another. We
//! usually select the pixel based on brightness (eg. if grayscale
//! brightness exceeds a threshold). If needed, the pixels can also be picked
//! by beloging to a certain object (eg. only blur pixels coming from light
//! sources, but not other bright surfaces), but this requires additional
//! information to be passed.
//!
//! ## Blur Passes
//!
//! Once we have the image containing the bright areas, we can perform gaussian
//! blur on it. This is done by convoluting a gaussian kernel on the image's pixels.
//!
//! When applying this 3x3 gaussian kernel to a pixel, its value will be the its
//! original value multiplied by the kernel's middle cell, summed with the kernel's
//! outer cells multiplied with their corresponding pixels in our image.
//!
//! 0.077847	0.123317	0.077847
//! 0.123317	0.195346	0.123317
//! 0.077847	0.123317	0.077847
//!
//! This gaussian kernel is already normalized, meaning its cells sum up to one,
//! so we don't end up with brighter pixels than we previously had. Note, that
//! for this 3x3 kernel we will need to perform 9 texture sample operations.
//!
//! To optimize, we can use 1d kernels instead of 2d, and blur in multiple render
//! passes (eg first horizontally, then vertically).
//! This reduces the number of sample operations from N ^ 2 to 2 * N, and is
//! a significant speedup for larger blur kernels.
//!
//! To achieve the right visual result, we can apply the pairs of blur passes
//! multiple times.
//!
//! ## Merge Pass
//!
//! In the final pass, we merge the blurred highlights back with the original
//! image. Bloom can sometimes be implemented together with HDR, but HDR itself
//! is not necessary, it only complements the bloom effect nicely. If we used HDR,
//! we would also apply tonemapping and gamma correcting here.
//!
//! # Optimizations
//!
//! Besides using 1d kernels for blur, we can also do the following.
//!
//! Use a fraction of the screen resolution for the blur images. Since we sample
//! the images anyway, they do not need to be the same size. This effectively
//! reduces the number of fragment shader runs. The resolution can easily be
//! halved before any visual degradation of output.
//!
//! Another optimization is sampling between two pixels and strinding
//! two pixels at once when blurring. This increases the blur's reach, and we
//! may potentially use a lesser number of passes achieve the same visual result.
//!

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

use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::buffer::CpuBufferPool;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::image::ImageUsage;
use vulkano::image::AttachmentImage;
use vulkano::sampler::Sampler;
use vulkano::format::Format;
use vulkano::format::ClearValue;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync as vk_sync;
use vulkano::sync::GpuFuture;


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