// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


// Welcome to the glTF example!

extern crate cgmath;
extern crate gltf;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

use std::env;
use std::sync::Arc;
use std::mem;

mod gltf_system;

fn main() {
    // These initialization steps are common to all examples. See the `triangle` example if you
    // want explanations.
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");

    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let queue = physical.queue_families().find(|&q| {
        q.supports_graphics() && window.surface().is_supported(q).unwrap_or(false)
    }).expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical, physical.supported_features(), &device_ext,
                    [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = window.surface().capabilities(physical)
                         .expect("failed to get surface capabilities");
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dims = window.window().get_inner_size_pixels().unwrap();
        Swapchain::new(device.clone(), window.surface().clone(), caps.min_image_count, format,
                       [dims.0, dims.1], 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };

    //
    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this
                // attachment at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw
                // in the actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to
                // be one of the types of the `vulkano::format` module (or alternatively one
                // of your structs that implements the `FormatDesc` trait). Here we use the
                // generic `vulkano::format::Format` enum because we don't know the format in
                // advance.
                format: swapchain.format(),
                // TODO:
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {}
        }
    ).unwrap());

    // Try loading our glTF model.
    let gltf = {
        let args: Vec<_> = env::args().collect();
        let path = args.get(1).map(|s| s.as_str()).unwrap_or("gltf/Duck.gltf");
        let import = gltf::Import::from_path(path);
        import.sync().expect("Error while loading glTF file")
    };

    let model = gltf_system::GltfModel::new(gltf, queue.clone(), Subpass::from(render_pass.clone(), 0).unwrap());


    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        let dimensions = {
            let (new_width, new_height) = window.window().get_inner_size_pixels().unwrap();
            [new_width, new_height]
        };

        if recreate_swapchain {
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(images[image_num].clone())
            .unwrap()
            .build()
            .unwrap());

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false,
                               vec![[0.0, 0.0, 1.0, 1.0].into()])
            .unwrap();

        builder = model.draw_default_scene(dimensions, builder);

        let command_buffer = builder.end_render_pass()
            .unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        previous_frame_end = Box::new(future) as Box<_>;

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                _ => ()
            }
        });
        if done { return; }
    }
}
