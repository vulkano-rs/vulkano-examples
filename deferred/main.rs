// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![allow(dead_code)]

extern crate cgmath;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::device::Device;
use vulkano::instance::Instance;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

use cgmath::Matrix4;
use cgmath::SquareMatrix;
use cgmath::Vector3;

use std::mem;

mod ambient_lighting_system;
mod directional_lighting_system;
mod frame;
mod point_lighting_system;
mod triangle_draw_system;

fn main() {
    // Basic initialization. See the triangle example if you want more details about this.
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");


    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    // Get the dimensions of the viewport. These variables need to be mutable since the viewport
    // can change size.
    let dimensions = {
        let (width, height) = window.window().get_inner_size_pixels().unwrap();
        [width, height]
    };

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
        Swapchain::new(device.clone(), window.surface().clone(), caps.min_image_count, format,
                       dimensions, 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };

    let mut frame_system = frame::FrameSystem::new(queue.clone(), swapchain.format());
    let triangle_draw_system = triangle_draw_system::TriangleDrawSystem::new(queue.clone(),
                                                                             frame_system.deferred_subpass());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        // If the swapchain needs to be recreated, recreate it
        if recreate_swapchain {
            // Get the new dimensions for the viewport/framebuffers.
            let dimensions = {
                let (new_width, new_height) = window.window().get_inner_size_pixels().unwrap();
                [new_width, new_height]
            };
            
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);
            recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let future = previous_frame_end.join(acquire_future);
        let mut frame = frame_system.frame(future, images[image_num].clone(), Matrix4::identity());
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                frame::Pass::Deferred(mut draw_pass) => {
                    let cb = triangle_draw_system.draw(draw_pass.viewport_dimensions());
                    draw_pass.execute(cb);
                },
                frame::Pass::Lighting(mut lighting) => {
                    lighting.ambient_light([0.1, 0.1, 0.1]);
                    lighting.directional_light(Vector3::new(0.2, -0.1, -0.7), [0.6, 0.6, 0.6]);
                    lighting.point_light(Vector3::new(0.5, -0.5, -0.1), [1.0, 0.0, 0.0]);
                },
                frame::Pass::Finished(af) => {
                    after_future = Some(af);
                },
            }
        }

        let after_frame = after_future.unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        previous_frame_end = Box::new(after_frame) as Box<_>;

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
