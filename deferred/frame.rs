// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;
use cgmath::Matrix4;
use cgmath::SquareMatrix;
use cgmath::Vector3;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::AttachmentImage;
use vulkano::image::ImageAccess;
use vulkano::image::ImageUsage;
use vulkano::image::ImageViewAccess;
use vulkano::sync::GpuFuture;

use ambient_lighting_system::AmbientLightingSystem;
use directional_lighting_system::DirectionalLightingSystem;
use point_lighting_system::PointLightingSystem;

/// System that contains the necessary facilities for rendering a single frame.
pub struct FrameSystem {
    gfx_queue: Arc<Queue>,

    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    final_output_format: Format,

    diffuse_buffer: Arc<AttachmentImage>,
    normals_buffer: Arc<AttachmentImage>,
    depth_buffer: Arc<AttachmentImage>,
    
    ambient_lighting_system: AmbientLightingSystem,
    directional_lighting_system: DirectionalLightingSystem,
    point_lighting_system: PointLightingSystem,
}

impl FrameSystem {
    /// Initializes the frame system.
    ///
    /// Should be called at initialization, as it can take some time to build.
    pub fn new(gfx_queue: Arc<Queue>, final_output_format: Format) -> FrameSystem {
        let render_pass = Arc::new(
            ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                final_color: {
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                },
                diffuse: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10UnormPack32,
                    samples: 1,
                },
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16Sfloat,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            passes: [
                // Write diffuse and normals.
                {
                    color: [diffuse, normals],
                    depth_stencil: {depth},
                    input: []
                },
                // Apply lighting.
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: [diffuse, normals, depth]
                }
            ]
        ).unwrap(),
        );

        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        let diffuse_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                         [1, 1],
                                                         Format::A2B10G10R10UnormPack32,
                                                         atch_usage)
            .unwrap();
        let normals_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                         [1, 1],
                                                         Format::R16G16B16A16Sfloat,
                                                         atch_usage)
            .unwrap();
        let depth_buffer = AttachmentImage::with_usage(gfx_queue.device().clone(),
                                                       [1, 1],
                                                       Format::D16Unorm,
                                                       atch_usage)
            .unwrap();

        let ambient_lighting_system = {
            let subpass = Subpass::from(render_pass.clone(), 1).unwrap();
            AmbientLightingSystem::new(gfx_queue.clone(), subpass)
        };
        let directional_lighting_system = {
            let subpass = Subpass::from(render_pass.clone(), 1).unwrap();
            DirectionalLightingSystem::new(gfx_queue.clone(), subpass)
        };
        let point_lighting_system = {
            let subpass = Subpass::from(render_pass.clone(), 1).unwrap();
            PointLightingSystem::new(gfx_queue.clone(), subpass)
        };

        FrameSystem {
            gfx_queue,
            render_pass: render_pass as Arc<_>,
            diffuse_buffer,
            normals_buffer,
            depth_buffer,
            final_output_format,
            ambient_lighting_system,
            directional_lighting_system,
            point_lighting_system,
        }
    }

    /// Returns the subpass of the render pass where the rendering should write info to gbuffers.
    ///
    /// Has two outputs: the diffuse color (3 components) and the normals in world coordinates
    /// (3 components). Also has a depth attachment.
    #[inline]
    pub fn deferred_subpass(&self) -> Subpass<Arc<RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    /// Starts drawing a new frame.
    pub fn frame<F, I>(&mut self, before_future: F, final_image: I,
                       world_to_framebuffer: Matrix4<f32>) -> Frame
        where F: GpuFuture + 'static,
              I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static
    {
        let img_dims = ImageAccess::dimensions(&final_image).width_height();

        if ImageAccess::dimensions(&self.diffuse_buffer).width_height() != img_dims {
            let atch_usage = ImageUsage {
                transient_attachment: true,
                input_attachment: true,
                ..ImageUsage::none()
            };

            self.diffuse_buffer = AttachmentImage::with_usage(self.gfx_queue.device().clone(),
                                                              img_dims,
                                                              Format::A2B10G10R10UnormPack32,
                                                              atch_usage)
                .unwrap();
            self.normals_buffer = AttachmentImage::with_usage(self.gfx_queue.device().clone(),
                                                              img_dims,
                                                              Format::R16G16B16A16Sfloat,
                                                              atch_usage)
                .unwrap();
            self.depth_buffer = AttachmentImage::with_usage(self.gfx_queue.device().clone(),
                                                            img_dims,
                                                            Format::D16Unorm,
                                                            atch_usage)
                .unwrap();
        }

        let framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(final_image.clone())
            .unwrap()
            .add(self.diffuse_buffer.clone())
            .unwrap()
            .add(self.normals_buffer.clone())
            .unwrap()
            .add(self.depth_buffer.clone())
            .unwrap()
            .build()
            .unwrap());

        let command_buffer =
            Some(AutoCommandBufferBuilder::primary_one_time_submit(self.gfx_queue
                                                                       .device()
                                                                       .clone(),
                                                                   self.gfx_queue.family())
                     .unwrap());

        Frame {
            system: self,
            before_main_cb_future: Some(Box::new(before_future)),
            framebuffer,
            num_pass: 0,
            command_buffer,
            world_to_framebuffer,
        }
    }
}

/// Represents the active process of rendering a frame.
pub struct Frame<'a> {
    system: &'a mut FrameSystem,
    before_main_cb_future: Option<Box<GpuFuture>>,
    framebuffer: Arc<FramebufferAbstract + Send + Sync>,
    num_pass: u8,
    command_buffer: Option<AutoCommandBufferBuilder>,
    world_to_framebuffer: Matrix4<f32>,
}

impl<'a> Frame<'a> {
    /// Returns an enumeration containing the next pass of the rendering.
    pub fn next_pass<'f>(&'f mut self) -> Option<Pass<'f, 'a>> {
        match self.num_pass {
            0 => {
                self.num_pass += 1;

                self.command_buffer =
                    Some(self.command_buffer
                             .take()
                             .unwrap()
                             .begin_render_pass(self.framebuffer.clone(),
                                                true,
                                                vec![[0.0, 0.0, 0.0, 0.0].into(),
                                                [0.0, 0.0, 0.0, 0.0].into(),
                                                [0.0, 0.0, 0.0, 0.0].into(),
                                                1.0f32.into()])
                             .unwrap());

                Some(Pass::Deferred(DrawPass {
                                    frame: self,
                                }))
            },
            1 => {
                self.num_pass += 1;

                self.command_buffer = Some(
                    self.command_buffer
                        .take()
                        .unwrap()
                        .next_subpass(false)
                        .unwrap()
                );

                Some(Pass::Lighting(LightingPass {
                                    frame: self,
                                }))
            },
            2 => {
                self.num_pass += 1;

                let command_buffer = 
                    self.command_buffer
                        .take()
                        .unwrap()
                        .end_render_pass()
                        .unwrap()
                        .build()
                        .unwrap();

                let before_main_cb = self.before_main_cb_future.take().unwrap();
                let after_main_cb = before_main_cb
                    .then_execute(self.system.gfx_queue.clone(), command_buffer)
                    .unwrap();
                Some(Pass::Finished(Box::new(after_main_cb)))
            },
            _ => None,
        }
    }
}

pub enum Pass<'f, 's: 'f> {
    Deferred(DrawPass<'f, 's>),
    Lighting(LightingPass<'f, 's>),
    Finished(Box<GpuFuture>),
}

pub struct DrawPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> DrawPass<'f, 's> {
    /// Adds a secondary command buffer that performs drawing.
    #[inline]
    pub fn execute<C>(&mut self, command_buffer: C)
        where C: CommandBuffer + Send + Sync + 'static
    {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        unsafe {
            self.frame.command_buffer = Some(self.frame
                .command_buffer
                .take()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap());
        }
    }

    /// Returns the dimensions in pixels of the viewport.
    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        let dims = self.frame.framebuffer.dimensions();
        [dims[0], dims[1]]
    }

    /// Returns the 4x4 matrix that turns world coordinates into 2D coordinates on the framebuffer.
    #[inline]
    pub fn world_to_framebuffer_matrix(&self) -> Matrix4<f32> {
        self.frame.world_to_framebuffer
    }
}

pub struct LightingPass<'f, 's: 'f> {
    frame: &'f mut Frame<'s>,
}

impl<'f, 's: 'f> LightingPass<'f, 's> {
    /// Applies an ambient lighting to the scene.
    pub fn ambient_light(&mut self, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = self.frame.system.ambient_lighting_system.draw([dims[0], dims[1]], self.frame.system.diffuse_buffer.clone(), color);
            self.frame.command_buffer = Some(self.frame
                .command_buffer
                .take()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap());
        }
    }

    /// Applies an directional lighting to the scene.
    pub fn directional_light(&mut self, direction: Vector3<f32>, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = self.frame.system.directional_lighting_system.draw([dims[0], dims[1]], self.frame.system.diffuse_buffer.clone(), self.frame.system.normals_buffer.clone(), direction, color);
            self.frame.command_buffer = Some(self.frame
                .command_buffer
                .take()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap());
        }
    }

    /// Applies a spot lighting to the scene.
    pub fn point_light(&mut self, position: Vector3<f32>, color: [f32; 3]) {
        // Note that vulkano doesn't perform any safety check for now when executing secondary
        // command buffers, hence why it is unsafe. This operation will be safe in the future
        // however.
        unsafe {
            let dims = self.frame.framebuffer.dimensions();
            let command_buffer = {
                self.frame.system.point_lighting_system.draw([dims[0], dims[1]],
                    self.frame.system.diffuse_buffer.clone(),
                    self.frame.system.normals_buffer.clone(),
                    self.frame.system.depth_buffer.clone(),
                    self.frame.world_to_framebuffer.invert().unwrap(),
                    position, color)
            };
            self.frame.command_buffer = Some(self.frame
                .command_buffer
                .take()
                .unwrap()
                .execute_commands(command_buffer)
                .unwrap());
        }
    }
}
