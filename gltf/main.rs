// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


// Welcome to the glTF example!

extern crate gltf;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuBufferPool;
use vulkano::buffer::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor::DescriptorBufferDesc;
use vulkano::descriptor::descriptor::DescriptorBufferContentDesc;
use vulkano::descriptor::descriptor::DescriptorImageDesc;
use vulkano::descriptor::descriptor::DescriptorImageDescArray;
use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::descriptor::descriptor::DescriptorDescTy;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescNames;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::image::Dimensions;
use vulkano::image::immutable::ImmutableImage;
use vulkano::instance::Instance;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::shader::ShaderInterfaceDef;
use vulkano::pipeline::vertex::AttributeInfo;
use vulkano::pipeline::vertex::IncompatibleVertexDefinitionError;
use vulkano::pipeline::vertex::InputRate;
use vulkano::pipeline::vertex::VertexDefinition;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::Sampler;
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
use std::vec::IntoIter as VecIntoIter;

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

    let gltf_buffers = {
        let mut buffers = Vec::new();
        for buffer in gltf.buffers() {
            let (buf, future) = {
                ImmutableBuffer::from_iter(buffer.data().iter().cloned(),
                                           BufferUsage::all(),
                                           Some(queue.family()), queue.clone())
                                           .expect("Failed to create immutable buffer")
            };
            buffers.push(buf);
        }
        buffers
    };

    let gltf_textures = {
        // TODO: use the sampler defined by the JSON struct
        let sampler = Sampler::simple_repeat_linear(device.clone());

        let mut textures = Vec::new();
        for texture in gltf.textures() {
            let (dimensions, format, raw_pixels) = match texture.source().data() {
                // Note that the gltf crate doesn't allow us to extract the image data without
                // cloning.
                &gltf::import::data::DynamicImage::ImageLuma8(ref buf) => {
                    let dimensions = Dimensions::Dim2d { width: buf.width(), height: buf.height() };
                    (dimensions, Format::R8Srgb, buf.clone().into_raw())
                },
                &gltf::import::data::DynamicImage::ImageLumaA8(ref buf) => {
                    let dimensions = Dimensions::Dim2d { width: buf.width(), height: buf.height() };
                    (dimensions, Format::R8G8Srgb, buf.clone().into_raw())
                },
                &gltf::import::data::DynamicImage::ImageRgb8(ref buf) => {
                    // Since RGB is often not supported by Vulkan, convert to RGBA instead.
                    let dimensions = Dimensions::Dim2d { width: buf.width(), height: buf.height() };
                    let rgba = gltf::import::data::DynamicImage::ImageRgb8(buf.clone()).to_rgba();
                    (dimensions, Format::R8G8B8A8Srgb, rgba.into_raw())
                },
                &gltf::import::data::DynamicImage::ImageRgba8(ref buf) => {
                    let dimensions = Dimensions::Dim2d { width: buf.width(), height: buf.height() };
                    (dimensions, Format::R8G8B8A8Srgb, buf.clone().into_raw())
                },
            };

            let (img, future) = {
                ImmutableImage::from_iter(raw_pixels.into_iter(), dimensions, format,
                                          Some(queue.family()), queue.clone())
                                          .expect("Failed to create immutable image")
            };
            textures.push((img, sampler.clone()));
        }
        textures
    };

    let gltf_materials = {
        let mut params_buffer = CpuBufferPool::new(device.clone(), BufferUsage::uniform_buffer(),
                                                   Some(queue.family()));
        let pipeline_layout = Arc::new(MyPipelineLayout.build(device.clone()).unwrap());

        let dummy_sampler = Sampler::simple_repeat_linear(device.clone());
        let (dummy_texture, _) = 
                ImmutableImage::from_iter([0u8].iter().cloned(),
                                          Dimensions::Dim2d { width: 1, height: 1 },
                                          Format::R8Unorm, Some(queue.family()), queue.clone())
                                          .expect("Failed to create immutable image");

        let mut materials = Vec::new();
        for material in gltf.as_json().materials.iter() {
            let material_params = params_buffer.next(fs::ty::MaterialParams {
                base_color_factor: material.pbr_metallic_roughness.as_ref()
                                .map(|t| t.base_color_factor.0).unwrap_or([1.0, 1.0, 1.0, 1.0]),
                base_color_texture_tex_coord: material.pbr_metallic_roughness.as_ref()
                            .and_then(|t| t.base_color_texture.as_ref()).map(|t| t.tex_coord as i32).unwrap_or(-1),
                metallic_factor: material.pbr_metallic_roughness.as_ref()
                                                    .map(|t| t.metallic_factor.0).unwrap_or(1.0),
                roughness_factor: material.pbr_metallic_roughness.as_ref()
                                                    .map(|t| t.roughness_factor.0).unwrap_or(1.0),
                metallic_roughness_texture_tex_coord: material.pbr_metallic_roughness.as_ref()
                    .and_then(|t| t.metallic_roughness_texture.as_ref()).map(|t| t.tex_coord as i32).unwrap_or(-1),
                normal_texture_scale: material.normal_texture.as_ref()
                                                             .map(|t| t.scale).unwrap_or(0.0),
                normal_texture_tex_coord: material.normal_texture.as_ref()
                                                    .map(|t| t.tex_coord as i32).unwrap_or(-1),
                occlusion_texture_tex_coord: material.occlusion_texture.as_ref()
                                                     .map(|t| t.tex_coord as i32).unwrap_or(-1),
                occlusion_texture_strength: material.occlusion_texture.as_ref()
                                                    .map(|t| t.strength.0).unwrap_or(0.0),
                emissive_texture_tex_coord: material.emissive_texture.as_ref()
                                                    .map(|t| t.tex_coord as i32).unwrap_or(-1),
                emissive_factor: material.emissive_factor.0,
                _dummy0: [0; 12],
            });

            let base_color = material.pbr_metallic_roughness.as_ref()
                .and_then(|t| t.base_color_texture.as_ref())
                .map(|t| gltf_textures[t.index.value()].clone())
                .unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
            let metallic_roughness = material.pbr_metallic_roughness.as_ref()
                .and_then(|t| t.metallic_roughness_texture.as_ref())
                .map(|t| gltf_textures[t.index.value()].clone())
                .unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
            let normal_texture = material.normal_texture.as_ref()
                .map(|t| gltf_textures[t.index.value()].clone())
                .unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
            let occlusion_texture = material.occlusion_texture.as_ref()
                .map(|t| gltf_textures[t.index.value()].clone())
                .unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));
            let emissive_texture = material.emissive_texture.as_ref()
                .map(|t| gltf_textures[t.index.value()].clone())
                .unwrap_or((dummy_texture.clone(), dummy_sampler.clone()));

            let descriptor_set = 
                Arc::new(PersistentDescriptorSet::start(pipeline_layout.clone(), 0)
                    .add_buffer(material_params)
                    .unwrap()
                    .add_sampled_image(base_color.0, base_color.1)
                    .unwrap()
                    .add_sampled_image(metallic_roughness.0, metallic_roughness.1)
                    .unwrap()
                    .add_sampled_image(normal_texture.0, normal_texture.1)
                    .unwrap()
                    .add_sampled_image(occlusion_texture.0, occlusion_texture.1)
                    .unwrap()
                    .add_sampled_image(emissive_texture.0, emissive_texture.1)
                    .unwrap()
                    .build()
                    .unwrap());

            materials.push(descriptor_set);
        }
        materials
    };


    mod vs {
        #[derive(VulkanoShader)]
        #[ty = "vertex"]
        #[src = "
#version 450

layout(location = 0) in vec3 i_position;
layout(location = 1) in vec3 i_normal;
layout(location = 2) in vec4 i_tangent;
layout(location = 3) in vec2 i_texcoord_0;
layout(location = 4) in vec2 i_texcoord_1;
layout(location = 5) in vec4 i_color_0;
layout(location = 6) in vec4 i_joints_0;
layout(location = 7) in vec4 i_weights_0;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_texcoord_0;
layout(location = 3) out vec2 v_texcoord_1;

void main() {
    v_position = i_position;
    v_normal = i_normal;
    v_texcoord_0 = i_texcoord_0;
    v_texcoord_1 = i_texcoord_1;

    gl_Position = vec4(i_position, 1.0);
    gl_Position.xyz /= 300.0;
}
"]
        struct Dummy;
    }

    mod fs {
        #[derive(VulkanoShader)]
        #[ty = "fragment"]
        #[src = "
#version 450

layout(set = 0, binding = 0) uniform MaterialParams {
    vec4 base_color_factor;
    int base_color_texture_tex_coord;
    float metallic_factor;
    float roughness_factor;
    int metallic_roughness_texture_tex_coord;
    float normal_texture_scale;
    int normal_texture_tex_coord;
    int occlusion_texture_tex_coord;
    float occlusion_texture_strength;
    int emissive_texture_tex_coord;
    vec3 emissive_factor;
} u_material_params;

layout(set = 0, binding = 1) uniform sampler2D u_base_color;
layout(set = 0, binding = 2) uniform sampler2D u_metallic_roughness;
layout(set = 0, binding = 3) uniform sampler2D u_normal_texture;
layout(set = 0, binding = 4) uniform sampler2D u_occlusion_texture;
layout(set = 0, binding = 5) uniform sampler2D u_emissive_texture;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_texcoord_0;
layout(location = 3) in vec2 v_texcoord_1;

layout(location = 0) out vec4 f_color;

const float M_PI = 3.141592653589793;

float SmithG1_var2(float n_dot_v, float r) {
    float tanSquared = (1.0 - n_dot_v * n_dot_v) / max((n_dot_v * n_dot_v), 0.00001);
    return 2.0 / (1.0 + sqrt(1.0 + r * r * tanSquared));
}

void main() {
    // Load the metallic and roughness properties values.
    float metallic = 1.0;
    float perceptual_roughness = 1.0;
    if (u_material_params.base_color_texture_tex_coord == 0) {
        vec2 v = texture(u_metallic_roughness, v_texcoord_0).rg;
        metallic = v.r;
        perceptual_roughness = v.g;
    } else if (u_material_params.base_color_texture_tex_coord == 1) {
        vec2 v = texture(u_metallic_roughness, v_texcoord_1).rg;
        metallic = v.r;
        perceptual_roughness = v.g;
    }
    metallic *= u_material_params.metallic_factor;
    perceptual_roughness *= u_material_params.roughness_factor;

    // Load the base color of the material.
    vec4 base_color = vec4(0.0);
    if (u_material_params.base_color_texture_tex_coord == 0) {
        base_color.rgb = texture(u_base_color, v_texcoord_0).rgb;
    } else if (u_material_params.base_color_texture_tex_coord == 1) {
        base_color.rgb = texture(u_base_color, v_texcoord_1).rgb;
    }
    base_color *= u_material_params.base_color_factor;


    // TODO: temp ; move to uniform buffer
    vec3 u_LightColor = vec3(1.0);
    vec3 u_Camera = vec3(0.0, 0.0, 300.0);
    vec3 u_LightDirection = vec3(-0.4, 0.7, 0.2);


    // Complex maths here.
    
    vec3 n = v_normal;      // TODO:

    vec3 v = normalize(u_Camera - v_position);
    vec3 l = normalize(u_LightDirection);
    vec3 h = normalize(l + v);
    //vec3 reflection = -normalize(reflect(v, n));

    float n_dot_l = clamp(dot(n, l), 0.001, 1.0);
    float n_dot_v = abs(dot(n, v)) + 0.001;
    float n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    float l_dot_h = clamp(dot(l, h), 0.0, 1.0);
    float v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    vec3 diffuse_color = mix(base_color.rgb * (1 - 0.04), vec3(0.0), metallic);
    vec3 specular_color = mix(vec3(0.04), base_color.rgb, metallic);
    
    float reflectance = max(max(specular_color.r, specular_color.g), specular_color.b);
    vec3 specular_environment_r90 = vec3(1.0, 1.0, 1.0) * clamp(reflectance * 25.0, 0.0, 1.0);
    float alpha_roughness = perceptual_roughness * perceptual_roughness;

    vec3 fresnel_schlick_2 = specular_color + (specular_environment_r90 - specular_color) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    float geometric_occlusion_smith_ggx = SmithG1_var2(n_dot_l, alpha_roughness) * SmithG1_var2(n_dot_v, alpha_roughness);
    float ggx;
    {
        float roughness_sq = alpha_roughness * alpha_roughness;
        float f = (n_dot_h * roughness_sq - n_dot_h) * n_dot_h + 1.0;
        ggx = roughness_sq / (M_PI * f * f);
    }

    vec3 diffuse_contrib = (1.0 - fresnel_schlick_2) * base_color.rgb / M_PI;

    vec3 spec_contrib = fresnel_schlick_2 * geometric_occlusion_smith_ggx * ggx / (4.0 * n_dot_l * n_dot_v);

    f_color.rgb = n_dot_l * u_LightColor * (diffuse_contrib + spec_contrib);
    f_color.a = base_color.a;



    // Add the emissive color.
    vec4 emissive = vec4(0.0);
    if (u_material_params.emissive_texture_tex_coord == 0) {
        emissive.rgb = texture(u_emissive_texture, v_texcoord_0).rgb;
        emissive.a = 1.0;
    } else if (u_material_params.emissive_texture_tex_coord == 1) {
        emissive.rgb = texture(u_emissive_texture, v_texcoord_1).rgb;
        emissive.a = 1.0;
    }
    f_color.rgb += emissive.rgb * emissive.a;


    /*f_color.rgb = mix(f_color.rgb, fresnel_schlick_2, u_ScaleFGDSpec.x);
    f_color.rgb = mix(f_color.rgb, vec3(geometric_occlusion_smith_ggx), u_ScaleFGDSpec.y);
    f_color.rgb = mix(f_color.rgb, vec3(ggx), u_ScaleFGDSpec.z);
    f_color.rgb = mix(f_color.rgb, specContrib, u_ScaleFGDSpec.w);

    f_color.rgb = mix(f_color.rgb, diffuseContrib, u_ScaleDiffBaseMR.x);
    f_color.rgb = mix(f_color.rgb, baseColor.rgb, u_ScaleDiffBaseMR.y);
    f_color.rgb = mix(f_color.rgb, vec3(metallic), u_ScaleDiffBaseMR.z);
    f_color.rgb = mix(f_color.rgb, vec3(perceptualRoughness), u_ScaleDiffBaseMR.w);*/
}
"]
        struct Dummy;
    }

    let meshes = {
        let mut meshes = Vec::new();
        for (mesh_id, mesh) in gltf.as_json().meshes.iter().enumerate() {
            let mut mesh_prim_out = Vec::with_capacity(mesh.primitives.len());
            for (primitive_id, primitive) in mesh.primitives.iter().enumerate() {
                let runtime_def = RuntimeVertexDef::from_primitive(gltf.as_json(), mesh_id, primitive_id);
                let vertex_buffer_ids = runtime_def.vertex_buffer_ids().to_owned();

                let index_buffer = if let Some(indices) = primitive.indices.as_ref().map(|i| i.value()) {
                    let accessor = &gltf.as_json().accessors[indices];
                    let view = &gltf.as_json().buffer_views[accessor.buffer_view.value()];
                    let total_offset = accessor.byte_offset as usize + view.byte_offset as usize;
                    let num_indices = accessor.count;
                    let buf = view.buffer.value();
                    Some((buf, total_offset, num_indices))
                } else {
                    None
                };

                let primitive_topology = match primitive.mode.clone().unwrap() {
                    gltf::json::mesh::Mode::Points => PrimitiveTopology::PointList,
                    gltf::json::mesh::Mode::Lines => PrimitiveTopology::LineList,
                    gltf::json::mesh::Mode::LineLoop => panic!("LineLoop not supported"),
                    gltf::json::mesh::Mode::LineStrip => PrimitiveTopology::LineStrip,
                    gltf::json::mesh::Mode::Triangles => PrimitiveTopology::TriangleList,
                    gltf::json::mesh::Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
                    gltf::json::mesh::Mode::TriangleFan => PrimitiveTopology::TriangleFan,
                };

                let material_id = primitive.material.as_ref().expect("Default material not supported").value();

                // TODO: adjust some pipeline params based on material

                let pipeline = {
                    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
                    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");
                    Arc::new(GraphicsPipeline::start()
                        .vertex_input(runtime_def)
                        .vertex_shader(vs.main_entry_point(), ())
                        .primitive_topology(primitive_topology)
                        .viewports_dynamic_scissors_irrelevant(1)
                        .fragment_shader(fs.main_entry_point(), ())
                        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                        .build(device.clone())
                        .unwrap())
                };

                mesh_prim_out.push((pipeline, vertex_buffer_ids, index_buffer, gltf_materials[material_id].clone()));
            }
            meshes.push(mesh_prim_out);
        }
        meshes
    };


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

        for mesh in meshes.iter() {
            for &(ref pipeline, ref vertex_buffer_ids, ref index_buffer, ref material_ds) in mesh.iter() {
                let vertex_buffers = vertex_buffer_ids.iter().map(|&(vb_id, offset)| {
                    let buf = gltf_buffers[vb_id].clone();
                    let buf_len = buf.len();
                    Arc::new(buf.into_buffer_slice().slice(offset..buf_len).unwrap()) as Arc<_>
                }).collect();

                let dynamic_state = DynamicState {
                    viewports: Some(vec![Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                        depth_range: 0.0 .. 1.0,
                    }]),
                    .. DynamicState::none()
                };

                if let &Some((buf_id, total_offset, num_indices)) = index_buffer {
                    let ib = gltf_buffers[buf_id].clone();
                    let ib_len = ib.len();
                    let indices = ib.into_buffer_slice().slice(total_offset..ib_len).unwrap();
                    let indices: BufferSlice<[u16], Arc<ImmutableBuffer<[u8]>>> = unsafe { ::std::mem::transmute(indices) };     // TODO: add a function in vulkano that does that
                    let indices = indices.clone().slice(0..num_indices as usize).unwrap();
                    builder = builder.draw_indexed(pipeline.clone(),
                        dynamic_state,
                        vertex_buffers, indices, material_ds.clone(), ())
                        .unwrap();
                } else {
                    builder = builder.draw(pipeline.clone(),
                        dynamic_state,
                        vertex_buffers, material_ds.clone(), ())
                        .unwrap();
                }
            }
        }

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

pub struct RuntimeVertexDef {
    buffers: Vec<(u32, usize, InputRate)>,
    vertex_buffer_ids: Vec<(usize, usize)>,
    attributes: Vec<(String, u32, AttributeInfo)>,
    num_vertices: u32,
}

impl RuntimeVertexDef {
    pub fn from_primitive(gltf: &gltf::json::root::Root, mesh_id: usize, primitive_id: usize)
                          -> RuntimeVertexDef
    {
        let mut buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();
        let mut attributes = Vec::new();

        let mut num_vertices = u32::max_value();

        let primitive = &gltf.meshes[mesh_id].primitives[primitive_id];
        for (attribute_id, (attribute, accessor_id)) in primitive.attributes.iter().enumerate() {
            let accessor = &gltf.accessors[accessor_id.value()];

            if accessor.count < num_vertices {
                num_vertices = accessor.count;
            }

            let name = match attribute.clone().unwrap() {
                gltf::json::mesh::Semantic::Positions => "i_position".to_owned(),
                gltf::json::mesh::Semantic::Normals => "i_normal".to_owned(),
                gltf::json::mesh::Semantic::Tangents => "i_tangent".to_owned(),
                gltf::json::mesh::Semantic::Colors(0) => "i_color_0".to_owned(),
                gltf::json::mesh::Semantic::TexCoords(0) => "i_texcoord_0".to_owned(),
                gltf::json::mesh::Semantic::TexCoords(1) => "i_texcoord_1".to_owned(),
                gltf::json::mesh::Semantic::Joints(0) => "i_joints_0".to_owned(),
                gltf::json::mesh::Semantic::Weights(0) => "i_weights_0".to_owned(),
                _ => unimplemented!()
            };

            let infos = AttributeInfo {
                offset: 0,
                format: match (accessor.component_type.unwrap().0, accessor.type_.unwrap()) {
                    (gltf::json::accessor::ComponentType::I8,
                     gltf::json::accessor::Type::Scalar) => Format::R8Snorm,
                    (gltf::json::accessor::ComponentType::U8,
                     gltf::json::accessor::Type::Scalar) => Format::R8Unorm,
                    (gltf::json::accessor::ComponentType::F32,
                     gltf::json::accessor::Type::Vec2) => Format::R32G32Sfloat,
                    (gltf::json::accessor::ComponentType::F32,
                     gltf::json::accessor::Type::Vec3) => Format::R32G32B32Sfloat,
                    (gltf::json::accessor::ComponentType::F32,
                     gltf::json::accessor::Type::Vec4) => Format::R32G32B32A32Sfloat,
                    v => unimplemented!("{:?}", v)
                },
            };

            let view = &gltf.buffer_views[accessor.buffer_view.value()];

            buffers.push((attribute_id as u32, view.byte_stride.unwrap().0 as usize, InputRate::Vertex));
            attributes.push((name, attribute_id as u32, infos));
            vertex_buffer_ids.push((view.buffer.value(), view.byte_offset as usize + accessor.byte_offset as usize));
        }

        RuntimeVertexDef {
            buffers: buffers,
            vertex_buffer_ids: vertex_buffer_ids,
            num_vertices: num_vertices,
            attributes: attributes,
        }
    }

    /// Returns the indices of the buffers to bind as vertex buffers and the byte offset, when
    /// drawing the primitive.
    pub fn vertex_buffer_ids(&self) -> &[(usize, usize)] {
        &self.vertex_buffer_ids
    }
}

unsafe impl<I> VertexDefinition<I> for RuntimeVertexDef
    where I: ShaderInterfaceDef
{
    type BuffersIter = VecIntoIter<(u32, usize, InputRate)>;
    type AttribsIter = VecIntoIter<(u32, u32, AttributeInfo)>;
    
    fn definition(&self, interface: &I)
            -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError>
    {
        let buffers_iter = self.buffers.clone().into_iter();

        let mut attribs_iter = self.attributes.iter().map(|&(ref name, buffer_id, ref infos)| {
            let attrib_loc = interface
                .elements()
                .find(|e| e.name.as_ref().map(|n| &n[..]) == Some(&name[..]))
                .unwrap()
                .location.start;
            (attrib_loc as u32, buffer_id, AttributeInfo { offset: infos.offset, format: infos.format })
        }).collect::<Vec<_>>();

        // Add dummy attributes.
        for binding in interface.elements() {
            if attribs_iter.iter().any(|a| a.0 == binding.location.start) {
                continue;
            }

            attribs_iter.push((binding.location.start, 0,
                               AttributeInfo { offset: 0, format: binding.format }));
        }

        Ok((buffers_iter, attribs_iter.into_iter()))
    }
}

unsafe impl VertexSource<Vec<Arc<BufferAccess + Send + Sync>>> for RuntimeVertexDef {
    fn decode(&self, bufs: Vec<Arc<BufferAccess + Send + Sync>>)
        -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize)
    {
        (bufs.into_iter().map(|b| Box::new(b) as Box<_>).collect(), self.num_vertices as usize, 1)
    }
}

#[derive(Debug, Copy, Clone)]
struct MyPipelineLayout;

unsafe impl PipelineLayoutDesc for MyPipelineLayout {
    fn num_sets(&self) -> usize { 1 }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set { 0 => Some(6), _ => None, }
    }
    fn descriptor(&self, set: usize, binding: usize)
        -> Option<DescriptorDesc> {
        match (set, binding) {
            (0, 0) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::Buffer(DescriptorBufferDesc{dynamic:
                                                                                        Some(false),
                                                                                    storage:
                                                                                        false,
                                                                                    content:
                                                                                        DescriptorBufferContentDesc::F32,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            (0, 1) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                    true,
                                                                                                dimensions:
                                                                                                    DescriptorImageDescDimensions::TwoDimensional,
                                                                                                format:
                                                                                                    None,
                                                                                                multisampled:
                                                                                                    false,
                                                                                                array_layers:
                                                                                                    DescriptorImageDescArray::NonArrayed,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            (0, 2) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                    true,
                                                                                                dimensions:
                                                                                                    DescriptorImageDescDimensions::TwoDimensional,
                                                                                                format:
                                                                                                    None,
                                                                                                multisampled:
                                                                                                    false,
                                                                                                array_layers:
                                                                                                    DescriptorImageDescArray::NonArrayed,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            (0, 3) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                    true,
                                                                                                dimensions:
                                                                                                    DescriptorImageDescDimensions::TwoDimensional,
                                                                                                format:
                                                                                                    None,
                                                                                                multisampled:
                                                                                                    false,
                                                                                                array_layers:
                                                                                                    DescriptorImageDescArray::NonArrayed,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            (0, 4) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                    true,
                                                                                                dimensions:
                                                                                                    DescriptorImageDescDimensions::TwoDimensional,
                                                                                                format:
                                                                                                    None,
                                                                                                multisampled:
                                                                                                    false,
                                                                                                array_layers:
                                                                                                    DescriptorImageDescArray::NonArrayed,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            (0, 5) =>
            Some(DescriptorDesc{ty:
                                    DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc{sampled:
                                                                                                    true,
                                                                                                dimensions:
                                                                                                    DescriptorImageDescDimensions::TwoDimensional,
                                                                                                format:
                                                                                                    None,
                                                                                                multisampled:
                                                                                                    false,
                                                                                                array_layers:
                                                                                                    DescriptorImageDescArray::NonArrayed,}),
                                array_count: 1,
                                stages: ShaderStages { fragment: true, .. ShaderStages::none() },
                                readonly: true,}),
            _ => None,
        }
    }
    fn num_push_constants_ranges(&self) -> usize { 0 }
    fn push_constants_range(&self, num: usize)
        -> Option<PipelineLayoutDescPcRange> {
        if num != 0 || 0 == 0 { return None; }
        Some(PipelineLayoutDescPcRange{offset: 0,
                                        size: 0,
                                        stages: ShaderStages::all(),})
    }
}

unsafe impl PipelineLayoutDescNames for MyPipelineLayout {
    fn descriptor_by_name(&self, name: &str)
        -> Option<(usize, usize)> {
        match name {
            "u_material_params" => Some((0, 0)),
            "u_base_color" => Some((0, 1)),
            "u_metallic_roughness" => Some((0, 2)),
            "u_normal_texture" => Some((0, 3)),
            "u_occlusion_texture" => Some((0, 4)),
            "u_emissive_texture" => Some((0, 5)),
            _ => None,
        }
    }
}
