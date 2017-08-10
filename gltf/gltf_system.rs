// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


// Welcome to the glTF example!

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
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescNames;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::Dimensions;
use vulkano::image::immutable::ImmutableImage;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::shader::ShaderInterfaceDef;
use vulkano::pipeline::vertex::AttributeInfo;
use vulkano::pipeline::vertex::IncompatibleVertexDefinitionError;
use vulkano::pipeline::vertex::InputRate;
use vulkano::pipeline::vertex::VertexDefinition;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::Sampler;

use cgmath::Matrix4;
use cgmath::SquareMatrix;

use std::sync::Arc;
use std::vec::IntoIter as VecIntoIter;

use gltf;

/// Represents a fully-loaded glTF model, ready to be drawn.
pub struct GltfModel {
    // The main glTF document.
    gltf: gltf::gltf::Gltf,
    gltf_buffers: Vec<Arc<ImmutableBuffer<[u8]>>>,
    // Each mesh of the glTF scene is made of one or more primitives.
    gltf_meshes: Vec<Vec<PrimitiveInfo>>,
    // Buffer used to upload `InstanceParams` when drawing.
    instance_params_upload: CpuBufferPool<vs::ty::InstanceParams>,
    // Pipeline layout common to all the graphics pipeline of all the primitives.
    pipeline_layout: Arc<PipelineLayoutAbstract + Send + Sync>,
}

// Information about a primitive.
struct PrimitiveInfo {
    // The graphics pipeline used to draw the primitive.
    pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffers: Vec<(usize, usize)>,
    index_buffer: Option<(usize, usize, u32)>,
    // Descriptor set to bind to slot #0 when drawing.
    material: Arc<DescriptorSet + Send + Sync>,
}

impl GltfModel {
    /// Loads all the resources necessary to draw `gltf`.
    ///
    /// The `queue` parameter is the queue that will be used to submit data transfer commands as
    /// part of the loading.
    ///
    /// The `subpass` parameter is the render pass subpass that we will need to be in when drawing.
    pub fn new<R>(gltf: gltf::gltf::Gltf, queue: Arc<Queue>, subpass: Subpass<R>) -> GltfModel
        where R: RenderPassAbstract + Clone + Send + Sync + 'static
    {
        // The first step is to go through all the glTF buffer definitions and load them as
        // `ImmutableBuffer`.
        let gltf_buffers: Vec<Arc<ImmutableBuffer<[u8]>>> = {
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

        // Then we go through each glTF texture and load them.
        let gltf_textures = {
            // TODO: use the sampler defined by the JSON struct
            let sampler = Sampler::simple_repeat_linear(queue.device().clone());

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
        
        let pipeline_layout = Arc::new(MyPipelineLayout.build(queue.device().clone()).unwrap());

        let gltf_materials: Vec<Arc<DescriptorSet + Send + Sync>> = {
            // TODO: meh, we want some device-local thing here
            let params_buffer = CpuBufferPool::new(queue.device().clone(), BufferUsage::uniform_buffer(),
                                                    Some(queue.family()));

            let dummy_sampler = Sampler::simple_repeat_linear(queue.device().clone());
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
                    Arc::new(PersistentDescriptorSet::start(pipeline_layout.clone(), 1)
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

                materials.push(descriptor_set as Arc<_>);
            }
            materials
        };

        let gltf_meshes = {
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
                        let vs = vs::Shader::load(queue.device().clone()).expect("failed to create shader module");
                        let fs = fs::Shader::load(queue.device().clone()).expect("failed to create shader module");
                        Arc::new(GraphicsPipeline::start()
                            .vertex_input(runtime_def)
                            .vertex_shader(vs.main_entry_point(), ())
                            .primitive_topology(primitive_topology)
                            .viewports_dynamic_scissors_irrelevant(1)
                            .fragment_shader(fs.main_entry_point(), ())
                            .render_pass(subpass.clone())
                            .build(queue.device().clone())
                            .unwrap())
                    };

                    mesh_prim_out.push(PrimitiveInfo {
                        pipeline: pipeline as Arc<_>,
                        vertex_buffers: vertex_buffer_ids,
                        index_buffer: index_buffer,
                        material: gltf_materials[material_id].clone(),
                    });
                }
                meshes.push(mesh_prim_out);
            }
            meshes
        };

        GltfModel {
            gltf: gltf,
            gltf_buffers: gltf_buffers,
            gltf_meshes: gltf_meshes,
            instance_params_upload: CpuBufferPool::new(queue.device().clone(),
                                                       BufferUsage::uniform_buffer(),
                                                       Some(queue.family())),
            pipeline_layout: pipeline_layout,
        }
    }

    /// Draws the glTF scene by adding commands to `builder`.
    ///
    /// `viewport_dimensions` should be the dimensions of the framebuffer we're drawing to.
    ///
    /// The `builder` must be inside a subpass compatible with the one that was passed in `new`.
    pub fn draw_default_scene(&self, viewport_dimensions: [u32; 2],
                              builder: AutoCommandBufferBuilder) -> AutoCommandBufferBuilder
    {
        if let Some(ref scene) = self.gltf.as_json().scene {
            self.draw_scene(scene.value(), viewport_dimensions, builder)
        } else {
            builder
        }
    }

    /// Draws a single scene.
    ///
    /// # Panic
    ///
    /// - Panics if the scene is out of range.
    ///
    pub fn draw_scene(&self, scene_id: usize, viewport_dimensions: [u32; 2],
                      mut builder: AutoCommandBufferBuilder) -> AutoCommandBufferBuilder
    {
        for node in self.gltf.as_json().scenes[scene_id].nodes.iter() {
            builder = self.draw_node(node.value(), Matrix4::identity(), viewport_dimensions,
                                     builder);
        }

        builder
    }

    // Draws a single node.
    //
    // # Panic
    //
    // - Panics if the node is out of range.
    //
    fn draw_node(&self, node_id: usize, world_to_framebuffer: Matrix4<f32>,
                 viewport_dimensions: [u32; 2], mut builder: AutoCommandBufferBuilder)
                 -> AutoCommandBufferBuilder
    {
        let node = &self.gltf.as_json().nodes[node_id];

        let local_matrix = world_to_framebuffer * {
            let m = node.matrix;
            Matrix4::new(m[ 0], m[ 1], m[ 2], m[ 3],
                         m[ 4], m[ 5], m[ 6], m[ 7],
                         m[ 8], m[ 9], m[10], m[11],
                         m[12], m[13], m[14], m[15])
        };

        // TODO: handle TSR correctly

        if let Some(ref mesh) = node.mesh {
            builder = self.draw_mesh(mesh.value(), local_matrix, viewport_dimensions, builder);
        }

        if let Some(ref children) = node.children {
            for child in children {
                builder = self.draw_node(child.value(), local_matrix, viewport_dimensions, builder);
            }
        }

        builder
    }

    /// Draws a single mesh of the glTF document.
    ///
    /// # Panic
    ///
    /// - Panics if the mesh is out of range.
    ///
    pub fn draw_mesh(&self, mesh_id: usize, world_to_framebuffer: Matrix4<f32>,
                     viewport_dimensions: [u32; 2], mut builder: AutoCommandBufferBuilder)
                     -> AutoCommandBufferBuilder
    {
        let instance_params = {
            let buf = self.instance_params_upload.next(vs::ty::InstanceParams {
                world_to_framebuffer: world_to_framebuffer.into(),
            });
            
            Arc::new(PersistentDescriptorSet::start(self.pipeline_layout.clone(), 0)
                .add_buffer(buf)
                .unwrap()
                .build()
                .unwrap())
        };

        for primitive in self.gltf_meshes[mesh_id].iter() {
            let vertex_buffers = primitive.vertex_buffers.iter().map(|&(vb_id, offset)| {
                let buf = self.gltf_buffers[vb_id].clone();
                let buf_len = buf.len();
                Arc::new(buf.into_buffer_slice().slice(offset..buf_len).unwrap()) as Arc<_>
            }).collect();

            let dynamic_state = DynamicState {
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0 .. 1.0,
                }]),
                .. DynamicState::none()
            };

            if let Some((buf_id, total_offset, num_indices)) = primitive.index_buffer {
                let ib = self.gltf_buffers[buf_id].clone();
                let ib_len = ib.len();
                let indices = ib.into_buffer_slice().slice(total_offset..ib_len).unwrap();
                let indices: BufferSlice<[u16], Arc<ImmutableBuffer<[u8]>>> = unsafe { ::std::mem::transmute(indices) };     // TODO: add a function in vulkano that does that
                let indices = indices.clone().slice(0..num_indices as usize).unwrap();
                builder = builder.draw_indexed(primitive.pipeline.clone(),
                    dynamic_state,
                    vertex_buffers, indices, (instance_params.clone(), primitive.material.clone()), ())
                    .unwrap();
            } else {
                builder = builder.draw(primitive.pipeline.clone(),
                    dynamic_state,
                    vertex_buffers, (instance_params.clone(), primitive.material.clone()), ())
                    .unwrap();
            }
        }

        builder
    }
}

mod vs {
    #[derive(VulkanoShader)]
    #[allow(dead_code)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(set = 0, binding = 0) uniform InstanceParams {
    mat4 world_to_framebuffer;
} u_instance_params;

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

    gl_Position = u_instance_params.world_to_framebuffer * vec4(i_position, 1.0);
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

layout(set = 1, binding = 0) uniform MaterialParams {
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

layout(set = 1, binding = 1) uniform sampler2D u_base_color;
layout(set = 1, binding = 2) uniform sampler2D u_metallic_roughness;
layout(set = 1, binding = 3) uniform sampler2D u_normal_texture;
layout(set = 1, binding = 4) uniform sampler2D u_occlusion_texture;
layout(set = 1, binding = 5) uniform sampler2D u_emissive_texture;

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


    // Add ambient occlusion.
    {
        float ao = 1.0;
        if (u_material_params.occlusion_texture_tex_coord == 0) {
            ao = texture(u_occlusion_texture, v_texcoord_0).x;
        } else if (u_material_params.occlusion_texture_tex_coord == 1) {
            ao = texture(u_occlusion_texture, v_texcoord_1).x;
        }
        f_color.rgb = mix(f_color.rgb, f_color.rgb * ao,
                        u_material_params.occlusion_texture_strength);
    }

    // Add the emissive color.
    {
        vec4 emissive = vec4(0.0);
        if (u_material_params.emissive_texture_tex_coord == 0) {
            emissive.rgb = texture(u_emissive_texture, v_texcoord_0).rgb;
            emissive.a = 1.0;
        } else if (u_material_params.emissive_texture_tex_coord == 1) {
            emissive.rgb = texture(u_emissive_texture, v_texcoord_1).rgb;
            emissive.a = 1.0;
        }
        f_color.rgb += emissive.rgb * emissive.a;
    }


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
    fn num_sets(&self) -> usize { 2 }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set { 0 => Some(1), 1 => Some(6), _ => None, }
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
                                stages: ShaderStages { vertex: true, .. ShaderStages::none() },
                                readonly: true,}),
            (1, 0) =>
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
            (1, 1) =>
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
            (1, 2) =>
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
            (1, 3) =>
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
            (1, 4) =>
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
            (1, 5) =>
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
