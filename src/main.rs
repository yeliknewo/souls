#[macro_use]
extern crate gfx;
extern crate piston;
extern crate sdl2_window;
extern crate gfx_voxel;
extern crate gfx_device_gl;
extern crate image;
extern crate vecmath;
extern crate camera_controllers;
extern crate fps_counter;
extern crate shader_version;

use std::f32::consts::{PI};
use std::path::{Path, PathBuf};
use sdl2_window::{Sdl2Window};
use piston::event_loop::{Events, EventLoop};
use piston::window::{Size, Window, AdvancedWindow, OpenGLWindow, WindowSettings};
use vecmath::{vec3_add, vec3_scale, vec3_normalized, Matrix4};
use gfx::traits::{FactoryExt, Device};
use gfx_voxel::texture::{AtlasBuilder, ImageSize, Texture};
use shader_version::{OpenGL};

fn main() {
    let mut window: Sdl2Window = WindowSettings::new(
        "Loading".to_string(),
        Size {
            width: 600,
            height: 480
        }
    )
    .build()
    .unwrap();

    let (mut device, mut factory) = gfx_device_gl::create(|s|
        window.get_proc_address(s) as *const _
    );

    let Size {width: w, height:h} = window.size();

    let (target_view, depth_view) = gfx_device_gl::create_main_targets(
        (w as u16, h as u16, 1, (0 as gfx::tex::NumSamples).into())
    );

    let assets = Path::new("./assets");

    let mut atlas = AtlasBuilder::new(assets, 16, 16);

    let texture = atlas.complete(&mut factory);

    let mut renderer = Renderer::new(factory, target_view, depth_view, texture.surface.clone());

    let player_pos: [f32; 3] = [0.0; 3];
    let player_yaw = 0.0;
    let player_pitch = 0.0;

    let projection_mat = camera_controllers::CameraPerspective {
        fov: 70.0,
        near_clip: 0.1,
        far_clip: 1000.0,
        aspect_ratio: {
            let Size {width: w, height: h} = window.size();
            (w as f32) / (h as f32)
        },
    }.projection();
    renderer.set_projection(projection_mat);

    let mut first_person_settings = camera_controllers::FirstPersonSettings::keyboard_wasd();
    first_person_settings.speed_horizontal = 8.0;
    first_person_settings.speed_vertical = 4.0;
    let mut first_person = camera_controllers::FirstPerson::new(
        player_pos,
        first_person_settings
    );
    first_person.yaw = PI - player_yaw / 180.0 * PI;
    first_person.pitch = player_pitch / 180.0 * PI;

    let mut fps_counter = fps_counter::FPSCounter::new();

    let mut events = window.events().ups(120).max_fps(10_000);

    while let Some(e) = events.next(&mut window) {
        use piston::input::Button::Keyboard;
        use piston::input::Input::{Move, Press};
        use piston::input::keyboard::{Key};
        use piston::input::Motion::{MouseRelative};
        use piston::input::{Event};

        match e {
            Event::Render(_) => {
                let mut camera = first_person.camera(0.0);
                camera.position[1] += 1.62;
                let mut xz_forward = camera.forward;
                xz_forward[1] = 0.0;
                xz_forward = vec3_normalized(xz_forward);
                camera.position = vec3_add(
                    camera.position,
                    vec3_scale(xz_forward, 0.1)
                );

                let view_mat = camera.orthogonal();
                renderer.set_view(view_mat);
                renderer.clear();
                let fps = fps_counter.tick();
                let title = format!("FPS={}", fps);
                window.set_title(title);
            },
            Event::AfterRender(_) => {
                device.cleanup();
            },
            Event::Update(_) => {

            },
            _ => {},
        }

        first_person.event(&e);
    }
}

static VERTEX: &'static [u8] = b"
    #version 150 core
    uniform mat4 u_projection, u_view;
    in vec2 at_tex_coord;
    in vec3 at_color, at_position;
    out vec2 v_tex_coord;
    out vec3 v_color;
    void main() {
        v_tex_coord = at_tex_coord;
        v_color = at_color;
        gl_Position = u_projection * u_view * vec4(at_position, 1.0);
    }
";

static FRAGMENT: &'static [u8] = b"
    #version 150 core
    out vec4 out_color;
    uniform sampler2D s_texture;
    in vec2 v_tex_coord;
    in vec3 v_color;
    void main() {
        vec4 tex_color = texture(s_texture, v_tex_coord);
        if(tex_color.a == 0.0) // Discard transparent pixels.
            discard;
        out_color = tex_color * vec4(v_color, 1.0);
    }
";

gfx_pipeline!( pipe {
    vbuf: gfx::VertexBuffer<Vertex> = (),
    transform: gfx::Global<[[f32; 4]; 4]> = "u_projection",
    view: gfx::Global<[[f32; 4]; 4]> = "u_view",
    color: gfx::TextureSampler<[f32; 4]> = "s_texture",
    out_color: gfx::RenderTarget<gfx::format::Rgba8> = "out_color",
    out_depth: gfx::DepthTarget<gfx::format::DepthStencil> =
        gfx::preset::depth::LESS_EQUAL_WRITE,
});

gfx_vertex_struct!( Vertex {
    xyz: [f32; 3] = "at_position",
    uv: [f32; 2] = "at_tex_coord",
    rgb: [f32; 3] = "at_color",
});

struct Renderer<R: gfx::Resources, F: gfx::Factory<R>> {
    factory: F,
    pub pipe: gfx::PipelineState<R, pipe::Meta>,
    data: pipe::Data<R>,
    encoder: gfx::Encoder<R, F::CommandBuffer>,
    clear_color: [f32; 4],
    clear_depth: f32,
    clear_stencil: u8,
    slice: gfx::Slice<R>,
}

impl<R: gfx::Resources, F: gfx::Factory<R>> Renderer<R, F> {

    pub fn new(mut factory: F, target: gfx::handle::RenderTargetView<R, gfx::format::Rgba8>,
        depth: gfx::handle::DepthStencilView<R, (gfx::format::D24_S8, gfx::format::Unorm)>,
        tex: gfx::handle::Texture<R, gfx::format::R8_G8_B8_A8>) -> Renderer<R, F> {

        let sampler = factory.create_sampler(
                gfx::tex::SamplerInfo::new(
                    gfx::tex::FilterMethod::Scale,
                    gfx::tex::WrapMode::Tile
                )
            );

        let texture_view = factory.view_texture_as_shader_resource::<gfx::format::Rgba8>(
            &tex, (0, 0), gfx::format::Swizzle::new()).unwrap();

        let prog = factory.link_program(VERTEX, FRAGMENT).unwrap();

        let mut rasterizer = gfx::state::Rasterizer::new_fill(gfx::state::CullFace::Back);
        rasterizer.front_face = gfx::state::FrontFace::Clockwise;
        let pipe = factory.create_pipeline_from_program(&prog, gfx::Primitive::TriangleList,
            rasterizer, pipe::new()).unwrap();

        let (vbuf, slice) = factory.create_vertex_buffer(&[]);

        let data = pipe::Data {
            vbuf: vbuf,
            transform: vecmath::mat4_id(),
            view: vecmath::mat4_id(),
            color: (texture_view, sampler),
            out_color: target,
            out_depth: depth,
        };

        let encoder = factory.create_encoder();

        Renderer {
            factory: factory,
            pipe: pipe,
            data: data,
            encoder: encoder,
            clear_color: [0.81, 0.8, 1.0, 1.0],
            clear_depth: 1.0,
            clear_stencil: 0,
            slice: slice,
        }
    }

    pub fn set_projection(&mut self, proj_mat: Matrix4<f32>) {
        self.data.transform = proj_mat;
    }

    pub fn set_view(&mut self, view_mat: Matrix4<f32>) {
        self.data.view = view_mat;
    }

    pub fn clear(&mut self) {
        self.encoder.clear(&self.data.out_color, self.clear_color);
        self.encoder.clear_depth(&self.data.out_depth, self.clear_depth);
        self.encoder.clear_stencil(&self.data.out_depth, self.clear_stencil);
    }

    pub fn flush<D: gfx::Device<Resources=R, CommandBuffer=F::CommandBuffer> + Sized>(&mut self, device: &mut D) {
        device.submit(self.encoder.as_buffer());
        self.encoder.reset();
    }

    pub fn create_buffer(&mut self, data: &[Vertex]) -> gfx::handle::Buffer<R, Vertex> {
        let (vbuf, slice) = self.factory.create_vertex_buffer(data);
        self.slice = slice;

        vbuf
    }

    pub fn render(&mut self, buffer: &mut gfx::handle::Buffer<R, Vertex>) {
        self.data.vbuf = buffer.clone();
        self.slice.end = buffer.len() as u32;
        self.encoder.draw(&self.slice, &self.pipe, &self.data);
    }
}
