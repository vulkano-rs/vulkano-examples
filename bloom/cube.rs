#[derive(Debug, Clone)]
pub struct Vertex { a_position: [f32; 3] }
impl_vertex!(Vertex, a_position);

#[derive(Debug, Clone)]
pub struct Uv { a_uv: [f32; 2] }
impl_vertex!(Uv, a_uv);

pub const VERTICES: [Vertex; 24] = [
    // back
    Vertex { a_position: [-1.0, -1.0,  1.0] },
    Vertex { a_position: [ 1.0, -1.0,  1.0] },
    Vertex { a_position: [ 1.0,  1.0,  1.0] },
    Vertex { a_position: [-1.0,  1.0,  1.0] },

    // front
    Vertex { a_position: [-1.0, -1.0, -1.0] },
    Vertex { a_position: [-1.0,  1.0, -1.0] },
    Vertex { a_position: [ 1.0,  1.0, -1.0] },
    Vertex { a_position: [ 1.0, -1.0, -1.0] },

    // top
    Vertex { a_position: [-1.0,  1.0, -1.0] },
    Vertex { a_position: [-1.0,  1.0,  1.0] },
    Vertex { a_position: [ 1.0,  1.0,  1.0] },
    Vertex { a_position: [ 1.0,  1.0, -1.0] },

    // bottom
    Vertex { a_position: [-1.0, -1.0, -1.0] },
    Vertex { a_position: [ 1.0, -1.0, -1.0] },
    Vertex { a_position: [ 1.0, -1.0,  1.0] },
    Vertex { a_position: [-1.0, -1.0,  1.0] },

    // left
    Vertex { a_position: [ 1.0, -1.0, -1.0] },
    Vertex { a_position: [ 1.0,  1.0, -1.0] },
    Vertex { a_position: [ 1.0,  1.0,  1.0] },
    Vertex { a_position: [ 1.0, -1.0,  1.0] },

    // right
    Vertex { a_position: [-1.0, -1.0, -1.0] },
    Vertex { a_position: [-1.0, -1.0,  1.0] },
    Vertex { a_position: [-1.0,  1.0,  1.0] },
    Vertex { a_position: [-1.0,  1.0, -1.0] },
];

pub const UVS: [Uv; 24] = [
    // back
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },

    // front
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },

    // top
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },

    // bottom
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },

    // left
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },

    // right
    Uv { a_uv: [0.0, 0.0] },
    Uv { a_uv: [1.0, 0.0] },
    Uv { a_uv: [1.0, 1.0] },
    Uv { a_uv: [0.0, 1.0] },
];

pub const INDICES: [u16; 36] = [
    // back
    0, 1, 2,
    2, 3, 0,
    // front
    4, 5, 6,
    6, 7, 4,
    // top
    8, 9, 10,
    10, 11, 8,
    // bottom
    12, 13, 14,
    14, 15, 12,
    // left
    16, 17, 18,
    18, 19, 16,
    // right
    20, 21, 22,
    22, 23, 20,
];
