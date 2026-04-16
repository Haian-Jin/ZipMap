 #version 330 core

in vec2 in_vert;
out vec2 scr_coord;

void main() {
    scr_coord = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0., 1.);
}