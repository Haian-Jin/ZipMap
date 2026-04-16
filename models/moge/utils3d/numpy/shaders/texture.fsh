#version 330

uniform sampler2D tex;
uniform sampler2D uv;

in vec2 scr_coord;
out vecN tex_color;

void main() {
    tex_color = vecN(texture(tex, texture(uv, scr_coord).xy));
}