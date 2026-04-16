#version 330

uniform mat4 u_mvp;

in vec3 i_position;
in vecN i_attr;

out vecN v_attr;

void main() {
    gl_Position = u_mvp * vec4(i_position, 1.0);
    v_attr = i_attr;
}
