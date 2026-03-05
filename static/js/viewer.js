// === GLOBALS ===
let canvas;
let gl;

// === SHADERS & PROGRAMS ===

const vertexShaders = {
xyz_rgba: `#version 300 es
  precision highp float;
  in vec3 xyz;
  in vec4 rgba;
  uniform mat4 camera;
  uniform float point_size;
  uniform float minWidth;
  out vec4 color;
  in float index;

  void main(void) {
    gl_Position = camera * vec4(xyz, 1.0);

    float size = point_size / gl_Position.w;
    color = rgba;
    if (size < minWidth) {
      color.a *= size / minWidth;
      size = minWidth;
    }
    gl_PointSize = size;
  }`,

xy: `#version 300 es
  precision highp float;
  in vec2 xy;
  uniform mat4 camera;
  uniform mat4 pose;
  uniform float frustum_size;
  out vec2 v_uv;

  void main(void) {
    gl_Position = camera * pose * (vec4(frustum_size, frustum_size, frustum_size, 1.0) * vec4(xy, 1.0, 1.0));
    v_uv = xy;
  }`,

depth: `#version 300 es
  precision highp float;
  uniform mat4 camera;
  uniform mat4 pose;
  uniform sampler2D depth;
  uniform float depthscale;
  uniform float point_size;
  uniform int width;
  uniform int height;
  uniform int stride;
  uniform float max_grad;
  out vec2 v_uv;

  float d(vec2 p) {
    vec4 rgba = texture(depth, p);
    return depthscale * (rgba.r + rgba.g/256.0);
  }

  void main(void) {
    int x = gl_VertexID % (width / stride) * stride;
    int y = gl_VertexID / (width / stride) * stride;
    vec2 uv;
    uv.x = (float(x) + 0.5) / float(width);
    uv.y = (float(y) + 0.5) / float(height);
    highp float z = d(uv);
    // Cull points with high gradient.
    vec2 dx = vec2(1.0 / float(width), 0.0);
    vec2 dy = vec2(0.0, 1.0 / float(height));
    highp float gx = abs(d(uv + dx) - d(uv - dx));
    highp float gy = abs(d(uv + dy) - d(uv - dy));
    if (gx > max_grad * z || gy > max_grad * z) {
      z = 0.0;
    }
    gl_Position = camera * pose * vec4(uv.x * z, uv.y * z, z, 1.0);
    v_uv = uv;
    gl_PointSize = point_size * z / gl_Position[3];
  }`,

linesegment: `#version 300 es
  precision highp float;
  uniform mat4 camera;
  uniform float width;
  uniform float height;
  uniform float lineWidth;
  uniform float minWidth;
  in vec3 xyz0;
  in vec3 xyz1;
  in vec4 rgba;
  out vec4 color;

  void main(void) {
    vec4 zero4 = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 p0 = camera * vec4(xyz0, 1.0);
    vec4 p1 = camera * vec4(xyz1, 1.0);
    color = rgba;
    if (p0.w < 0.0 || p1.w < 0.0) {
      gl_Position = zero4;
      color = zero4;
      return;
    }

    // Project into 2d to do line width calculations.
    float p0w = p0.w;
    float p1w = p1.w;
    p0 /= p0w;
    p1 /= p1w;

    // Radii at each end
    float r0 = lineWidth / p0w;
    float r1 = lineWidth / p1w;

    // Alpha modifier at each end.
    float r0a = 1.0;
    float r1a = 1.0;
    if (r0 < minWidth) {
      r0a = r0 / minWidth;
      r0 = minWidth;
    }
    if (r1 < minWidth) {
      r1a = r1 / minWidth;
      r1 = minWidth;
    }

    // Length of segment (from center to center) in pixels,
    // and unit vector from p0 to p1.
    vec2 viewsize = vec2(width, height);
    vec2 unit = (p1.xy - p0.xy) * viewsize;
    float linelength = length(unit);
    unit /= linelength;

    // Angle of tangents. Positive angle means they tilt inwards
    // from p0 to p1.
    float theta = asin((r0 - r1) / linelength);

    vec4 p;
    float r;
    float side = float(2*(gl_VertexID % 2) - 1);
    if (gl_VertexID < 2) {
      p = p0;
      r = r0;
      color.a *= r0a;
    } else {
      p = p1;
      r = r1;
      color.a *= r1a;
    }

    vec2 offset = vec2(-unit.y, unit.x);
    gl_Position = p + vec4(
      (unit * (sin(theta) * r) + offset * cos(theta) * side * r) / viewsize,
      0.0, 0.0);
  }`,
}

const fragmentShaders = {
vcolor: `#version 300 es
  precision highp float;
  in vec4 color;
  out vec4 outColor;

  void main(void) {
    outColor = color;
  }`,

tex: `#version 300 es
  precision highp float;
  in highp vec2 v_uv;
  uniform sampler2D image;
  uniform float alpha;
  out vec4 color;

  void main(void) {
    color = texture(image, v_uv);
    color.a *= alpha;
  }`,

roundpoint: `#version 300 es
  precision highp float;
  in vec4 color;
  out vec4 outColor;

  void main(void) {
    vec2 d = 2.0*gl_PointCoord - vec2(1.0, 1.0);
    float d2 = dot(d, d);
    if (d2 > 1.0) {
      discard;
    }
    outColor = color;
  }`
}


const programs = {
  screen: ['xy', 'tex'],    // For drawing texture on screen. Use corners buffer.
  cloud: ['depth', 'tex'],  // For drawing point cloud. Use points, one per pixel.
  linequads: ['linesegment', 'vcolor'],   // Two triangles per line segment.
  roundpoints: ['xyz_rgba', 'roundpoint'],  // For drawing round points.
};

function gridBuffer() {
  const s = 100;
  const p = [];
  const y = 2;
  for (i = -s; i <= s; i++) {
    for (j = -s; j <= s; j++) {
      p.push(i, y, j, i+1, y, j);
      p.push(i, y, j, i, y, j+1);
    }
  }
  return p;
}

const buffers = {
  frustum: [0,0,0, 0,0,1, 0,0,0, 1,0,1, 0,0,0, 0,1,1, 0,0,0, 1,1,1,
            0,0,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1, 0,1,1, 0,1,1, 0,0,1],
  frustum_points: [0,0,0, 0,0,1, 0,1,1, 1,0,1, 1,1,1],
  corners: [0,0, 1,0, 0,1, 1,1],
  grid: gridBuffer(),
};

const cameraInternal = {
  near: .01,
  far: 100,
  aspect_ratio: 1,
  xfrac: 1,
};

// Matrix mapping world to clip space.
function cameraMatrix(camera, pose) {
  var d = 1 / (cameraInternal.far - cameraInternal.near);
  var a = (cameraInternal.near + cameraInternal.far) * d;
  var b = - 2 * (cameraInternal.near * cameraInternal.far) * d;

  var w = camera.zoom;
  var h = camera.zoom * cameraInternal.aspect_ratio;
  var px = cameraInternal.xfrac - 1;
  var perspective = [
    w,  0, px, 0,
    0, -h,  0, 0,
    0,  0,  a, b,
    0,  0,  1, 0
  ];
  var target = [camera.pan_x, camera.pan_y, -camera.forward];  // Point to look at
  
  const follow_position = matT([-pose[0][3], -pose[1][3], -pose[2][3]]);
  let follow_rotation;
  if (camera.follow_rotation) {
    follow_rotation = [
      pose[0][0], pose[1][0], pose[2][0], 0,
      pose[0][1], pose[1][1], pose[2][1], 0,
      pose[0][2], pose[1][2], pose[2][2], 0,
      0, 0, 0, 1];
  } else {
    follow_rotation = matI();
  }
  return matCompose(
    perspective,
    matT([0, camera.elevation * camera.distance, camera.distance]),  // View distance
    matRx(camera.rx), matRy(camera.ry),  // View angle (controlled with mouse)
    matT(target),
    follow_rotation,
    follow_position);
}

// Maps camera-coords to world coords.
function poseMatrix(data, i) {
  const intrinsics = data.intrinsics[i];
  const px = intrinsics[2];
  const py = intrinsics[3];
  const ifx = 1.0 / intrinsics[0];
  const ify = 1.0 / intrinsics[1];
  const tex_to_cam = [
    ifx, 0, -px*ifx, 0,
    0, ify, -py*ify, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
  const cp = data.poses[i];
  const pose = [
    cp[0][0], cp[0][1], cp[0][2], cp[0][3],
    cp[1][0], cp[1][1], cp[1][2], cp[1][3],
    cp[2][0], cp[2][1], cp[2][2], cp[2][3],
    0, 0, 0, 1];
  const result = matCompose(pose, tex_to_cam);
  return result;
}

// Matrix utilities

/**
 * Multiply together any number of matrices. If there are none, returns an
 * identity matrix.
 * @param {...*} var_args The matrices to be multiplied.
 * @return {Array<number>}
 */
function matCompose(var_args) {
  var l = arguments.length;
  if (l == 0) {
    return matI();
  }
  var m = arguments[0];
  for (var i = 1; i < l; i++) {
    m = matMM(m, arguments[i]);
  }
  return m;
}

function matMM(a, b) {
  var c = [];
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      var k = j*4;
      c.push(a[k]*b[i] + a[k+1]*b[i+4] + a[k+2]*b[i+8] + a[k+3]*b[i+12]);
    }
  }
  return c;
}

function matMV(m, v) {
  var c = [];
  for (var j = 0; j < 4; j++) {
    var k = j*4;
    c.push(m[k]*v[0] + m[k+1]*v[1] + m[k+2]*v[2] + m[k+3]*v[3]);
  }
  return c;
}

function matMV3(m, v) {
  var c = [];
  for (var j = 0; j < 3; j++) {
    var k = j*4;
    c.push(m[k]*v[0] + m[k+1]*v[1] + m[k+2]*v[2]);
  }
  return c;
}

function vec3VdotV(a, b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

function vec4Lerp(a, b, p) {
  const q = 1-p
  return [a[0]*q+b[0]*p, a[1]*q+b[1]*p, a[2]*q+b[2]*p, a[3]*q+b[3]*p]
}

function matRx(t) {
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    1, 0, 0, 0,
    0, c,-s, 0,
    0, s, c, 0,
    0, 0, 0, 1
  ];
}

function matRy(t) {
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0,
     0, 0, 0, 1 
  ];
}

function matRz(t) {
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    c,-s, 0, 0,
    s, c, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1 
  ];
}

// Rotate by t about unit vector v (which can be a 3- or 4-vector).
function matRV(v, t) {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  const c = Math.cos(t);
  const s = Math.sin(t);
  const m = 1 - c;
  return [
    x*x*m + c,   x*y*m - z*s, x*z*m + y*s, 0,
    x*y*m + z*s, y*y*m + c,   y*z*m - x*s, 0,
    x*z*m - y*s, y*z*m + x*s, z*z*m + c,   0,
    0, 0, 0, 1
  ];
}

function matScale(t) {
  return [
    t, 0, 0, 0,
    0, t, 0, 0,
    0, 0, t, 0,
    0, 0, 0, 1
  ];
}

function matScaleV(v) {
  return [
    v[0], 0,    0,    0,
    0,    v[1], 0,    0,
    0,    0,    v[2], 0,
    0,    0,    0,    1
  ];
}

function matI() {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ];
}

function matT([x, y, z]) {
  return [
    1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1
  ];
}

function matTranspose(m) {
  return [
    m[0], m[4], m[ 8], m[12],
    m[1], m[5], m[ 9], m[13],
    m[2], m[6], m[10], m[14],
    m[3], m[7], m[11], m[15]
  ]
}


// === WEBGL SETUP ===

function initShaders(shaders, type) {
  const compiled = {};
  for (let i in shaders) {
    const s = gl.createShader(type);
    gl.shaderSource(s, shaders[i]);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.log('Compiling ' + i, gl.getShaderInfoLog(s));
    }
    compiled[i] = s;
  }
  return compiled;
}

function initPrograms(vertexShaders, fragmentShaders, programs) {
  for (let i in programs) {
    const p = gl.createProgram();
    gl.attachShader(p, vertexShaders[programs[i][0]]);
    gl.attachShader(p, fragmentShaders[programs[i][1]]);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.log('Linking ' + i, gl.getProgramInfoLog(p));
    }
    const attribute = {};
    const uniform = {};
    let n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < n; ++i) {
      const info = gl.getActiveAttrib(p, i);
      const loc = gl.getAttribLocation(p, info.name);
      if (loc >= 0) {
        attribute[info.name] = loc;
      }
    }
    n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; ++i) {
      const info = gl.getActiveUniform(p, i);
      const loc = gl.getUniformLocation(p, info.name);
      if (loc) {
        uniform[info.name] = loc;
      }
    }
    programs[i].name = i;
    programs[i].program = p;
    programs[i].attribute = attribute;
    programs[i].uniform = uniform;
  }
}

function initBuffer(b) {
  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  const data = new Float32Array(b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  b.vb = vb;
  b.data = data;
}

function initBuffers(buffers) {
  for (const b in buffers) {
    initBuffer(buffers[b]);
  }
}
function glFailed() {
  document.getElementById('glfailed').style.display = "block";
}

let remaining_to_load = 0;
let total_to_load = 0;
let dirty = true;

function updateLoading() {
  if (remaining_to_load <= 0) {
    document.getElementById('loading').style.display = 'none';
  }
}

function leafName(path) {
  const i = path.lastIndexOf('/');
  if (i > 0) {
    return path.slice(i + 1);
  } else {
    return path;
  }
}

// Returns a buffer with the camera path. xyz 12 bytes each.
function buildPathBuffer(poses) {
  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  const floats = new Float32Array(poses.length * 3);
  let index = 0;
  for (const p of poses) {
    floats[index] = p[0][3];
    floats[index+1] = p[1][3];
    floats[index+2] = p[2][3];
    index += 3;
  }
  gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);
  return {vb: vb, n: poses.length};
}

function pad5(i) {
  return i.toString().padStart(5, '0');
}

async function loadScene(packedurl, override) {
  document.getElementById('loading').style.display = 'block';
  lastOverlayFrame = -1;
  data = {};
  resetState(state, parameterSpec.state, override);
  resetState(camera, parameterSpec.camera, override);
  console.log(`Loading scene: ${packedurl}`);
  const packed_data = await fetchPacked(`static/megaview/${packedurl}.packed`);
  data = JSON.parse(await packed_data['data.json'].text());
  data.rgb = [];
  data.depth = [];
  data.depth_scale = 20;
  data.path = buildPathBuffer(data.poses);

  // Apply scene-specific viewer hints embedded in data.json.
  // These are only used when not already overridden by a URL/dataset parameter.
  if (data.frustum_size !== undefined && !('frustum_size' in override)) {
    state.frustum_size = data.frustum_size;
  }
  const frustumSlider = document.getElementById('frustum-size-slider');
  if (frustumSlider) frustumSlider.value = state.frustum_size;
  const pointSizeSlider = document.getElementById('point-size-slider');
  if (pointSizeSlider) pointSizeSlider.value = state.point_size;
  const otherPointSizeSlider = document.getElementById('other-point-size-slider');
  if (otherPointSizeSlider) otherPointSizeSlider.value = state.other_point_size;

  if (data.camera_dist !== undefined && !('distance' in override)) {
    camera.distance = data.camera_dist;
  }
  if (data.camera_forward !== undefined && !('forward' in override)) {
    camera.forward = data.camera_forward;
  }

  for (let i = 0; i < data.poses.length; i++) {
    data.rgb[i] = loadTexture(packed_data[`rgb_${pad5(i)}.png`]);
    data.depth[i] = loadDepth(packed_data[`depthrgb_${pad5(i)}.png`]);
  }
}

function loadTexture(blob) {
  const t = gl.createTexture();
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = function() {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    t.image = image;  // Keep image reference for video overlay
    t.ready = true;
    URL.revokeObjectURL(url);
    dirty = true;
    remaining_to_load--;
    updateLoading();
  }
  total_to_load++;
  remaining_to_load++;
  image.src = url;
  return t;
}
function loadDepth(blob) {
  const t = gl.createTexture();
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = function() {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    t.ready = true;
    URL.revokeObjectURL(url);
    dirty = true;
    remaining_to_load--;
    updateLoading();
  }
  total_to_load++;
  remaining_to_load++;
  updateLoading();
  image.src = url;
  return t;
}

function clamp(i, base, limit) {
  return (i < base ? base : i >= limit ? limit-1 : i);
}

function prepareDrawFrustum(size_factor) {
  const p = programs['linequads'];
  program(p);
  gl.uniform1f(p.uniform.width, canvas.width);
  gl.uniform1f(p.uniform.height, canvas.height);
  gl.uniform1f(p.uniform.minWidth, size_factor);
  gl.uniform1f(p.uniform.zOffset, 0.0);
  gl.uniform1i(p.uniform.maxLength, 1);
  gl.uniform1i(p.uniform.maxSolidLength, 1);
  gl.uniform1i(p.uniform.recolor, 0);
  gl.uniform1i(p.uniform.index_stride, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.frustum.vb);
  gl.enableVertexAttribArray(p.attribute.xyz0);
  gl.enableVertexAttribArray(p.attribute.xyz1);
  gl.vertexAttribDivisor(p.attribute.xyz0, 1);
  gl.vertexAttribDivisor(p.attribute.xyz1, 1);
  gl.vertexAttrib1f(p.attribute.segmentLength, 0.0);
  gl.vertexAttrib1f(p.attribute.index, 0.0);
  gl.vertexAttribPointer(p.attribute.xyz0, 3, gl.FLOAT, false, 24, 0);
  gl.vertexAttribPointer(p.attribute.xyz1, 3, gl.FLOAT, false, 24, 12);
}

function drawFrustum(camera_matrix, i, color, width) {
  const p = programs['linequads'];
  gl.uniformMatrix4fv(p.uniform.camera, true, matCompose(
      camera_matrix,
      poseMatrix(data, i),
      matScale(state.frustum_size)));
  gl.vertexAttrib4fv(p.attribute.rgba, color);
  gl.uniform1f(p.uniform.lineWidth, width);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 8);
}

function prepareDrawFrustumPoints(size_factor) {
  const p = programs['roundpoints'];
  program(p);
  gl.uniform1f(p.uniform.width, canvas.width);
  gl.uniform1f(p.uniform.height, canvas.height);
  gl.uniform1f(p.uniform.minWidth, size_factor);
  gl.uniform1f(p.uniform.zOffset, 0.0);
  gl.uniform1i(p.uniform.recolor, 0);
  gl.uniform1i(p.uniform.index_stride, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.frustum_points.vb);
  gl.enableVertexAttribArray(p.attribute.xyz);
  gl.vertexAttrib1f(p.attribute.index, 0.0);
  gl.vertexAttribPointer(p.attribute.xyz, 3, gl.FLOAT, false, 12, 0);
}

function drawFrustumPoints(camera_matrix, i, color, size) {
  const p = programs['roundpoints'];
  gl.uniformMatrix4fv(p.uniform.camera, true, matCompose(
      camera_matrix,
      poseMatrix(data, i),
      matScale(state.frustum_size)));
  gl.vertexAttrib4fv(p.attribute.rgba, color);
  gl.uniform1f(p.uniform.point_size, size);
  gl.drawArrays(gl.POINTS, 0, 5);
}


function drawPath(camera_matrix, color, size_factor, width, dash) {
  const p = programs['linequads'];
  gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
  gl.vertexAttrib4fv(p.attribute.rgba, color);
  gl.uniform1f(p.uniform.minWidth, size_factor);
  gl.uniform1f(p.uniform.zOffset, 0.0);
  gl.uniform1f(p.uniform.lineWidth, width);
  gl.bindBuffer(gl.ARRAY_BUFFER, data.path.vb);

  const step = (2*dash) || 1;
  gl.vertexAttribPointer(p.attribute.xyz0, 3, gl.FLOAT, false, step * 12, 0);
  gl.vertexAttribPointer(p.attribute.xyz1, 3, gl.FLOAT, false, step * 12, (dash || 1) * 12);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, Math.floor((data.path.n-1) / step));
}

function prepareDrawImage(camera_matrix) {
  const p = programs['screen'];
  program(p);
  gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.corners.vb);
  gl.enableVertexAttribArray(p.attribute.xy);
  gl.vertexAttribPointer(p.attribute.xy, 2, gl.FLOAT, false, 8, 0);
  gl.uniform1f(p.uniform.frustum_size, state.frustum_size*1.001);
}
function drawImage(frame, tex, alpha) {
  if (!tex.ready) {
    return;
  }
  const p = programs['screen'];
  gl.uniformMatrix4fv(p.uniform.pose, true, poseMatrix(data, frame));
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(p.uniform.image, 0)
  gl.uniform1f(p.uniform.alpha, alpha)
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, buffers.corners.length/2);
}

function prepareDrawPoints(camera_matrix, stride, size, size_factor) {
  const p = programs['cloud'];
  program(p);
  gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
  gl.uniform1i(p.uniform.image, 0)
  gl.uniform1i(p.uniform.depth, 1)
  gl.uniform1f(p.uniform.depthscale, data.depth_scale);
  gl.uniform1f(p.uniform.max_grad, state.z_clamp * 2);
  gl.uniform1f(p.uniform.point_size, size * size_factor);
  gl.uniform1i(p.uniform.width, data.width);
  gl.uniform1i(p.uniform.height, data.height);
  gl.uniform1i(p.uniform.stride, stride);
}
function drawPoints(i, stride, alpha) {
  const depth_tex = data.depth[i]; 
  if (!data.rgb[i].ready || !depth_tex.ready) {
    return;
  }
  const p = programs['cloud'];
  gl.uniform1f(p.uniform.alpha, alpha);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, data.rgb[i]);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, depth_tex);
  gl.uniformMatrix4fv(p.uniform.pose, true, poseMatrix(data, i));
  gl.drawArrays(gl.POINTS, 0, totalPoints(stride));
}

function totalPoints(stride) {
  const w = data.width;
  const h = data.height;
  return (w/stride | 0) * (h/stride | 0);
}

function program(p) {
  gl.useProgram(p.program);
  for (const a of Object.values(p.attribute)) {
    gl.disableVertexAttribArray(a);
    gl.vertexAttribDivisor(a, 0);
  }
}

function* other_frames() {
  const step = state.every_nth || 1;
  let limit;
  if (state.other_points === 'past') {
    limit = state.frame + 1;
  } else if (state.reveal) {
    limit = state.frame;
  } else {
    limit = data.poses.length;
  }
  for (let i = state.other_offset; i < limit; i += step) {
    let fade = 1.0;
    if (state.reveal && i > state.frame - step) {
      fade = (state.frame - i) / step;
    }
    yield [i, fade];
  }
}

function drawScene(camera_matrix, size_factor, opaque) {
  if (!state.show_back_facing) {
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.FRONT);
  }

  // Other points
  switch (state.other_points) {
    case 'points':
    case 'past':
      prepareDrawPoints(camera_matrix, state.other_stride, state.other_point_size, size_factor);
      for (const [i, fade] of other_frames()) {
        if ((fade == 1.0) == opaque) {
          drawPoints(i, state.other_stride, fade * state.other_points_alpha);
        }
      }
      break;
  }

  if (opaque) {
    // Current frame
    switch (state.show_points) {
      case 'points':
        prepareDrawPoints(camera_matrix, state.stride, state.point_size, size_factor);
        drawPoints(state.frame, state.stride, state.points_alpha);
        break;
    }
  }
  gl.disable(gl.CULL_FACE);
}
  
//

function draw(state) {
  const back = state.background;
  gl.clearColor(back, back, back, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const bg = back < 0.5 ? 0 : 255;
  const fg = 255 - bg;
  document.body.style.setProperty('--bg', `${bg}`);
  document.body.style.setProperty('--fg', `${fg}`);
  
  if (!data || !data.poses) { return; }
  const size_factor = state.size_factor;

  const camera_matrix = cameraMatrix(camera, data.poses[state.camera_frame]);
  let depth = data.depth;
  const rgb = data.rgb[state.frame];

  gl.disable(gl.CULL_FACE);

  drawScene(camera_matrix, size_factor, true); // opaque parts

  // Solid: frusta
  const other_colors = [
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [0.75, 0.75, 0.0, 1.0],
    [0.0, 0.75, 0.0, 1.0],
    [0.0, 0.25, 0.75, 1.0],
  ];

  prepareDrawFrustum(size_factor);
  if (state.other_frusta) {
    const other_color_start = rgba(colors.other_frusta);
    const other_color_end = rgba(colors.other_frusta_end);
    let j = 0;
    for (const [i, fade] of other_frames()) {
      if (i != state.frame || !state.draw_frustum) {
        drawFrustum(
            camera_matrix, i,
            vec4Lerp(other_color_start, other_color_end, i / (data.poses.length-1)),
            state.other_frusta_width * size_factor);
        j++;
      }
    }
  }
  if (state.draw_frustum) {
    drawFrustum(camera_matrix, state.frame, rgba(colors.frustum), state.frustum_width * size_factor);
  }
  if (state.draw_path) {
    drawPath(camera_matrix, rgba(colors.path), size_factor, state.path_width * size_factor, state.path_dash);
  }

  prepareDrawFrustumPoints(size_factor);
  if (state.other_frusta) {
    const other_color_start = rgba(colors.other_frusta);
    const other_color_end = rgba(colors.other_frusta_end);
    let j = 0;
    for (const [i, fade] of other_frames()) {
      if (i != state.frame || !state.draw_frustum) {
        drawFrustumPoints(
            camera_matrix, i,
            vec4Lerp(other_color_start, other_color_end, i / (data.poses.length-1)),
            state.other_frusta_width * size_factor);
        j++;
      }
    }
  }
  if (state.draw_frustum) {
    drawFrustumPoints(camera_matrix, state.frame, rgba(colors.frustum), state.frustum_width * size_factor);
  }

  // Solid: images/depth in frusta
  prepareDrawImage(camera_matrix);
  if (state.other_image != 'empty') {
    for (const [i, fade] of other_frames()) {
      if (i != state.frame || state.image == 'empty') {
        drawImage(i, data.rgb[i], 1.0);
      }
    }
  }
  if (state.image != 'empty') {
    drawImage(state.frame, data.rgb[state.frame], 1.0);
  }
  drawScene(camera_matrix, size_factor, false); // non-opaque parts
}

function rgba(text, int=false) {
  function f(a, b) {
    const x = parseInt(a, 16);
    const y = b ? parseInt(b, 16) : x;
    if (isNaN(x) || isNaN(y)) {
      return 128;
    }
    return 16*x + y;
  }
  if (text[0] != '#') {
    text = '#' + text;
  }
  let col = [255,0,255,255];
  switch (text.length) {
    case 4: col = [f(text[1]), f(text[2]), f(text[3]), 255]; break;
    case 7: col = [f(text[1], text[2]), f(text[3], text[4]), f(text[5], text[6]), 255]; break;
  }
  if (int) {
    return col;
  }
  return [col[0]/255, col[1]/255, col[2]/255, col[3]/255];
}

const parameterSpec = {
  camera: {
    distance:  2,
    forward:   0,
    pan_x:     0,
    pan_y:     0,
    elevation: 0.2,
    zoom:      1,
    follow:    false,
    follow_rotation: true,
    rx:        0.5,
    ry:        0,
  },
  state: {
    draw_frustum:    true,
    image:           'image',
    show_points:     'points',
    points_alpha:    1.0,
    stride:          1,
    point_size:      2,
    frustum_size:    0.25,
    z_clamp:         0.02,
    show_back_facing: false,
    // Others
    every_nth:       8,
    reveal:          false,
    other_offset:    0,
    other_frusta:    true,
    other_image:     'empty',
    other_points:    'past',
    other_points_alpha: 1.0,
    other_stride:    2,
    other_point_size: 1,
    // Animation
    playing:         true, 
    fps:             30,
    frame:           0,
    camera_frame:    0,
    // Misc
    background:      0.95,
    // Style
    frustum_width:   2,
    other_frusta_width: 1,
    draw_path:       false,
    path_width:      2,
    path_dash:       0,
  },
  colors: {
    frustum:         '#ff0000',
    other_frusta:    '#6666ff',
    other_frusta_end: '#6666ff',
    path:            '#888888',
  }
}

async function fetchPacked(url) {
  const results = {};
  const response = await fetch(url);
  if (response.status != 200) {
    const error = await response.text();
    console.log('Status: ', response.status);
    console.log('Error: ', error);
    return results;
  }
  const blob = await response.blob();
  const prefix_size = new DataView(await blob.slice(0, 8).arrayBuffer()).getUint32(0);
  // Map name to start,end offsets (not including json_size).
  const json = JSON.parse(await blob.slice(8, prefix_size).text());
  for (const [key, [start, end, content_type]] of Object.entries(json)) {
    results[key] = blob.slice(start + prefix_size, end + prefix_size, content_type);
  }
  return results;
}

function resetState(target, spec, override={}) {
  for (const key in spec) {
    if (key in override) {
      const val = override[key];
      switch (typeof(spec[key])) {
        case "number": target[key] = +val; break;
        case "boolean": target[key] = (val === true || val === 'true' || val === '1'); break;
        default: target[key] = val; break;
      }
    } else {
     target[key] = spec[key];
   }
  }
}

const state = {};  resetState(state, parameterSpec.state);
const camera = {}; resetState(camera, parameterSpec.camera);
const colors = {}; resetState(colors, parameterSpec.colors);

let data = false;
let frame_time = Date.now();

function update(state) {
  const time = Date.now();
  const period = 1000.0 / state.fps;
  if (time - frame_time <= period) {
    return;
  }
  frame_time = time;
  if (state.playing && data.poses && data.poses.length) {
    state.frame = (state.frame + 1) % data.poses.length;
    dirty = true;
  }
}

// === VIDEO OVERLAY ===

let videoOverlayCanvas = null;
let videoOverlayCtx = null;
let showVideoOverlay = true;
let lastOverlayFrame = -1;

function initVideoOverlay() {
  videoOverlayCanvas = document.getElementById('video-overlay');
  if (videoOverlayCanvas) {
    videoOverlayCtx = videoOverlayCanvas.getContext('2d');
  }
}

function updateVideoOverlay() {
  if (!showVideoOverlay || !data || !data.rgb || !videoOverlayCanvas || !videoOverlayCtx) return;
  const frame = state.frame;
  if (frame === lastOverlayFrame) return;
  lastOverlayFrame = frame;
  const tex = data.rgb[frame];
  if (!tex || !tex.ready || !tex.image) return;
  const img = tex.image;
  if (videoOverlayCanvas.width !== img.naturalWidth || videoOverlayCanvas.height !== img.naturalHeight) {
    videoOverlayCanvas.width = img.naturalWidth;
    videoOverlayCanvas.height = img.naturalHeight;
    const cssWidth = Math.min(280, Math.round(canvas.clientWidth * 0.36));
    videoOverlayCanvas.style.width = cssWidth + 'px';
    videoOverlayCanvas.style.height = Math.round(cssWidth * img.naturalHeight / img.naturalWidth) + 'px';
  }
  videoOverlayCtx.drawImage(img, 0, 0);
}

function toggleVideoOverlay() {
  showVideoOverlay = !showVideoOverlay;
  const btn = document.getElementById('video-overlay-btn');
  const overlay = document.getElementById('video-overlay');
  if (!overlay) return;
  if (showVideoOverlay) {
    overlay.style.display = 'block';
    if (btn) { btn.textContent = 'Hide'; btn.classList.add('active-frusta'); }
    lastOverlayFrame = -1;
    dirty = true;
  } else {
    overlay.style.display = 'none';
    if (btn) { btn.textContent = 'Show'; btn.classList.remove('active-frusta'); }
  }
}

function tick() {
  window.requestAnimationFrame(tick);
  if (remaining_to_load > 0) {
    return;
  }
  if (state.playing) {
    update(state);
  }
  if (dirty) {
    dirty = false;
    if (data) {
      if (camera.follow) {
        state.camera_frame = state.frame;
      } else {
        state.camera_frame = 0;
      }
    }
    draw(state);
    updateVideoOverlay();
  }
}

function addHandlers(canvas) {
  let rotating = false;
  let panning = false;
  let ox, oy, rx, ry, px, py;  // Drag origin; rotate/pan start values.
  const rotSpeed = Math.PI/500;  // Radians per pixel.

  canvas.addEventListener("pointerdown", (event) => {
    ox = event.clientX;
    oy = event.clientY;
    if (event.button === 2) {  // Right-click: pan
      panning = true;
      px = camera.pan_x;
      py = camera.pan_y;
    } else {                   // Left-click: rotate
      rotating = true;
      rx = camera.rx;
      ry = camera.ry;
    }
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    const dx = event.clientX - ox;
    const dy = event.clientY - oy;
    if (rotating) {
      camera.rx = rx + dy * rotSpeed;
      camera.ry = ry - dx * rotSpeed;
      dirty = true;
    } else if (panning) {
      const panSpeed = camera.distance * 2 / canvas.clientWidth;
      camera.pan_x = px - dx * panSpeed;
      camera.pan_y = py + dy * panSpeed;
      dirty = true;
    }
  });
  const stopDrag = () => { rotating = false; panning = false; };
  canvas.addEventListener("pointerup",     stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
  canvas.addEventListener("pointerout",    stopDrag);
  canvas.addEventListener("pointerleave",  stopDrag);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  var wheel = function(e) {
    const scale = Math.pow(0.998, e.deltaY);
    // Dolly the orbit center forward so the camera flies into the scene,
    // not just orbits tighter around a fixed point.
    camera.forward += camera.distance * (1 - scale);
    camera.distance *= scale;
    camera.distance = Math.max(0.001, Math.min(20, camera.distance));
    dirty = true;
    e.preventDefault();
    e.stopPropagation();
  }
  canvas.addEventListener('wheel', wheel, {passive: false});
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || document.activeElement != document.body) {
      return;
    }
    if (handleKey(e.key, e.shiftKey)) {
      dirty = true;
      e.preventDefault();
      e.stopPropagation();
    }
  });
}

function incFrame(i) {
  setPlaying(false);
  state.frame = (state.frame + i + 2*data.poses.length) % data.poses.length;
  dirty = true;
}

function setPlaying(v) {
  state.playing = v;
}

function handleKey(key, shift) {
  switch (key) {
    case 'ArrowLeft': if (shift) { incFrame(-10); } else { incFrame(-1); }; return true;
    case 'ArrowRight': if (shift) { incFrame(10); } else { incFrame(1); }; return true;
    case ' ': setPlaying(!state.playing); return true;
  }
  return false;
}

function resize() {
  const dp = window.devicePixelRatio;
  state.size_factor = dp;
  const w = canvas.clientWidth * dp;
  const h = canvas.clientHeight * dp;
  canvas.width = w;
  canvas.height = h;
  gl.viewport(0, 0, w, h);
  cameraInternal.aspect_ratio = w/h;
  dirty = true;
}

async function init() {
  canvas = document.getElementById('megaview');
  gl = canvas.getContext('webgl2', {antialias: false, alpha: false, preserveDrawingBuffer: true});
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) {
    glFailed();
  }
  resize();

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const v = initShaders(vertexShaders, gl.VERTEX_SHADER);
  const f = initShaders(fragmentShaders, gl.FRAGMENT_SHADER);
  initPrograms(v, f, programs);
  initBuffers(buffers);
  addHandlers(canvas);
  initVideoOverlay();
  dirty = true;
  window.addEventListener('resize', resize);
  window.requestAnimationFrame(tick);
  document.querySelector('.load').addEventListener('click', (event) => {
    if (event.target.dataset && event.target.dataset.scene) {
        loadScene(event.target.dataset.scene, event.target.dataset);
    }
  });
  document.querySelector('.load img').click();
}

window.addEventListener('load', init);
