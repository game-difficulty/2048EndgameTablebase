// Photo-space landmarks (240 x 240). Outer edges and the planted foot stay fixed.
export const CHICKEN_CYCLE_MS = { chicken: 2800, serious: 3200 };
const smooth = value => value * value * (3 - 2 * value);
const clamp = value => Math.max(0, Math.min(1, value));
const tracks = {
  chicken: [[0,0,0], [.12,-.22,-.12], [.36,1,1], [.51,.85,.72], [.72,-.55,-.35], [.86,-.2,.12], [1,0,0]],
  serious: [[0,0,0], [.14,-.12,-.2], [.4,1,1], [.64,1,.85], [.83,-.08,.12], [1,0,0]],
};

export function chickenPose(id, milliseconds) {
  const frames = tracks[id] || tracks.chicken;
  const duration = CHICKEN_CYCLE_MS[id] || CHICKEN_CYCLE_MS.chicken;
  const phase = ((milliseconds % duration) + duration) % duration / duration;
  const index = frames.findIndex((frame, i) => i > 0 && phase <= frame[0]);
  const a = frames[index - 1], b = frames[index];
  const blend = smooth((phase - a[0]) / (b[0] - a[0]));
  return [a[1] + (b[1] - a[1]) * blend, a[2] + (b[2] - a[2]) * blend];
}

function influence(x, y, cx, cy, rx, ry) {
  const distance = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
  return distance >= 1 ? 0 : (1 - distance) ** 2;
}

export function deformChickenPoint(id, x, y, pose) {
  const [motion, follow] = pose;
  let dx = 0, dy = 0;
  const move = (cx, cy, rx, ry, tx, ty, angle = 0, pivotX = cx, pivotY = cy) => {
    const weight = influence(x, y, cx, cy, rx, ry);
    const px = x - pivotX, py = y - pivotY;
    dx += weight * (tx + px * (Math.cos(angle) - 1) - py * Math.sin(angle));
    dy += weight * (ty + px * Math.sin(angle) + py * (Math.cos(angle) - 1));
  };
  if (id === 'serious') {
    // The raised orange limbs are feet, not hands. The feathered wings rest on the floor.
    move(121,146,49,61,0,7 * motion);
    move(122,94,35,32,0,1.4 * motion);
    move(39,126,39,41,-4 * follow,7 * follow,-.12 * follow,76,163);
    move(202,124,37,42,4 * follow,7 * follow,.12 * follow,167,160);
    move(77,187,34,27,-2.8 * motion,.7 * motion);
    move(167,187,34,27,2.8 * motion,.7 * motion);
  } else {
    move(126,139,48,57,2 * motion,2.3 * motion);
    move(87,79,37,37,-1.8 * follow,-1.6 * motion,-.045 * follow,108,110);
    move(51,112,45,24,-2 * follow,-6 * follow,.12 * follow,99,117);
    move(157,105,43,33,2 * follow,-5 * follow,-.12 * follow,116,120);
    move(183,74,22,47,-3 * motion,-5 * motion,-.07 * motion,168,126);
    move(143,173,20,30,-motion,motion);
  }
  const edge = smooth(clamp(Math.min(x, 240 - x, y, 224 - y) / 14));
  return [x + dx * edge, y + dy * edge];
}

export function createChickenMesh(segments = 32) {
  const points = [], indices = [];
  for (let row = 0; row <= segments; row++) {
    for (let col = 0; col <= segments; col++) points.push(col / segments * 240, row / segments * 240);
  }
  for (let row = 0; row < segments; row++) {
    for (let col = 0; col < segments; col++) {
      const a = row * (segments + 1) + col, b = a + segments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { points: new Float32Array(points), indices: new Uint16Array(indices) };
}

// One textured draw per frame; no video downloads, frame swapping or per-triangle canvas draws.
export function createChickenRenderer(canvas, image, id) {
  const gl = canvas.getContext('webgl', { alpha:false, antialias:false, depth:false, preserveDrawingBuffer:false });
  if (!gl) return null;
  const vertex = gl.createShader(gl.VERTEX_SHADER), fragment = gl.createShader(gl.FRAGMENT_SHADER);
  const program = gl.createProgram(), position = gl.createBuffer(), uv = gl.createBuffer(), element = gl.createBuffer();
  const texture = gl.createTexture();
  const dispose = () => {
    for (const buffer of [position, uv, element]) gl.deleteBuffer(buffer);
    gl.deleteTexture(texture); gl.deleteProgram(program); gl.deleteShader(vertex); gl.deleteShader(fragment);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  };
  gl.shaderSource(vertex, 'attribute vec2 a_position; attribute vec2 a_uv; varying vec2 v_uv; void main(){v_uv=a_uv;gl_Position=vec4(a_position.x/120.0-1.0,1.0-a_position.y/120.0,0.0,1.0);}');
  gl.shaderSource(fragment, 'precision mediump float; varying vec2 v_uv; uniform sampler2D u_photo; void main(){gl_FragColor=texture2D(u_photo,v_uv);}');
  gl.compileShader(vertex); gl.compileShader(fragment);
  gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { dispose(); return null; }
  const mesh = createChickenMesh(), vertices = new Float32Array(mesh.points.length);
  gl.useProgram(program);
  for (const [buffer, name, data, usage] of [
    [position, 'a_position', mesh.points, gl.DYNAMIC_DRAW],
    [uv, 'a_uv', mesh.points.map(value => value / 240), gl.STATIC_DRAW],
  ]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
  }
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, element); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
  gl.viewport(0, 0, canvas.width, canvas.height);
  return {
    draw(milliseconds) {
      const pose = chickenPose(id, milliseconds);
      for (let i = 0; i < vertices.length; i += 2) {
        const [x, y] = deformChickenPoint(id, mesh.points[i], mesh.points[i + 1], pose);
        vertices[i] = x; vertices[i + 1] = y;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, position); gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertices);
      gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0);
    },
    dispose,
  };
}
