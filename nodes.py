import numpy as np
import torch
import OpenGL.GL as gl
import glfw
import nodes as comfy_nodes

def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetShaderInfoLog(shader))
    return shader

def compile_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

def setup_framebuffer(width, height):
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer is not complete")

    return fbo, texture

def setup_render_resources(width, height, fragment_source: str):
    if not glfw.init():
        raise RuntimeError("GLFW did not init")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden
    window = glfw.create_window(width, height, "hidden", None, None)
    if not window:
        raise RuntimeError("GLFW did not init window")

    glfw.make_context_current(window)

    vertex_source = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    void main()
    {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """
    shader = compile_program(vertex_source, fragment_source)

    vertices = np.array([
        -1.0, -1.0,
         3.0, -1.0,
        -1.0,  3.0
    ], dtype=np.float32)
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)

    fbo, texture = setup_framebuffer(width, height)

    return (fbo, shader, vao)

def render_resources_cleanup():
    # assume all other resources get cleaned up here
    glfw.terminate()

def render(width, height, fbo, shader, vao):
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    gl.glUseProgram(shader)
    gl.glBindVertexArray(vao)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = image[::-1, :, :]
    image = np.array(image).astype(np.float32) / 255.0

    return image

SHADERTOY_HEADER = """
#version 440

precision highp float;

uniform vec3	iResolution;
uniform vec4	iMouse;
uniform float	iTime;
uniform float	iTimeDelta;
uniform float	iFrameRate;
uniform int	    iFrame;

#define texture2D texture

"""

SHADERTOY_FOOTER = """

layout(location = 0) out vec4 _fragColor;

void main() 
{ 
	mainImage(_fragColor, gl_FragCoord.xy); 
}
"""

SHADERTOY_DEFAULT = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}
"""

def shadertoy_vars_update(shader, width, height, time, time_delta, frame_rate, frame):
    gl.glUseProgram(shader)
    iResolution_location = gl.glGetUniformLocation(shader, "iResolution")
    gl.glUniform3f(iResolution_location, width, height, 0)
    iMouse_location = gl.glGetUniformLocation(shader, "iMouse")
    gl.glUniform4f(iMouse_location, 0, 0, 0, 0)
    iTime_location = gl.glGetUniformLocation(shader, "iTime")
    gl.glUniform1f(iTime_location, time)
    iTimeDelta_location = gl.glGetUniformLocation(shader, "iTimeDelta")
    gl.glUniform1f(iTimeDelta_location, time_delta)
    iFrameRate_location = gl.glGetUniformLocation(shader, "iFrameRate")
    gl.glUniform1f(iFrameRate_location, frame_rate)
    iFrame_location = gl.glGetUniformLocation(shader, "iFrame")
    gl.glUniform1i(iFrame_location, frame)

class Shadertoy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "frame_count": ("INT", {"default": 1, "min": 1, "max": 262144}),
                              "fps": ("INT", {"default": 1, "min": 1, "max": 120}),
                              "source": ("STRING", {"default": SHADERTOY_DEFAULT, "multiline": True, "dynamicPrompts": False})}}
    
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "Audio Reactor"
    FUNCTION = "render"

    def render(self, width: int, height: int, frame_count: int, fps: int, source: str):
        fragment_source = SHADERTOY_HEADER
        fragment_source += source
        fragment_source += SHADERTOY_FOOTER
        print(fragment_source)

        fbo, shader, vao = setup_render_resources(width, height, fragment_source)

        images = []
        frame = 0
        for _ in range(frame_count):
            shadertoy_vars_update(shader, width, height, frame * (1.0 / fps), (1.0 / fps), fps, frame)

            image = render(width, height, fbo, shader, vao)
            image = torch.from_numpy(image)[None,]
            images.append(image)

            frame += 1
        
        render_resources_cleanup()

        return (torch.cat(images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "Shadertoy": Shadertoy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Shadertoy": "Shadertoy",
}
