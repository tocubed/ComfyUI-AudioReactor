import numpy as np
import torch
import librosa
import nodes as comfy_nodes

import os
headless = os.name == "posix" and os.environ.get("DISPLAY", "") == ""

if headless:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import OpenGL.EGL as egl
    import OpenGL.GL as gl

    def render_surface_and_context_init(width, height): # type: ignore
        egl_display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        if egl_display == egl.EGL_NO_DISPLAY:
            raise RuntimeError("No EGL display connection available")
        if not egl.eglInitialize(egl_display, None, None):
            raise RuntimeError("Unable to initialize EGL")

        config_attribs = [
             egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
             egl.EGL_BLUE_SIZE, 8,
             egl.EGL_GREEN_SIZE, 8,
             egl.EGL_RED_SIZE, 8,
             egl.EGL_DEPTH_SIZE, 24,
             egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT,
             egl.EGL_NONE
        ]
        num_configs = egl.EGLint()
        if not egl.eglChooseConfig(egl_display, config_attribs, None, 0, num_configs) or num_configs.value == 0:
            raise RuntimeError("No suitable EGL configuration was found")
        
        egl_config = (egl.EGLConfig * num_configs.value)()
        egl.eglChooseConfig(egl_display, config_attribs, egl_config, 1, num_configs)
        egl_config = egl_config[0]

        egl_context = egl.eglCreateContext(egl_display, egl_config, egl.EGL_NO_CONTEXT, None)
        if egl_context == egl.EGL_NO_CONTEXT:
            raise RuntimeError("Unable to create EGL context")

        pbuffer_attribs = [
             egl.EGL_WIDTH, width,
             egl.EGL_HEIGHT, height,
             egl.EGL_NONE
        ]
        egl_surface = egl.eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs)
        if egl_surface == egl.EGL_NO_SURFACE:
            raise RuntimeError("Unable to create offscreen EGL surface")

        if not egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
            raise RuntimeError("Unable to make EGL context current")

        return {"egl_display": egl_display, "egl_surface": egl_surface, "egl_context": egl_context}
    
    def render_surface_and_context_deinit(egl_display, egl_surface, egl_context): # type: ignore
        egl.eglMakeCurrent(egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT)
        egl.eglDestroySurface(egl_display, egl_surface)
        egl.eglDestroyContext(egl_display, egl_context)
        egl.eglTerminate(egl_display)
else:
    import OpenGL.GL as gl
    import glfw

    def render_surface_and_context_init(width, height):
        if not glfw.init():
            raise RuntimeError("GLFW did not init")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden
        window = glfw.create_window(width, height, "hidden", None, None)
        if not window:
            raise RuntimeError("GLFW did not init window")

        glfw.make_context_current(window)
        return {}
    
    def render_surface_and_context_deinit(**kwargs):
        glfw.terminate()

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
    ctx = render_surface_and_context_init(width, height)

    vertex_source = """
    #version 330 core
    void main()
    {
        vec2 verts[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
        gl_Position = vec4(verts[gl_VertexID], 0, 1);
    }
    """
    shader = compile_program(vertex_source, fragment_source)

    fbo, texture = setup_framebuffer(width, height)

    textures = gl.glGenTextures(4)

    return (ctx, fbo, shader, textures)

def render_resources_cleanup(ctx):
    # assume all other resources get cleaned up with the context
    render_surface_and_context_deinit(**ctx)

def render(width, height, fbo, shader):
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    gl.glUseProgram(shader)
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

uniform sampler2D   iChannel0;
uniform sampler2D   iChannel1;
uniform sampler2D   iChannel2;
uniform sampler2D   iChannel3;

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

def shadertoy_texture_update(texture, image, frame):
    if len(image.shape) == 4:
        image = image[frame]
    image = image.cpu().numpy()
    image = image[::-1, :, :]
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, image.shape[1], image.shape[0], 0, gl.GL_RGB, gl.GL_FLOAT, image)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

def shadertoy_texture_bind(shader, textures):
    gl.glUseProgram(shader)
    for i in range(4):
        gl.glActiveTexture(gl.GL_TEXTURE0 + i) # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D, textures[i])
        iChannel_location = gl.glGetUniformLocation(shader, f"iChannel{i}")
        gl.glUniform1i(iChannel_location, i)

class Shadertoy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": comfy_nodes.MAX_RESOLUTION, "step": 8}),
                              "frame_count": ("INT", {"default": 1, "min": 1, "max": 262144}),
                              "fps": ("INT", {"default": 1, "min": 1, "max": 120}),
                              "source": ("STRING", {"default": SHADERTOY_DEFAULT, "multiline": True, "dynamicPrompts": False})},
                "optional": { "channel_0": ("IMAGE",),
                              "channel_1": ("IMAGE",),
                              "channel_2": ("IMAGE",),
                              "channel_3": ("IMAGE",)}}
    
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "Audio Reactor"
    FUNCTION = "render"

    def render(self, width: int, height: int, frame_count: int, fps: int, source: str, 
               channel_0: torch.Tensor|None=None, channel_1: torch.Tensor|None=None,
               channel_2: torch.Tensor|None=None, channel_3: torch.Tensor|None=None):
        fragment_source = SHADERTOY_HEADER
        fragment_source += source
        fragment_source += SHADERTOY_FOOTER

        ctx, fbo, shader, textures = setup_render_resources(width, height, fragment_source)

        images = []
        frame = 0
        for _ in range(frame_count):
            shadertoy_vars_update(shader, width, height, frame * (1.0 / fps), (1.0 / fps), fps, frame)
            if channel_0 != None: shadertoy_texture_update(textures[0], channel_0, frame)
            if channel_1 != None: shadertoy_texture_update(textures[1], channel_1, frame)
            if channel_2 != None: shadertoy_texture_update(textures[2], channel_2, frame)
            if channel_3 != None: shadertoy_texture_update(textures[3], channel_3, frame)
            shadertoy_texture_bind(shader, textures)

            image = render(width, height, fbo, shader)
            image = torch.from_numpy(image)[None,]
            images.append(image)

            frame += 1
        
        render_resources_cleanup(ctx)

        return (torch.cat(images, dim=0),)

class AudioLoadPath:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "path": ("STRING", {"default": "X://insert/path/here.mp4"}),
                              "sample_rate": ("INT", {"default": 22050, "min": 6000, "max": 192000, "step": 1}),
                              "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6, "step": 0.001}),
                              "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6, "step": 0.001})}}
    
    RETURN_TYPES = ("AUDIO", )
    CATEGORY = "Audio Reactor"
    FUNCTION = "load"

    def load(self, path: str, sample_rate: int, offset: float, duration: float|None):
        if duration == 0.0: duration = None
        audio, _ = librosa.load(path, sr=sample_rate, offset=offset, duration=duration)
        audio = torch.from_numpy(audio)[None,:,None]
        return (audio,)

class AudioFrameTransformShadertoy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO",),
                              "sample_rate": ("INT", {"default": 22050, "min": 6000, "max": 192000, "step": 1}),
                              "frame_count": ("INT", {"default": 1, "min": 1, "max": 262144}),
                              "fps": ("INT", {"default": 1, "min": 1, "max": 120})}}

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "Audio Reactor"
    FUNCTION = "transform"

    def transform(self, audio: torch.Tensor, sample_rate: int, frame_count: int, fps: int):
        timestamps = np.arange(0, frame_count) * (1 / fps)

        samples = audio.cpu().numpy()[0, :, 0]
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=48000)

        # referencing https://gist.github.com/soulthreads/2efe50da4be1fb5f7ab60ff14ca434b8
        # frames and smoothed fft
        frame_idxs = librosa.time_to_frames(timestamps, sr=48000, hop_length=512)
        frames = librosa.util.frame(samples, frame_length=2048, hop_length=512, axis=0)[frame_idxs, :]
        blackman = librosa.filters.get_window("blackman", 2048, fftbins=True)
        fft_frames = np.fft.rfft(frames * blackman[None,], axis=1)
        fft_frames = np.abs(fft_frames) / 2048
        fft_smoothed = np.zeros_like(fft_frames)
        k_factor = 0.8 ** (60 / fps)
        fft_smoothed[0] = (1 - k_factor) * fft_frames[0]
        for i in range(1, fft_frames.shape[0]): fft_smoothed[i] = k_factor * fft_smoothed[i - 1] + (1 - k_factor) * fft_frames[i]
        fft_frames = 20 * np.log10(fft_smoothed) # type: ignore

        # conversion
        frames = np.clip(0.5 * (1 + frames), 0, 1) # type: ignore
        fft_min = -100
        fft_max = -30
        fft_frames = np.divide(fft_frames - fft_min, fft_max - fft_min, out=np.zeros_like(fft_frames), where=fft_max!=fft_min) # type: ignore
        fft_frames = np.clip(fft_frames, 0, 1) # type: ignore

        frames = torch.from_numpy(frames[:, :512]).unsqueeze(-1).expand(-1, -1, 3)
        fft_frames = torch.from_numpy(fft_frames[:, :512]).unsqueeze(-1).expand(-1, -1, 3)
        return (torch.cat([frames.unsqueeze(1), fft_frames.unsqueeze(1)], dim=1),)

class AudioFrameTransformBeats:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO",),
                              "sample_rate": ("INT", {"default": 22050, "min": 6000, "max": 192000, "step": 1}),
                              "frame_count": ("INT", {"default": 1, "min": 1, "max": 262144}),
                              "fps": ("INT", {"default": 1, "min": 1, "max": 120})}}

    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "Audio Reactor"
    FUNCTION = "transform"

    def transform(self, audio: torch.Tensor, sample_rate: int, frame_count: int, fps: int):
        timestamps = np.arange(0, frame_count) * (1 / fps)

        samples = audio.cpu().numpy()[0, :, 0]
        tempo, beats = librosa.beat.beat_track(y=samples, sr=sample_rate, hop_length=512)

        beat_timestamps = librosa.frames_to_time(beats, sr=sample_rate, hop_length=512)
        matches = librosa.util.match_events(beat_timestamps, timestamps)

        beats = np.isin(np.arange(frame_count), matches).astype(np.float32) # type: ignore

        beats_smoothed = np.zeros_like(beats)
        k_factor = 0.8 ** (60 / fps)
        beats_smoothed[0] = beats[0]
        for i in range(1, beats.shape[0]): beats_smoothed[i] = max(k_factor * beats_smoothed[i - 1], beats[i])

        return (torch.from_numpy(beats_smoothed)[:, None, None, None].expand(-1, -1, -1, 3),)


NODE_CLASS_MAPPINGS = {
    "Shadertoy": Shadertoy,
    "AudioLoadPath": AudioLoadPath,
    "AudioFrameTransformShadertoy": AudioFrameTransformShadertoy,
    "AudioFrameTransformBeats": AudioFrameTransformBeats,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Shadertoy": "Shadertoy",
    "AudioLoadPath": "Load Audio (from Path)",
    "AudioFrameTransformShadertoy": "Audio Frame Transform (Shadertoy)",
    "AudioFrameTransformBeats": "Audio Frame Transform (Beats)",
}
