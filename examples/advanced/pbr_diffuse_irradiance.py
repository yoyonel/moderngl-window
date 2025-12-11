"""
- https://learnopengl.com/PBR/IBL/Diffuse-irradiance
"""
import logging
from pathlib import Path

import OpenEXR
import glm
import moderngl
from OpenGL import GL
from imgui_bundle import imgui

from base import CameraWindow
# from base import OrbitDragCameraWindow
from moderngl_window import geometry
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer

# inherit from moderngl_window root logger
logger = logging.getLogger("moderngl_window.exemple.pbr_direct_lighting")


# logger.setLevel(logging.DEBUG)


class PBRWithDiffuseIrradiance(CameraWindow):
    """Example Physic Base Rendering with Diffuse Irradiance"""

    title = "Example Physic Base Rendering with Diffuse Irradiance"
    resource_dir: Path = (Path(__file__) / "../../resources").resolve()
    window_size = 1280, 720
    aspect_ratio = window_size[0] / window_size[1]
    vsync = False
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True

        self.frame_time_decay_factor = 0.995
        self.average_frame_time = 0.01666

        self.cube = geometry.cube(size=(100, 100, 100))

        # // pbr: load the HDR environment map
        # // ---------------------------------
        # hdri_name = "qwantani_noon_4k"
        # hdri_name = "klippad_sunrise_2_4k"
        # https://polyhaven.com/a/klippad_dawn_2
        hdri_name = "klippad_dawn_2_4k"
        self.hdr_texture = self.load_hdr_env_map(hdri_name)
        # // pbr: convert HDR equirectangular environment map to cubemap equivalent
        # // ----------------------------------------------------------------------
        self.env_cubemap = self.build_env_cubemap(self.hdr_texture, size=1024)
        # // pbr: create an irradiance cubemap, and re-scale capture FBO to irradiance scale.
        # // --------------------------------------------------------------------------------
        # // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
        # // -----------------------------------------------------------------------------
        self.irradiance_map_cubemap = self.build_irradiance_cubemap_cs(self.env_cubemap, size=32)

        self.sphere = geometry.sphere(radius=1.0, sectors=32, rings=32)
        self.prog_pbr_lighting = self.load_program("programs/PBR/pbr_diffuse_irradiance.glsl")
        self.prog_pbr_lighting["albedo"] = (0.5, 0.0, 0.0)
        try:
            self.prog_pbr_lighting["ao"] = 1.0
        except KeyError:
            pass

        self.backgroundShader = self.load_program("programs/PBR/background.glsl")

        # Set up imgui.
        imgui.create_context()
        if self.wnd.ctx.error != "GL_NO_ERROR":
            logger.error(self.wnd.ctx.error)
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.imgui.io.want_capture_mouse = False

        self.camera.set_position(0, 0, 10)

    def on_render(self, time, frame_time):
        self.average_frame_time = (
            self.frame_time_decay_factor * self.average_frame_time
            + (1.0 - self.frame_time_decay_factor) * frame_time
        )

        self.ctx.enable_only(moderngl.DEPTH_TEST)

        self.wnd.use()

        self.render_spheres()

        self.render_skybox(self.env_cubemap)
        # self.render_skybox(self.irradiance_map_cubemap)

        assert GL.glGetError() == GL.GL_NO_ERROR

        self.render_ui()

    def load_hdr_env_map(self, hdri_name: str) -> moderngl.Texture:
        path_to_image = (
            PBRWithDiffuseIrradiance.resource_dir /
            f"textures/hdr/{hdri_name}.exr"
        ).as_posix()
        logger.info(f"Loading: {path_to_image}")
        with OpenEXR.File(path_to_image) as infile:
            channel = list(infile.channels().values())[0]
            nd_pixels = channel.pixels
            result = self.ctx.texture(
                size=nd_pixels.shape[:2][::-1],
                components=nd_pixels.shape[-1],
                data=nd_pixels.tobytes(),
                dtype=f'f{nd_pixels.dtype.alignment}',
            )
            result.repeat_x = True
            result.repeat_y = True
            result.filter = moderngl.LINEAR, moderngl.LINEAR
        return result

    def build_env_cubemap(
        self,
        hdr_texture: moderngl.Texture,
        size: int = 512,
        dtype_precision="f2",
    ) -> moderngl.TextureCube:
        result = self.ctx.texture_cube(
            size=(size, size),
            components=hdr_texture.components,
            data=None,
            dtype=dtype_precision,
        )

        prog_equirect2cube = self.load_compute_shader("programs/IBL/equirect2cube.glsl")
        # config for compute shader
        w, h = result.size
        gw, gh = 32, 32
        nx, ny, nz = int(w / gw), int(h / gh), 6
        #
        hdr_texture.use(0)
        result.bind_to_image(0, read=False, write=True)
        prog_equirect2cube.run(nx, ny, nz)

        # RELEASE
        prog_equirect2cube.release()

        return result

    def build_irradiance_cubemap_cs(
        self,
        env_cubemap: moderngl.TextureCube,
        size: int = 32,
        dtype_precision: str = "f2",
    ) -> moderngl.TextureCube:
        """Compute Irradiance Diffuse Map with Compute Shader on CubeMap"""
        irradiance_map_texture = self.ctx.texture_cube(
            size=(size, size),
            components=env_cubemap.components,
            data=None,
            dtype=dtype_precision,
        )

        # repeat_{x|y|z} : False <=> GL_CLAMP_TO_EDGE
        irradiance_map_texture.repeat_x = False
        irradiance_map_texture.repeat_y = False
        irradiance_map_texture.repeat_z = False
        #
        irradiance_map_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        # irradiance_map_texture.filter = moderngl.NEAREST, moderngl.NEAREST

        irradiance_map_shader = self.load_compute_shader("programs/IBL/irmap.glsl")
        # config for compute shader
        w, h = irradiance_map_texture.size
        gw, gh = min(32, irradiance_map_texture.size[0]), min(32, irradiance_map_texture.size[1])
        nx, ny, nz = int(w / gw), int(h / gh), 6
        #
        env_cubemap.use(location=0)
        irradiance_map_texture.bind_to_image(0, read=False, write=True)
        irradiance_map_shader.run(nx, ny, nz)

        # RELEASE
        irradiance_map_shader.release()

        return irradiance_map_texture

    def build_irradiance_cubemap_rtt(
        self,
        env_cubemap: moderngl.TextureCube,
        size: int = 32,
        dtype_precision: str = "f2",
    ):
        """Doesn't work, ModernGL seems not to have Render To Target active for rendering in CubeMap ... """
        result = self.ctx.texture_cube(
            size=(size, size),
            components=env_cubemap.components,
            data=None,
            dtype=dtype_precision,
        )

        # repeat_{x|y|z} : False <=> GL_CLAMP_TO_EDGE
        result.repeat_x = False
        result.repeat_y = False
        result.repeat_z = False
        #
        result.filter = moderngl.LINEAR, moderngl.LINEAR

        irradianceMap = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, irradianceMap)
        for i in range(6):
            GL.glTexImage2D(
                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                GL.GL_RGB16F,
                32, 32,
                0,
                GL.GL_RGB,
                GL.GL_FLOAT,
                None
            )

        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        captureFBO = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, captureFBO)

        # // pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
        # // ----------------------------------------------------------------------------------------------
        captureProjection = glm.perspective(glm.radians(90.0), 1.0, 0.1, 10.0)
        captureViews = [
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0), glm.vec3(0.0, 0.0, 1.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0), glm.vec3(0.0, 0.0, -1.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0), glm.vec3(0.0, -1.0, 1.0)),
            glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, -1.0), glm.vec3(0.0, -1.0, 0.0)),
        ]

        irradianceShader = self.load_program("programs/PBR/irradiance_convolution.glsl")
        irradianceShader["environmentMap"].value = 0
        irradianceShader["projection"].write(captureProjection)

        GL.glViewport(0, 0, 32, 32)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, captureFBO)
        env_cubemap.use()
        for i, captureView in enumerate(captureViews):
            irradianceShader["view"].write(captureView)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                irradianceMap,
                0
            )
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self.cube.render(irradianceShader)

        result.glo = irradianceMap

        return result

    def render_spheres(self):
        self.prog_pbr_lighting["projection"].write(self.camera.projection.matrix)
        self.prog_pbr_lighting["view"].write(self.camera.matrix)
        self.prog_pbr_lighting["camPos"].write(self.camera.position)

        self.irradiance_map_cubemap.use(location=0)

        nrRows = 7
        nrColumns = 7
        spacing = 2.5
        # // render rows*column number of spheres with varying metallic/roughness values scaled by rows and columns respectively
        for row in range(nrRows):
            self.prog_pbr_lighting["metallic"].value = float(row) / float(nrRows)
            for col in range(nrColumns):
                # // we clamp the roughness to 0.05 - 1.0 as perfectly smooth surfaces (roughness of 0.0) tend to look a bit off
                # // on direct lighting.
                try:
                    self.prog_pbr_lighting["roughness"].value = glm.clamp(float(col) / float(nrColumns), 0.05, 1.0)
                except:
                    pass
                # logger.debug(f'{self.prog_pbr_lighting["roughness"].value=}')

                model = glm.mat4(1.0)
                model = glm.translate(model, glm.vec3(
                    (col - (nrColumns / 2)) * spacing,
                    (row - (nrRows / 2)) * spacing,
                    0.0
                ))
                self.prog_pbr_lighting["model"].write(model)
                self.prog_pbr_lighting["normalMatrix"].write(glm.transpose(glm.inverse(glm.mat3(model))))

                self.sphere.render(self.prog_pbr_lighting)

    def render_skybox(self, cubemap: moderngl.TextureCube):
        skybox_cam = self.camera.matrix
        # Purge camera translation
        skybox_cam[3][0] = 0
        skybox_cam[3][1] = 0
        skybox_cam[3][2] = 0

        # self.ctx.disable(moderngl.DEPTH_TEST)
        self.backgroundShader["m_proj"].write(self.camera.projection.matrix)
        self.backgroundShader["m_camera"].write(skybox_cam)
        cubemap.use(location=0)
        self.cube.render(self.backgroundShader)

    def on_resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def render_ui(self):
        imgui.new_frame()

        imgui.begin("Debug Panel", True)
        imgui.text(f"Frame time: {1000.0 * self.average_frame_time:.1f} ms")
        imgui.text(f"FPS: {1.0 / self.average_frame_time:.1f}")
        imgui.text(f"Camera Position: {str(self.camera.position)}")

        imgui.end()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)
        if not self.imgui.io.want_capture_mouse:
            super().on_mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x: int, y: int, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)
        if not self.imgui.io.want_capture_mouse:
            super().on_mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)
        if not self.imgui.io.want_capture_mouse:
            super().on_mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        self.imgui.mouse_release_event(x, y, button)

    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)
        if not self.imgui.io.want_capture_keyboard:
            super().on_key_event(key, action, modifiers)


if __name__ == "__main__":
    PBRWithDiffuseIrradiance.run()
