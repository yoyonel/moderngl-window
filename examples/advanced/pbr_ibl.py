"""
- https://moderngl.readthedocs.io/en/5.8.2/reference/compute_shader.html?highlight=compute%20shader#moderngl.ComputeShader.run
- https://github.com/moderngl/moderngl/issues/345
- https://www.khronos.org/opengl/wiki/Compute_Shader
- https://registry.khronos.org/OpenGL-Refpages/gl4/html/glCopyImageSubData.xhtml
- https://moderngl.readthedocs.io/en/5.8.2/reference/texture_cube.html
- https://moderngl.readthedocs.io/en/5.8.2/reference/texture.html?highlight=bind_to_image#moderngl.Texture.bind_to_image
- https://polyhaven.com/a/qwantani_noon
- https://github.com/nbertoa/BRE12/blob/master/BRE/ToneMappingPass/Shaders/PS.hlsl
- https://github.com/srcres258/learnopengl-rust/tree/master
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
logger = logging.getLogger(f"moderngl_window.exemple.pbr_ibl")


# logger.setLevel(logging.DEBUG)


class PBRWithIBL(CameraWindow):
    # class PBRWithIBL(OrbitDragCameraWindow):
    """Example Physic Base Rendering with Image Based Lighting"""

    title = "Example Physic Base Rendering with Image Based Lighting"
    resource_dir = (Path(__file__) / "../../resources").resolve()
    window_size = 1280, 720
    aspect_ratio = window_size[0] / window_size[1]
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        # self.wnd.mouse_exclusivity = False

        self.frame_time_decay_factor = 0.995
        self.average_frame_time = 0.01666

        # // Parameters
        kEnvMapSize = 512
        # ❗️avec 32 ça fait des artefacts ...
        kIrradianceMapSize = 32
        kBRDF_LUT_Size = 256
        # Precision used for render targets
        # (16 bits floats)
        dtype_precision = "f2"
        # (32 bits floats)
        # dtype_precision = "f4"

        self.cube = geometry.cube(size=(20, 20, 20))
        self.sphere = geometry.sphere(radius=5.0, sectors=64, rings=32)

        # texture_name = "limestone-cliffs"
        texture_name = "cheap-plywood1"
        texture_rootpath = f"textures/PBR/{texture_name}-bl"
        self.sphere_textures = {
            "albedo": self.load_texture_2d(f"{texture_rootpath}/{texture_name}_albedo.png"),
            "ao": self.load_texture_2d(f"{texture_rootpath}/{texture_name}_ao.png"),
            "roughness": self.load_texture_2d(f"{texture_rootpath}/{texture_name}_roughness.png"),
            "metallic": self.load_texture_2d(f"{texture_rootpath}/{texture_name}_metallic.png"),
        }

        logger.info("Load & convert equirectangular environment map to a cubemap texture.")
        # self.prog_render_hdri_env_map = self.load_program("programs/IBL/hdricube.glsl")
        self.prog_render_cubemap = self.load_program("programs/cubemap.glsl")

        # hdri_name = "qwantani_noon_4k"
        hdri_name = "klippad_sunrise_2_4k"
        # hdri_name = "newport_loft"
        path_to_image = str(PBRWithIBL.resource_dir / f"textures/hdr/{hdri_name}.exr")
        with OpenEXR.File(path_to_image) as infile:
            # assert "RGBA" in infile.channels()
            # channel = infile.channels()["RGBA"]
            channel = list(infile.channels().values())[0]
            nd_pixels = channel.pixels
            env_texture_equirect = self.ctx.texture(
                size=nd_pixels.shape[:2][::-1],
                components=nd_pixels.shape[-1],
                data=nd_pixels.tobytes(),
                dtype=f'f{nd_pixels.dtype.alignment}',
            )
        components = env_texture_equirect.components

        env_texture_unfiltered = self.ctx.texture_cube(
            size=(kEnvMapSize, kEnvMapSize),
            components=components,
            data=None,
            dtype=dtype_precision,
        )

        prog_equirect2cube = self.load_compute_shader("programs/IBL/equirect2cube.glsl")
        # config for compute shader
        w, h = env_texture_unfiltered.size
        gw, gh = 32, 32
        nx, ny, nz = int(w / gw), int(h / gh), 6
        #
        env_texture_equirect.use(0)
        env_texture_unfiltered.bind_to_image(0, read=False, write=True)
        prog_equirect2cube.run(nx, ny, nz)
        ################################################################################################################
        # RELEASE
        ################################################################################################################
        env_texture_equirect.release()
        prog_equirect2cube.release()
        ################################################################################################################
        #
        env_texture_unfiltered.build_mipmaps()

        logger.info("Compute pre-filtered specular environment map.")
        prog_spmap = self.load_compute_shader("programs/IBL/spmap.glsl")
        self.env_texture = self.ctx.texture_cube(
            size=(kEnvMapSize, kEnvMapSize),
            components=components,
            data=None,
            dtype=dtype_precision,
        )
        self.env_texture.build_mipmaps()
        # TODO: integrate into moderngl
        logger.info("Copy 0th mipmap level into destination environment map.")
        GL.glCopyImageSubData(
            env_texture_unfiltered.glo, GL.GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
            self.env_texture.glo, GL.GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0,
            *self.env_texture.size, 6
        )
        logger.info("Pre-filter rest of the mip chain.")
        levels = int(glm.log2(kEnvMapSize))
        deltaRoughness = 1.0 / max(float(levels - 1), 1.0)
        size = kEnvMapSize / 2
        for level in range(1, levels):
            numGroups = int(max(1, size / 32))
            self.env_texture.bind_to_image(1, read=False, write=True, level=level)
            env_texture_unfiltered.use(location=0)
            prog_spmap["roughnessValue"] = level * deltaRoughness
            logger.debug(f'{prog_spmap["roughnessValue"].value=}')
            prog_spmap.run(numGroups, numGroups, 6)
            size //= 2
        ################################################################################################################
        # RELEASE
        ################################################################################################################
        prog_spmap.release()
        env_texture_unfiltered.release()
        ################################################################################################################

        logger.info("Compute diffuse irradiance cubemap.")
        #
        self.irradianceMap_texture = self.ctx.texture_cube(
            size=(kIrradianceMapSize, kIrradianceMapSize),
            components=components,
            data=None,
            dtype=dtype_precision,
        )
        self.irradianceMap_texture.build_mipmaps(max_level=1)
        self.irradianceMap_texture.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        # self.irradianceMap_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        # repeat_{x|y} : False <=> GL_CLAMP_TO_EDGE
        self.irradianceMap_texture.repeat_x = False
        self.irradianceMap_texture.repeat_y = False
        self.irradianceMap_texture.repeat_z = False

        prog_irmap = self.load_compute_shader("programs/IBL/irmap.glsl")
        prog_irmap["envMap"].value = 0
        self.env_texture.use(location=0)
        self.irradianceMap_texture.bind_to_image(0, read=False, write=True, level=0)
        prog_irmap.run(int(self.irradianceMap_texture.size[0] / 32), int(self.irradianceMap_texture.size[1] / 32), 6)
        ################################################################################################################
        # RELEASE
        ################################################################################################################
        prog_irmap.release()
        ################################################################################################################

        logger.info("Compute Cook-Torrance BRDF 2D LUT for split-sum approximation.")
        prog_spBRDF = self.load_compute_shader("programs/IBL/spbrdf.glsl")
        self.spBRDF_LUT = self.ctx.texture(
            size=(kBRDF_LUT_Size, kBRDF_LUT_Size),
            components=2,
            data=None,
            dtype='f2',
        )
        # repeat_{x|y} : False <=> GL_CLAMP_TO_EDGE
        self.spBRDF_LUT.repeat_x = False
        self.spBRDF_LUT.repeat_y = False
        self.spBRDF_LUT.filter = moderngl.LINEAR, moderngl.LINEAR
        self.spBRDF_LUT.bind_to_image(0, read=False, write=True, level=0)
        prog_spBRDF.run(int(self.spBRDF_LUT.size[0] / 32), int(self.spBRDF_LUT.size[1] / 32), 1)
        ################################################################################################################
        # RELEASE
        ################################################################################################################
        prog_spBRDF.release()
        ################################################################################################################

        logger.info("Wait for all drawing commands to finish ...")
        self.ctx.finish()

        assert GL.glGetError() == GL.GL_NO_ERROR

        logger.info("Initialization complete.")

        # Load the shading program
        # PBR/IBL
        self.prog_pbr_ibl_light = self.load_program("programs/IBL/ibl_light.glsl")
        if "prefilteredEnvMap" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["prefilteredEnvMap"].value = 0
        if "diffuseIrradianceMap" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["diffuseIrradianceMap"].value = 1
        if "brdfConvolutionMap" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["brdfConvolutionMap"].value = 2
        # Scene Material
        if "textureAlbedo" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["textureAlbedo"].value = 3
        if "textureAmbientOcclusion" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["textureAmbientOcclusion"].value = 4
        if "textureRoughness" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["textureRoughness"].value = 5
        if "textureMetallic" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["textureMetallic"].value = 6
        # Light Material
        if "lightColor" in self.prog_pbr_ibl_light:
            self.prog_pbr_ibl_light["lightColor"].value = glm.vec3(243.0, 242.0, 240.0) / 255.0
            logger.info(f'lightColor={self.prog_pbr_ibl_light["lightColor"].value}')

        # Set up imgui.
        imgui.create_context()
        if self.wnd.ctx.error != "GL_NO_ERROR":
            print(self.wnd.ctx.error)
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.imgui.io.want_capture_mouse = False

        self.camera.look_at(glm.vec3(0, 0, 0), (0, -5, -32))

    def on_render(self, time, frame_time):
        self.average_frame_time = (
            self.frame_time_decay_factor * self.average_frame_time
            + (1.0 - self.frame_time_decay_factor) * frame_time
        )

        # self.ctx.enable_only(moderngl.CULL_FACE)
        # self.ctx.front_face = "cw"

        skybox_cam = self.camera.matrix
        # Purge camera translation
        skybox_cam[3][0] = 0
        skybox_cam[3][1] = 0
        skybox_cam[3][2] = 0

        # environment (mip)map
        # self.env_texture_unfiltered.use(location=0)
        #
        # pre-filtered specular environment map
        self.env_texture.use(location=0)
        #
        # diffuse irradiance cubemap
        # self.irradianceMap_texture.use(location=0)
        #
        # split-sum BRDF lookup table
        # self.spBRDF_LUT.use(location=0)

        self.prog_render_cubemap["m_proj"].write(self.camera.projection.matrix)
        self.prog_render_cubemap["m_camera"].write(skybox_cam)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.cube.render(self.prog_render_cubemap)

        # positionnement de la sphère dans la scène
        scene_pos = glm.vec3(0, -5, -32)

        self.wnd.use()

        self.prog_pbr_ibl_light["m_proj"].write(self.camera.projection.matrix)
        self.prog_pbr_ibl_light["m_camera"].write(self.camera.matrix)
        self.prog_pbr_ibl_light["m_model"].write(glm.translate(glm.vec3(scene_pos)))

        # if "cameraPosition" in self.prog_pbr_ibl_light:
        self.prog_pbr_ibl_light["cameraPosition"].value = self.camera.position
        # if "lightPosition" in self.prog_pbr_ibl_light:
        self.prog_pbr_ibl_light["lightPosition"].value = glm.vec3(10.0, 0.0, 0.0)

        # diffuse irradiance cubemap
        self.env_texture.use(location=0)
        self.irradianceMap_texture.use(location=1)
        self.spBRDF_LUT.use(location=2)
        # scene material
        if "textureAlbedo" in self.prog_pbr_ibl_light:
            self.sphere_textures["albedo"].use(location=self.prog_pbr_ibl_light["textureAlbedo"].value)
        if "textureMetallic" in self.prog_pbr_ibl_light:
            self.sphere_textures["metallic"].use(location=self.prog_pbr_ibl_light["textureMetallic"].value)
        if "textureRoughness" in self.prog_pbr_ibl_light:
            self.sphere_textures["roughness"].use(location=self.prog_pbr_ibl_light["textureRoughness"].value)
        if "textureAmbientOcclusion" in self.prog_pbr_ibl_light:
            self.sphere_textures["ao"].use(location=self.prog_pbr_ibl_light["textureAmbientOcclusion"].value)

        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.sphere.render(self.prog_pbr_ibl_light)

        assert GL.glGetError() == GL.GL_NO_ERROR

        self.render_ui()

    def on_resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def render_ui(self):
        imgui.new_frame()

        imgui.begin("Debug Panel", True)
        imgui.text(f"Frame time: {1000.0 * self.average_frame_time:.1f} ms")
        imgui.text(f"FPS: {1.0 / self.average_frame_time:.1f}")

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
    PBRWithIBL.run()
