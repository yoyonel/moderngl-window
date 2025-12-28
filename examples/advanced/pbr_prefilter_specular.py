"""
- https://learnopengl.com/PBR/IBL/Diffuse-irradiance
- https://google.github.io/filament/main/filament.html

TODO:
- replace triangulates spheres with billboards
"""

import functools
import logging
import json
import pathlib
import time
from pathlib import Path
from typing import Optional, Callable, Any, Final

import OpenEXR
import glm
import moderngl
from OpenGL import GL
from imgui_bundle import imgui

from base import CameraWindow
from moderngl_window import geometry
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer

# inherit from moderngl_window root logger
logger = logging.getLogger("moderngl_window.exemple.pbr_prefilter_specular")


# logger.setLevel(logging.DEBUG)


class PBRWithPrefilteredSpecular(CameraWindow):
    """Example Physic Base Rendering with Prefiltered Specular"""

    title = "Example Physic Base Rendering with Prefiltered Specular"
    gl_version = (4, 5)
    window_size = 1280, 720
    # window_size = 1920, 1080
    aspect_ratio = None
    resizable = True
    vsync = False
    samples = 8
    resource_dir: Path = (Path(__file__) / "../../resources").resolve()

    # exhib some bugs (dark pixel) at around spheres if the size is too low (for example size=32)
    res_for_brdf_lut: Final[int] = 512
    # env cubemap resolution for skybox rendering
    res_for_env_map_hires: Final[int] = 2048
    # env cubemap resolution for IBL computations
    res_for_env_map: Final[int] = 512
    # env cubemap resolution for precomputing irradiance convolutions
    res_for_irradiance_map: Final[int] = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera.projection.update(near=0.001, far=100)

        self.MATERIAL_PRESETS = self._load_material_presets()
        self.ui_render_mode_options = [
            "Grid (Metallic/Roughness Interpolation)",
            "Material Presets",
        ]
        self.ui_render_mode = 0
        self.ui_show_labels = True

        self.wnd.mouse_exclusivity = True
        self.wnd.fullscreen_key = self.wnd.keys.F

        self.frame_time_decay_factor = 0.995
        self.average_frame_time = 0.01666

        self.cube = geometry.cube(size=(100, 100, 100))
        self.ui_sphere_subdivisions = 3
        self.ui_sphere_irregularity = 0.0
        self.sphere = geometry.icosphere(
            radius=1.0,
            subdivisions=self.ui_sphere_subdivisions,
            randomization=self.ui_sphere_irregularity,
        )
        self.quad = geometry.quad_2d(size=(2.0, 2.0))
        self.ui_use_billboarding = True
        # with cubes no black pixels problem, certainly a problem a mesh definition/precision
        # self.sphere = geometry.cube(size=(2.0, 2.0, 2.0))

        self.hdr_texture: Optional[moderngl.Texture] = None
        self.env_cubemap_hires: Optional[moderngl.TextureCube] = None
        self.env_cubemap: Optional[moderngl.TextureCube] = None
        self.prefiltered_specular_map: Optional[moderngl.TextureCube] = None
        self.irradiance_map_cubemap: Optional[moderngl.TextureCube] = None

        # Load Compute Shaders once (before use in precompute)
        self.prog_equirect2cube = self.load_compute_shader("programs/IBL/equirect2cube.glsl")
        self.prog_spmap = self.load_compute_shader("programs/IBL/spmap.glsl")
        self.prog_irmap = self.load_compute_shader("programs/IBL/irmap.glsl")

        # // pbr: generate a (static/constant) 2D LUT from the BRDF equations used.
        self.brdf_lut_texture = self.build_brdf_lut_texture(
            size=PBRWithPrefilteredSpecular.res_for_brdf_lut
        )

        self.argv.path_to_hdr_env_map = self.argv.path_to_hdr_env_map.expanduser().resolve()
        assert self.argv.path_to_hdr_env_map.exists(), self.argv.path_to_hdr_env_map
        self.hdri_names = [
            hdri_path.stem for hdri_path in self.argv.path_to_hdr_env_map.glob("*.exr")
        ]
        self.ui_hdri_id = 0

        self.ui_irradiance_method_options = ["Monte Carlo (Fast)", "Convolution (Stable)"]
        self.ui_irradiance_method = 1  # 0: Monte Carlo, 1: Convolution
        self.ui_irradiance_clamp = (
            50.0  # Default higher than 1.0 to preserve HDR but limit huge spikes
        )

        self.elapsed_time = 0.0
        self.precompute_from_hdr_env_map(self.hdri_names[self.ui_hdri_id])
        logger.info(f"Time spent on the GPU: {self.elapsed_time / 1_000_000.0:.2f} ms")

        self.prog_pbr_lighting = self.load_program("programs/PBR/pbr_prefiltered_specular.glsl")
        self.prog_pbr_lighting["irradianceMap"].value = 0
        self.prog_pbr_lighting["prefilterMap"].value = 1
        self.prog_pbr_lighting["brdfLUT"].value = 2

        self.ui_albedo = [0.50, 0.50, 0.50]
        self.ui_ao = 1.0
        self.ui_exposure = 1.2
        self.prog_pbr_lighting["albedo"] = self.ui_albedo
        self.prog_pbr_lighting["ao"] = self.ui_ao
        self.prog_pbr_lighting["pbr_exposure"] = self.ui_exposure
        self.prog_pbr_lighting["use_billboarding"] = self.ui_use_billboarding

        self.backgroundShader = self.load_program("programs/PBR/background.glsl")
        self.backgroundShader["environmentMap"].value = 0
        self.backgroundShader["environmentMap"].value = 0
        self.backgroundShader["blur_lod"].value = 0.0

        self.backgroundShader["blur_lod"].value = 0.0

        self.ui_nr_rows = 7
        self.ui_nr_columns = 7
        self.ui_spacing = 2.5

        # Set up imgui.
        imgui.create_context()
        if self.wnd.ctx.error != "GL_NO_ERROR":
            logger.error(self.wnd.ctx.error)
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.imgui.io.want_capture_mouse = False
        self._update_imgui_resources()

        self.camera.set_position(0, 0, 10)

        self.ui_skybox_enabled = True
        self.ui_clear_color = (0, 1, 0)
        self.clear_color = self.ui_clear_color
        self.ui_wireframe_enabled = False

        self.ui_visualization_modes = ["Standard", "False Color (Luminance)"]
        self.ui_visualization_mode = 0

        self.ui_debug_skybox_options = ["High Res", "Low Res", "Irradiance", "Prefilter"]
        self.ui_debug_skybox_id = 0

        # Uniform state trackers to avoid GPU stalls via .value
        self._last_exposure = self.ui_exposure
        self._last_debug_mode = self.ui_visualization_mode
        self._last_albedo = tuple(self.ui_albedo)
        self._last_ao = self.ui_ao
        self._last_use_billboarding = self.ui_use_billboarding

    @classmethod
    def add_arguments(cls, parser):
        # Mandatory positional argument for the file to load
        parser.add_argument(
            "--path_to_hdr_env_map",
            type=pathlib.Path,
            default=(cls.resource_dir / "textures/hdr/"),
            help="Path to HDRi Environment Map",
        )

    @staticmethod
    def gl_time_elapsed(ogl_func: Callable[[Any, ...], Any]):
        """
        Decorator to perform a OpenGL timer query
        Result:
            side effect on self.elapsed_time:
        """

        @functools.wraps(ogl_func)
        def wrap(self, *args, **kwargs):
            # https://www.lighthouse3d.com/tutorials/opengl-timer-query/
            # // generate two queries
            query = GL.glGenQueries(1)[0]
            GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
            returned_value = ogl_func(self, *args, **kwargs)
            GL.glEndQuery(GL.GL_TIME_ELAPSED)
            # // wait until the results are available
            stopTimerAvailable = 0
            while not stopTimerAvailable:
                stopTimerAvailable = GL.glGetQueryObjectiv(query, GL.GL_QUERY_RESULT_AVAILABLE)
            # // get query results
            # UNSIGNED INT 32 bits work :-)
            self.elapsed_time = GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT)
            return returned_value

        return wrap

    def _release_textures(self):
        for ogl_object in (
            self.hdr_texture,
            self.env_cubemap_hires,
            self.env_cubemap,
            self.prefiltered_specular_map,
            self.irradiance_map_cubemap,
        ):
            if ogl_object is not None:
                ogl_object.release()
                assert type(ogl_object.mglo) is moderngl.mgl.InvalidObject
                ogl_object = None

    # @gl_time_elapsed # Removed decorator to have fine grained profiling
    def precompute_from_hdr_env_map(
        self,
        hdri_name: str,
        release: bool = True,
        wait_for_finish: bool = True,
    ):
        if release:
            self._release_textures()

        # Measure CPU/IO time for loading EXR
        t0 = time.perf_counter()
        # // pbr: load the HDR environment map
        # // ---------------------------------
        self.hdr_texture = self.load_hdr_env_map(hdri_name)
        t1 = time.perf_counter()
        logger.info(f"Asset Loading Time (CPU + Upload): {(t1 - t0) * 1000:.2f} ms")

        # Measure GPU Compute time for PBR generation
        query = GL.glGenQueries(1)[0]
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)

        # build hires cubemap for the skybox from HDR equirectangular environment map
        self.env_cubemap_hires = self.build_env_cubemap(
            self.hdr_texture, size=PBRWithPrefilteredSpecular.res_for_env_map_hires
        )
        # self.ctx.finish() # Remove intermediate finish to let driver pipeline

        # // pbr: convert HDR equirectangular environment map to cubemap equivalent
        # // ----------------------------------------------------------------------
        self.env_cubemap = self.build_env_cubemap(
            self.hdr_texture, size=PBRWithPrefilteredSpecular.res_for_env_map
        )
        self.env_cubemap.build_mipmaps()
        # self.ctx.finish()

        # // pbr: create a pre-filter cubemap, and re-scale capture FBO to pre-filter scale.
        # // --------------------------------------------------------------------------------
        self.prefiltered_specular_map = self.build_prefiltered_specular_map(self.env_cubemap)
        # self.ctx.finish()

        # // pbr: create an irradiance cubemap, and re-scale capture FBO to irradiance scale.
        # // --------------------------------------------------------------------------------
        # // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
        # // -----------------------------------------------------------------------------
        self.irradiance_map_cubemap = self.build_irradiance_cubemap(
            self.env_cubemap, size=PBRWithPrefilteredSpecular.res_for_irradiance_map
        )
        # self.ctx.finish()

        GL.glEndQuery(GL.GL_TIME_ELAPSED)

        if wait_for_finish:
            # logger.info("Wait for all computing commands to finish ...")
            self.ctx.finish()

        # Retrieve query result
        elapsed_gpu = GL.glGetQueryObjectuiv(query, GL.GL_QUERY_RESULT)
        self.elapsed_time = elapsed_gpu
        logger.info(f"PBR Generation Time (GPU Compute): {self.elapsed_time / 1_000_000.0:.2f} ms")
        GL.glDeleteQueries(1, [query])

    def _load_material_presets(self):
        """Load material presets from a JSON resource file."""
        resource_path = self.resource_dir / "materials" / "pbr_materials.json"
        try:
            with open(resource_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load material presets: {e}")
            # Fallback to a minimal list if loading fails
            return [
                {"name": "Default", "albedo": [0.5, 0.5, 0.5], "metallic": 0.0, "roughness": 0.5}
            ]

    def on_render(self, time, frame_time):
        self.average_frame_time = (
            self.frame_time_decay_factor * self.average_frame_time
            + (1.0 - self.frame_time_decay_factor) * frame_time
        )

        self.ctx.enable_only(moderngl.DEPTH_TEST)

        self.wnd.use()

        self.ctx.clear(*self.clear_color)

        if self.ui_wireframe_enabled:
            self.ctx.wireframe = True
        self.render_spheres()
        if self.ui_wireframe_enabled:
            self.ctx.wireframe = False

        if self.ui_skybox_enabled:
            # Debug Skybox selection
            if self.ui_debug_skybox_id == 0:
                self.render_skybox(self.env_cubemap_hires, lod=0.0)
            elif self.ui_debug_skybox_id == 1:
                self.render_skybox(
                    self.env_cubemap, lod=1.0
                )  # Show a bit of blur/mip level 1 for LowRes
            elif self.ui_debug_skybox_id == 2:
                self.render_skybox(
                    self.irradiance_map_cubemap, lod=0.0
                )  # Irradiance map has no mips
            elif self.ui_debug_skybox_id == 3:
                self.render_skybox(
                    self.prefiltered_specular_map, lod=1.0
                )  # Show a mip level for prefilter

        assert self.ctx.error == "GL_NO_ERROR", self.ctx.error

        self.render_ui()

    def load_hdr_env_map(self, hdri_name: str) -> moderngl.Texture:
        path_to_hdr_env_map = (self.argv.path_to_hdr_env_map / f"{hdri_name}.exr").as_posix()
        logger.info(f"Loading: {path_to_hdr_env_map}")
        with OpenEXR.File(path_to_hdr_env_map) as infile:
            channel = list(infile.channels().values())[0]
            nd_pixels = channel.pixels
            result = self.ctx.texture(
                size=nd_pixels.shape[:2][::-1],
                components=nd_pixels.shape[-1],
                data=nd_pixels.tobytes(),
                dtype=f"f{nd_pixels.dtype.alignment}",
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
            # https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)
            components=4,
            data=None,
            dtype=dtype_precision,
        )

        # prog_equirect2cube = self.load_compute_shader("programs/IBL/equirect2cube.glsl")
        # config for compute shader
        w, h = result.size
        gw, gh = 32, 32
        nx, ny, nz = int(w / gw), int(h / gh), 6
        #
        hdr_texture.use(0)
        result.bind_to_image(1, read=False, write=True)
        self.prog_equirect2cube.run(nx, ny, nz)

        # RELEASE
        # prog_equirect2cube.release()

        return result

    def build_irradiance_cubemap(
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

        # irradiance_map_shader = self.load_compute_shader("programs/IBL/irmap.glsl")
        # config for compute shader
        w, h = irradiance_map_texture.size
        gw, gh = min(32, irradiance_map_texture.size[0]), min(32, irradiance_map_texture.size[1])
        nx, ny, nz = int(w / gw), int(h / gh), 6
        #
        env_cubemap.use(location=0)
        irradiance_map_texture.bind_to_image(unit=1, read=False, write=True)
        # uniform int method; 0: Monte Carlo, 1: Convolution
        self.prog_irmap["method"] = self.ui_irradiance_method
        self.prog_irmap["max_intensity"] = self.ui_irradiance_clamp
        self.prog_irmap.run(nx, ny, nz)

        # RELEASE
        # irradiance_map_shader.release()

        return irradiance_map_texture

    def build_prefiltered_specular_map(
        self,
        env_cubemap: moderngl.TextureCube,
        dtype_precision: str = "f2",
    ):
        prefiltered_specular_texture = self.ctx.texture_cube(
            size=env_cubemap.size,
            components=env_cubemap.components,
            data=None,
            dtype=dtype_precision,
        )
        # repeat_{x|y|z} : False <=> GL_CLAMP_TO_EDGE
        prefiltered_specular_texture.repeat_x = False
        prefiltered_specular_texture.repeat_y = False
        prefiltered_specular_texture.repeat_z = False
        # // enable pre-filter mipmap sampling (combatting visible dots artifact)
        prefiltered_specular_texture.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR

        # // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
        prefiltered_specular_texture.build_mipmaps()

        # // pbr: run a quasi monte-carlo simulation on the environment lighting
        # // to create a prefilter (cube)map.
        # // --------------------------------------------------------------------------------------
        # TODO: integrate into moderngl
        logger.info("Copy 0th mipmap level into destination environment map.")
        self.ctx.copy_texture_cube(prefiltered_specular_texture, env_cubemap)
        assert self.ctx.error == "GL_NO_ERROR", self.ctx.error

        logger.info("Pre-filter rest of the mip chain.")
        levels = int(glm.log2(env_cubemap.size[0]))
        delta_roughness = 1.0 / max(float(levels), 1.0)
        mipmap_size = env_cubemap.size[0] // 2
        mipmap_size = mipmap_size
        for level in range(1, levels + 1):
            logger.debug(f"Level {level}")
            logger.debug(f'{self.prog_spmap["roughnessValue"].value=}')

            self.prog_spmap["roughnessValue"] = level * delta_roughness

            # config for compute shader
            w, h = mipmap_size, mipmap_size
            gw, gh = 32, 32
            # Ensure at least one workgroup is dispatched even for small mip levels (< 32px)
            nx, ny, nz = (w + gw - 1) // gw, (h + gh - 1) // gh, 6
            #
            env_cubemap.use(location=0)
            prefiltered_specular_texture.bind_to_image(1, read=False, write=True, level=level)
            self.prog_spmap.run(nx, ny, nz)

            mipmap_size //= 2

        # RELEASE
        # compute_shader.release()

        return prefiltered_specular_texture

    def build_brdf_lut_texture(
        self,
        size: int = 512,
        dtype_precision: str = "f2",
    ):
        # // pre-allocate enough memory for the LUT texture.
        brdf_lut_texture = self.ctx.texture((size, size), 2, dtype=dtype_precision)
        # // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
        brdf_lut_texture.repeat_x = False
        brdf_lut_texture.repeat_y = False
        #
        brdf_lut_texture.filter = moderngl.LINEAR, moderngl.LINEAR

        brdf_shader = self.load_compute_shader("programs/IBL/spbrdf.glsl")
        # config for compute shader
        w, h = size, size
        gw, gh = 32, 32
        nx, ny, nz = int(w / gw), int(h / gh), 1
        #
        brdf_lut_texture.bind_to_image(0, read=False, write=True, level=0)
        brdf_shader.run(nx, ny, nz)
        # RELEASE
        brdf_shader.release()

        return brdf_lut_texture

    def _update_imgui_resources(self):
        # We need to register textures for ImGui
        # This is a bit manual, but required by moderngl-window's integration
        self.imgui.register_texture(self.brdf_lut_texture)

    def render_spheres(self):
        self.prog_pbr_lighting["projection"].write(self.camera.projection.matrix)
        self.prog_pbr_lighting["view"].write(self.camera.matrix)
        self.prog_pbr_lighting["camPos"].write(self.camera.position)

        # Optimize uniforms: only update if changed (local state tracking)
        if self._last_exposure != self.ui_exposure:
            self.prog_pbr_lighting["pbr_exposure"].value = self.ui_exposure
            self._last_exposure = self.ui_exposure

        if self._last_debug_mode != self.ui_visualization_mode:
            self.prog_pbr_lighting["debug_mode"].value = self.ui_visualization_mode
            self._last_debug_mode = self.ui_visualization_mode

        albedo_tuple = tuple(self.ui_albedo)
        if self._last_albedo != albedo_tuple:
            self.prog_pbr_lighting["albedo"].value = albedo_tuple
            self._last_albedo = albedo_tuple

        if self._last_ao != self.ui_ao:
            self.prog_pbr_lighting["ao"].value = self.ui_ao
            self._last_ao = self.ui_ao

        if self._last_use_billboarding != self.ui_use_billboarding:
            self.prog_pbr_lighting["use_billboarding"].value = self.ui_use_billboarding
            self._last_use_billboarding = self.ui_use_billboarding

        self.irradiance_map_cubemap.use(location=0)
        self.prefiltered_specular_map.use(location=1)
        self.brdf_lut_texture.use(location=2)

        if self.ui_render_mode == 0:
            # Grid Mode: Varying metallic/roughness values
            nr_rows = self.ui_nr_rows
            nr_columns = self.ui_nr_columns
            spacing = self.ui_spacing
            for row in range(nr_rows):
                self.prog_pbr_lighting["metallic"].value = float(row) / float(nr_rows)
                for col in range(nr_columns):
                    model = glm.translate(
                        glm.vec3(
                            (col - (nr_columns / 2)) * spacing, (row - (nr_rows / 2)) * spacing, 0.0
                        )
                    )
                    self.prog_pbr_lighting["roughness"].value = glm.clamp(
                        float(col) / float(nr_columns), 0.05, 1.0
                    )
                    self.prog_pbr_lighting["model"].write(model)
                    self.prog_pbr_lighting["normalMatrix"].write(
                        glm.transpose(glm.inverse(glm.mat3(model)))
                    )
                    if self.ui_use_billboarding:
                        self.quad.render(self.prog_pbr_lighting)
                    else:
                        self.sphere.render(self.prog_pbr_lighting)
        else:
            # Presets Mode: Fixed material properties
            spacing = self.ui_spacing
            cols = self.ui_nr_columns
            rows = self.ui_nr_rows
            for i, mat in enumerate(self.MATERIAL_PRESETS):
                if i >= cols * rows:
                    break
                row = i // cols
                col = i % cols
                model = glm.translate(
                    glm.vec3((col - (cols / 2)) * spacing, (row - (rows / 2)) * spacing, 0.0)
                )
                self.prog_pbr_lighting["albedo"].value = mat["albedo"]
                self.prog_pbr_lighting["metallic"].value = mat["metallic"]
                self.prog_pbr_lighting["roughness"].value = mat["roughness"]
                self.prog_pbr_lighting["model"].write(model)
                self.prog_pbr_lighting["normalMatrix"].write(
                    glm.transpose(glm.inverse(glm.mat3(model)))
                )
                if self.ui_use_billboarding:
                    self.quad.render(self.prog_pbr_lighting)
                else:
                    self.sphere.render(self.prog_pbr_lighting)

    def render_skybox(self, cubemap: moderngl.TextureCube, lod: float = 0.0):
        skybox_cam = self.camera.matrix
        # Purge camera translation
        skybox_cam[3][0] = 0
        skybox_cam[3][1] = 0
        skybox_cam[3][2] = 0

        # self.ctx.disable(moderngl.DEPTH_TEST)
        self.backgroundShader["m_proj"].write(self.camera.projection.matrix)
        self.backgroundShader["m_camera"].write(skybox_cam)
        self.backgroundShader["blur_lod"].value = lod
        cubemap.use(location=0)
        self.cube.render(self.backgroundShader)

    def on_resize(self, width: int, height: int):
        # Calculate aspect ratio directly from arguments to avoid querying window state
        # which might be inconsistent during the event.
        if width > 0 and height > 0:
            self.camera.projection.update(aspect_ratio=width / height)

        self.imgui.resize(width, height)

    def render_ui(self):
        imgui.new_frame()

        imgui.begin("Debug Panel", True)
        imgui.text(f"OpenGL version: {self.ctx.version_code}")
        imgui.text(f"Frame time: {1000.0 * self.average_frame_time:.2f} ms")
        imgui.text(f"FPS: {1.0 / self.average_frame_time:.2f}")

        _, self.ui_wireframe_enabled = imgui.checkbox(
            "Enabled Wireframe", self.ui_wireframe_enabled
        )
        _, self.ui_skybox_enabled = imgui.checkbox("Enabled SkyBox", self.ui_skybox_enabled)
        changed, self.ui_clear_color = imgui.color_edit3("Clear Color", list(self.ui_clear_color))
        if changed:
            self.clear_color = self.ui_clear_color

        # reload a new HDR env map
        changed, self.ui_hdri_id = imgui.combo(
            f"HDRI Environment Map ({self.elapsed_time / 1_000_000.0:.2f} ms)",
            self.ui_hdri_id,
            self.hdri_names,
        )
        if changed:
            self.precompute_from_hdr_env_map(self.hdri_names[self.ui_hdri_id])
            logger.info(f"Time spent on the GPU: {self.elapsed_time / 1_000_000.0:.2f} ms")

        imgui.separator()
        imgui.text("Debug Views")
        _, self.ui_visualization_mode = imgui.combo(
            "Visualization", self.ui_visualization_mode, self.ui_visualization_modes
        )
        _, self.ui_debug_skybox_id = imgui.combo(
            "Skybox Texture", self.ui_debug_skybox_id, self.ui_debug_skybox_options
        )

        changed, self.ui_irradiance_method = imgui.combo(
            "Irradiance Gen Method", self.ui_irradiance_method, self.ui_irradiance_method_options
        )
        changed_clamp, self.ui_irradiance_clamp = imgui.slider_float(
            "Irradiance Clamp", self.ui_irradiance_clamp, 1.0, 100.0
        )

        if changed or changed_clamp:
            logger.info(
                f"Regenerating Irradiance Map using "
                f"{self.ui_irradiance_method_options[self.ui_irradiance_method]} "
                f"and clamp {self.ui_irradiance_clamp}"
            )
            # Release previous texture to be clean? (not strictily necessary if overwriting,
            # but good practice if recreating object)
            if self.irradiance_map_cubemap:
                self.irradiance_map_cubemap.release()
            self.irradiance_map_cubemap = self.build_irradiance_cubemap(
                self.env_cubemap, size=PBRWithPrefilteredSpecular.res_for_irradiance_map
            )
            # Ensure we wait for it to be done if we want to measure time accurately
            # or avoid glitches
            self.ctx.finish()

        if imgui.collapsing_header("BRDF LUT"):
            # Inspect BRDF LUT
            width = 256
            height = 256
            # Flip UVs for correct display if needed, but simple image is fine
            imgui.image(
                imgui.ImTextureRef(self.brdf_lut_texture.glo), (width, height), (0, 1), (1, 0)
            )
        imgui.separator()

        _, self.ui_render_mode = imgui.combo(
            "Render Mode", self.ui_render_mode, self.ui_render_mode_options
        )
        _, self.ui_show_labels = imgui.checkbox("Show Labels", self.ui_show_labels)

        if self.ui_render_mode == 0:
            _, self.ui_albedo = imgui.color_edit3("Albedo", self.ui_albedo)

        _, self.ui_nr_rows = imgui.slider_int("Number of Rows", self.ui_nr_rows, 1, 10)
        _, self.ui_nr_columns = imgui.slider_int("Number of Columns", self.ui_nr_columns, 1, 10)
        _, self.ui_spacing = imgui.slider_float("Spacing", self.ui_spacing, 1.0, 10.0)
        _, self.ui_ao = imgui.slider_float("Ambient Occlusion", self.ui_ao, 0.05, 10.0)
        _, self.ui_exposure = imgui.slider_float("Exposure", self.ui_exposure, 1.0, 1.5)

        imgui.separator()
        _, self.ui_use_billboarding = imgui.checkbox("Raytraced Billboards", self.ui_use_billboarding)

        imgui.text("Sphere Mesh (Legacy Settings)")
        changed1, self.ui_sphere_subdivisions = imgui.slider_int(
            "Subdivisions", self.ui_sphere_subdivisions, 0, 6
        )
        changed2, self.ui_sphere_irregularity = imgui.slider_float(
            "Irregularity (Planet)", self.ui_sphere_irregularity, 0.0, 0.5
        )
        if changed1 or changed2:
            self.sphere.release()
            self.sphere = geometry.icosphere(
                radius=1.0,
                subdivisions=self.ui_sphere_subdivisions,
                randomization=self.ui_sphere_irregularity,
            )
        imgui.text(
            f"Camera Position: ({self.camera.position.x:.2f}, "
            f"{self.camera.position.y:.2f}, {self.camera.position.z:.2f})"
        )

        imgui.end()

        # Draw labels for Materiel Presets or Grid Coordinates
        if self.ui_show_labels:
            draw_list = imgui.get_foreground_draw_list()
            spacing = self.ui_spacing

            # View-Projection for labels
            mvp = self.camera.projection.matrix * self.camera.matrix
            width, height = self.wnd.size

            if self.ui_render_mode == 1:
                # Presets Mode Labels
                cols = self.ui_nr_columns
                rows = self.ui_nr_rows
                for i, mat in enumerate(self.MATERIAL_PRESETS):
                    if i >= cols * rows:
                        break
                    row = i // cols
                    col = i % cols
                    world_pos = glm.vec3(
                        (col - (cols / 2)) * spacing, (row - (rows / 2)) * spacing, 0.0
                    )
                    self._draw_label(draw_list, mvp, world_pos, width, height, mat["name"])
            else:
                # Grid Mode Labels (Roughness / Metallic)
                nr_rows = self.ui_nr_rows
                nr_cols = self.ui_nr_columns
                for row in range(nr_rows):
                    m = float(row) / float(nr_rows)
                    for col in range(nr_cols):
                        r = glm.clamp(float(col) / float(nr_cols), 0.05, 1.0)
                        world_pos = glm.vec3(
                            (col - (nr_cols / 2)) * spacing, (row - (nr_rows / 2)) * spacing, 0.0
                        )
                        label = f"M:{m:.2f} R:{r:.2f}"
                        # Only show labels for first/last or every few if grid is large to
                        # avoid clutter? For now, show all if requested.
                        self._draw_label(draw_list, mvp, world_pos, width, height, label)

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def _draw_label(self, draw_list, mvp, world_pos, width, height, text):
        # Project to clip space
        clip_pos = mvp * glm.vec4(world_pos, 1.0)

        # Check if visible (in front of camera)
        if clip_pos.w > 0:
            # Normalized Device Coordinates (NDC)
            ndc = glm.vec3(clip_pos) / clip_pos.w

            # Check if within screen bounds (roughly)
            if -1.0 <= ndc.x <= 1.0 and -1.0 <= ndc.y <= 1.0:
                # Convert to screen coordinates (Imgui uses pixel coords from top-left)
                screen_x = (ndc.x + 1.0) * 0.5 * width
                screen_y = (1.0 - ndc.y) * 0.5 * height

                # Draw centered text slightly below the sphere
                text_size = imgui.calc_text_size(text)
                # Increased offset for labels
                pos = imgui.ImVec2(screen_x - text_size.x * 0.5, screen_y + 35)

                # Draw shadow for readability
                draw_list.add_text(
                    imgui.ImVec2(pos.x + 1, pos.y + 1),
                    imgui.get_color_u32(imgui.ImVec4(0, 0, 0, 1)),
                    text,
                )
                draw_list.add_text(pos, imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 1)), text)

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)
        super().on_mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x: int, y: int, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)
        super().on_mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)
        super().on_mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        self.imgui.mouse_release_event(x, y, button)

    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)
        # if not self.imgui.io.want_capture_keyboard:
        super().on_key_event(key, action, modifiers)


if __name__ == "__main__":
    PBRWithPrefilteredSpecular.run()
