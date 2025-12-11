"""
- https://learnopengl.com/PBR/Lighting
- https://learnopengl.com/code_viewer_gh.php?code=src/6.pbr/1.1.lighting/lighting.cpp
- https://learnopengl.com/code_viewer_gh.php?code=src/6.pbr/1.1.lighting/1.1.pbr.vs
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
logger = logging.getLogger(f"moderngl_window.exemple.pbr_direct_lighting")


# logger.setLevel(logging.DEBUG)


class PBRWithDirectLighting(CameraWindow):
    """Example Physic Base Rendering with Direct Lighting"""

    title = "Example Physic Base Rendering with Direct Lighting"
    resource_dir = (Path(__file__) / "../../resources").resolve()
    window_size = 1280, 720
    aspect_ratio = window_size[0] / window_size[1]
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True

        self.frame_time_decay_factor = 0.995
        self.average_frame_time = 0.01666

        self.sphere = geometry.sphere(radius=1.0, sectors=32, rings=32)
        self.prog_pbr_lighting = self.load_program("programs/PBR/pbr_direct_lighting.glsl")
        self.prog_pbr_lighting["albedo"] = (0.5, 0.0, 0.0)
        self.prog_pbr_lighting["ao"] = 1.0

        # Set up imgui.
        imgui.create_context()
        if self.wnd.ctx.error != "GL_NO_ERROR":
            logger.error(self.wnd.ctx.error)
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.imgui.io.want_capture_mouse = False

        self.camera.set_position(0, 0, 20)

    def on_render(self, time, frame_time):
        self.average_frame_time = (
            self.frame_time_decay_factor * self.average_frame_time
            + (1.0 - self.frame_time_decay_factor) * frame_time
        )

        self.ctx.enable_only(moderngl.DEPTH_TEST)

        self.wnd.use()

        self.prog_pbr_lighting["projection"].write(self.camera.projection.matrix)
        self.prog_pbr_lighting["view"].write(self.camera.matrix)
        try:
            self.prog_pbr_lighting["camPos"].write(self.camera.position)
        except:
            pass

        nrRows = 7
        nrColumns = 7
        spacing = 2.5
        # // render rows*column number of spheres with varying metallic/roughness values scaled by rows and columns respectively
        for row in range(nrRows):
            try:
                self.prog_pbr_lighting["metallic"].value = float(row) / float(nrRows)
                # logger.debug(f'{self.prog_pbr_lighting["metallic"].value=}')
            except:
                pass
            for col in range(nrColumns):
                # // we clamp the roughness to 0.05 - 1.0 as perfectly smooth surfaces (roughness of 0.0) tend to look a bit off
                # // on direct lighting.
                try:
                    self.prog_pbr_lighting["roughness"].value = glm.clamp(float(col) / float(nrColumns), 0.05, 1.0)
                    # logger.debug(f'{self.prog_pbr_lighting["roughness"].value=}')
                except:
                    pass

                model = glm.mat4(1.0)
                model = glm.translate(model, glm.vec3(
                    (col - (nrColumns / 2)) * spacing,
                    (row - (nrRows / 2)) * spacing,
                    0.0
                ))
                self.prog_pbr_lighting["model"].write(model)
                try:
                    self.prog_pbr_lighting["normalMatrix"].write(glm.transpose(glm.inverse(glm.mat3(model))))
                except:
                    pass

                self.sphere.render(self.prog_pbr_lighting)

        assert GL.glGetError() == GL.GL_NO_ERROR

        self.render_ui()

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
    PBRWithDirectLighting.run()
