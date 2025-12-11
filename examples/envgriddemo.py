from pathlib import Path

import glm
import moderngl
import numpy as np
from imgui_bundle import imgui

import moderngl_window
from base import OrbitDragCameraWindow
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer


class EnvGridDemo(OrbitDragCameraWindow):
    """A demo of
    """

    title = "EnvGrid"
    resource_dir = (Path(__file__) / "../resources").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.wnd.mouse_exclusivity = True

        self.camera.projection.update(near=0.1, far=50.0)
        self.camera.radius = 2.0
        self.camera.angle_x = 290.0
        self.camera.angle_y = -80.0
        self.camera.velocity = 7.0
        self.camera.target = (0.0, 0.0, 0.0)
        self.camera.mouse_sensitivity = 1.0
        self.camera.zoom_sensitivity = 0.3

        self.frame_time_decay_factor = 0.995
        self.average_frame_time = 0.01666

        # Load the geometry program.
        self.geometry_program = self.load_program("programs/envgrid/geometry.glsl")

        # Generate a fullscreen quad.
        self.quad_fs = moderngl_window.geometry.quad_fs()

        # Set up imgui.
        imgui.create_context()
        if self.wnd.ctx.error != "GL_NO_ERROR":
            print(self.wnd.ctx.error)
        self.imgui = ModernglWindowRenderer(self.wnd)

    def on_render(self, time: float, frametime: float):
        self.average_frame_time = (
            self.frame_time_decay_factor * self.average_frame_time
            + (1.0 - self.frame_time_decay_factor) * frametime
        )

        projection_matrix = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection_matrix * camera_matrix
        imvp = glm.inverse(mvp)
        camera_pos = (
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        )

        self.geometry_program["imvp"].write(imvp)
        # self.geometry_program["m_camera"].write(camera_matrix)
        self.quad_fs.render(self.geometry_program)

        self.render_ui()

    def render_ui(self):
        imgui.new_frame()

        imgui.begin("Debug Panel", False)
        imgui.text(f"Frame time: {1000.0 * self.average_frame_time:.1f} ms")
        imgui.text(f"FPS: {1.0 / self.average_frame_time:.1f}")

        imgui.end()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

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
    EnvGridDemo.run()
