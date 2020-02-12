from pathlib import Path

import moderngl

import moderngl_window
from moderngl_window import geometry
from moderngl_window import resources

resources.register_dir((Path(__file__).parent / 'resources').resolve())


class QuadFullscreen(moderngl_window.WindowConfig):
    window_size = 512, 512
    aspect_ratio = 512 / 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = geometry.quad_fs()
        #
        self.texture = self.load_texture_2d('textures/grid.png')
        # self.texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.texture.filter = moderngl.NEAREST, moderngl.LINEAR
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        #
        self.texture.build_mipmaps(max_level=5)
        # TODO: bug, can't set nearest with mipmap !
        # self.texture.filter = moderngl.NEAREST_MIPMAP_NEAREST, moderngl.NEAREST_MIPMAP_NEAREST
        self.texture.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_LINEAR

        self.prog = self.load_program('programs/interpolation.glsl')

    def render(self, time: float, frame_time: float):
        self.ctx.clear()

        self.texture.use(location=0)
        self.prog['texture0'].value = 0
        self.quad.render(self.prog)


if __name__ == '__main__':
    moderngl_window.run_window_config(QuadFullscreen)
