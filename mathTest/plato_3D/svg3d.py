# svg3d :: https://prideout.net/blog/svg_wireframes/
# Single-file Python library for generating 3D wireframes in SVG format.
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.

import numpy as np
import pyrr
import svgwrite

from typing import NamedTuple, Callable, Sequence

# 首先，需要定义在几乎任何3D渲染器中都会找到的经典成分：视口、相机、网格和场景的类：
# 视图类
class Viewport(NamedTuple):
    # 定义相机投影到的最终图像中的矩形区域。除非图像包含多个面板，否则可以将其设置为默认值。
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_aspect(cls, aspect_ratio: float):
        return cls(-aspect_ratio / 2.0, -0.5, aspect_ratio, 1.0)

    @classmethod
    def from_string(cls, string_to_parse):
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)

# 相机包含视图矩阵和投影矩阵。
# 我们可以使用pyrr来生成这些；它提供create_look_at和create_perspective_projection功能
class Camera(NamedTuple):
    view: np.ndarray
    projection: np.ndarray

# 网格有一个面列表、一个着色器和一个应用于表示网格的SVG组的样式字典。
# 网格还包含一个称为面的三维 numpy 数组，其形状为 n⨯m⨯3，其中n是面数，m是每个面的顶点数（例如，对于四边形网格，m=4）。最后一个轴的长度为3，因为网格由 XYZ 坐标组成。
class Mesh(NamedTuple):
    faces: np.ndarray
    shader: Callable[[int, float], dict] = None
    style: dict = None
    circle_radius: float = 0

# 场景类
class Scene(NamedTuple):
    meshes: Sequence[Mesh]

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)


class View(NamedTuple):
    camera: Camera
    scene: Scene
    viewport: Viewport = Viewport()


class Engine:
    # 负责使用场景描述并生成SVG文件。
    def __init__(self, views, precision=5):
        self.views = views
        self.precision = precision

    def render(self, filename, size=(512, 512), viewBox="-0.5 -0.5 1.0 1.0", **extra):
        drawing = svgwrite.Drawing(filename, size, viewBox=viewBox, **extra)
        self.render_to_drawing(drawing)
        drawing.save()

    def render_to_drawing(self, drawing):
        for view in self.views:
            # numpy.dot请注意将一个 4x4 矩阵与另一个矩阵相乘的用法。生成的矩阵将用于将齐次坐标投影到观察平面上。
            projection = np.dot(view.camera.view, view.camera.projection)

            clip_path = drawing.defs.add(drawing.clipPath())
            clip_min = view.viewport.minx, view.viewport.miny
            clip_size = view.viewport.width, view.viewport.height
            clip_path.add(drawing.rect(clip_min, clip_size))

            for mesh in view.scene.meshes:
                g = self._create_group(drawing, projection, view.viewport, mesh)
                g["clip-path"] = clip_path.get_funciri()
                drawing.add(g)

    def _create_group(self, drawing, projection, viewport, mesh):
        faces = mesh.faces
        shader = mesh.shader or (lambda face_index, winding: {})
        default_style = mesh.style or {}

        # Extend each point to a vec4, then transform to clip space.
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, projection)

        # Reject trivially clipped polygons.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        accepted = np.logical_and(np.greater(xyz, -w), np.less(xyz, +w))
        accepted = np.all(accepted, 2)  # vert is accepted if xyz are all inside
        accepted = np.any(accepted, 1)  # face is accepted if any vert is inside
        degenerate = np.less_equal(w, 0)[:, :, 0]  # vert is bad if its w <= 0
        degenerate = np.any(degenerate, 1)  # face is bad if any of its verts are bad
        accepted = np.logical_and(accepted, np.logical_not(degenerate))
        faces = np.compress(accepted, faces, axis=0)

        # Apply perspective transformation.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        faces = xyz / w

        # Sort faces from back to front.
        face_indices = self._sort_back_to_front(faces)
        faces = faces[face_indices]

        # Apply viewport transform to X and Y.
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * viewport.width / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * viewport.height / 2
        faces[:, :, 0:1] += viewport.minx
        faces[:, :, 1:2] += viewport.miny

        # Compute the winding direction of each polygon.
        windings = np.zeros(faces.shape[0])
        if faces.shape[1] >= 3:
            p0, p1, p2 = faces[:, 0, :], faces[:, 1, :], faces[:, 2, :]
            normals = np.cross(p2 - p0, p1 - p0)
            np.copyto(windings, normals[:, 2])

        group = drawing.g(**default_style)

        # Create circles.
        if mesh.circle_radius > 0:
            for face_index, face in enumerate(faces):
                style = shader(face_indices[face_index], 0)
                if style is None:
                    continue
                face = np.around(face[:, :2], self.precision)
                for pt in face:
                    group.add(drawing.circle(pt, mesh.circle_radius, **style))
            return group

        # Create polygons and lines.
        for face_index, face in enumerate(faces):
            style = shader(face_indices[face_index], windings[face_index])
            if style is None:
                continue
            face = np.around(face[:, :2], self.precision)
            if len(face) == 2:
                group.add(drawing.line(face[0], face[1], **style))
            else:
                group.add(drawing.polygon(face, **style))

        return group

    def _sort_back_to_front(self, faces):
        z_centroids = -np.sum(faces[:, :, 2], axis=1)
        for face_index in range(len(z_centroids)):
            z_centroids[face_index] /= len(faces[face_index])
        return np.argsort(z_centroids)
