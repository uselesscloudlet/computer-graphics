import pygame
import pygame_widgets
from pygame_widgets.button import Button
import numpy as np
from enum import Enum
from numba import jit


class Mode(Enum):
    TRANSLATING = 0
    ROTATING = 1
    CAMERA_TRANSLATING = 2
    CAMERA_ROTATING = 3


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    FORWARD = 4
    BACKWARD = 5


class Shading(Enum):
    WIREFRAME = 0
    LAMBERT = 1
    GOURAUD = 2
    PHONG = 3


SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 512
FOV = np.radians(110)
FOC_LENGTH = SCREEN_WIDTH / 2 * np.cos(FOV / 2) / np.sin(FOV / 2)
FPS = 60


class Point3D():
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_arr(self):
        return np.array([self.x, self.y, self.z])

    def rotate_X(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        y = self.y * cosA - self.z * sinA
        z = self.y * sinA + self.z * cosA
        return Point3D(self.x, y, z)

    def rotate_Y(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        z = self.z * cosA - self.x * sinA
        x = self.z * sinA + self.x * cosA

        return Point3D(x, self.y, z)

    def rotate_Z(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        x = self.x * cosA - self.y * sinA
        y = self.x * sinA + self.y * cosA
        return Point3D(x, y, self.z)

    def project(self, x=0, y=0, z=0):
        if (self.z + z) > 0:
            factor = FOC_LENGTH / (self.z + z)
            x = (self.x + x) * factor + SCREEN_WIDTH / 2
            y = (self.y + y) * factor + SCREEN_HEIGHT / 2
        else:
            x = y = 0
        return Point3D(x, y, self.z)

    def __determ_sincos(self, angle):
        rad = angle * np.pi / 180
        cosA = np.cos(rad)
        sinA = np.sin(rad)

        return sinA, cosA


class PointLight():
    def __init__(self, point, color):
        self.point = point
        self.color = color


class Cube():
    def __init__(self, size=1):
        self.vertices = [
            Point3D(-1 * size, 1 * size, -1 * size),
            Point3D(1 * size, 1 * size, -1 * size),
            Point3D(1 * size, -1 * size, -1 * size),
            Point3D(-1 * size, -1 * size, -1 * size),
            Point3D(-1 * size, 1 * size, 1 * size),
            Point3D(1 * size, 1 * size, 1 * size),
            Point3D(1 * size, -1 * size, 1 * size),
            Point3D(-1 * size, -1 * size, 1 * size)
        ]
        self.faces = [(0, 1, 2, 3),
                      (1, 5, 6, 2),
                      (5, 4, 7, 6),
                      (4, 0, 3, 7),
                      (0, 4, 5, 1),
                      (3, 2, 6, 7)]

        self.angles = [0, 0, 0]
        self.pos = [0, 0, 0]
        self.lambert_colors = np.zeros((len(self.faces), 3), dtype=int)
        self.gouraud_colors = np.zeros((len(self.vertices), 3), dtype=int)

    def lambert_make_color(self, point_lights):
        intensities = self.lambert_lighting(point_lights)
        for i in range(len(self.lambert_colors)):
            self.lambert_colors[i] = intensities[i] * 255

    def gouraud_make_color(self, point_lights):
        intensities = self.gouraud_lighting(point_lights)
        for i in range(len(self.gouraud_colors)):
            self.gouraud_colors[i] = intensities[i] * 255

    def draw_cube(self, camera_pos, camera_angles, point_lights):
        t_vertices = self.transform_vertices(camera_pos, camera_angles, False)
        tf_vertices = self.transform_vertices(camera_pos, camera_angles, True)
        self.lambert_make_color(point_lights)
        self.gouraud_make_color(point_lights)
        avg_Z = self.calculate_avg_z(tf_vertices)
        polygons = []

        _, normals = self.get_c_and_norm(tf_vertices, True)
        pos = Point3D(*self.pos).project(0, 0, 0)
        vis = np.dot([pos.x, pos.y, pos.z + 100000, 0], normals.T)
        for z_val in sorted(avg_Z, key=lambda x: x[1], reverse=True):
            if vis[z_val[0]] >= 0:
                continue
            if (z_val[1]) < (3):
                continue
            f_index = z_val[0]
            f = self.faces[f_index]

            point_list = np.array([
                (tf_vertices[f[0]].x, tf_vertices[f[0]].y,
                    t_vertices[f[0]].z, f[0]),
                (tf_vertices[f[1]].x, tf_vertices[f[1]].y,
                    t_vertices[f[1]].z, f[1]),
                (tf_vertices[f[2]].x, tf_vertices[f[2]].y,
                    t_vertices[f[2]].z, f[2]),
            ])

            polygons.append((point_list, f_index))

            point_list = np.array([
                (tf_vertices[f[0]].x, tf_vertices[f[0]].y,
                    t_vertices[f[0]].z, f[0]),
                (tf_vertices[f[2]].x, tf_vertices[f[2]].y,
                    t_vertices[f[2]].z, f[2]),
                (tf_vertices[f[3]].x, tf_vertices[f[3]].y,
                    t_vertices[f[3]].z, f[3]),
            ])
            polygons.append((point_list, f_index))
        return polygons

    def translate_cube(self, x, y, z):
        self.pos[0] += x
        self.pos[1] += y
        self.pos[2] += z

    def rotate_cube(self, direction):
        if direction == Direction.UP:
            self.angles[0] += 2
        elif direction == Direction.DOWN:
            self.angles[0] -= 2
        elif direction == Direction.LEFT:
            self.angles[1] += 2
        elif direction == Direction.RIGHT:
            self.angles[1] -= 2
        elif direction == Direction.FORWARD:
            self.angles[2] += 2
        elif direction == Direction.BACKWARD:
            self.angles[2] -= 2

    def transform_vertices(self, canera_pos, camera_angles, project=True):
        t_vertices = []
        for vertex in self.vertices:
            rotation = vertex.rotate_X(self.angles[0]).rotate_Y(
                self.angles[1]).rotate_Z(self.angles[2])
            rotation.x += canera_pos[0] + self.pos[0]
            rotation.y += canera_pos[1] + self.pos[1]
            rotation.z += canera_pos[2] + self.pos[2]
            rotation = rotation.rotate_X(camera_angles[0]).rotate_Y(
                camera_angles[1]).rotate_Z(camera_angles[2])
            if project:
                projection = rotation.project(0, 0, 0)
            else:
                projection = rotation
            t_vertices.append(projection)
        return t_vertices

    def calculate_avg_z(self, vertices):
        avg_z = []
        for idx, face in enumerate(self.faces):
            z = (vertices[face[0]].z +
                 vertices[face[1]].z +
                 vertices[face[2]].z +
                 vertices[face[3]].z) / 4.0
            avg_z.append([idx, z])

        return avg_z

    def lambert_lighting(self, point_lights):
        intensities = np.zeros((len(self.faces), 3))
        transformed_vertices = self.transform_vertices(
            [0, 0, 0], [0, 0, 0], False)
        centers, normals = self.get_c_and_norm(transformed_vertices)
        for light in point_lights:
            light_vectors = [light.point - center for center in centers]
            for i in np.arange(len(light_vectors)):
                light_vector = light_vectors[i].to_arr(
                ) / np.linalg.norm(light_vectors[i].to_arr())
                normal = normals[i] / np.linalg.norm(normals[i])
                intensity_coef = np.maximum(np.dot(light_vector, normal), 0)
                for j in np.arange(3):
                    intensities[i][j] += 1 * light.color[j] * intensity_coef
        return np.array(intensities)

    def gouraud_lighting(self, point_lights):
        intensities = np.zeros((len(self.vertices), 3))
        transformed_vertices = self.transform_vertices(
            [0, 0, 0], [0, 0, 0], False)
        _, faces_normals = self.get_c_and_norm(transformed_vertices)
        vertices_normals = []
        for _ in range(len(self.vertices)):
            vertices_normals.append([])
        for i, face in enumerate(self.faces):
            for j in face:
                vertices_normals[j].append(faces_normals[i])

        vertices_normals = np.mean(np.array(vertices_normals), 1)

        for light in point_lights:
            light_vectors = np.array(
                [light.point - vert for vert in transformed_vertices])

            for i in np.arange(len(light_vectors)):
                light_vector = light_vectors[i].to_arr(
                ) / np.linalg.norm(light_vectors[i].to_arr())
                normal = vertices_normals[i] / \
                    np.linalg.norm(vertices_normals[i])
                intensity_coef = np.maximum(np.dot(light_vector, normal), 0)
                for j in np.arange(3):
                    intensities[i][j] += 1 * light.color[j] * intensity_coef
        return np.array(intensities)

    def get_c_and_norm(self, transformed_vertices, return_d=False):
        normals = []
        centers = []
        for face in self.faces:
            sum_x = 0
            sum_y = 0
            sum_z = 0
            for vertex in face:
                point = transformed_vertices[vertex]
                sum_x += point.x
                sum_y += point.y
                sum_z += point.z
            center = Point3D(sum_x / 4, sum_y / 4, sum_z / 4)
            p1 = transformed_vertices[face[0]].to_arr()
            p2 = transformed_vertices[face[1]].to_arr()
            p3 = transformed_vertices[face[2]].to_arr()
            v1 = p3 - p1
            v2 = p2 - p1
            normal = -np.cross(v1, v2)

            if return_d:
                d = np.dot(normal, p1)
                normals.append(np.array([*normal, d]))
            else:
                normals.append(normal)
            centers.append(center)

        return np.array(centers), np.array(normals)


@jit(nopython=True)
def bilinear_interpolation(x: np.number, y: np.number, x1: np.number, y1: np.number, q1: np.number, x2: np.number, y2: np.number, q2: np.number, x3: np.number, y3: np.number, q3: np.number):
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))

    if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))):
        w1 = np.full_like(w1, 0)
        w2 = np.full_like(w2, 0)

    w3 = 1 - w1 - w2

    return np.outer(q1, w1) + np.outer(q2, w2) + np.outer(q3, w3)


@jit(nopython=True)
def area(x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                   + x3 * (y1 - y2)) / 2.0)


@jit(nopython=True)
def is_inside_triangle(x: np.number, y: np.number, x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    A = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    A3 = area(x1, y1, x2, y2, x, y)
    return (A1 + A2 + A3) - A < 0.001


@jit(nopython=True)
def get_triangle_points(x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    xs = np.array([x1, x2, x3])
    ys = np.array([y1, y2, y3])
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    x_delta = x_max - x_min + 1
    y_delta = y_max - y_min + 1

    if x_max < 0 or x_min > SCREEN_WIDTH or y_max < 0 or y_min > SCREEN_HEIGHT:
        x_delta = y_delta = 1

    if x_min == x_max or y_min == y_max:
        x_delta = y_delta = 1

    X = np.repeat(np.arange(x_delta), y_delta)
    Y = np.array([*np.arange(y_delta)] * x_delta)

    screen_mask = ((x_min + X) > 0) & ((x_min + X) < SCREEN_WIDTH) & (
        (y_min + Y) > 0) & ((y_min + Y) < SCREEN_HEIGHT)
    X = X[screen_mask]
    Y = Y[screen_mask]

    XY_mask = is_inside_triangle(X, Y,
                                 x1 - x_min,
                                 y1 - y_min,
                                 x2 - x_min,
                                 y2 - y_min,
                                 x3 - x_min,
                                 y3 - y_min)
    X = X[XY_mask]
    Y = Y[XY_mask]

    X = X + x_min
    Y = Y + y_min

    return X, Y


@jit(nopython=True)
def screen_space(draw_buffer, x, y):
    mask = (x >= 0) & (y >= 0) & (
        x < draw_buffer.shape[0]) & (y < draw_buffer.shape[1])
    return x[mask].astype(np.int32), y[mask].astype(np.int32)


@jit(nopython=True)
def draw_wireframe(draw_buffer, triangle):
    for i in range(len(triangle)):
        x1, y1, _, _ = triangle[i]
        x2, y2, _, _ = triangle[(i + 1) % len(triangle)]
        l = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        x = np.linspace(x1, x2, l)
        y = np.linspace(y1, y2, l)
        x, y = screen_space(draw_buffer, x, y)

        for _x, _y in zip(x, y):
            draw_buffer[_x, _y] = [255, 255, 255]


@jit(nopython=True)
def draw_lumbert(draw_buffer, z_buffer, triangle, color):
    XY = get_triangle_points(
        triangle[0, 0], triangle[0, 1], triangle[1, 0], triangle[1, 1], triangle[2, 0], triangle[2, 1])
    if XY is not None and len(XY[0]) > 0:
        X, Y = XY
        z_key_points = []
        for x, y, z, _ in triangle:
            z_key_points.append(
                (x, y, [z]))

        Z = bilinear_interpolation(
            X, Y, *z_key_points[0], *z_key_points[1], *z_key_points[2])[0]

        for _x, _y, _z in zip(X, Y, Z):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = color
                z_buffer[_x, _y] = _z


@jit(nopython=True)
def draw_gouraud(draw_buffer, z_buffer, triangle, colors):
    XY = get_triangle_points(
        triangle[0, 0], triangle[0, 1], triangle[1, 0], triangle[1, 1], triangle[2, 0], triangle[2, 1])
    if XY is not None and len(XY[0]) > 0:
        X, Y = XY
        key_points = []

        for x, y, z, v in triangle:
            v = int(v)
            key_points.append(
                (x, y, [z, *colors[v]]))

        K = bilinear_interpolation(
            X, Y, *key_points[0], *key_points[1], *key_points[2])

        for _x, _y, _k in zip(X, Y, K.T):
            _z = _k[0]
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = _k[1:]
                z_buffer[_x, _y] = _z


@jit(nopython=True, fastmath=True)
def draw_phong(draw_buffer, z_buffer, triangle, plane, color, lights):
    XY = get_triangle_points(
        triangle[0, 0], triangle[0, 1], triangle[1, 0], triangle[1, 1], triangle[2, 0], triangle[2, 1])
    if XY is not None and len(XY[0]) > 0:
        X, Y = XY
        z_key_points = []
        for x, y, z, _v in triangle:
            z_key_points.append(
                (x, y, [z]))

        Z = bilinear_interpolation(
            X, Y, *z_key_points[0], *z_key_points[1], *z_key_points[2])[0]

        xs = X
        ys = Y

        factor = FOC_LENGTH / Z
        X = (X - SCREEN_WIDTH / 2) / factor
        Y = (Y - SCREEN_HEIGHT / 2) / factor

        AMBIENT_COEFF = 0.05
        DIFFUSE_COEFF = 1
        SPECULAR_COEFF = 0.2

        diffuse = np.zeros((len(X), 3))
        specular = np.zeros((len(X), 3))
        for light in lights:
            light_color = light[3:]
            light_pos = light[:3]
            xyz = np.zeros((3, len(xs)))
            xyz[0] = X
            xyz[1] = Y
            xyz[2] = Z
            l = light_pos - xyz.T

            n = plane[:3]
            n_norm = np.sqrt(np.sum(n ** 2))
            n /= n_norm

            l_norm = np.sqrt(np.sum(l ** 2, axis=1))
            l[:, 0] /= l_norm
            l[:, 1] /= l_norm
            l[:, 2] /= l_norm

            r = 2 * n - l
            r_norm = np.sqrt(np.sum(r ** 2, axis=1))
            r[:, 0] /= r_norm
            r[:, 2] /= r_norm
            r[:, 1] /= r_norm

            d = np.outer(np.maximum(np.dot(n, l.T), 0), light_color)
            diffuse += d * DIFFUSE_COEFF

            if np.max(d) < 0.001:
                continue

            neg_i = -xyz.T
            neg_i_norm = np.sqrt(np.sum(neg_i ** 2, axis=1))
            neg_i[:, 0] /= neg_i_norm
            neg_i[:, 1] /= neg_i_norm
            neg_i[:, 2] /= neg_i_norm

            specular += np.outer(np.power(np.maximum(np.sum(r *
                                 neg_i, axis=1), 0), 50), light_color) * SPECULAR_COEFF

        c = np.clip(color * (AMBIENT_COEFF + diffuse + specular), 0, 1) * 255

        for _x, _y, _z, _c in zip(xs, ys, Z, c):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = _c
                z_buffer[_x, _y] = _z


class Simulation():
    def __init__(self, screen_width, screen_height, objects=[]):
        pygame.init()
        pygame.display.set_caption('BumbleBee Flight')
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.current_mode = Mode.ROTATING
        self.current_shading = Shading.WIREFRAME
        self._objects = objects
        self._selected_object = None
        self.phong_coeff = (360 - FOC_LENGTH) / 360

        self.camera_pos = np.zeros(3)
        self.camera_pos[2] = 5
        self.camera_angles = np.zeros(3)

    def add_cube_btn(self):
        Button(
            self.screen,
            SCREEN_WIDTH - 100,
            0,
            100,
            50,
            text='Добавить куб',
            fontSize=20,
            inactiveColour=(0, 250, 154),
            hoverColour=(128, 128, 128),
            pressedColour=(105, 105, 105),
            onClick=self.create_cube
        )

    def create_cube(self):
        new_cube = Cube()
        self._selected_object = new_cube
        self._objects.append(new_cube)
        
    def simple_ray_tracer(self, x, y, plane):
        phong_coeff = FOC_LENGTH / SCREEN_WIDTH 

        ray_origin = np.array([0, 0, 0])

        x1, y1, z1 = ray_origin
        x2 = x
        y2 = y
        z2 = -phong_coeff

        a, b, c, d = plane

        t = (-d - a * x1 - b * y1 - c * z1) / (a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1))
        
        return np.abs(t)

    def run(self):
        self.add_cube_btn()
        a = 2
        d = -1.5
        point_lights = [
            PointLight(Point3D(0, a, d), [1, 0, 0]),
            PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), [0, 0, 1]),
        ]
        info_text = 'Вращение куба'
        
        while True:
            self.clock.tick(FPS)
            events = pygame.event.get()
            mc = False
            mouse_pos = None
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    mc = True
            keys = pygame.key.get_pressed()
            draw_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 20)
            z_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), np.inf)

            for light in point_lights:
                cube = Cube(0.1)
                cube.translate_cube(*light.point.to_arr())
                triangles = cube.draw_cube(
                    self.camera_pos, self.camera_angles, [])
                for polygon in triangles:
                    triangle = polygon[0]
                    draw_lumbert(draw_buffer, z_buffer, triangle,
                                 np.array(light.color) * 255)
                    
            new_selected_obj = None
            max = np.Inf
            for _, obj in enumerate(self._objects):
                triangles = obj.draw_cube(self.camera_pos, self.camera_angles,
                                          point_lights)

                _, normals = obj.get_c_and_norm(
                    obj.transform_vertices(self.camera_pos, self.camera_angles, False), True)
                
                for polygon in triangles:
                    triangle = polygon[0]
                    face_ind = polygon[1]
                    if self.current_shading == Shading.WIREFRAME:
                        draw_wireframe(draw_buffer, triangle)
                    elif self.current_shading == Shading.LAMBERT:
                        draw_lumbert(draw_buffer, z_buffer, triangle,
                                     obj.lambert_colors[face_ind])
                    elif self.current_shading == Shading.GOURAUD:
                        draw_gouraud(draw_buffer, z_buffer,
                                     triangle, obj.gouraud_colors)
                    elif self.current_shading == Shading.PHONG:
                        lights = np.zeros((len(point_lights), 6))
                        for i, light in enumerate(point_lights):
                            light_pos = Point3D(*light.point.to_arr())
                            light_pos.x += self.camera_pos[0]
                            light_pos.y += self.camera_pos[1]
                            light_pos.z += self.camera_pos[2]
                            rotation = light_pos.rotate_X(self.camera_angles[0]).rotate_Y(
                                self.camera_angles[1]).rotate_Z(self.camera_angles[2])
                            lights[i] = [*rotation.to_arr(), *light.color]
                        draw_phong(draw_buffer, z_buffer,
                                   triangle, normals[face_ind], np.array([1, 1, 1]), lights)
                    
                    if obj == self._selected_object:
                        draw_wireframe(draw_buffer, triangle)
                    
                    if mc and is_inside_triangle(*mouse_pos, *triangle[:, :2].reshape(-1)):
                        mouse_pos_x = (mouse_pos[0] - SCREEN_WIDTH / 2) / SCREEN_WIDTH
                        mouse_pos_y = (mouse_pos[1] - SCREEN_HEIGHT / 2) / SCREEN_WIDTH
                        intersect = self.simple_ray_tracer(mouse_pos_x, mouse_pos_y, normals[face_ind])
                        if intersect < max:
                            max = intersect
                            new_selected_obj = obj
                    
                if obj == self._selected_object:
                    if self.current_mode == Mode.ROTATING:
                        if keys[pygame.K_w]:
                            obj.rotate_cube(Direction.UP)
                        elif keys[pygame.K_s]:
                            obj.rotate_cube(Direction.DOWN)
                        elif keys[pygame.K_a]:
                            obj.rotate_cube(Direction.LEFT)
                        elif keys[pygame.K_d]:
                            obj.rotate_cube(Direction.RIGHT)
                        elif keys[pygame.K_q]:
                            obj.rotate_cube(Direction.FORWARD)
                        elif keys[pygame.K_e]:
                            obj.rotate_cube(Direction.BACKWARD)
                    elif self.current_mode == Mode.TRANSLATING:
                        if keys[pygame.K_w]:
                            obj.translate_cube(0, 0, 0.05)
                        elif keys[pygame.K_s]:
                            obj.translate_cube(0, 0, -0.05)
                        elif keys[pygame.K_a]:
                            obj.translate_cube(-0.05, 0, 0)
                        elif keys[pygame.K_d]:
                            obj.translate_cube(0.05, 0, 0)
                        elif keys[pygame.K_q]:
                            obj.translate_cube(0, -0.05, 0)
                        elif keys[pygame.K_e]:
                            obj.translate_cube(0, 0.05, 0)
                            
            if new_selected_obj is not None:
                self._selected_object = new_selected_obj
                            
            if keys[pygame.K_r]:
                self.current_mode = Mode.ROTATING
                info_text = 'Вращение куба'
            elif keys[pygame.K_t]:
                self.current_mode = Mode.TRANSLATING
                info_text = 'Перемещение куба'
            elif keys[pygame.K_z]:
                self.current_mode = Mode.CAMERA_ROTATING
                info_text = 'Вращение камеры'
            elif keys[pygame.K_x]:
                self.current_mode = Mode.CAMERA_TRANSLATING
                info_text = 'Перемещение камеры'
            elif keys[pygame.K_1]:
                self.current_shading = Shading.WIREFRAME
            elif keys[pygame.K_2]:
                self.current_shading = Shading.LAMBERT
            elif keys[pygame.K_3]:
                self.current_shading = Shading.GOURAUD
            elif keys[pygame.K_4]:
                self.current_shading = Shading.PHONG
            elif keys[pygame.K_i]:
                self._selected_object = None
                
            if self.current_mode == Mode.CAMERA_ROTATING:
                if keys[pygame.K_s]:  # ротейт вверх
                    self.camera_angles[0] += 3
                if keys[pygame.K_w]:  # ротейт вниз
                    self.camera_angles[0] -= 3
                if keys[pygame.K_a]:  # ротейет влево
                    self.camera_angles[1] += 3
                if keys[pygame.K_d]:  # ротейет вправо
                    self.camera_angles[1] -= 3
            elif self.current_mode == Mode.CAMERA_TRANSLATING:
                forward = Point3D(
                    0, 0, -1).rotate_X(-self.camera_angles[0]).rotate_Y(-self.camera_angles[1]).rotate_Z(-self.camera_angles[2])
                right = Point3D(-1, 0, 0).rotate_X(-self.camera_angles[0]).rotate_Y(
                    -self.camera_angles[1]).rotate_Z(-self.camera_angles[2])
                top = Point3D(0, -1, 0).rotate_X(-self.camera_angles[0]).rotate_Y(
                    -self.camera_angles[1]).rotate_Z(-self.camera_angles[2])

                if keys[pygame.K_w]:  # назад
                    self.camera_pos += 0.5 * forward.to_arr()
                if keys[pygame.K_s]:  # вперед
                    self.camera_pos -= 0.5 * forward.to_arr()
                if keys[pygame.K_d]:  # вправо
                    self.camera_pos += 0.5 * right.to_arr()
                if keys[pygame.K_a]:  # влево
                    self.camera_pos -= 0.5 * right.to_arr()
                if keys[pygame.K_e]:  # вниз
                    self.camera_pos += 0.5 * top.to_arr()
                if keys[pygame.K_q]:  # вверх
                    self.camera_pos -= 0.5 * top.to_arr()

            surf = pygame.surfarray.make_surface(draw_buffer)
            self.screen.blit(surf, (0, 0))

            font = pygame.font.Font(None, 26)
            fps_text = font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, np.array([255, 255, 255]))
            info_print = font.render(info_text, True, np.array([255, 255, 255]))
            place = fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            info_place = info_place = info_print.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)
            self.screen.blit(info_print, info_place)

            pygame_widgets.update(events)
            pygame.display.update()


Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
