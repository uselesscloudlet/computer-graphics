import pygame
from pygame import draw
import pygame_widgets
from pygame_widgets.button import Button, ButtonArray
import numpy as np
from enum import Enum
import skimage.draw
import time


class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (211, 211, 211)


class Mode(Enum):
    MOVING = 0
    ROTATING = 1


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Shading(Enum):
    WIREFRAME = 0
    LAMBERT = 1
    GOURAUD = 2
    PHONG = 3


SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 512
FOV = 360
VIEWER_DISTANCE = 4
FPS = 60

seva_colors = np.array([(230, 25, 75), (60, 180, 75), (255,
                       225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)])


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

    def project(self, x, y, z):
        factor = FOV / (VIEWER_DISTANCE + self.z + z)
        x = (self.x + x) * factor + SCREEN_WIDTH / 2
        y = (-self.y + y) * factor + SCREEN_HEIGHT / 2
        return Point3D(x, y, self.z)

    def __determ_sincos(self, angle):
        rad = angle * np.pi / 180
        cosA = np.cos(rad)
        sinA = np.sin(rad)

        return sinA, cosA


class PointLight(Point3D):
    def __init__(self, point, color):
        self.point = point
        self.color = color


class Cube():
    def __init__(self):
        self.vertices = [
            Point3D(-1, 1, -1),
            Point3D(1, 1, -1),
            Point3D(1, -1, -1),
            Point3D(-1, -1, -1),
            Point3D(-1, 1, 1),
            Point3D(1, 1, 1),
            Point3D(1, -1, 1),
            Point3D(-1, -1, 1)
        ]
        self.faces = [(0, 1, 2, 3),
                      (1, 5, 6, 2),
                      (5, 4, 7, 6),
                      (4, 0, 3, 7),
                      (0, 4, 5, 1),
                      (3, 2, 6, 7)]
        # self.lambert_colors = [(np.random.randint(0, 255, 3)) for _ in np.arange(6)]
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

    def draw_cube(self, point_lights):
        t_vertices = self.transform_vertices()
        tf_vertices = self.transform_vertices(True)
        self.lambert_make_color(point_lights)
        self.gouraud_make_color(point_lights)
        avg_Z = self.calculate_avg_z(t_vertices)
        polygons = []

        _, normals = self.get_c_and_norm(tf_vertices, True)
        pos = Point3D(*self.pos).project(0, 0, 0)
        vis = np.dot([pos.x, pos.y, pos.z + 100000, 0], normals.T)
        for z_val in sorted(avg_Z, key=lambda x: x[1], reverse=True):
            if vis[z_val[0]] >= 0:
                continue
            if (z_val[1] + self.pos[2]) < (-VIEWER_DISTANCE + 1.5):
                continue
            f_index = z_val[0]
            f = self.faces[f_index]

            point_list = np.array([
                (t_vertices[f[0]].x, t_vertices[f[0]].y),
                (t_vertices[f[1]].x, t_vertices[f[1]].y),
                (t_vertices[f[2]].x, t_vertices[f[2]].y),
                (t_vertices[f[3]].x, t_vertices[f[3]].y),
            ])
            polygons.append((np.array(point_list), f_index))
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

    def transform_vertices(self, project=True):
        t_vertices = []
        for vertex in self.vertices:
            rotation = vertex.rotate_X(self.angles[0]).rotate_Y(
                self.angles[1]).rotate_Z(self.angles[2])
            if project:
                projection = rotation.project(
                    self.pos[0], self.pos[1], self.pos[2])
            else:
                projection = rotation
                projection.x += self.pos[0]
                projection.y += self.pos[1]
                projection.z += self.pos[2]
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
        transformed_vertices = self.transform_vertices(False)
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
        transformed_vertices = self.transform_vertices(False)
        _, faces_normals = self.get_c_and_norm(transformed_vertices)
        vertices_normals = []
        for _ in range(len(self.vertices)):
            vertices_normals.append([])
        for i, face in enumerate(self.faces):
            for j in face:
                vertices_normals[j].append(faces_normals[i])
        vertices_normals = np.mean(np.array(vertices_normals), 1)

        for light in point_lights:
            light_vectors = [light.point - vert for vert in self.vertices]
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
            normal = np.cross(v1, v2)

            if return_d:
                d = np.dot(normal, p1)
                normals.append(np.array([*normal, d]))
            else:
                normals.append(normal)
            centers.append(center)

        return np.array(centers), np.array(normals)


class Simulation():
    def __init__(self, screen_width, screen_height, objects=[]):
        pygame.init()
        pygame.display.set_caption('BumbleBee Flight')
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.current_mode = Mode.ROTATING
        self.current_shading = Shading.WIREFRAME
        self._objects = objects

    def add_cube_btn(self):
        button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            SCREEN_WIDTH - 100,  # X-coordinate of top left corner
            0,  # Y-coordinate of top left corner
            100,  # Width
            50,  # Height

            # Optional Parameters
            text='Добавить куб',  # Text to display
            fontSize=20,  # Size of font
            inactiveColour=(0, 250, 154),
            hoverColour=(128, 128, 128),
            pressedColour=(105, 105, 105),
            onClick=self.create_cube  # Function to call when clicked on
        )

    def create_cube(self):
        if len(self._objects) > 2:
            return
        self._objects.append(Cube())

    def trianglething(self, x, y, points):
        pts = points
        (x1, y1, q1), (x2, y2, q2), (x3, y3, q3) = pts
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / \
            ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / \
            ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w3 = 1 - w1 - w2

        return np.outer(q1, w1) + np.outer(q2, w2) + np.outer(q3, w3)

    def area(self, x1, y1, x2, y2, x3, y3):
        return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                       + x3 * (y1 - y2)) / 2.0)

    def is_inside(self, x1, y1, x2, y2, x3, y3, x, y):
        A = self.area(x1, y1, x2, y2, x3, y3)
        A1 = self.area(x, y, x2, y2, x3, y3)
        A2 = self.area(x1, y1, x, y, x3, y3)
        A3 = self.area(x1, y1, x2, y2, x, y)
        return (A1 + A2 + A3) - A < 0.00001

    def line_plane_collision(self, planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            #raise RuntimeError("no intersection or line is within plane")
            return False

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi
        # return True

    def run(self):
        self.add_cube_btn()
        a = 1
        d = 1
        point_lights = [
            # PointLight(Point3D(1000, 1000, 0), [0, 1, 0]),
            # PointLight(Point3D(-1000, 1000, 0), [0, 0, 1]),
            # PointLight(Point3D(0, -1000, 0), [1, 0, 0]),
            PointLight(Point3D(0, a, d), [1, 0, 0]),
            PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), [0, 0, 1]),]
        
        a = 1.5
        for i in range(-2, 3):
            for j in range(-2, 3):
                cube = Cube()
                cube.translate_cube(i, j, 10)
                self._objects.append(cube)
        
        # cube2 = Cube()
        # cube2.translate_cube(-a / 2, -np.sqrt(3) / 2 * a, 0)
        # cube3 = Cube()
        # cube3.translate_cube(a / 2, -np.sqrt(3) / 2 * a, 0)
        
        
        # self._objects.append(cube2)
        # self._objects.append(cube3)
        while True:
            self.clock.tick(FPS)
            self.screen.fill(Color.WHITE.value)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            keys = pygame.key.get_pressed()
            draw_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 0)
            for obj in self._objects:
                polygons = obj.draw_cube(point_lights)

                if self.current_shading == Shading.GOURAUD or self.current_shading == Shading.PHONG:
                    t_vertices = obj.transform_vertices()
                    centers, normals = obj.get_c_and_norm(
                        obj.transform_vertices(False), True)
                for polygon in polygons:
                    figure = polygon[0]
                    color = polygon[1]
                    if self.current_shading == Shading.WIREFRAME:
                        pygame.draw.lines(
                            self.screen, Color.BLACK.value, True, figure)
                    elif self.current_shading == Shading.LAMBERT:
                        pygame.draw.polygon(
                            self.screen, obj.lambert_colors[color], figure)
                        # pygame.draw.polygon(
                        #     self.screen, seva_colors[color], figure)
                    elif self.current_shading == Shading.GOURAUD:
                        x_min = int(figure[:, 0].min())
                        y_min = int(figure[:, 1].min())
                        x_max = int(figure[:, 0].max())
                        y_max = int(figure[:, 1].max())
                        x_delta = x_max - x_min + 1
                        y_delta = y_max - y_min + 1

                        if x_max < 0 or x_min > SCREEN_WIDTH or y_max < 0 or y_min > SCREEN_HEIGHT:
                            continue

                        for i in np.arange(2):
                            X = np.repeat(np.arange(x_delta), y_delta)
                            Y = np.array([*np.arange(y_delta)] * x_delta)

                            screen_mask = ((x_min + X) > 0) & ((x_min + X) < SCREEN_WIDTH) & (
                                (y_min + Y) > 0) & ((y_min + Y) < SCREEN_HEIGHT)
                            X = X[screen_mask]
                            Y = Y[screen_mask]

                            XY_mask = self.is_inside(figure[(0 + i * 2) % 4, 0] - x_min,
                                                     figure[(0 + i * 2) %
                                                            4, 1] - y_min,
                                                     figure[(1 + i * 2) %
                                                            4, 0] - x_min,
                                                     figure[(1 + i * 2) %
                                                            4, 1] - y_min,
                                                     figure[(2 + i * 2) %
                                                            4, 0] - x_min,
                                                     figure[(2 + i * 2) %
                                                            4, 1] - y_min,
                                                     X, Y)
                            X = X[XY_mask]
                            Y = Y[XY_mask]

                            points_r = []

                            for v in obj.faces[color]:
                                pos = t_vertices[v]
                                c = obj.gouraud_colors[v]
                                points_r.append(
                                    (pos.x - x_min, pos.y - y_min, c))
                            if len(X) == 0:
                                continue

                            points_r = np.array(points_r)

                            if i == 0:
                                c = self.trianglething(X, Y, points_r[:3])
                            else:
                                c = self.trianglething(
                                    X, Y, points_r[[0, 2, 3]])

                            draw_buffer[x_min + X, y_min + Y] = c.T
                    elif self.current_shading == Shading.PHONG:
                        plane = normals[color]

                        ray_origin = np.array([0, 0, 1])

                        # range(0, SCREEN_WIDTH, 10):

                        H, W = 2, 1  # SCREEN_HEIGHT / FOV / 2, SCREEN_WIDTH / FOV / 2

                        xxx = np.linspace(-W / 2, W / 2, SCREEN_WIDTH)
                        yyy = np.linspace(-H / 2, H / 2, SCREEN_HEIGHT)
                        X = np.repeat(xxx, len(yyy))
                        Y = np.array([*yyy] * len(xxx))

                        x1, y1, z1 = ray_origin
                        x2 = X
                        y2 = Y
                        z2 = 0

                        a, b, c, d = plane

                        ts = (-d - a * x1 - b * y1 - c * z1) / \
                            (a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1))

                        xs = ts * (x2 - x1) + x1
                        ys = ts * (y2 - y1) + y1
                        zs = ts * (z2 - z1) + z1

                        flags = ts > 0
                        for plane1 in normals:
                            a1, b1, c1, d1 = plane1

                            if np.sum(np.abs(plane + plane1)) < 0.01:
                                continue

                            flags &= (
                                (a1 * xs + b1 * ys + c1 * zs + d1) < 0.01)

                        # print(np.max(flags))
                        xs = xs[flags]
                        ys = ys[flags]
                        zs = zs[flags]

                        if (len(xs) == 0):
                            continue

                        diffuse = np.zeros((len(xs), 3))
                        specular = np.zeros((len(xs), 3))
                        for light in point_lights:
                            light_pos = light.point.to_arr()
                            light_pos[1] = -light_pos[1]
                            l = (light_pos - np.array([xs, ys, zs]).T)
                            l = l / np.linalg.norm(l.T, axis=1)
                            r = 2 * plane[:3] - l
                            r = r / np.linalg.norm(r.T, axis=1)
                            specular_coef = -np.array([xs, ys, zs]).T
                            specular_coef = specular_coef / \
                                np.linalg.norm(specular_coef.T, axis=1)
                            diffuse = diffuse + \
                                np.clip(np.outer(np.dot(plane[:3], l.T), np.array(
                                    light.color)), 0, 1) * 255 * 10

                            wee_wee_1 = r * specular_coef
                            specular = specular + \
                                np.power(np.maximum(wee_wee_1, 0),
                                         1) * 255 * 100

                        #print(np.max(specular))

                        factor = FOV / zs
                        xs = xs * factor + SCREEN_WIDTH / 2
                        ys = -ys * factor + SCREEN_HEIGHT / 2
                        xs = np.round(xs).astype(int)
                        ys = np.round(ys).astype(int)

                        screen_mask = (xs > 0) & (xs < SCREEN_WIDTH) & (
                            ys > 0) & (ys < SCREEN_HEIGHT)
                        xs = xs[screen_mask]
                        ys = ys[screen_mask]
                        zs = zs[screen_mask]
                        diffuse = diffuse[screen_mask] + specular[screen_mask]

                        r = plane * 2 - 1
                        r = r / np.linalg.norm(r)

                        coeff = 8
                        for i in range(coeff):
                            for j in range(coeff):
                                xxs = np.clip(xs + i - coeff // 2,
                                              0, SCREEN_WIDTH - 1)
                                yys = np.clip(
                                    SCREEN_HEIGHT - ys - j + coeff // 2, 0, SCREEN_HEIGHT - 1)
                                # seva_colors[color]
                                draw_buffer[xxs, yys] = diffuse

                if keys[pygame.K_r]:
                    self.current_mode = Mode.ROTATING
                elif keys[pygame.K_t]:
                    self.current_mode = Mode.MOVING
                elif keys[pygame.K_1]:
                    self.current_shading = Shading.WIREFRAME
                elif keys[pygame.K_2]:
                    self.current_shading = Shading.LAMBERT
                elif keys[pygame.K_3]:
                    self.current_shading = Shading.GOURAUD
                elif keys[pygame.K_4]:
                    self.current_shading = Shading.PHONG

                if self.current_mode == Mode.ROTATING:
                    if keys[pygame.K_w]:
                        obj.rotate_cube(Direction.UP)
                    elif keys[pygame.K_s]:
                        obj.rotate_cube(Direction.DOWN)
                    elif keys[pygame.K_a]:
                        obj.rotate_cube(Direction.LEFT)
                    elif keys[pygame.K_d]:
                        obj.rotate_cube(Direction.RIGHT)
                elif self.current_mode == Mode.MOVING:
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

            if self.current_shading == Shading.GOURAUD:
                surf = pygame.surfarray.make_surface(draw_buffer)
                self.screen.blit(surf, (0, 0))
            elif self.current_shading == Shading.PHONG:
                surf = pygame.surfarray.make_surface(draw_buffer)
                self.screen.blit(surf, (0, 0))

            font = pygame.font.Font(None, 26)
            fps_text = font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, Color.BLACK.value)
            place = fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)

            pygame_widgets.update(events)
            pygame.display.update()


Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
