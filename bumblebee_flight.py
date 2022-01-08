import pygame
from pygame import draw
import pygame_widgets
from pygame_widgets.button import Button, ButtonArray
import numpy as np
from enum import Enum
import skimage.draw


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
FOV = 90
VIEWER_DISTANCE = 4
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
        # print(self.pos)
        pos = Point3D(*self.pos).project(0, 0, 0)
        #vis = np.dot([self.pos[0], self.pos[1], -(VIEWER_DISTANCE + self.pos[2]), 0], normals.T)
        vis = np.dot([pos.x, pos.y, pos.z + 100000, 0], normals.T)
        # print([pos.x, pos.y, pos.z + 100000, 0])
        # print(self.pos[2])
        # print(avg_Z)
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

    def cubes_bttns(self):
        objs_count = len(self._objects) + 1
        buttonArray = ButtonArray(
            # Mandatory Parameters
            self.screen,  # Surface to place button array on
            SCREEN_WIDTH - 100,  # X-coordinate
            70,  # Y-coordinate
            100,  # Width
            50 * objs_count,  # Height
            (1, objs_count),  # Shape: 2 buttons wide, 2 buttons tall
            border=0,  # Distance between buttons and edge of array
            # Sets the texts of each button (counts left to right then top to bottom)
            texts=('1', '2', '3'),
            # When clicked, print number
            onClicks=(lambda: print('1'), lambda: print(
                '2'), lambda: print('3'))
        )

    def create_cube(self):
        if len(self._objects) > 2:
            return
        self.cubes_bttns()
        self._objects.append(Cube())

    def generate_gradient(from_color, to_color, height, width):
        channels = []
        for channel in range(3):
            from_value, to_value = from_color[channel], to_color[channel]
            channels.append(
                np.tile(
                    np.linspace(from_value, to_value, width), [height, 1],
                ),
            )
        return np.dstack(channels)

    def binterpolation(self, x, y, points):
        pts = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = pts

        # if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        #     print('The given points do not form a rectangle')
        #     raise "NO"

        P = (q11 * (x2 - x) * (y2 - y) +
             q21 * (x - x1) * (y2 - y) +
             q12 * (x2 - x) * (y - y1) +
             q22 * (x - x1) * (y - y1)
             ) / ((x2 - x1) * (y2 - y1) + 0.01)

        return P

    def trainglething(self, x, y, points):
        pts = points
        #print(points)
        (x1, y1, q1), (x2, y2, q2), (x3, y3, q3), (_x, _y, _z) = pts
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w3 = 1 - w1 - w2
        
        p1 = np.array([x1, y1])
        res1 = q1 * w1 + q2 * w2 + q3 * w3
        
        (x1, y1, q1), (_x, _y, _q), (x3, y3, q3), (x2, y2, q2) = pts
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w3 = 1 - w1 - w2
        
        
        p2 = np.array([x3, y3])
        res2 = q1 * w1 + q2 * w2 + q3 * w3
                
        above = np.cross(np.array(list(zip(x, y))) - p1, p1 - p2) < 0
        
    
        #return np.where(above, np.full_like(res1, 255), np.full_like(res2, 0))
        return np.where(above, res1, res2)


    def run(self):
        self.add_cube_btn()
        point_lights = [PointLight(Point3D(50, 5, 0), [0, 1, 0]), PointLight(
            Point3D(-50, 5, 0), [0, 0, 1]), PointLight(
            Point3D(0, -5, 0), [1, 0, 0])]
        while True:
            self.clock.tick(FPS)
            self.screen.fill(Color.WHITE.value)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            keys = pygame.key.get_pressed()
            draw_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 100)
            for obj in self._objects:
                polygons = obj.draw_cube(point_lights)
                t_vertices = obj.transform_vertices()
                for poly_ind, polygon in enumerate(polygons):
                    figure = polygon[0]
                    color = polygon[1]
                    if self.current_shading == Shading.WIREFRAME:
                        pygame.draw.aalines(
                            self.screen, Color.BLACK.value, True, figure)
                    elif self.current_shading == Shading.LAMBERT:
                        pygame.draw.polygon(
                            self.screen, obj.lambert_colors[color], figure)
                    elif self.current_shading == Shading.GOURAUD:
                        x_min = int(figure[:, 0].min())
                        y_min = int(figure[:, 1].min())
                        x_max = int(figure[:, 0].max())
                        y_max = int(figure[:, 1].max())

                        x_delta = x_max - x_min + 1
                        y_delta = y_max - y_min + 1

                        rr, cc = skimage.draw.polygon(
                            figure[:, 0] - x_min, figure[:, 1] - y_min)

                        points_r = []
                        points_g = []
                        points_b = []

                        corners = [[[], []], [[], []]]
                        
                        for i, v in enumerate(obj.faces[color]):
                            pos = t_vertices[v]
                            c = obj.gouraud_colors[v]

                            x = 0 if abs(
                                pos.x - x_min) < abs(pos.x - x_max) else 1
                            y = 0 if abs(
                                pos.y - y_min) < abs(pos.y - y_max) else 1
                            
                            
                            points_r.append((pos.x - x_min, pos.y - y_min, c))
                            points_g.append((pos.x - x_min, pos.y - y_min, c[1]))
                            points_b.append((pos.x - x_min, pos.y - y_min, c[2]))
                            corners[x][y].append(c)

                            

                        for x in [0, 1]:
                            for y in [0, 1]:
                                if len(corners[x][y]) > 0:
                                    corners[x][y] = np.mean(corners[x][y], axis=0)
                        for x in [0, 1]:
                            for y in [0, 1]:
                                if len(corners[x][y]) == 0:
                                    if len(corners[(x + 1) % 2][y]) == 0:
                                        corners[x][y] = corners[x][(y + 1) % 2]
                                    elif len(corners[x][(y + 1) % 2]) == 0:
                                        corners[x][y] = corners[(x + 1) % 2][y]
                                    else:
                                        corners[x][y] = np.mean([corners[(x + 1) % 2][y], corners[x][(y + 1) % 2]], axis=0)


                        # points_r.append((0, 0, corners[0][0][0]))
                        # points_r.append((x_delta, 0, corners[1][0][0]))
                        # points_r.append((x_delta, y_delta, corners[1][1][0]))
                        # points_r.append((0, y_delta, corners[0][1][0]))
                        
                        # points_g.append((0, 0, corners[0][0][1]))
                        # points_g.append((x_delta, 0, corners[1][0][1]))
                        # points_g.append((x_delta, y_delta, corners[1][1][1]))
                        # points_g.append((0, y_delta, corners[0][1][1]))
                        
                        # points_b.append((0, 0, corners[0][0][2]))
                        # points_b.append((x_delta, 0, corners[1][0][2]))
                        # points_b.append((x_delta, y_delta, corners[1][1][2]))
                        # points_b.append((0, y_delta, corners[0][1][2]))
                            
                        X = np.repeat(np.arange(x_delta), y_delta)
                        Y = np.array([*range(y_delta)] * x_delta)


                        # rs = np.clip(self.binterpolation(
                        #     X, Y, points_r), 0, 255)
                        # gs = np.clip(self.binterpolation(
                        #     X, Y, points_g), 0, 255)
                        # bs = np.clip(self.binterpolation(
                        #     X, Y, points_b), 0, 255)
                        
                        rs = self.trainglething(X, Y, points_r)
                        gs = self.trainglething(X, Y, points_g)
                        bs = self.trainglething(X, Y, points_b)
                        
                        draw_buffer[x_min + rr, y_min + cc] = np.array([rs[rr * y_delta + cc], gs[rr * y_delta + cc], bs[rr * y_delta + cc]]).T

                if keys[pygame.K_F1]:
                    self.current_mode = Mode.ROTATING
                elif keys[pygame.K_F2]:
                    self.current_mode = Mode.MOVING
                elif keys[pygame.K_1]:
                    self.current_shading = Shading.WIREFRAME
                elif keys[pygame.K_2]:
                    self.current_shading = Shading.LAMBERT
                elif keys[pygame.K_3]:
                    self.current_shading = Shading.GOURAUD

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
            
            font = pygame.font.Font(None, 26)
            fps_text = font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, Color.BLACK.value)
            place = fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)

            pygame_widgets.update(events)
            pygame.display.update()


# objects = [Cube()]
Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
