import pygame
import pygame_widgets
from pygame_widgets.button import Button, ButtonArray
import numpy as np
from enum import Enum


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


SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 512
FOV = 260
VIEWER_DISTANCE = 4
FPS = 60


class Point3D():
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

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

    def project(self):
        factor = FOV / (VIEWER_DISTANCE + self.z)
        x = self.x * factor + SCREEN_WIDTH / 2
        y = -self.y * factor + SCREEN_HEIGHT / 2
        return Point3D(x, y, self.z)

    def __determ_sincos(self, angle):
        rad = angle * np.pi / 180
        cosA = np.cos(rad)
        sinA = np.sin(rad)

        return sinA, cosA


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
        self.colors = [(np.random.randint(0, 255, 3)) for _ in np.arange(6)]
        self.angles = [0, 0, 0]

    def draw_cube(self):
        t_vertices = self.transform_vertices()
        avg_Z = self.calculate_avg_z(t_vertices)
        polygons = []
        for z_val in sorted(avg_Z, key=lambda x: x[1], reverse=True):
            f_index = z_val[0]
            f = self.faces[f_index]
            point_list = [
                (t_vertices[f[0]].x, t_vertices[f[0]].y),
                (t_vertices[f[1]].x, t_vertices[f[1]].y),
                (t_vertices[f[1]].x, t_vertices[f[1]].y),
                (t_vertices[f[2]].x, t_vertices[f[2]].y),
                (t_vertices[f[2]].x, t_vertices[f[2]].y),
                (t_vertices[f[3]].x, t_vertices[f[3]].y),
                (t_vertices[f[3]].x, t_vertices[f[3]].y),
                (t_vertices[f[0]].x, t_vertices[f[0]].y)
            ]
            polygons.append((point_list, f_index))
        return polygons

    def rotate_cube(self, direction):
        if direction == Direction.UP:
            self.angles[0] += 2
        elif direction == Direction.DOWN:
            self.angles[0] -= 2
        elif direction == Direction.LEFT:
            self.angles[1] += 2
        elif direction == Direction.RIGHT:
            self.angles[1] -= 2
            
    def translate_cube(self, x, y, z):
        new_points = []
        for vertex in self.vertices:
            p = [vertex.x, vertex.y, vertex.z, 1]
            t = np.identity(4)
            t[3][0], t[3][1], t[3][2] = x, y, z
            d = np.matmul(p, t)
            new_points.append(Point3D(d[0], d[1], d[2]))
        # print(self.vertices)
        # print(new_points)
        self.vertices = new_points

    def transform_vertices(self):
        t_vertices = []
        for vertex in self.vertices:
            rotation = vertex.rotate_X(self.angles[0]).rotate_Y(
                self.angles[1]).rotate_Z(self.angles[2])
            projection = rotation.project()
            t_vertices.append(projection)
        # print(f'{self.vertices} -> {t_vertices}')
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


class Simulation():
    def __init__(self, screen_width, screen_height, objects=[]):
        pygame.init()
        pygame.display.set_caption('BumbleBee Flight')
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.current_mode = Mode.ROTATING
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
            onClicks=(lambda: print('1'), lambda: print('2'), lambda: print('3'))
        )


    def create_cube(self):
        if len(self._objects) > 2:
            return
        self.cubes_bttns()
        self._objects.append(Cube())

    def run(self):
        self.add_cube_btn()
        while True:
            self.clock.tick(FPS)
            self.screen.fill(Color.WHITE.value)
            events=pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            keys=pygame.key.get_pressed()
            for obj in self._objects:
                polygons=obj.draw_cube()
                for polygon in polygons:
                    figure=polygon[0]
                    color=polygon[1]
                    # pygame.draw.polygon(self.screen, obj.colors[color], figure)
                    pygame.draw.aalines(self.screen, Color.BLACK.value, True, figure)
                if keys[pygame.K_UP]:
                    obj.rotate_cube(Direction.UP)
                elif keys[pygame.K_DOWN]:
                    obj.rotate_cube(Direction.DOWN)
                elif keys[pygame.K_LEFT]:
                    obj.rotate_cube(Direction.LEFT)
                elif keys[pygame.K_RIGHT]:
                    obj.rotate_cube(Direction.RIGHT)
                elif keys[pygame.K_F2]:
                    obj.translate_cube(0, 0.05, 0)

            font=pygame.font.Font(None, 26)
            fps_text=font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, Color.BLACK.value)
            place=fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)

            pygame_widgets.update(events)
            pygame.display.update()


# objects = [Cube()]
Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
