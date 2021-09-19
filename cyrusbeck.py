import pygame
import pygame.gfxdraw
import pygame_widgets as pw
from pygame_widgets.button import Button
import numpy as np

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 512
FPS = 60

COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'GRAY': (211, 211, 211),
}


class Dot(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((3, 3))
        self.image.fill(COLORS['RED'])
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def translate(self, delta):
        x, y = self.rect.center
        self.rect.center = (x + delta[0], y + delta[1])

    def rotate(self, coords):
        self.rect.center = coords


def get_dots_coordinates(group):
    dots_list = group.sprites()
    dots_centers = []
    for i in dots_list:
        dots_centers.append(i.rect.center)
    return dots_centers


def pressed_button(button):
    global db_isPressed
    global mb_isPressed
    global rb_isPressed
    global sb_isPressed
    global tb_isPressed
    if button == 'db':
        draw_button.setInactiveColour((0, 250, 154))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        db_isPressed = True
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False
    elif button == 'mb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((0, 250, 154))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = True
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False
    elif button == 'rb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((0, 250, 154))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = True
        sb_isPressed = False
        tb_isPressed = False
    elif button == 'sb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((0, 250, 154))
        tabulate_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = True
        tb_isPressed = False
    elif button == 'tb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((0, 250, 154))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = True


def ray_tracing(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_polygon_center(dots):
    x_sum = np.sum([dot[0] for dot in dots])
    y_sum = np.sum([dot[1] for dot in dots])
    dots_count = len(dots)
    center = (x_sum / dots_count, y_sum / dots_count)
    return center


pygame.init()
pygame.display.set_caption("Cyrus-Beck Algorithm")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

dots_group = pygame.sprite.Group()
group_counter = 0
all_dots = [(dots_group, COLORS['GREEN'], group_counter)]
needed_figure = None
left_top = None
right_bot = None
font = pygame.font.Font(None, 26)

db_isPressed = True
mb_isPressed = False
rb_isPressed = False
sb_isPressed = False
tb_isPressed = False

draw_button = Button(win=screen,
                     x=SCREEN_WIDTH-100,
                     y=0,
                     width=100,
                     height=50,
                     text='DRAW',
                     inactiveColour=(0, 250, 154),
                     hoverColour=(128, 128, 128),
                     pressedColour=(105, 105, 105),
                     onClick=lambda: pressed_button('db'))

move_button = Button(win=screen,
                     x=SCREEN_WIDTH-100,
                     y=50,
                     width=100,
                     height=50,
                     text='MOVE',
                     inactiveColour=(169, 169, 169),
                     hoverColour=(128, 128, 128),
                     pressedColour=(105, 105, 105),
                     onClick=lambda: pressed_button('mb'))
rotate_button = Button(win=screen,
                       x=SCREEN_WIDTH-100,
                       y=100,
                       width=100,
                       height=50,
                       text='ROTATE',
                       inactiveColour=(169, 169, 169),
                       hoverColour=(128, 128, 128),
                       pressedColour=(105, 105, 105),
                       onClick=lambda: pressed_button('rb'))
scale_button = Button(win=screen,
                      x=SCREEN_WIDTH-100,
                      y=150,
                      width=100,
                      height=50,
                      text='SCALE',
                      inactiveColour=(169, 169, 169),
                      hoverColour=(128, 128, 128),
                      pressedColour=(105, 105, 105),
                      onClick=lambda: pressed_button('sb'))
tabulate_button = Button(win=screen,
                         x=SCREEN_WIDTH-100,
                         y=200,
                         width=100,
                         height=50,
                         text='TABULATE',
                         inactiveColour=(169, 169, 169),
                         hoverColour=(128, 128, 128),
                         pressedColour=(105, 105, 105),
                         onClick=lambda: pressed_button('tb'))

while True:
    clock.tick(FPS)
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if pos[0] < SCREEN_WIDTH - 100:
                if db_isPressed:
                    if event.button == 1:
                        all_dots[group_counter][0].add(Dot(*pos))
                    elif event.button == 3:
                        dots_group = pygame.sprite.Group()
                        rand_color = np.random.randint(0, 255, 3)
                        group_counter += 1
                        all_dots.append(
                            (dots_group, rand_color, group_counter))
                elif mb_isPressed:
                    for sample in reversed(all_dots[:-1]):
                        dots = get_dots_coordinates(sample[0])
                        if ray_tracing(*pos, dots):
                            needed_figure = sample
                            min_x = np.min([dot[0] for dot in dots])
                            max_x = np.max([dot[0] for dot in dots])
                            min_y = np.min([dot[1] for dot in dots])
                            max_y = np.max([dot[1] for dot in dots])
                            left_top = -np.array([min_x, min_y])
                            right_bot = np.array(
                                [SCREEN_WIDTH - 100 - max_x,
                                 SCREEN_HEIGHT - max_y]
                                )
                            break
                elif rb_isPressed:
                    dots = get_dots_coordinates(all_dots[0][0])
                    center = get_polygon_center(dots)
                    print(pos)
                    pass
            else:
                break
        if event.type == pygame.MOUSEBUTTONUP:
            needed_figure = None
        if event.type == pygame.QUIT:
            pygame.quit()

    if mb_isPressed and needed_figure is not None:
        final_pos = pygame.mouse.get_pos()
        delta = np.array(final_pos) - np.array(pos)
        delta[0] = np.clip(delta[0], left_top[0], right_bot[0])
        delta[1] = np.clip(delta[1], left_top[1], right_bot[1])
        pos = final_pos
        for dot in needed_figure[0]:
            dot.translate(delta)
        left_top -= delta
        right_bot -= delta

    screen.fill(COLORS['WHITE'])
    menu_rect = pygame.Rect(SCREEN_WIDTH-100, 0, 100, SCREEN_HEIGHT)
    pygame.draw.rect(screen, COLORS['GRAY'], menu_rect)
    for sample in all_dots:
        dots = get_dots_coordinates(sample[0])
        if len(dots) == 2:
            pygame.draw.aalines(screen, sample[1], True, dots)
        elif len(dots) > 2:
            pygame.gfxdraw.aapolygon(screen, dots, sample[1])
            pygame.gfxdraw.filled_polygon(screen, dots, sample[1])
        sample[0].draw(screen)
    fps_text = font.render(
        f'FPS: {np.round(clock.get_fps())}', True, COLORS['WHITE'])
    place = fps_text.get_rect(center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
    screen.blit(fps_text, place)
    pw.update(events)
    pygame.display.update()
