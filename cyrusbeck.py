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

    def scale_up(self, mult, center):
        x, y = self.rect.center
        new_x = mult * (x - center[0]) + center[0]
        new_y = mult * (y - center[1]) + center[1]
        self.rect.center = (new_x, new_y)

    def scale_down(self, mult, center):
        x, y = self.rect.center
        new_x = (x - center[0]) / mult + center[0]
        new_y = (y - center[1]) / mult + center[1]
        self.rect.center = (new_x, new_y)


def cyrus_beck(figure, line):
    figure_len = len(figure)
    D = np.array(line[1]) - np.array(line[0])
    pei = []
    normal = []
    for i in range(figure_len):
        pei.append(figure[i])
        x = figure[i][1] - figure[(i + 1) % figure_len][1]
        y = figure[(i + 1) % figure_len][0] - figure[i][0]
        normal.append((x, y))
    t = 0
    tE = 0
    tL = 1
    for i in range(figure_len):
        dot_prod = np.dot(normal[i], D)
        if dot_prod != 0:
            diff = np.array(line[0]) - np.array(pei[i])
            t = np.dot(normal[i], diff) / (-dot_prod)
            if dot_prod < 0:
                tE = np.maximum(tE, t)
            else:
                tL = np.minimum(tL, t)
    if tE > tL:
        print('Линия снаружи')
        return
    l1 = np.array(line[0])
    l2 = np.array(line[1])
    p1 = l1 + (l2 - l1) * tE
    p2 = l1 + (l2 - l1) * tL
    return p1, p2


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
    global cbb_isPressed
    if button == 'db':
        draw_button.setInactiveColour((0, 250, 154))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        cyrus_beck_button.setInactiveColour((169, 169, 169))
        db_isPressed = True
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False
        cbb_isPressed = False
    elif button == 'mb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((0, 250, 154))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        cyrus_beck_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = True
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False
        cbb_isPressed = False
    elif button == 'rb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((0, 250, 154))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        cyrus_beck_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = True
        sb_isPressed = False
        tb_isPressed = False
        cbb_isPressed = False
        cyrus_beck_button.setInactiveColour((169, 169, 169))
    elif button == 'sb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((0, 250, 154))
        tabulate_button.setInactiveColour((169, 169, 169))
        cyrus_beck_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = True
        tb_isPressed = False
        cbb_isPressed = False
    elif button == 'tb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((0, 250, 154))
        cyrus_beck_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = True
        cbb_isPressed = False
    elif button == 'cbb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        cyrus_beck_button.setInactiveColour((0, 250, 154))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False
        cbb_isPressed = True


def ray_tracing(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > np.minimum(p1y, p2y):
            if y <= np.maximum(p1y, p2y):
                if x <= np.maximum(p1x, p2x):
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


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.ceil(qx), np.ceil(qy)


pygame.init()
pygame.display.set_caption("Cyrus-Beck Algorithm")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

dots_group = pygame.sprite.Group()
cb_points = pygame.sprite.Group()
group_counter = 0
click_count = 0
all_dots = [(dots_group, np.array(COLORS['GREEN']), group_counter)]
needed_figure = None
cb_figure = None
left_top = None
right_bot = None
line = None
font = pygame.font.Font(None, 26)

db_isPressed = True
mb_isPressed = False
rb_isPressed = False
sb_isPressed = False
tb_isPressed = False
cbb_isPressed = False

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
cyrus_beck_button = Button(win=screen,
                           x=SCREEN_WIDTH-100,
                           y=250,
                           width=100,
                           height=50,
                           text='CYRUS-BECK',
                           inactiveColour=(169, 169, 169),
                           hoverColour=(128, 128, 128),
                           pressedColour=(105, 105, 105),
                           onClick=lambda: pressed_button('cbb'))

while True:
    clock.tick(FPS)
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if pos[0] < SCREEN_WIDTH - 100:
                if not cbb_isPressed:
                    click_count = 0
                    cb_points.empty()
                    cb_figure = None
                    line = None
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
                    for sample in reversed(all_dots[:-1]):
                        dots = get_dots_coordinates(sample[0])
                        if ray_tracing(*pos, dots):
                            needed_figure = sample
                            prim_center = get_polygon_center(dots)
                            break
                elif sb_isPressed:
                    for sample in reversed(all_dots[:-1]):
                        dots = get_dots_coordinates(sample[0])
                        if ray_tracing(*pos, dots):
                            needed_figure = sample
                            prim_center = get_polygon_center(dots)
                            break
                elif tb_isPressed:
                    needed_figures = []
                    for sample in reversed(all_dots[:-1]):
                        dots = get_dots_coordinates(sample[0])
                        if ray_tracing(*pos, dots):
                            needed_figures.append(sample)
                    min = np.iinfo(np.int16).max
                    lowest_figure = None
                    for figure in needed_figures:
                        if figure[2] < min:
                            lowest_figure = figure
                    idx = all_dots.index(needed_figures[0])
                    all_dots.insert(0, all_dots.pop(idx))
                elif cbb_isPressed:
                    if event.button == 1:
                        if click_count == 0:
                            for sample in reversed(all_dots[:-1]):
                                dots = get_dots_coordinates(sample[0])
                                if ray_tracing(*pos, dots):
                                    cb_figure = dots
                                    break
                        elif click_count == 1:
                            pos1 = pos
                            cb_points.add(Dot(*pos1))
                        elif click_count == 2:
                            pos2 = pos
                            cb_points.add(Dot(*pos2))
                            line = cyrus_beck(cb_figure, (pos1, pos2))
                        click_count += 1
                        if not cb_figure:
                            click_count = 0
                    if event.button == 3:
                        click_count = 0
                        cb_points.empty()
                        cb_figure = None
                        line = None
            else:
                break
        if event.type == pygame.MOUSEBUTTONUP:
            needed_figure = None
            prim_center = None
        if event.type == pygame.QUIT:
            pygame.quit()

    if needed_figure is not None:
        final_pos = pygame.mouse.get_pos()
        if mb_isPressed:
            delta = np.array(final_pos) - np.array(pos)
            delta[0] = np.clip(delta[0], left_top[0], right_bot[0])
            delta[1] = np.clip(delta[1], left_top[1], right_bot[1])
            pos = final_pos
            for dot in needed_figure[0]:
                dot.translate(delta)
            left_top -= delta
            right_bot -= delta

        if rb_isPressed:
            dots = get_dots_coordinates(needed_figure[0])
            center = get_polygon_center(dots)
            diff = np.array(prim_center) - np.array(center)
            angle = final_pos[1] - pos[1]
            for ind, dot in enumerate(needed_figure[0]):
                final_coord = rotate(center, dots[ind], angle / 20)
                dot.rotate(final_coord)
                dot.translate(diff)
            pos = pygame.mouse.get_pos()

        if sb_isPressed:
            dots = np.array(get_dots_coordinates(needed_figure[0]))
            mouse_difference = pos[1] - final_pos[1]
            mult = 1.05
            for dot in needed_figure[0]:
                if mouse_difference > 0:
                    dot.scale_up(mult, prim_center)
                elif mouse_difference < 0:
                    dot.scale_down(mult, prim_center)
                center = get_polygon_center(dots)
                diff = np.array(prim_center) - np.array(center)
                dot.translate(diff)
            pos = pygame.mouse.get_pos()

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
    if line:
        pygame.draw.aalines(screen, COLORS['RED'], True, line)
    if cb_points:
        cb_points.draw(screen)

    place = fps_text.get_rect(center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
    screen.blit(fps_text, place)
    pw.update(events)
    pygame.display.update()
