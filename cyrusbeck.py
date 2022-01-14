import pygame
import pygame.gfxdraw
import pygame_widgets as pw
from pygame_widgets.button import Button
import numpy as np
import tripy

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
        else:
            a = figure[i]
            b = figure[(i + 1) % figure_len]
            c = line[0]
            if (b[0] - a[0]) * (c[1] - a[1]) * (c[0] - a[0]) < 0:
                tE = 1
                tL = -1
    if tE > tL:
        return
    l1 = np.array(line[0])
    l2 = np.array(line[1])
    p1 = l1 + (l2 - l1) * tE
    p2 = l1 + (l2 - l1) * tL
    return list(p1), list(p2)


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
        cbb_isPressed = False
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
    elif button == 'cbb':
        draw_button.setInactiveColour((169, 169, 169))
        move_button.setInactiveColour((169, 169, 169))
        rotate_button.setInactiveColour((169, 169, 169))
        scale_button.setInactiveColour((169, 169, 169))
        tabulate_button.setInactiveColour((169, 169, 169))
        db_isPressed = False
        mb_isPressed = False
        rb_isPressed = False
        sb_isPressed = False
        tb_isPressed = False


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
all_dots = [(dots_group, np.array(COLORS['BLACK']), group_counter)]
needed_figure = None
cb_figure = None
left_top = None
right_bot = None
lines = None
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
                    lines = None
                if db_isPressed:
                    if event.button == 1:
                        all_dots[group_counter][0].add(Dot(*pos))
                    elif event.button == 3:
                        dots_group = pygame.sprite.Group()
                        color = COLORS['BLACK']
                        group_counter += 1
                        all_dots.append(
                            (dots_group, color, group_counter))
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
                # elif tb_isPressed:
                #     needed_figures = []
                #     for sample in reversed(all_dots[:-1]):
                #         dots = get_dots_coordinates(sample[0])
                #         if ray_tracing(*pos, dots):
                #             needed_figures.append(sample)
                #     min = np.iinfo(np.int16).max
                #     lowest_figure = None
                #     for figure in needed_figures:
                #         if figure[2] < min:
                #             lowest_figure = figure
                #     idx = all_dots.index(needed_figures[0])
                #     all_dots.insert(0, all_dots.pop(idx))
            else:
                break
        if event.type == pygame.MOUSEBUTTONUP:
            needed_figure = None
            prim_center = None
        if event.type == pygame.QUIT:
            pygame.quit()

    lines = []
    if len(all_dots) > 2:
        first_figure = get_dots_coordinates(all_dots[-2][0])
        figure_triangles = [tuple(reversed(dts)) for dts in tripy.earclip(first_figure)]
        for i in range(len(figure_triangles)):
            for j in range(1, len(all_dots) - 1):
                cur_figure = get_dots_coordinates(all_dots[j][0])
                for k in range(len(all_dots[j][0])):
                    lines.append(cyrus_beck(figure_triangles[i], (cur_figure[k], cur_figure[(k + 1) % len(cur_figure)])))
                    
        
        
    
    
    # first_figure = None
    # cb_lines = []
    # all_lines = []
    # for idx, sample in enumerate(reversed(all_dots[:-1])):
    #     current_figure = get_dots_coordinates(sample[0])
    #     lines = list(zip(current_figure[:-1], current_figure[1:])) + [[current_figure[-1], current_figure[0]]]
        
    #     for idx1, sample1 in enumerate(reversed(all_dots[idx + 1:-1])):
    #         hiding_figure = get_dots_coordinates(sample1[0])
    #         cb_triangles = [tuple(reversed(dts)) for dts in tripy.earclip(hiding_figure)]
    #         new_lines = []
    #         for triangle in cb_triangles:
    #             for line in lines:
    #                 invisible_part = cyrus_beck(triangle, line)
    #                 if invisible_part:
    #                     l1 = (line[0], invisible_part[0])
    #                     l2 = (invisible_part[1], line[1])
    #                     new_lines += [l1, l2]
    #                 else:
    #                     new_lines += [line]

    #         lines = new_lines
            
    #     all_lines += lines
        
    #     print(lines)

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
            pygame.draw.aalines(screen, sample[1], False, dots, blend=1)
        elif len(dots) > 2:
            pygame.gfxdraw.aapolygon(screen, dots, sample[1])
        sample[0].draw(screen)
    fps_text = font.render(
        f'FPS: {np.round(clock.get_fps())}', True, COLORS['WHITE'])
    if lines is not None:
        for line in lines:
            pygame.draw.aalines(screen, COLORS['RED'], True, line)
    if cb_points:
        cb_points.draw(screen)

    place = fps_text.get_rect(center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
    screen.blit(fps_text, place)
    pw.update(events)
    pygame.display.update()
