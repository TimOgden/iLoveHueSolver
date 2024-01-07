import cv2
import numpy as np
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from src.alias_types import Color
from src.vision_utils import Anchor


def display_img(img: np.array) -> None:
    plt.imshow(img)
    plt.show()


def load_img(filename: pathlib.Path) -> np.array:
    return np.clip(np.asarray(Image.open(filename)), 0, 255)


def apply_anchors(img: np.array, anchors: list[Anchor], color: Color = None) -> np.array:
    img2 = img.copy()
    for anchor in anchors:
        if not color:
            anchor_color = anchor.color.tolist() if anchor.color is not None else (255, 255, 255)
        else:
            anchor_color = color
        cv2.circle(img2, (anchor.x, anchor.y), anchor.radius,
                   anchor_color, -1)
    return img2


def draw_lerped_pieces(img: np.array, line,
                       start_t: float, end_t: float, num_between: int) -> np.array:
    def lerp_color(color_a: Color, color_b: Color, t_: float) -> tuple[int, ...]:
        return tuple([int(lerp(a, b, t_)) for a, b in zip(color_a, color_b)])

    def lerp(a: float, b: float, t_: float) -> float:
        return a * (1 - t_) + b * t_

    piece_a, piece_b = line.pieces
    x_1, y_1 = piece_a.pos
    x_2, y_2 = piece_b.pos

    c_1, c_2 = piece_a.color, piece_b.color

    step = (end_t - start_t) / num_between

    for t in np.arange(start_t, end_t + step, step):
        x = lerp(x_1, x_2, t)
        y = lerp(y_1, y_2, t)
        color = lerp_color(c_1, c_2, t)
        cv2.circle(img, (int(x), int(y)), 30, color, -1)

        cv2.circle(img, (int(x), int(y)), 33, (255, 255, 255), 3)
    cv2.circle(img, (x_1, y_1), 10, (0, 0, 0), -1)
    cv2.circle(img, (x_2, y_2), 10, (0, 0, 0), -1)
    return img


def draw_lerped_pieces_from_points(img: np.array, point_a, point_b,
                                   color_a, color_b,
                                   start_t: float, end_t: float, num_between: int) -> np.array:
    def lerp_color(color_a: Color, color_b: Color, t_: float) -> tuple[int, ...]:
        return tuple([int(lerp(a, b, t_)) for a, b in zip(color_a, color_b)])

    def lerp(a: float, b: float, t_: float) -> float:
        return a * (1 - t_) + b * t_

    x_1, y_1 = int(point_a[0]), int(point_a[1])
    x_2, y_2 = int(point_b[0]), int(point_b[1])

    c_1, c_2 = color_a, color_b

    step = (end_t - start_t) / num_between

    for t in np.arange(start_t, end_t + step, step):
        x = lerp(x_1, x_2, t)
        y = lerp(y_1, y_2, t)
        color = lerp_color(c_1, c_2, t)
        cv2.circle(img, (int(x), int(y)), 20, color, -1)

        cv2.circle(img, (int(x), int(y)), 22, (255, 255, 255), 2)
    # cv2.circle(img, (x_1, y_1), 10, (0, 0, 0), -1)
    # cv2.circle(img, (x_2, y_2), 10, (0, 0, 0), -1)
    return img


def plot_point(img: np.array, x: int, y: int, color: tuple[int, int, int] = None, size: int = 12) -> np.array:
    img = img.copy()
    cv2.circle(img, (x, y), size, color, -1)
    cv2.circle(img, (x, y), size + 3, (0, 0, 0), 3)
    return img


def draw_line(img: np.array, x_1: float, y_1: float, x_2: float, y_2: float, num_points) -> np.array:

    img = img.copy()
    if x_2 == x_1:
        # do nothing
        return img
    step = (x_2 - x_1) / num_points
    for x in np.arange(x_1, x_2, step):
        y = ((y_2 - y_1) / (x_2 - x_1)) * (x - x_2) + y_2
        # print(f'{x=}, {y=}')
        cv2.circle(img, (int(x), int(y)), 8, (68, 120, 82), -1)
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 0), 2)
    return img
