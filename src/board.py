import logging
import pathlib
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np

from src import img_utils, vision_utils, geometry_utils
from src.vision_utils import Anchor, Point
from src.alias_types import Color
from src.geometry_utils import line_intersection
from functools import cached_property


def lerp(a: float, b: float, t_: float) -> float:
    return a * (1 - t_) + b * t_


@dataclass
class Piece:
    center: Point
    color: Color
    contour: tuple[np.array]
    is_anchor: bool
    verified: bool = False

    def __eq__(self, other: Self) -> bool:
        return self.center.x == other.center.x and self.center.y - other.center.y

    @cached_property
    def area(self):
        return cv2.contourArea(self.contour)

    @property
    def pos(self) -> tuple[int, int]:
        return self.center.x, self.center.y

    @property
    def x(self) -> int:
        return self.center.x

    @property
    def y(self) -> int:
        return self.center.y

    def is_same_shape(self, other: Self) -> bool:
        if self.area == 0:
            raise ValueError(f'{str(self)} has no area')

        if len(self.contour) != len(other.contour):
            return False

        if other.area == 0:
            return False

        if abs(1 - abs(self.area / other.area)) > 1E-1:
            return False
        return True


@dataclass
class LinePieces:
    pieces: tuple[Piece, Piece]

    @property
    def slope(self) -> float:
        try:
            return (self.pieces[1].y - self.pieces[0].y) / (self.pieces[1].x - self.pieces[0].x)
        except ZeroDivisionError:
            return np.inf

    @property
    def b(self) -> float:
        return self.pieces[0].y - self.slope * self.pieces[0].x

    def eval(self, x: float) -> float:
        return self.slope * x + self.b

    def color_at(self, t: float) -> Color:
        c_a, c_b = self.pieces[0].color, self.pieces[1].color
        return tuple(int(lerp(channel_a, channel_b, t)) for channel_a, channel_b in zip(c_a, c_b))

    def point_of_intersection(self, other: Self) -> Point:
        if self.slope == other.slope:
            raise ValueError('No intersection between parallel lines')
        x = (other.b - self.b) / (self.slope - other.slope)
        y = self.eval(x)
        return Point(int(x), int(y))

    def t_to_point(self, t: float) -> tuple[float, float]:
        x, y = lerp(self.pieces[0].x, self.pieces[1].x, t), lerp(self.pieces[0].y, self.pieces[1].y, t)
        return x, y


class Board:
    def __init__(self, filename: pathlib.Path) -> None:
        self.original_img = img_utils.load_img(filename)
        self.current_img = self.original_img.copy()
        self.grayscale_img = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY)
        self.pieces = self.initialize_pieces()

    @cached_property
    def anchors(self) -> list[Piece]:
        return [piece for piece in self.pieces if piece.is_anchor]

    @property
    def floating_pieces(self) -> list[Piece]:
        return [piece for piece in self.pieces if not piece.is_anchor and not piece.verified]

    @property
    def is_solved(self) -> bool:
        return len(self.floating_pieces) == 0

    def initialize_pieces(self) -> list[Piece]:
        logging.info('Initializing pieces...')
        contours = vision_utils.find_contours(self.original_img)
        pieces = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius < 5:
                continue

            color = self.original_img[int(y), int(x), :]
            is_anchor = all(channel == 0 for channel in color)

            if is_anchor:
                # grab color offset by bubble
                color = self.original_img[int(y) + 12, int(x), :]
            new_piece = Piece(center=Point(int(x), int(y)), color=color, is_anchor=is_anchor, contour=contour)
            if new_piece not in pieces:
                pieces.append(new_piece)
        return pieces

    def get_basis_lines(self) -> tuple[LinePieces, LinePieces]:
        pieces = np.random.choice(self.anchors, 4, replace=False)
        line_a, line_b = LinePieces((pieces[0], pieces[1])), LinePieces((pieces[2], pieces[3]))
        return line_a, line_b

    def display(self) -> None:
        img_utils.display_img(self.current_img)

    def draw_pieces(self, img: np.array, draw_centers: bool = True, draw_contours: bool = True) -> np.array:
        img = img.copy()
        for piece in self.pieces:
            img = self.draw_piece(img, piece, draw_centers, draw_contours)
        return img

    def draw_piece(self, img: np.array, piece: Piece, draw_center: bool = True, draw_contour: bool = True) -> np.array:
        img = img.copy()
        if draw_center:
            if piece.is_anchor:
                cv2.circle(img, (piece.center.x, piece.center.y), 10, (0, 0, 0), -1)
            else:
                cv2.circle(img, (piece.center.x, piece.center.y), 10, piece.color.tolist(), -1)
            cv2.circle(img, (piece.center.x, piece.center.y), 13, (0, 0, 0), 3)
        if draw_contour:
            cv2.drawContours(img, [piece.contour], 0, (255, 0, 0), 3)
        return img

    def draw_same_shapes(self, img: np.array, comparing_piece: Piece) -> np.array:
        img = img.copy()
        for piece in self.pieces:
            if piece.is_same_shape(comparing_piece):
                cv2.fillPoly(img, [piece.contour], (0, 0, 255))
        cv2.fillPoly(img, [comparing_piece.contour], (0, 255, 0))
        return img

    def draw_interpolated_colors(self, img: np.array) -> np.array:
        img = img.copy()
        for piece in self.pieces:
            cv2.circle(img, (piece.center.x, piece.center.y), 10,
                       self.screen_coordinate_to_color(piece.center.x, piece.center.y), -1)
        return img

    def draw_interpolated_points_of_interest(self, img: np.array, pieces: list[Piece]) -> np.array:
        img = img.copy()
        for previous_piece, next_piece in zip(pieces[:-1], pieces[1:]):
            img_utils.draw_lerped_pieces(img, previous_piece, next_piece, 0, 1, 5)
        img_utils.draw_lerped_pieces(img, pieces[-1], pieces[0], 0, 1, 5)
        return img

    def draw_basis_pieces(self, img: np.array, line: LinePieces, num_points: int) -> np.array:
        img = img.copy()
        img = img_utils.draw_lerped_pieces(img, line, -3, 4, num_points)
        return img

    def draw_solving_mechanism(self, img: np.array, point: tuple,
                               basis_pieces: tuple[LinePieces, LinePieces]) -> np.array:
        img = img.copy()
        (t1, t2), result = geometry_utils.optimal_lerp_line(*basis_pieces, *point)
        # print(t1)
        # print(result)
        # print('-' * 30)
        for basis_line in basis_pieces:
            img = self.draw_basis_pieces(img, basis_line, 75)
        p1, p2 = basis_pieces[0].t_to_point(t1), basis_pieces[1].t_to_point(t2)
        img = img_utils.draw_line(img, *p1, *p2, 100)
        img = img_utils.plot_point(img, *point)
        return img

    def sample_color(self, x: int, y: int, max_samples: int = 10) -> tuple[Color, tuple[LinePieces, LinePieces]]:
        result = {'success': False}
        t1, t2 = 0, 0
        t_x_y = np.inf
        basis_lines = []
        n_samples = 0

        while n_samples < max_samples and not result['success'] or (t_x_y < -.5 or t_x_y > 1.5):
            basis_lines = self.get_basis_lines()
            try:
                (t1, t2), result = geometry_utils.optimal_lerp_line(*basis_lines, x, y)
            except ValueError:
                pass  # no valid solution for these pair of lines
            p1 = basis_lines[0].t_to_point(t1)
            p2 = basis_lines[1].t_to_point(t2)
            t_x_y = geometry_utils.get_t_value(p1, p2, (x, y))
            n_samples += 1

        c1 = basis_lines[0].color_at(t1)
        c2 = basis_lines[1].color_at(t2)
        # print('t_val:', t_x_y)
        c_x_y = tuple(int(lerp(channel_a, channel_b, t_x_y)) for channel_a, channel_b in zip(c1, c2))
        return c_x_y, basis_lines

    def closest_colored_piece(self, color: tuple[int, int, int]) -> Piece:
        pieces_to_consider = self.floating_pieces
        return min(pieces_to_consider, key=lambda x: np.mean(np.abs(x.color - color)))

    def screen_coordinate_to_color(self, x: int, y: int):
        color, basis_lines = self.sample_color(x, y)
        return color, basis_lines

    def color_to_screen_coordinate(self, color: Color) -> tuple[int, int]:
        raise NotImplementedError()

