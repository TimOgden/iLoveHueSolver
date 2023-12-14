from typing import Self

import cv2
import numpy as np
import tqdm as tqdm

from src import img_utils
from alias_types import Color
import pathlib
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Anchor(Point):
    radius: float
    color: Color = None

    def linear_interpolate(self, other: Self, num_points: int) -> tuple[np.array, np.array]:
        x_linear_space = np.linspace(self.x, other.x, num_points)
        y_linear_space = np.linspace(self.y, other.y, num_points)

        red_linear_space = np.linspace(self.color[0], other.color[0], num_points)
        green_linear_space = np.linspace(self.color[1], other.color[1], num_points)
        blue_linear_space = np.linspace(self.color[2], other.color[2], num_points)

        positional_linear_space = np.column_stack([x_linear_space, y_linear_space]).astype(int)
        color_linear_space = np.column_stack([red_linear_space, green_linear_space, blue_linear_space]).astype(int)
        return positional_linear_space, color_linear_space


def find_anchors(img: np.array) -> list[Anchor]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 50, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    anchors = [contour for contour in contours if 70 <= cv2.contourArea(contour) <= 85]
    points = [cv2.minEnclosingCircle(anchor) for anchor in anchors]
    points = [Anchor(int(anchor[0][0]), int(anchor[0][1]), int(anchor[1])) for anchor in points]
    return points


def approximate_contour(contour: np.array, arc_length_percent: float = .01) -> np.array:
    if arc_length_percent <= 0:
        return contour
    epsilon = arc_length_percent * cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
    return approximated_contour


def _find_contour(masked_img: np.array, arc_length_percent: float = .01) -> tuple[np.array]:
    gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parent_hierarchies = np.reshape(hierarchy[:, :, 3] == -1, (-1,))

    valid_contours = []
    for contour, parent_hierarchy in zip(contours, parent_hierarchies):
        if parent_hierarchy:
            valid_contours.append(approximate_contour(contour, arc_length_percent))

    return tuple(valid_contours)


def find_contours(img: np.array, arc_length_percent: float = .01) -> tuple[np.array]:
    contours = []
    colors = np.unique(np.reshape(img, (-1, img.shape[2])), axis=0)
    for color in tqdm.tqdm(colors):
        if all(channel == 0 for channel in color):
            continue
        fuzziness = 5
        lower_bound_color = np.array([np.clip(channel - fuzziness, 0, 255) for channel in color])
        if all(channel == 0 for channel in lower_bound_color):
            continue
        upper_bound_color = np.array([np.clip(channel + fuzziness, 0, 255) for channel in color])
        mask = cv2.inRange(img, lower_bound_color, upper_bound_color)
        if np.count_nonzero(mask) < 256:
            continue
        contours.extend(_find_contour(cv2.bitwise_and(img, img, mask=mask), arc_length_percent=arc_length_percent))
    return contours


def sample_color(img: np.array, point: Anchor) -> Color:
    location = [point.x + int(point.radius * 1.75), point.y]
    return img[*location[::-1]]


def assign_colors_to_anchors(img: np.array, anchors: list[Anchor]) -> None:
    for anchor in anchors:
        anchor.color = sample_color(img, anchor)


def main():
    # filename = pathlib.Path('../imgs/confusion_7.png')
    filename = pathlib.Path('../imgs/test_003.png')
    img = img_utils.load_img(filename)
    contours = find_contours(img)
    img_utils.display_img(cv2.drawContours(img, contours, -1, (255, 0, 0), 3))
    # anchors = find_anchors(img)
    #
    # assign_colors_to_anchors(img, anchors)
    # for anchor in anchors:
    #     print(anchor)
    # img_utils.display_img(img_utils.apply_anchors(img, anchors, color=(0, 0, 255)))


if __name__ == '__main__':
    main()
