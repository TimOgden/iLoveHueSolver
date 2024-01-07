import numpy as np
from typing import TYPE_CHECKING

import scipy as scipy

# from board import LinePieces
from alias_types import LinePoints


def slope(line: LinePoints) -> float:
    (x1, y1), (x2, y2) = line
    return (y2 - y1) / (x2 - x1)


def project_onto_line(point: tuple[int, int], line: LinePoints) -> tuple[int, int]:
    original_slope = slope(line)
    perpendicular_slope = -original_slope**-1

    x1, y1 = point
    _, (x2, y2) = line

    # goal: find x s.t. original_slope * (x - x2) + y2 = perpendicular_slope * (x - x1) + y1
    new_x = (y1 - y2 + original_slope * x2 - perpendicular_slope * x1) / (original_slope - perpendicular_slope)
    new_y = perpendicular_slope * (new_x - x1) + y1
    return int(new_x), int(new_y)


def line_intersection(line_a: LinePoints, line_b: LinePoints) -> bool:
    # Unpack the points from the input tuples
    point1_a, point2_a = line_a
    point1_b, point2_b = line_b

    # Calculate the slope of lines A and B
    try:
        slope_a = (point2_a[1] - point1_a[1]) / (point2_a[0] - point1_a[0])
        slope_b = (point2_b[1] - point1_b[1]) / (point2_b[0] - point1_b[0])
    except ZeroDivisionError:
        return False

    # If the slopes are not equal, the lines will intersect
    return np.abs(slope_a - slope_b) > 1E-2


def line_two_points(x_1: int, y_1: int, x_2: int, y_2: int, x: float) -> float:
    return ((y_2 - y_1) / (x_2 - x_1)) * (x - x_1) + y_1


def optimal_lerp_line(line_a, line_b, x: int, y: int) -> tuple[tuple[float, float], scipy.optimize.OptimizeResult]:
    def objective_function(parameters):
        t_a, t_b = parameters
        x_a, y_a = line_a.t_to_point(t_a)
        x_b, y_b = line_b.t_to_point(t_b)

        distance_a_b = (x_b - x_a)**2 + (y_b - y_a)**2

        return distance_a_b

    def constraint_function(parameters):
        t_a, t_b = parameters
        x_a, y_a = line_a.t_to_point(t_a)
        x_b, y_b = line_b.t_to_point(t_b)

        return ((y_b - y_a) / (x_b - x_a)) * (x - x_b) + y_b - y

    parameters0 = np.array([0, 0])
    bounds = [(None, None), (None, None)]

    constraints = {'type': 'eq', 'fun': constraint_function}

    result = scipy.optimize.minimize(objective_function, parameters0, bounds=bounds, constraints=constraints)

    if any(np.isnan(result['x'])):
        raise ValueError('No valid solution')
    return result['x'], result


def get_t_value(p1: tuple[float, float], p2: tuple[float, float], p: tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x, y = p

    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return t

