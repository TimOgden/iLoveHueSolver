import logging
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from src import board, img_utils


logging.getLogger(__name__)


class Solver:
    def __init__(self, game_board: board.Board) -> None:
        self.game_board = game_board
        self.swaps = []

    def solve_puzzle(self, display_lerp: bool = False) -> list[tuple[board.Piece, board.Piece]]:
        while not self.game_board.is_solved:
            for piece in self.game_board.floating_pieces:
                try:
                    new_piece = self.solve_one_piece(piece, display=display_lerp)
                except ValueError:
                    print('Optimization of piece failed...')
                    continue
                if piece != new_piece:
                    self.swaps.append((piece, new_piece))
                piece.verified = True
            if yet_to_be_solved := len(self.game_board.floating_pieces):
                print(f'{yet_to_be_solved} pieces yet to be solved...')
        return self.swaps
    
    def swap_piece_locations(self, piece_a: board.Piece, piece_b: board.Piece) -> None:
        piece_a.center, piece_b.center = piece_b.center, piece_a.center

    def solve_one_piece(self, piece: board.Piece, display: bool = False) -> board.Piece:
        point = piece.pos
        # draw_solving_mechanism(self.game_board, point)
        logging.info(f'Solving randomized point {point}...')
        point_color, basis_lines = self.game_board.screen_coordinate_to_color(*point)
        img = self.game_board.original_img.copy()
        img = self.game_board.draw_solving_mechanism(img, point, basis_lines)

        correct_piece = self.game_board.closest_colored_piece(piece, point_color)
        if display:
            img = self.game_board.draw_piece(img, correct_piece, draw_center=False)
            img = img_utils.plot_point(img, *point, point_color, size=30)
            img_utils.display_img(img)
        self.swap_piece_locations(piece, correct_piece)
        return correct_piece
    
    def solve_all_pieces(self) -> None:
        img = self.game_board.original_img.copy()
    
        logging.info('Solving all floating pieces...')
        for piece in tqdm(self.game_board.floating_pieces):
            point = piece.center.x, piece.center.y
            point_color, basis_lines = self.game_board.screen_coordinate_to_color(*point)
            # img = self.game_board.draw_solving_mechanism(point, basis_lines)
            img = img_utils.plot_point(img, *point, point_color, size=25)
        img_utils.display_img(img)


def main():
    filename = pathlib.Path('../imgs/test_005.png')
    game_board = board.Board(filename)
    solver = Solver(game_board)
    print(f'Found {len(game_board.pieces)} pieces, {len([piece for piece in game_board.pieces if piece.is_anchor])}'
          f' of which are anchors.')
    # game_board.display()
    # solver.solve_one_piece(np.random.choice(game_board.floating_pieces, 1)[0], display=True)
    swaps = solver.solve_puzzle(display_lerp=False)
    print(len(swaps))
    # print('\n'.join([f'{original.pos} -> {new.pos}' for original, new in swaps]))
    for swap in swaps:
        img = game_board.current_img.copy()
        cv2.arrowedLine(img, swap[0].pos, swap[1].pos, (0, 0, 0), 8)
        plt.imshow(img)
        plt.show()

        # replace the color
        img = game_board.current_img.copy()
        c1, c2 = swap[0].color.tolist(), swap[1].color.tolist()
        contour_1, contour_2 = swap[0].contour, swap[1].contour
        cv2.drawContours(img, [contour_1], 0, c2, -1)
        cv2.drawContours(img, [contour_2], 0, c1, -1)
        game_board.current_img = img
    img_utils.display_img(game_board.current_img)


if __name__ == '__main__':
    main()
