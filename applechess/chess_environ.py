#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import chess

from applechess.utils import get_king_safety, get_game_phase


class ChessEnvironment:
    """The Chess Environment the agent works in."""

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    MAX_CONTROL = {
        chess.PAWN: 2,
        chess.KNIGHT: 8,
        chess.BISHOP: 13,
        chess.ROOK: 14,
        chess.QUEEN: 27,
        chess.KING: 8
    }

    def calculate_material(self, color: chess.WHITE | chess.BLACK) -> int:
        """
        Calculates the amount of material the color specified has.

        :param color: The color to calculate the amount of material.
        :return: The amount of material the color has.
        """

        return sum(self.PIECE_VALUES[piece_type]
                   for piece_type in range(1, 7)
                   for _square in self.board.pieces(piece_type, color))

    def __init__(self):
        self.board = chess.Board()

    def reset(self) -> chess.Board:
        """Resets the board to its original state."""
        self.board = chess.Board()
        return self.board

    def step(self, cmove: chess.Move) -> (chess.Board, int | float, bool):
        """
        Adds a move to the chess game.

        :param cmove: The move to add to the board.
        :return: The updated board, the reward as calculated by the get_reward function, and whether the game is over.
        """

        if self.board.is_legal(cmove):
            color = self.board.turn
            self.board.push(cmove)
            reward = self.get_reward(color)
            done = self.board.is_game_over()
            return self.board, reward, done
        else:
            # Invalid move, penalize heavily
            return self.board, -10, True

    def get_reward(self, color: chess.WHITE | chess.BLACK) -> int | float:
        """
        Gets the reward or punishment based on the move the agent made.

        :param color: The color the agent is playing.
        :return: The reward (positive) or punishment (negative) given to the agent.
        """

        if self.board.is_checkmate():
            if self.board.turn == color:
                return -10
            else:
                return 10
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return -0.5

        # Small negative to encourage faster games
        reward = -0.1

        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        game_phase = get_game_phase(self.board)
        if game_phase in ["opening", "midgame"]:
            center_control = sum(1 for square in center_squares if
                                 self.board.piece_at(square) and self.board.piece_at(square).color == color)
            reward += (center_control / 4)

        king_safety = get_king_safety(self.board, color) / 6
        reward += king_safety

        white_material = self.calculate_material(chess.WHITE)
        black_material = self.calculate_material(chess.BLACK)
        material_difference = white_material - black_material if color == chess.WHITE else black_material - white_material

        reward += (material_difference / 2)

        return reward
