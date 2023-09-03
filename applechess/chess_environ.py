#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import chess


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
            prev_board = self.board
            self.board.push(cmove)
            reward = self.get_reward(prev_board, self.board.turn)
            done = self.board.is_game_over()
            return self.board, reward, done
        else:
            # Invalid move, penalize heavily
            return self.board, -10, True

    def get_reward(self, prev_board: chess.Board, color: chess.WHITE | chess.BLACK) -> int | float:
        """
        Gets the reward or punishment based on the move the agent made.

        :param prev_board: The board before the move was made.
        :param color: The color the agent is playing.
        :return: The reward (positive) or punishment (negative) given to the agent, between -10 and 10.
        """

        if self.board.is_checkmate():
            if self.board.turn == color:
                return -50
            else:
                return 50
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return -0.5

        # Small negative to encourage faster games
        reward = -0.001

        white_material = self.calculate_material(chess.WHITE)
        black_material = self.calculate_material(chess.BLACK)
        material_difference = white_material - black_material if color == chess.WHITE else black_material - white_material

        # Reward or penalize for captures
        for piece_type in self.PIECE_VALUES.keys():
            prev_count = len(prev_board.pieces(piece_type, color))
            current_count = len(self.board.pieces(piece_type, color))
            # If a piece of the bots color is missing, it was captured.
            if prev_count > current_count:
                reward -= (self.PIECE_VALUES[piece_type] / 3)

            prev_count_opp = len(prev_board.pieces(piece_type, not color))
            current_count_opp = len(self.board.pieces(piece_type, not color))
            # If a piece of the opposite color is missing, the bot made a capture.
            if prev_count_opp > current_count_opp:
                reward += (self.PIECE_VALUES[piece_type] / 3)

        reward += (material_difference / 2)

        return reward
