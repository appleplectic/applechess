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
                return -10
            else:
                return 10
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.1

        reward = 0.01

        # Reward or penalize for captures
        for piece_type in self.PIECE_VALUES.keys():
            prev_count = len(prev_board.pieces(piece_type, color))
            current_count = len(self.board.pieces(piece_type, color))
            # If a piece of the bots color is missing, it was captured.
            if prev_count > current_count:
                reward -= (self.PIECE_VALUES[piece_type] / 2)

            prev_count_opp = len(prev_board.pieces(piece_type, not color))
            current_count_opp = len(self.board.pieces(piece_type, not color))
            # If a piece of the opposite color is missing, the bot made a capture.
            if prev_count_opp > current_count_opp:
                reward += (self.PIECE_VALUES[piece_type] / 2)

        # Reward promotions
        for square, piece in self.board.piece_map().items():
            if not prev_board.piece_at(square) and piece.piece_type != chess.PAWN and self.board.color_at(
                    square) == color:
                reward += 1.5

        # Reward piece development
        for piece_pos in self.board.pieces(chess.KNIGHT, color):
            if (color == chess.WHITE and chess.square_rank(piece_pos) == 0) or (
                    color == chess.BLACK and chess.square_rank(piece_pos) == 7):
                reward += 0.5

        # Penalize pieces that don't control many squares
        for square, piece in self.board.piece_map().items():
            if piece.color == color:
                controlled_squares = len(list(self.board.attacks(square)))
                control_ratio = controlled_squares / self.MAX_CONTROL[piece.piece_type]
                if control_ratio > 1:
                    control_ratio = 1
                reward += self.PIECE_VALUES[piece.piece_type] * control_ratio

        # Penalize doubled pawns
        for file in range(8):
            if sum(1 for square in self.board.pieces(chess.PAWN, color) if chess.square_file(square) == file) > 1:
                reward -= 0.3

        # Reward passed pawns
        for pawn_pos in self.board.pieces(chess.PAWN, color):
            if not any(self.board.pieces(chess.PAWN, not color) & chess.BB_FILES[chess.square_file(pawn_pos)]):
                reward += 0.7

        return float(reward)
