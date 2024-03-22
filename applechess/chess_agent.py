#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# applechess chess_agent.py

"""
chess_agent.py
~~~~~~~~~~~~~~

This module defines a ChessAgent class that uses the minimax algorithm with alpha-beta pruning to play chess.

Classes:
    ChessAgent: A class that represents the board and the agent that plays chess.
Dependencies:
    chess (formerly python-chess): A chess library for Python, which provides move generation, validation, and more.
"""

import random
from enum import Enum

import chess


class ChessAgent:
    """Class that represents the board and the agent that plays chess."""
    def __init__(self, board: chess.Board | str | None = None):
        """
        Initialize the ChessAgent class with a board.
        :param board: the board to initialize the class with. If None, a new board is created from the default position.
        If a string, the board is created from the FEN string. If a chess.Board object, the board is set to that object.
        """
        if board is None:
            self.board = chess.Board()
        elif isinstance(board, str):
            self.board = chess.Board(fen=board)
        else:
            self.board = board

    class GamePhase(Enum):
        """
        Enum class to represent the game phase.
        Opening is defined as the first 9 moves.
        Midgame is defined as the phase between the opening and the endgame.
        Endgame is defined as the phase when there are less than 24 points of material on the board.
        """
        OPENING = 1
        MIDGAME = 2
        ENDGAME = 3

    @staticmethod
    def get_value(piece: chess.Piece) -> int:
        """
        Get the value of a piece. The value is based on the piece type. If an unknown piece is passed, 0 is returned.
        :param piece: the piece to get the value of.
        :return: the value of the piece.
        """
        match piece.piece_type:
            case chess.BISHOP:
                return 3
            case chess.KNIGHT:
                return 3
            case chess.QUEEN:
                return 9
            case chess.ROOK:
                return 5
            case chess.PAWN:
                return 1
            case _:
                # unknown piece (probably king), return 0 because it doesn't really mean anything
                return 0

    def _get_heuristic(self, board: chess.Board, color: chess.Color) -> int:
        """
        Get the heuristic value of a board state for a given color, using various factors such as material,
        game phase, pawn structure, etc.
        :param board: the board to get the heuristic value of.
        :param color: color to get the heuristic value for.
        :return: the heuristic value of the board state for the given color.
        """
        score = 0
        outcome = board.outcome()

        if outcome is None:
            # game has not ended yet
            # point value of material for both players
            agent_material = 0
            opponent_material = 0

            for _sq, piece in board.piece_map().items():
                if piece.color == color:
                    agent_material += self.get_value(piece)
                else:
                    opponent_material += self.get_value(piece)

            # reward difference of material
            score += agent_material - opponent_material

            if len(board.move_stack) < 9:
                # if less than 9 moves have been played, it is probably the opening
                game_phase = self.GamePhase.OPENING
            elif agent_material + opponent_material < 24:
                # if there is less than 24 points of material, it is probably the endgame
                game_phase = self.GamePhase.ENDGAME
            else:
                game_phase = self.GamePhase.MIDGAME

            if game_phase == self.GamePhase.OPENING:
                # reward bishop/knight development
                # initial squares for bishops and knights based on color
                if color == chess.WHITE:
                    initial_squares = [chess.B1, chess.G1, chess.C1, chess.F1]
                else:
                    initial_squares = [chess.B8, chess.G8, chess.C8, chess.F8]
                for square in initial_squares:
                    piece = board.piece_at(square)
                    if piece is not None and piece.color == color and (
                            piece.piece_type in (chess.BISHOP, chess.KNIGHT)):
                        # was not developed
                        score -= 0.5

                # reward castling
                if board.is_castling(board.peek()):
                    score += 0.25

                # reward control of the center
                center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                for square in center_squares:
                    num_attackers = len(board.attackers(color, square))
                    score += 0.07 * num_attackers

            elif game_phase == self.GamePhase.MIDGAME:
                ...
            elif game_phase == self.GamePhase.ENDGAME:
                ...
        else:
            if outcome.winner is None:
                # game was drawn
                score += 1
            elif outcome.winner == color:
                # agent won the game
                score += 1e6
            else:
                # agent lost the game
                score -= 1e6

        return score

    @staticmethod
    def play_move(board: chess.Board, move: chess.Move) -> chess.Board:
        """
        Play a move on a board and return the new board.
        :param board: the board to play the move on.
        :param move: the move to play.
        :return: a new board with the move played.
        """
        new_board = board.copy()
        new_board.push(move)
        return new_board

    def _minimax(self, node: chess.Board, depth: int, color: chess.Color, maximizing_player: bool) -> int:
        """
        Minimax algorithm to find the best move for a given color.
        :param node: the node to evaluate.
        :param depth: the depth of the search tree.
        :param color: the color to evaluate the node for.
        :param maximizing_player: whether the current node is a maximizing player.
        The maximizing player is the agent, while the minimizing player is the opponent.
        :return: the heuristic value of the node.
        """
        # depth is zero or it is a terminal node
        if depth == 0 or node.outcome() is not None:
            return self._get_heuristic(node, color)

        if maximizing_player:
            value = float("-inf")
            for move in list(node.legal_moves):
                child = self.play_move(node, move)
                value = max(value, self._minimax(child, depth - 1, color, False))
            return value

        value = float("inf")
        for move in list(node.legal_moves):
            child = self.play_move(node, move)
            value = min(value, self._minimax(child, depth - 1, color, True))
        return value

    def _score_move(self, board: chess.Board, move: chess.Move, depth: int) -> int:
        """
        Score a move based on the minimax algorithm.
        :param board: board to score the move on.
        :param move: move to score.
        :param depth: depth of the search tree.
        :return: the score of the move.
        """
        color = board.turn
        next_board = self.play_move(board, move)
        score = self._minimax(next_board, depth - 1, color, False)
        return score

    def get_best_move(self, depth: int) -> chess.Move:
        """
        Agent that plays a move based on the minimax algorithm.
        :param depth: depth of the search tree.
        :return: the best move to play.
        """
        scores = {}
        for move in list(self.board.legal_moves):
            scores[move] = self._score_move(self.board, move, depth)

        best_moves = [move for move in scores.keys() if scores[move] == max(scores.values())]
        return random.choice(best_moves)

    def pprint(self) -> None:
        """Pretty print the board."""
        print(self.board.unicode())

    def push_move(self, move: str) -> None:
        """
        Push a move in SAN/UCI format.
        :param move: string of the move.
        """
        try:
            self.board.push_uci(move)
        except ValueError:
            self.board.push_san(move)

    def agent_gen_push_move(self, depth: int = 3) -> chess.Move:
        """
        Generate the best move for the agent and push it.
        :param depth: depth of the search tree.
        """
        move = self.get_best_move(depth)
        self.board.push(move)
        return move

    def interactive_terminal(self, color: chess.Color = chess.WHITE) -> None:
        """
        Play an interactive game in the terminal against the agent.
        :param color: color to play as.
        """
        while self.board.outcome() is None:
            if color == chess.WHITE:
                self.pprint()
                move = input("Input SAN: ")
                self.push_move(move)
                self.agent_gen_push_move()
            else:
                self.agent_gen_push_move()
                self.pprint()
                move = input("Input SAN: ")
                self.push_move(move)

        outcome = self.board.outcome()
        if outcome.winner is None:
            print("Draw")
        elif outcome.winner == chess.WHITE:
            print("You won!")
        else:
            print("You lost.")
