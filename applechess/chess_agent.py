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
import os

import random
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        Opening is defined as the first 6 moves.
        Midgame is defined as the phase between the opening and the endgame.
        Endgame is defined as the phase when there are less than 11 pieces on the board.
        """
        OPENING = 1
        MIDGAME = 2
        ENDGAME = 3

    @staticmethod
    def get_value(piece: chess.Piece) -> int:
        """
        Get the value of a piece based on the piece type. If an unknown piece is passed, 0 is returned.
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

    @staticmethod
    def __pawns_per_column(board: chess.Board, color: chess.Color) -> list[int]:
        pawns_per_column = [0, 0, 0, 0, 0, 0, 0, 0]
        pawns = board.pieces(chess.PAWN, color)
        for pawn in pawns:
            if pawn % 8 == 0:    # A file
                pawns_per_column[0] += 1
            elif pawn % 8 == 1:  # B file
                pawns_per_column[1] += 1
            elif pawn % 8 == 2:  # C file
                pawns_per_column[2] += 1
            elif pawn % 8 == 3:  # D file
                pawns_per_column[3] += 1
            elif pawn % 8 == 4:  # E file
                pawns_per_column[4] += 1
            elif pawn % 8 == 5:  # F file
                pawns_per_column[5] += 1
            elif pawn % 8 == 6:  # G file
                pawns_per_column[6] += 1
            elif pawn % 8 == 7:  # H file
                pawns_per_column[7] += 1
        return pawns_per_column

    def _get_heuristic(self, board: chess.Board, color: chess.Color) -> float:
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
            if board.can_claim_draw():
                return 0.5
            # game has not ended yet
            if len(board.move_stack) < 12:
                # if less than 6 moves have been played, it is probably the opening
                game_phase = self.GamePhase.OPENING
            elif len(board.piece_map()) < 13:
                # if there are less than 11 pieces on the board, it is probably the endgame
                game_phase = self.GamePhase.ENDGAME
            else:
                game_phase = self.GamePhase.MIDGAME

            if game_phase in (self.GamePhase.OPENING, self.GamePhase.MIDGAME):
                # reward good king position
                king_square = board.king(color)
                if color == chess.WHITE:
                    if king_square in (chess.G1, chess.C1, chess.B1):
                        score += 0.1
                else:
                    if king_square in (chess.G8, chess.C8, chess.B8):
                        score += 0.1

                # reward control of the center
                center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                for square in center_squares:
                    if board.piece_at(square) is None:
                        num_attackers = len(board.attackers(color, square))
                        score += 0.15 * num_attackers
                    elif board.piece_at(square).color == color:
                        score += 0.4
                    elif board.piece_at(square).color != color:
                        score -= 0.4

                pawns_per_column = self.__pawns_per_column(board, color)
                # punish double pawns
                for pawns in pawns_per_column:
                    if pawns > 1:
                        score -= 0.3

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
                        score -= 0.1

                # penalize excessive queen movement
                queen = board.pieces(chess.QUEEN, color)
                if color == chess.WHITE:
                    squares = chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_2)
                else:
                    squares = chess.SquareSet(chess.BB_RANK_7 | chess.BB_RANK_8)
                if len(queen) == 1 and list(queen)[0] in squares:
                    score += 0.15

            elif game_phase == self.GamePhase.MIDGAME:
                pawns_per_column = self.__pawns_per_column(board, color)
                opponent_pawns_per_column = self.__pawns_per_column(board, not color)

                # reward passed pawns
                for i, pawns in enumerate(pawns_per_column):
                    if pawns > 0 and opponent_pawns_per_column[i] == 0:
                        score += 0.2
                # punish opponent's passed pawns
                for i, pawns in enumerate(opponent_pawns_per_column):
                    if pawns > 0 and pawns_per_column[i] == 0:
                        score -= 0.2

                # punish isolated pawns
                for i, pawns in enumerate(pawns_per_column):
                    if i == 0:
                        if pawns > 0 and pawns_per_column[i + 1] == 0:
                            score -= 0.15
                    elif i == 7:
                        if pawns > 0 and pawns_per_column[i - 1] == 0:
                            score -= 0.15
                    elif pawns > 0 and (pawns_per_column[i - 1] == 0 and pawns_per_column[i + 1] == 0):
                        score -= 0.2

            elif game_phase == self.GamePhase.ENDGAME:
                pawns_per_column = self.__pawns_per_column(board, color)
                opponent_pawns_per_column = self.__pawns_per_column(board, not color)

                # reward passed pawns
                for i, pawns in enumerate(pawns_per_column):
                    if pawns > 0 and opponent_pawns_per_column[i] == 0:
                        score += 1
                # punish opponent's passed pawns
                for i, pawns in enumerate(opponent_pawns_per_column):
                    if pawns > 0 and pawns_per_column[i] == 0:
                        score -= 1

                # reward king activity
                for move in board.legal_moves:
                    if board.piece_at(move.from_square).piece_type == chess.KING:
                        score += 0.1

            # point value of material for both players
            agent_material = 0
            opponent_material = 0

            for _sq, piece in board.piece_map().items():
                if piece.color == color:
                    agent_material += self.get_value(piece)
                else:
                    opponent_material += self.get_value(piece)

            # reward difference of material
            diff_material = agent_material - opponent_material
            if game_phase == self.GamePhase.OPENING:
                score += diff_material
            elif game_phase == self.GamePhase.MIDGAME:
                score += 1.5 * diff_material
            else:
                score += 3 * diff_material
        else:
            if outcome.winner is None:
                # game was drawn
                score += 0.5
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

    def get_ordered_legal_moves(self, board: chess.Board) -> list[chess.Move]:
        """
        Get the legal moves of a board, ordered by the agent's preference for faster pruning.
        This function utilizes the MVV-LVA heuristic.
        :param board: the board to get the legal moves of.
        :return: the ordered legal moves.
        """
        moves = board.legal_moves
        sorted_moves = []

        for move in moves:
            if board.is_capture(move):
                try:
                    victim_value = self.get_value(board.piece_at(move.to_square))
                    attacker_value = self.get_value(board.piece_at(move.from_square))
                    score = victim_value - attacker_value
                except AttributeError:
                    score = 10
            else:
                score = -10

            if move.promotion is not None:
                score += 50
            if board.is_into_check(move):
                score += 5
            sorted_moves.append((score, move))

        sorted_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in sorted_moves]

    def _alphabeta(
            self,
            node: chess.Board,
            depth: int,
            color: chess.Color,
            maximizing_player: bool,
            a: float = float("-inf"),
            b: float = float("inf"),
            parallelize: bool = True
    ) -> float:
        """
        Alpha-beta pruning algorithm to find the best move for a given board state.
        :param node: the board state to find the best move for.
        :param depth: the depth of the search tree.
        :param color: the color to find the best move for.
        :param maximizing_player: the player to maximize the score for. If True, it is the agent's turn.
        Otherwise, it is the opponent's turn.
        :param a: alpha value.
        :param b: beta value.
        :param parallelize: whether to parallelize the search.
        :return: the score of the best move.
        """
        # depth is zero or it is a terminal node
        if depth == 0 or node.outcome() is not None:
            return self._get_heuristic(node, color)

        if parallelize:
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                futures = []
                for move in self.get_ordered_legal_moves(node):
                    child = self.play_move(node, move)
                    futures.append(executor.submit(self._alphabeta, child, depth - 1,
                                                   color, not maximizing_player, a, b, False))

                values = [future.result() for future in as_completed(futures)]
                if maximizing_player:
                    return max(values)
                return min(values)
        else:
            if maximizing_player:
                value = float("-inf")
                for move in self.get_ordered_legal_moves(node):
                    child = self.play_move(node, move)
                    value = max(value, self._alphabeta(child, depth - 1, color, False, a, b, False))
                    a = max(a, value)
                    if value > b:
                        break
                return value

            value = float("inf")
            for move in self.get_ordered_legal_moves(node):
                child = self.play_move(node, move)
                value = min(value, self._alphabeta(child, depth - 1, color, True, a, b, False))
                b = min(b, value)
                if value <= a:
                    break
            return value

    def get_best_move(self, depth: int, parallelize: bool = False) -> chess.Move:
        """
        Agent that plays a move based on the alpha/beta-minimax algorithm.
        :param depth: depth of the search tree.
        :param parallelize: whether to parallelize the search.
        :return: the best move to play.
        """
        color = self.board.turn
        scores = {}
        if parallelize:
            with ThreadPoolExecutor() as executor:
                future_to_move = {}
                for move in list(self.board.legal_moves):
                    next_board = self.play_move(self.board, move)
                    future_to_move = {executor.submit(self._alphabeta, next_board, depth - 1, color,
                                                      False, float("-inf"), float("inf"), True): move}
                for future in as_completed(future_to_move):
                    move = future_to_move[future]
                    score = future.result()
                    scores[move] = score

        else:
            for move in list(self.board.legal_moves):
                next_board = self.play_move(self.board, move)
                scores[move] = self._alphabeta(next_board, depth - 1, color, False, True)

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

    def agent_gen_push_move(self, depth: int) -> chess.Move:
        """
        Generate the best move for the agent and push it.
        :param depth: depth of the search tree.
        """
        move = self.get_best_move(depth)
        self.board.push(move)
        return move

    def interactive_terminal(self, color: chess.Color = chess.WHITE, depth: int = 3) -> None:
        """
        Play an interactive game in the terminal against the agent.
        :param depth: depth of the search tree.
        :param color: color to play as.
        """
        while self.board.outcome() is None:
            if color == chess.WHITE:
                self.pprint()
                move = input("Input SAN: ")
                self.push_move(move)
                self.agent_gen_push_move(depth)
            else:
                self.agent_gen_push_move(depth)
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
