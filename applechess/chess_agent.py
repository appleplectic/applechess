#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import random

import chess
import chess.engine

import torch
import torch.nn as nn
import torch.nn.functional as f

from applechess.chess_environ import ChessEnvironment
from applechess.utils import get_board_representation


class ChessAgent(nn.Module):
    """
    The agent for the Deep Reinforcement Learning model.
    It extends torch.nn.Module.
    """

    def __init__(self):
        super(ChessAgent, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc = nn.Linear(8 * 8 * 128, 1024)
        self.policy_head = nn.Linear(1024, 8 * 8 * 73)
        self.value_head = nn.Linear(1024, 1)

    def forward(self, board_state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Processes the given board state through the layers.

        :param board_state: A tensor representing the state of the board.
        :return: The processed tensors after passing through the layers.
        """
        x = f.relu(self.conv1(board_state))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # x.size(0) is the batch size

        x = f.relu(self.fc(x))

        # Policy and value heads
        policy = self.policy_head(x)
        value = torch.tanh(
            self.value_head(x))  # Using tanh to ensure output is in range [-1, 1], representing game outcome

        return policy, value

    def select_move(self, board: chess.Board, device: torch.device, epsilon: int = 0.1) -> chess.Move:
        """
        An Epsilon-greedy strategy to select a move.

        :param board: the board to select a move for.
        :param device: The device the tensors should be stored on.
        :param epsilon: The percentage of random moves to be used. Set to 0 if evaluating performance.
        :return: The move the agent has selected.
        """

        legal_moves = list(board.legal_moves)

        if random.random() < epsilon:
            return random.choice(legal_moves)

        best_value = float('-inf')
        best_move = None

        for move in legal_moves:
            board.push(move)
            board_rep = get_board_representation(board, device)
            _, value = self(board_rep)
            if value > best_value:
                best_value = value
                best_move = move
            board.pop()

        return best_move

    def play_against_agent(self, device: torch.device) -> None:
        """
        Play against the agent.

        :param device: The device the tensors are stored on.
        """

        env = ChessEnvironment()
        board = env.reset()

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                print(board)
                print("===============")
                move = self.select_move(board, device, 0)  # use zero to achieve full performance
                print(board.san(move))
                board.push(move)
                print("===============")
            else:
                print(board)
                san_move = input("Enter your move (in SAN format): ")
                try:
                    board.push_san(san_move)
                except ValueError:
                    print("Invalid move format. Try again.")

        print("Game over!")
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Black wins!")
            else:
                print("White wins!")
        else:
            print("It's a draw!")

    def save_model(self, filename: str) -> None:
        """
        Saves the model.

        :param filename: The filepath to save the model to.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename: str) -> None:
        """
        Load the model from a filepath.

        :param filename: The filepath to load the model from.
        """
        self.load_state_dict(torch.load(filename))
