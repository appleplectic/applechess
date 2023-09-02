#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import torch
import chess


def get_board_representation(board: chess.Board, device: torch.device) -> torch.FloatTensor:
    """
    Gets the board representation in a Float Tensor.

    :param board: The board to get the representation from.
    :param device: The device to place the Tensors on.
    :return: The board representation in a Float Tensor.
    """
    piece_to_idx = {
        'P': 0,  # White Pawn
        'N': 1,  # White Knight
        'B': 2,  # White Bishop
        'R': 3,  # White Rook
        'Q': 4,  # White Queen
        'K': 5,  # White King
        'p': 6,  # Black Pawn
        'n': 7,  # Black Knight
        'b': 8,  # Black Bishop
        'r': 9,  # Black Rook
        'q': 10,  # Black Queen
        'k': 11,  # Black King
    }

    board_str = str(board).replace('\n', '').replace(' ', '')
    one_hot = []

    for char in board_str:
        # Create a zero vector of length 12
        vec = [0] * 12
        if char in piece_to_idx:
            vec[piece_to_idx[char]] = 1
        one_hot.extend(vec)

    return torch.FloatTensor(one_hot).to(device)
