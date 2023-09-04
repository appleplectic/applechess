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

    tensor_representation = torch.FloatTensor(one_hot).view(1, 12, 8, 8).to(device)

    return tensor_representation


def get_game_phase(board: chess.Board) -> str:
    # Assuming starting material is 39 for each side (1 king + 1 queen + 2 rooks + 2 bishops + 2 knights + 8 pawns)
    total_material = sum(
        [len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)) for pt in range(1, 7)])

    if total_material > 60:  # 75% of the total material
        return "opening"
    elif total_material > 30:  # 37.5% of the total material
        return "midgame"
    else:
        return "endgame"


def surrounding_squares(square: int) -> list:
    """
    Gets all surrounding squares of a given square on a chessboard.

    :param square: The square to find the surrounding squares.
    :return: A list of squares surrounding the given square.
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    # Calculate surrounding rank and file values
    surrounding_ranks = [rank + i for i in (-1, 0, 1) if 0 <= rank + i < 8]
    surrounding_files = [file + i for i in (-1, 0, 1) if 0 <= file + i < 8]

    # Convert rank and file back to squares and filter out the original square
    surrounding = [chess.square(file, rank) for rank in surrounding_ranks for file in surrounding_files]
    surrounding.remove(square)

    return surrounding


def get_king_safety(board: chess.Board, color: chess.WHITE | chess.BLACK) -> int:
    king_position = board.king(color)
    safety = 0

    # Get surrounding squares
    surrounding_square = surrounding_squares(king_position)
    for square in surrounding_square:
        # Count defenders
        if board.attackers(color, square):
            safety += 1
        # Count attackers
        if board.attackers(not color, square):
            safety -= 1

    return safety


def load_games_from_pgn(pgn_path):
    games = []
    with open(pgn_path, 'r') as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
            except Exception as e:
                print(f"Exception occurred when parsing the PGN file: {e}")

    return games


def uci_to_index(uci_move):
    if "O-O" in uci_move:
        if uci_move == "O-O":
            return 4096  # White kingside castling
        elif uci_move == "O-O-O":
            return 4097  # White queenside castling
        elif uci_move == "o-o":
            return 4098  # Black kingside castling
        elif uci_move == "o-o-o":
            return 4099  # Black queenside castling
        elif len(uci_move) == 5:  # Pawn promotion
            file_from = ord(uci_move[0]) - ord('a')
            file_to = ord(uci_move[2]) - ord('a')
            piece = uci_move[4]
            promotion_offset = {"q": 0, "r": 1, "b": 2, "n": 3}.get(piece, 0)
            return 4100 + file_from * 8 + file_to + promotion_offset * 64

    else:
        file_from = ord(uci_move[0]) - ord('a')
        rank_from = int(uci_move[1]) - 1
        file_to = ord(uci_move[2]) - ord('a')
        rank_to = int(uci_move[3]) - 1
        return (file_from * 8 + rank_from) * 64 + file_to * 8 + rank_to


def uci_to_one_hot(uci_move):
    index = uci_to_index(uci_move)
    one_hot = torch.zeros(4164)
    one_hot[index] = 1
    return one_hot


def extract_training_data_from_game(game, device):
    board = game.board()
    training_data = []
    result = game.headers["Result"]
    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0  # Assuming draw for any other result

    for move in game.mainline_moves():
        board_representation = get_board_representation(board, device).squeeze(0)
        policy_target = uci_to_one_hot(move.uci())
        training_data.append((board_representation, policy_target, value))
        board.push(move)

    return training_data
