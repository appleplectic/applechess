#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# DRL-python
import sys

import chess
import chess.engine
from chess.pgn import Game
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import csv


class ChessEnvironment:
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

    def reset(self):
        self.board = chess.Board()
        return self.board

    def step(self, cmove):
        if self.board.is_legal(cmove):
            prev_board = self.board
            self.board.push(cmove)
            reward = self.get_reward(prev_board, self.board.turn)
            done = self.board.is_game_over()
            return self.board, reward, done
        else:
            # Invalid move, penalize heavily
            return self.board, -10, True

    def get_reward(self, prev_board, color):
        if self.board.is_checkmate():
            # If the current player's turn is the same as the given color, then our bot is checkmated
            if self.board.turn == color:
                return -10
            else:
                # Opponent is checkmated
                return 10
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            # Draw scenarios
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

        # Promotions
        for square, piece in self.board.piece_map().items():
            if not prev_board.piece_at(square) and piece.piece_type != chess.PAWN and self.board.color_at(
                    square) == color:
                reward += 1.5

        # Piece Development
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

        return reward


class SimpleChessAgent(nn.Module):
    def __init__(self):
        super(SimpleChessAgent, self).__init__()
        self.fc = nn.Linear(8 * 8 * 12, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, board_state):
        x = self.fc(board_state)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def get_board_representation(board):
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


def select_move(agent, board):
    # Epsilon-greedy strategy
    epsilon = 0.1
    legal_moves = list(board.legal_moves)

    if random.random() < epsilon:
        return random.choice(legal_moves)

    best_value = float('-inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        board_rep = get_board_representation(board)
        value = agent(board_rep)
        if value > best_value:
            best_value = value
            best_move = move
        board.pop()

    return best_move


def save_model(agent, filename):
    torch.save(agent.state_dict(), filename)


def load_model(agent, filename):
    agent.load_state_dict(torch.load(filename))
    agent.eval()  # Set the model to evaluation mode


def play_against_agent(agent):
    env = ChessEnvironment()
    board = env.reset()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print(board)
            move = select_move(agent, board)
            board.push(move)
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


def append_metrics_to_csv(metrics, filename="metrics.csv"):
    file_exists = False
    try:
        with open(filename, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    # Write to the CSV file
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        # If the file didn't exist before, write the header (column names)
        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def train(train_old=False, num_games=1000):
    # Initialize environment and agent
    env = ChessEnvironment()
    agent = SimpleChessAgent().to(device)
    if train_old:
        agent.load_state_dict(torch.load("chess_model.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for game in tqdm(range(num_games), desc="Training games", unit="game"):
        board = env.reset()
        done = False

        while not done:
            move = select_move(agent, board)
            next_board, reward, done = env.step(move)

            board_rep = get_board_representation(board)
            next_board_rep = get_board_representation(next_board)

            # Q-learning update
            if done:
                target = torch.tensor([reward], device=device)
            else:
                target = torch.tensor([reward], device=device) + 0.99 * agent(next_board_rep).detach()
            prediction = agent(board_rep)

            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            board = next_board

        if game % 100 == 0 and game != 0:
            try:
                save_model(agent, "chess_model.pth")
            except KeyboardInterrupt:
                print("Saving and exiting...")
                save_model(agent, "chess_model.pth")
                sys.exit(1)
            try:
                eval_env = ChessEnvironment()  # New environment for evaluation
                eval_board = eval_env.reset()
                eval_done = False

                while not eval_done:
                    eval_board_rep = get_board_representation(eval_board)
                    with torch.no_grad():  # Ensure we're in evaluation mode
                        eval_q_values = agent(eval_board_rep)
                    best_move_index = torch.argmax(eval_q_values).item()
                    _, eval_reward, eval_done, _ = eval_env.step(best_move_index)
                    eval_board = eval_env.board

                game_pgn = Game.from_board(eval_env.board)
                metrics = analyze_game_with_stockfish(game_pgn, "stockfish.exe")
                append_metrics_to_csv(metrics)
            except Exception as e:
                print(f"Exception occurred: {e}")


def analyze_game_with_stockfish(game_pgn, stockfish_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = game_pgn.board()

    evaluations = []
    stockfish_moves = []

    for move in game_pgn.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        evaluations.append(info["score"].relative.score())
        stockfish_moves.append(info["pv"][0])

    engine.quit()

    acl = sum([abs(evalu) for evalu in evaluations]) / len(evaluations)

    blunders = sum(1 for evalu in evaluations if abs(evalu) > 100)
    mistakes = sum(1 for evalu in evaluations if 50 < abs(evalu) <= 100)
    inaccuracies = sum(1 for evalu in evaluations if 20 < abs(evalu) <= 50)
    perfect_moves = sum(
        1 for actual, recommended in zip(game_pgn.mainline_moves(), stockfish_moves) if actual == recommended)

    metrics = {
        "ACL": acl,
        "Blunders": blunders,
        "Mistakes": mistakes,
        "Inaccuracies": inaccuracies,
        "Perfect Moves": perfect_moves
    }

    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaded_agent = SimpleChessAgent().to(device)
    # load_model(loaded_agent, "chess_model.pth")
    # play_against_agent(loaded_agent)

    train(True, 50000)
