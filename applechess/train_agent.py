#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import torch
import torch.nn as nn
import torch.optim as optim

from chess.pgn import Game

from tqdm import tqdm

from applechess.analysis import analyze_game_with_stockfish, append_metrics_to_csv
from applechess.chess_environ import ChessEnvironment
from applechess.utils import get_board_representation
from applechess.chess_agent import ChessAgent


def train_agent(device: torch.device, num_games: int, train_old: str = "", calculate_metrics: int = 100,
                stockfish: str = "applechess/data/stockfish.exe",
                save_path: str = "applechess/data/chess_model.pth") -> None:
    """
    Trains the agent using self-play.

    :param device: The device used to train the agent.
    :param train_old: If set, the filepath to the old model to continue training.
    :param num_games: How many games to train the model on.
    :param calculate_metrics: Each x games, calculate metrics, evaluate performance, and save the model.
    :param stockfish: The filepath to the stockfish executable.
    :param save_path: The filepath to save the model.
    """

    env = ChessEnvironment()
    agent = ChessAgent().to(device)
    if train_old != "":
        agent.load_state_dict(torch.load(train_old))
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for game in tqdm(range(num_games), desc="Training games", unit="game"):
        board = env.reset()
        done = False

        while not done:
            move = agent.select_move(board, device)
            next_board, reward, done = env.step(move)

            board_rep = get_board_representation(board, device)
            next_board_rep = get_board_representation(next_board, device)

            # Q-learning update
            target = reward + 0.99 * agent(next_board_rep)
            prediction = agent(board_rep)

            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            board = next_board

        if game % calculate_metrics == 0 and game != 0:
            try:
                agent.save_model(save_path)
            except KeyboardInterrupt:
                print("Saving and exiting...")
                agent.save_model(save_path)
                raise KeyboardInterrupt("Interrupted while saving.")
            try:
                # Evaluating the performance of the model
                eval_env = ChessEnvironment()
                eval_board = eval_env.reset()
                eval_done = False

                while not eval_done:
                    eval_board_rep = get_board_representation(eval_board, device)
                    with torch.no_grad():
                        eval_q_values = agent(eval_board_rep)
                    best_move_index = torch.argmax(eval_q_values).item()
                    _, eval_reward, eval_done = eval_env.step(best_move_index)
                    eval_board = eval_env.board

                game_pgn = Game.from_board(eval_env.board)
                metrics = analyze_game_with_stockfish(game_pgn, stockfish)
                append_metrics_to_csv(metrics)
            except Exception as e:
                print(f"Exception occurred: {e}")
