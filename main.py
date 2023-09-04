#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import argparse
import os
import torch

from applechess.chess_agent import ChessAgent
from applechess.exceptions import GPUUnsupportedException
from applechess.train_agent import train_agent_self_play, train_agent_on_pgns

# TODO: Add documentation for newly added functions
# TODO: Fix RAM usage
# TODO: Add Lichess bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple chess engine.")
    parser.add_argument("-g", "--force-gpu", action="store_true", help="Force usage of the CUDA.")
    parser.add_argument("-c", "--force-cpu", action="store_true", help="Force usage of the CPU.")
    parser.add_argument("-m", "--model", required=True, type=str, help="The model to train or play. If training a new model, put the filepath to where you want the model to be stored.")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the AI.")
    train_parser.add_argument("-s", "--stockfish", help="Path to the StockFish executable; default is current directory or PATH.")
    train_parser.add_argument("-e", "--metrics", type=int, help="How many games before saving and calculating metrics; default 50.")
    train_parser.add_argument("-n", "--num-games", type=int, help="How many games to train; default 1000.")
    train_parser.add_argument("--use-pgn", type=str, help="If set, this will train the AI on the PGN instead of self-play. Provide a filepath.")

    play_parser = subparsers.add_parser("play", help="Play the applechess bot.")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.force_gpu:
        raise GPUUnsupportedException("PyTorch does not support your GPU. Ensure you have a supported GPU and a correct version of CUDA is installed.")

    if args.force_cpu:
        device = torch.device("cpu")
    elif args.force_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "play":
        loaded_agent = ChessAgent().to(device)
        loaded_agent.load_model(args.model)
        loaded_agent.play_against_agent(device)
    elif args.command == "train":
        if args.use_pgn:
            if not os.path.isfile(args.use_pgn):
                raise FileNotFoundError("Please provide a valid PGN file.")
            train_agent_on_pgns(pgn_path=args.use_pgn, device=device, save_path=args.model)

        else:
            if not args.metrics:
                args.metrics = 50
            if not args.num_games:
                args.num_games = 1000
            if not args.stockfish:
                if os.name == "nt":
                    ext = ".exe"
                else:
                    ext = ""
                args.stockfish = "stockfish" + ext
            if os.path.isfile(args.model):
                train_old = args.model
            else:
                train_old = ""

            train_agent_self_play(device=device, num_games=args.num_games, train_old=train_old, calculate_metrics=args.metrics, stockfish=args.stockfish, save_path=args.model)
    else:
        print("what.")
