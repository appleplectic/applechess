#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import argparse
import torch

from applechess.chess_agent import ChessAgent
from applechess.exceptions import GPUUnsupportedException
from applechess.train_agent import train_agent

# TODO: Fix metrics
# TODO: Add arguments for various parameters in train_agent function
# TODO: Add logging
# TODO: Add Lichess bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple chess engine.")
    parser.add_argument("-s", "--stockfish", required=True, help="Path to the StockFish executable.")
    parser.add_argument("-g", "--force-gpu", action="store_true", help="Force usage of the CUDA.")
    parser.add_argument("-c", "--force-cpu", action="store_true", help="Force usage of the CPU.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--play", action="store_true", help="Play the applechess bot.")
    group.add_argument("--train", type=int, help="Train via self-play.")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.force_gpu:
        raise GPUUnsupportedException(
            "PyTorch does not support your GPU. Ensure you have a supported GPU and a correct version of CUDA is installed.")

    if args.force_cpu:
        device = torch.device("cpu")
    elif args.force_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.play:
        loaded_agent = ChessAgent().to(device)
        loaded_agent.load_model(args.model)
        loaded_agent.play_against_agent(device)
    elif args.train:
        train_agent(device=device, num_games=args.train, train_old="applechess/data/chess_model.pth", calculate_metrics=100)
