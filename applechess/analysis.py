#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# applechess

import chess
import csv


def append_metrics_to_csv(metrics: dict, filename: str = "metrics.csv") -> None:
    """
    Append metrics from training to a CSV file.

    :param metrics: The metrics to append.
    :param filename: The CSV filepath to append the metrics to.
    """
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


def analyze_game_with_stockfish(game_pgn: chess.pgn.Game, stockfish_path: str) -> dict:
    """
    Analyzes a game with StockFish and returns various metrics.

    :param game_pgn: The game to analyze.
    :param stockfish_path: The filepath that stockfish is located at.
    :return: The various metrics calculated in this function.
    """

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = game_pgn.board()

    evaluations = []
    stockfish_moves = []

    for move in game_pgn.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        evaluations.append(info["score"].relative.score())
        try:
            stockfish_moves.append(info["pv"][0])
        except KeyError:
            # Weird pv error
            return {}

    engine.quit()

    if len(evaluations) != 0 and None not in evaluations:
        acl = sum([abs(evalu) for evalu in evaluations]) / len(evaluations)
    else:
        # Weird NoneType error
        return {}

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
