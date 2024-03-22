#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# applechess lichess_bot.py

"""
lichess_bot.py
~~~~~~~~~~~~~~

This module defines a Game class that plays chess on Lichess using the berserk library.
It uses the ChessAgent class to generate moves.

See README.md for instructions on how to run the bot.

Classes:
    Game: A class that represents a game of chess on Lichess.

Dependencies:
    berserk: A Python library for the Lichess API.

    dotenv: A Python library that reads .env files and adds them to the environment.

    threading: A Python library that provides thread-based parallelism.

    typing: A Python library that provides support for type hints.
"""

import os
import time
import threading
from typing import Callable

import berserk
from berserk.exceptions import ApiError
import chess
from dotenv import dotenv_values

from applechess.chess_agent import ChessAgent

# Depth of the minimax algorithm
DEPTH = 3


class Game(threading.Thread):
    """
    A class that represents a game of chess on Lichess. It plays the game using the ChessAgent class.
    It extends from the threading.Thread class to run the game in a separate thread using the Lichess bot API.
    """
    def __init__(self, bot_client: berserk.Client, game_id: str, **kwargs):
        """
        Initialize the Game class with the Lichess bot client, the game ID, and optional keyword arguments.
        :param bot_client: Lichess bot client to interact with the API.
        :param game_id: ID of the game to play.
        :param kwargs: Other keyword arguments to pass to the threading.Thread class.
        """
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = bot_client
        self.stream = bot_client.bots.stream_game_state(game_id)
        self.agent = ChessAgent()
        self.color = chess.WHITE  # should be overwritten

    def run(self) -> None:
        """
        Overrides method of class threading.Thread to run the game loop to play chess on Lichess.
        It handles incoming game events and makes moves using the agent.
        """
        for bot_event in self.stream:
            if bot_event["type"] == "gameState":
                self._handle_state_change(bot_event)
            elif bot_event["type"] == "gameFull":
                # if player ID of the white player is the same as the bots ID, bot should be white
                if bot_event["white"]["id"] == self.client.account.get()["id"]:
                    self.color = chess.WHITE
                else:
                    self.color = chess.BLACK
                self._handle_state_change(bot_event["state"])

    def _handle_state_change(self, game_state: dict[str, str | int]) -> None:
        """
        Handle the state change of the game. It updates the board state and makes a move if it is the bots turn.
        :param game_state: the state of the game as a dictionary received from the Lichess API.
        """
        if game_state["status"] != "created" and game_state["status"] != "started":
            # the game ended, should gracefully exit
            return

        moves = game_state["moves"].split(" ")
        self.agent.board = chess.Board()
        for move in moves:
            if move != "":
                self.agent.push_move(move)
        if len(self.agent.board.move_stack) % 2 == 0:
            # even number of moves on the board, it is white's turn
            if self.color == chess.WHITE:
                # it is the agent's move
                move = self.agent.get_best_move(DEPTH)
                self.noexcept(self.client.bots.make_move, self.game_id, move.uci())
        else:
            # odd number of moves, it is black's turn
            if self.color == chess.BLACK:
                # it is the agent's move
                move = self.agent.get_best_move(DEPTH)
                self.noexcept(self.client.bots.make_move, self.game_id, move.uci())

    def noexcept(self, func: Callable, *args) -> None:
        """
        Calls a function, and if it throws an ApiError, wait 500ms and try again.
        Useful if the connection momentarily breaks.
        :param func: the function to be called.
        :param args: any arguments that should be passed to the function.
        """
        try:
            func(*args)
        except ApiError:
            print("ApiError handled...")
            time.sleep(0.8)
            func(*args)
            # reinitialize the stream, just in case
            self.stream = self.client.bots.stream_game_state(self.game_id)


if __name__ == "__main__":
    config = {
        **dotenv_values("../.env"),
        **dotenv_values("../.env"),  # load lichess API key from either applechess/ folder or root directory
        **os.environ                 # override loaded values with environment variables
    }
    api_key = config["LICHESS_API_KEY"]

    session = berserk.TokenSession(api_key)
    client = berserk.Client(session)

    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            if event["challenge"]["timeControl"] == "bullet" or event["challenge"]["timeControl"] == "ultraBullet":
                client.bots.decline_challenge(event["challenge"]["id"], "tooFast")
            elif event["challenge"]["variant"]["name"] != "Standard":
                client.bots.decline_challenge(event["challenge"]["id"], "variant")
            else:
                client.bots.accept_challenge(event["challenge"]["id"])
        elif event["type"] == "gameStart":
            game = Game(client, event["game"]["gameId"])
            game.start()
