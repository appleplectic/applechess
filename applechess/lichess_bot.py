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
import logging
from typing import Callable

import berserk
from berserk.exceptions import ApiError, ResponseError
import chess
from dotenv import dotenv_values

from chess_agent import ChessAgent


class Game(threading.Thread):
    """
    A class that represents a game of chess on Lichess. It plays the game using the ChessAgent class.
    It extends from the threading.Thread class to run the game in a separate thread using the Lichess bot API.
    """
    def __init__(self, bot_client: berserk.Client, game_id: str, depth: int, **kwargs):
        """
        Initialize the Game class with the Lichess bot client, the game ID, and optional keyword arguments.
        :param bot_client: Lichess bot client to interact with the API.
        :param game_id: ID of the game to play.
        :param depth: Depth of the minimax search algorithm for the agent.
        :param kwargs: Other keyword arguments to pass to the threading.Thread class.
        """
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = bot_client
        self.stream = bot_client.bots.stream_game_state(game_id)
        self.agent = ChessAgent()
        self.depth = depth
        self.color = chess.WHITE  # should be overwritten
        logging.debug("Successfully initialized Game() class.")

    def run(self) -> None:
        """
        Overrides method of class threading.Thread to run the game loop to play chess on Lichess.
        It handles incoming game events and makes moves using the agent.
        """
        for bot_event in self.stream:
            if bot_event["type"] == "gameState":
                logging.debug("Received gameState request.")
                self._handle_state_change(bot_event)
            elif bot_event["type"] == "gameFull":
                logging.debug("Received gameFull request.")
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
            logging.info(f"Game successfully ended, closing thread {threading.current_thread().ident}")
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
                move = self.agent.get_best_move(self.depth)
                self.noexcept(self.client.bots.make_move, self.game_id, move.uci())
        else:
            # odd number of moves, it is black's turn
            if self.color == chess.BLACK:
                # it is the agent's move
                move = self.agent.get_best_move(self.depth)
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
            logging.debug("Successfully moved piece (noexcept).")
        except ResponseError:
            logging.error("ResponseError received, assuming game is over; ending thread...")
            return
        except ApiError:
            logging.error("ApiError received, assuming connection lost; restarting...")
            time.sleep(0.8)
            func(*args)
            # reinitialize the stream, just in case
            self.stream = self.client.bots.stream_game_state(self.game_id)
            logging.warning("Successfully handled ApiError.")


def main(api_key: str) -> None:
    """
    Main function to run the Lichess bot. It initializes the client with the API key and listens for incoming events.
    :param api_key: the API key for the Lichess bot. Do not hardcode the key, use a .env file or similar.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    session = berserk.TokenSession(api_key)
    client = berserk.Client(session)

    logging.info("Successfully initialized client with API key.")

    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            if event["challenge"]["timeControl"] == "ultraBullet":
                client.bots.decline_challenge(event["challenge"]["id"], "tooFast")
                logging.info(f"Declined challenge {event['challenge']['id']} due to time control.")
            elif event["challenge"]["variant"]["name"] != "Standard":
                client.bots.decline_challenge(event["challenge"]["id"], "variant")
                logging.info(f"Declined challenge {event['challenge']['id']} due to variant.")
            else:
                client.bots.accept_challenge(event["challenge"]["id"])
        elif event["type"] == "gameStart":
            time_control = event["game"]["perf"]
            if time_control == "bullet":
                set_depth = 2
            elif time_control == "blitz":
                set_depth = 3
            else:
                set_depth = 4

            game = Game(client, event["game"]["gameId"], set_depth)
            game.start()
            logging.info(f"Accepted challenge {event['game']['gameId']}; deferring to thread {game.ident}")


if __name__ == "__main__":
    config = {
        **dotenv_values("../.env"),
        **dotenv_values(".env"),      # load lichess API key from either applechess/ folder or root directory
        **os.environ                  # override loaded values with environment variables
    }
    key = config["LICHESS_API_KEY"]
    main(key)
