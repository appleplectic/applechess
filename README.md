# Applechess

Applechess is a simple chess engine, written using the minimax and alpha-beta pruning algorithms along with a handcrafted heuristics function.

See more information about the two algorithms on the Wikipedia pages: [minimax](https://en.wikipedia.org/wiki/Minimax) and [alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning).

The easiest way to play the engine is to [play the bot on Lichess](https://lichess.org/@/apple-chess) by sending it a challenge request (variants and bullet/ultrabullet are not yet supported).
The engine may run slow if too many people are playing it. 

## Running Applechess Locally

In order to run the engine locally, first install Python from [python.org](https://python.org) or your package manager.
Python 3.8+ is recommended.

Then, clone and install the requirements. You can optionally create a virtual environment before installing the packages.
```bash
git clone https://github.com/appleplectic/applechess.git
cd applechess
python -m pip install -r requirements.txt
```

To use the `ChessAgent` class:
```python
from applechess.chess_agent import ChessAgent

agent = ChessAgent()
agent.interactive_terminal()
```
Check the documentation for usage details. Note that the API is not stable, and will change frequently.

### Running the Lichess bot locally

Follow the same instructions above to clone and install dependencies.

Next, create a Lichess bot account by first [registering on lichess.org](https://lichess.org/signup) and then obtaining a personal access token under settings.

Upgrade the account to a bot account by running the following command (an account with any games on it cannot become a bot account):
```bash
curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <yourTokenHere>"
```

Finally, create a `.env` file in either the root directory or the applechess/ directory with the following contents:
```
LICHESS_API_KEY=<yourTokenHere>
```

The bot should be ready to run:
```bash
cd applechess
python lichess_bot.py
```

### Building the Documentation
First, install `sphinx` and `myst-parser` (to handle Markdown):
```bash
python -m pip install -U sphinx myst-parser
```

Now, build the docs:
```bash
sphinx-build -M html docs/source/ docs/build/
```

The built docs should then be available in `docs/build/html/`.

## Licensing and Contributing

Contributions are encouraged - create a pull request or issue!

See the `LICENSE` file for licensing information. All contributions will be under the same license.
