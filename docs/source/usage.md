# Usage

The easiest way to play the engine is to [play the bot on Lichess](https://lichess.org/@/apple-chess) by sending it a challenge request (variants and ultrabullet are not yet supported).
The engine may run slow if too many people are playing it. If the bot is offline, you can run it locally.

## Running Applechess Locally

Applechess is now a package on PyPI:
```bash
pip install applechess
```

To use the `ChessAgent` class:
```python
from applechess.chess_agent import ChessAgent

agent = ChessAgent()
agent.interactive_terminal()
```
Check the documentation for usage details. Note that the API is not stable, and will change frequently.

### Building from Source

In order to run the engine locally, first install Python from [python.org](https://python.org) or your package manager.
Python 3.10+ is required.

Then, clone and install the requirements. You can optionally create a virtual environment before installing the packages.
```bash
git clone https://github.com/appleplectic/applechess.git
cd applechess
python -m pip install -r requirements.txt
```

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
First, follow the previous instructions to download the dependencies.

Now, build the docs:
```bash
sphinx-build -M html docs/source/ docs/build/
```

The built docs should then be available in `docs/build/html/`.

## Licensing and Contributing

Contributions are encouraged - create a pull request or issue!

See the `LICENSE` file for licensing information. All contributions will be under the same license.
