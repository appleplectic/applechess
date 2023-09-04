# Applechess - a simple chess engine
Applechess is a WIP.

## Setup
Python 3.6+ with pip is required. All versions remain untested besides `Python 3.10.6 (tags/v3.10.6:9c7b4bd) [MSC v.1932 64 bit (AMD64)] on win32`.

Run `pip install -r requirements.txt` to install the required Python packages.

Obtain a StockFish binary at https://stockfishchess.org/ and place it in the PATH or in the current working directory.

## Usage
To train the AI:
```shell
python main.py -m chess_model.pth train [-e METRICS] [-n NUM_GAMES]
```

To play the AI in the terminal:
```shell
python main.py -m chess_model.pth play
```

If you are experiencing this error: `Exception occurred: [WinError 2] The system cannot find the file specified`, you likely do not have StockFish installed.

See full usage details in the help message by running `python main.py --help` or `python main.py train --help`.

## Contributing
Please file an issue if you encounter one.

Contributions are welcome; please file a pull request. If you're looking for something to do, the TODOs can guide you.

## Licensing
This program was made by Levin Ma, under the MIT license.
See LICENSE for more details.
