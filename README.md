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

See full usage details below.

```
usage: main.py [-h] [-g] [-c] -m MODEL
               {train,play} ...

A simple chess engine.

positional arguments:
  {train,play}
    train         Train via self-play.
    play          Play the applechess
                  bot.

options:
  -h, --help      show this help
                  message and exit
  -g, --force-gpu
                  Force usage of the
                  CUDA.
  -c, --force-cpu
                  Force usage of the
                  CPU.
  -m MODEL, --model MODEL
                  The model to train
                  or play. If training
                  a new model, put the
                  filepath to where
                  you want the model
                  to be stored.


usage: main.py train [-h] -s STOCKFISH [-m METRICS]
                     [-n NUM_GAMES]

options:
  -h, --help            show this help message and exit   
  -s STOCKFISH, --stockfish STOCKFISH
                        Path to the StockFish
                        executable; default PATH or current directory.
  -e METRICS, --metrics METRICS
                        How many games before saving and  
                        calculating metrics; default 50.  
  -n NUM_GAMES, --num-games NUM_GAMES
                        How many games to train; default  
                        1000.


usage: main.py play [-h]

options:
  -h, --help  show this help message and exit
```

## Contributing
Please file an issue if you encounter one.

Contributions are welcome; please file a pull request. If you're looking for something to do, the TODOs can guide you.

## Licensing
This program was made by Levin Ma, under the MIT license.
See LICENSE for more details.
