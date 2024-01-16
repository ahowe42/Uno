# Uno Game Simulator
This is a simulator for the card game Uno. This simulator is designed to run an experiment with two major playing strategies, each with several parameterized variants. Currently, players are randomly assigned a strategy and associated parameters in each simulated game. I may add the ability to define this as part of the simulator's parameters. *The goal of developing this simulator is to identify an optimal play strategy*. Simulated games follow the rules of Uno. There's no possibility of bluffing and playing a draw 4 wild, so no possibility of challenging a bluff.

Each simulated game saves text output about what's going on to a timestamped logged file. A plethora of game information and results is also pickled to a .p file with the same timestamp. In addition, progress and summary text output is saved to a timestamped experiment file. Experiment results are also pickled to a .p file with the same timestamp.

## Python requirements
I have developed and tested this with python 3.10 on Ubuntu 22 LTS. See requirements.txt.

## Execution
This simulator is designed to be run from the command line with several arguments as `python3 ./Uno.py`, with arguments as listed below. Running `python3 ./Uno.py --help` will print some brief help information on the arguments. None of the arguments are validated at all. If they don't make sense, the results will be crap or the simulator will crash.

### Simulation Parameters
- `--sims`: This is the number of simulated games in the experiment.
- `--player`: Each player is just defined as a player's name, for however many `--player` arguments are passed. It allows for a variable number of players, expecting 2, but doesn't otherwise restrict the number. The game itself says it's good for up to 8 players. The name associated with each `--player` argument will be put into a list in order.
- `--startPlayer`: This is the index into the list of players of who should start. If not passed, the starting player is randomly chosen in each game.
- `--points`: Uno is largely a points-based game, but can be played ignoring points (whoever finishes their hand wins, everyone else loses, no ranking). Passing `--points False` will simulated playing games in this latter fashion.
- `--eDescrip`: This is a brief text description that will be part of the experiment output filenames. If not passed, 'Test' will be used.
- `--gDescrip`: This is a brief text description that will be part of the game output filenames. If not passed, 'Test' will be used.
- `--seed`: This is the seed for the `numpy` random number generator to be used. If not passed, a seed will be generated at the start of the experiment, based on the current time.
- `--debug`: By default, the logging level is `info` or higher. Setting `--debug True` reduces it to `debug` or higher.

Executing, for example `python3 ./Uno.py --sims 10 --player 'Ben Dover' --player 'Hugh Jass' --points False --seed 42 --debug True` will run an experiment of Ben and Hugh playing 10 games, not counting points. The PRNG starting seed will be set to 42, and extra debug logging will be output.

## Strategies
*stratfinishCurrentColor*: Using this strategy, a player will try to play cards of the game's current color as long as possible (i.e., to finish the current color). If there are only 2 players, the strategy will first try to play a skip / reverse to get another turn. Otherwise, any other special card of the same color will be preferred, followed by a value card. If not possible, a wild card is next preferred, followed by the best (see `countPoints`) card from another color.
### Parameters
- `hurtFirst`: Consider a situation in which a player has an option to play a draw 4 wild vs. regular wild, or a draw 2 special card vs. any other special card. If `hurtFirst` is `True`, the strategy will choose the option that forces the next player to draw cards.
- `hailMary`: If the next player has two or fewer cards, and is close to winning, the current player may, if possible act to hinder the next player. Setting `hailMary` to `True` causes this to happen. Under this strategy, the preference order of cards to play is: current color draw 2 > current color skip/reverse > draw 4 > different color draw 2 > different color skip / reverse. Note that this can't defend against a player **other than the next** close to winning.
- `countNotPoint`: Sometimes, a player will not have any cards with the current color, but have several options of different color cards to play. If `countNotPoint` is `False` the strategy will select the best color and best card to play based on the points of cards in the hand. Otherwise, only the number of cards matters. This has no effect if the simulation is run with `--points False`.

*stratSwitchMaxColor*: This strategy selects the best color to play as the color with the most points in the hand. If the best color is the same as the current color, it will switch to the other strategy. Once the best color to play is selected, it will choose a card to play using the priority order of wilds > special cards > value cards (using the highest value of course).
### Parameters
- `hurtFirst`: see above
- `hailMary`: For this strategy, the preferred defense card is a draw 4 wild. If no draw 4 wild is playable, any color draw 2 is preferred, followed by any color reverse / skip. 
- `countNotPoint`: see above
- `addWildPoints`: With `addWildPoints` set to `True`, when counting points to determine the best color to switch to, if a wild must be played first, it's points are added to that color's total points. Note that `countNotPoint` is `True` or the simulation is run with `--points False`, this only adds 1 card.

# Output