# Uno Game Simulator
This is a simulator for the card game Uno. This simulator is designed to run an experiment with two major playing strategies, each with several parameterized variants. Players can either have their strategies defined by the experimenter, or assigned randomly. Simulated games follow the rules of Uno with one exception. There's no possibility of bluffing and playing a draw 4 wild, so no possibility of challenging a bluff and having to draw extra cards from losing a challenge.

Each simulated game saves text output about what's going on to a timestamped logged file. A plethora of game information and results is also pickled to a .p file with the same timestamp. In addition, progress and summary text output is saved to a timestamped experiment file. Experiment results are also pickled to a .p file with the same timestamp.

*The goal of developing this simulator is to identify an optimal play strategy*. 

## Python Requirements
I have developed and tested this with python 3.10 on Ubuntu 22 LTS. See [package requirements file](./requirements.txt).

## Execution
This simulator - wholly contained in [Uno.py](./Uno.py) is designed to be run from the command line with several arguments as `python3 ./Uno.py`, with arguments as listed below. Running `python3 ./Uno.py --help` will print some brief help information on the arguments. None of the arguments are validated at all. If they don't make sense, the results will be crap or the simulator will crash.

On a computer with multiple cores, running games in parallel can significantly accelerate an experiment. For example, on my laptop, an experiment with 500 simulations of 4 players with debug loggin on takes about 1.3 minutes in parallel, or 2.3 minutes in serial.

### Simulation Parameters
- `--config`: Here the experimenter can pass a path + filename of a config file to read instead of requiring the below arguments. A sample [config file][./config.env] is included.
- `--para`: This flag tells the simulator to run games in parallel (True) or sequence (False)
- `--sims`: This is the number of simulated games in the experiment.
- `--player`: Each player is just defined as a player's name, for however many `--player` arguments are passed. It allows for a variable number of players, expecting 2, but doesn't otherwise restrict the number. The game itself says it's good for up to 8 players. The name associated with each `--player` argument will be put into a list in order.
- `--startPlayer`: This is the index into the list of players of who should start. If not passed, the starting player is randomly chosen in each game.
- `--points`: Uno is largely a points-based game, but can be played ignoring points (whoever finishes their hand wins, everyone else loses, no ranking). Passing `--points False` will simulated playing games in this latter fashion.
- `--eDescrip`: This is a brief text description that will be part of the experiment output filenames. If not passed, 'Test' will be used.
- `--gDescrip`: This is a brief text description that will be part of the game output filenames. If not passed, 'Test' will be used.
- `--seed`: This is the seed for the `numpy` random number generator to be used. If not passed, a seed will be generated at the start of each game, based on the current time. Note that the seed is used for each game - it's not for the entire experiment.
- `--debug`: By default, the logging level is `info` or higher. Setting `--debug True` reduces it to `debug` or higher.

Executing, for example `python3 ./Uno.py --para True --sims 10 --player 'Ben Dover' --player 'Hugh Jass' --points False --seed 42 --debug True` will run an experiment of Ben and Hugh playing 10 games, not counting points. The PRNG starting seed will be set to 42, and extra debug logging will be output.

A benefit of using a config file instead of command-line arguments is that the experimenter can define strategies & parameterizations for any of the players. Those players who don't have a defined strategy will have their randomly selected from all possibilities.

## Strategies
The simulator has two major playing strategies, each with several parameterized variants. If any player does not have a strategy defined, it and it's parameters will be randomly selected from all possible combinations of strategies & parameterizations.

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

## Output
### Game
Game logs record a lot of information about how the game was setup (players, random seed, etc.), along with turn-by-turn details. Whether run in parallel or serially, games are set up and run from the `setupRunGame` function. This function returns four objects:

- `stratParams`: This is a list of strategy data used by each player in a string holding the strategy function, parameters dict, a string representation, and the parameters' index into the strategy design.
- `gameResults`: This is just passing out the summary results returned by `Game.Play()`; it is a dictionary including:
    - tuple of timing information
    - integer index into the list of players of the winner
    - list summary of played cards from `Game.cardSummary()`
    - list summary of the cards in players' hands per player from `Game.cardSummary()`
    - list summary of the cards played per player from `Game.cardSummary()`
    - ranked list of player indices, sorted by points remaining in their hands from `Game.postGameSummary()`
    - random seed used for the game
    - integer number of times the deck was rebuilt from running out of cards
    - integer index into the list players of the starting player
- `resultsDF`: This is a single row dataframe indexed by the simulation number detailing a plethora of information gleaned from `gameResults`, including overall game statistics, player characteristics and statistics, and winner statistics. At the end of the simulation, these dataframes are all concatenated together for experiment analysis.
- `filName`: This is a string file name (sans extension) for both the game log and pickle file.


### Experiment
The experiment log records input experiment parameters, log and pickle filenames from each game, some summary analysis, and run statistics. The pickled experiment results dictionary includes input experiment parameters, game details, and summary analysis:

- all input parameters
- `timing`: tuple of timing information
- `design`: dict of list of dicts holding all strategy & parameter combinations
- `designUseCounts` dict of parameter set use counts across entire experiment
- `resultsDF`: dataframe of concatenated `resultsDF` dataframes from all games
- `allResults`: list of `gameResults` dicts from each game
- `winSummaries`: list of dataframe groupby winning counts, by strategy & starting, strategy, starting
- `gameRunFiles`: list of string file names (sans extension) for the game log and pickle files