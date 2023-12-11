'''
# TODO: setup to run from command line with args
# TODO: improve talking
# TODO: test test test
# TODO: define player strategies
# TODO: add games running with multiprocessing (can this work with the logg as it is setup?)
with multiprocessing.Pool() as pool:
    for (indx, expd) in pool.imap(expandDates, thisData.itertuples(name=None)):
        results[indx] = expd
'''
''' strategy ideas: prefer finish color, switch max color '''
from itertools import product
import logging
import time
import datetime as dt
import multiprocessing
import numpy as np
import ipdb
import re
from collections import OrderedDict
import pickle
import pandas as pd

pd.set_option('display.max_columns', None)



# globals
COLORS = ['red', 'green', 'blue', 'yellow']
SPECIALS = ['rev', 'skp', '+2']
WILDS = ['wld', 'wld+4']
SPECIALSWILDS = SPECIALS + WILDS
LENCOLORS = len(COLORS)
LENSPECIALS = len(SPECIALS)
LENWILDS = len(WILDS)
LENSPECIALSWILDS = len(SPECIALSWILDS)
LENVALUES = 10
NUMBERS_COUNT = 2
ZEROS_COUNT = 1
SPECIALS_COUNT = 2
WILDS_COUNT = 4
SPECIALS_POINTS = [20, 20, 20]
WILDS_POINTS = [50, 100]


def getCreateLogger(name:str, file:str=None, level:int=0):
    '''
    Get a logging object, creating it if non-existent.
    :param name: name of the logger
    :param file: optional file to where to store the log; required
        if creating a new logger object
    :param level: optional (default = 0) integer min level of logging; choices
        are 0 = 'NOTSET', 10 = 'DEBUG', 20 = 'INFO', 30 = 'WARN', 40 = 'ERROR',
        50 = 'CRITICAL'
    :return logger: logging object
    '''

    logger = logging.getLogger(name)

    if len(logger.handlers) == 0:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(file)

        # Set handler levels
        c_handler.setLevel(level)
        f_handler.setLevel(level)

        # Create formatters and add it to handlers
        dfmt = '%Y%m%d_%H%M%S'
        c_format = logging.Formatter(datefmt=dfmt,
                                     fmt='%(name)s@%(asctime)s@%(levelname)s@%(message)s')
        f_format = logging.Formatter(datefmt=dfmt,
                                     fmt='%(process)d@%(asctime)s@%(name)s@%(levelname)s@%(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # finish with the logger
        logger.setLevel(level)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


class Card():
    def __init__(self, color:int, value:int, specialWild:int=None):
        '''
        Define a card.
        :param color: integer in allowed range
        :param value: integer in allowed range
        :param specialWild: integer in allowed range
        '''

        # check consistency between special & value
        if (value is None) & (specialWild is None):
            raise ValueError("Value and Special can't both be empty!")
        if not ((value is not None) ^ (specialWild is not None)):
            raise ValueError('At least one of Value or Special/Wild should be provided.')

        # check consistency between special & color
        if (color is not None) & (specialWild is not None):
            if specialWild >= LENSPECIALS:
                raise ValueError('Color should be unspecified for wild cards.')

        # init the card indices now
        self.colorIndex = color
        self.valueIndex = value
        self.specialIndex = None
        self.wildIndex = None
        self.specialWildIndex = None

        # check valid inputs & define the card
        if color is not None:
            if (color < 0) | (color >= LENCOLORS):
                raise ValueError('Color must be in [0,...,%d]!'%(LENCOLORS-1))
            self.color = COLORS[color]

        if value is not None:
            if (value < 0) | (value >= LENVALUES):
                raise ValueError('Value must in [0...,%d]'%(LENVALUES-1))

        if specialWild is not None:
            if (specialWild < 0) | (specialWild >= LENSPECIALSWILDS):
                raise ValueError('Special/Wild must in [0...,%d]'%
                                 (LENSPECIALSWILDS-1))
            if specialWild >= LENSPECIALS:
                self.wild = WILDS[specialWild-LENSPECIALS]
                self.wildIndex = specialWild-LENSPECIALS
            else:
                self.special = SPECIALS[specialWild]
                self.specialIndex = specialWild

        # get the card points
        if self.wildIndex is not None:
            self.points = WILDS_POINTS[self.wildIndex]
        elif self.specialIndex is not None:
            self.points = SPECIALS_POINTS[self.specialIndex]
        else:
            self.points = self.valueIndex

        # give the card a name
        if self.colorIndex is not None:
            self.name = self.color + ' '
        else:
            self.name = ''
        if self.valueIndex is None:
            self.name += SPECIALSWILDS[specialWild]
        else:
            self.name += str(self.valueIndex)
        self.name += ' (%s)'%self.points

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Card(%r, %r, %r, %r, %s, %d)'%(self.colorIndex, self.valueIndex,
            self.specialIndex, self.wildIndex, self.name, self.points)

    def __eq__(self, other):
        return (self.colorIndex == other.colorIndex) &\
            (self.valueIndex == other.valueIndex) &\
            (self.specialWildIndex == other.specialWildIndex)

    def canPlace(self, other):
        '''
        Determine if this card can be placed on another.
        :param other: a card, presumably on the discard deck
        :return place: boolean match flag
        :return reason: index into reasons for matching; None if place is False
            0 = same card
            1 = this is a wild card
            2 = other is a wild card
            3 = same color, different number
            4 = different color, same number
            5 = same color, different special
            6 = different color, same special
        '''

        place = False
        reason = None

        # short circuit test for equality
        if self == other:
            place = True
            reason = 0
        # technically, anything can go on a wild, which can also go on anything
        elif self.wildIndex is not None:
            place = True
            reason = 1
        elif other.wildIndex is not None:
            place = True
            reason = 2
        # color cards
        elif self.colorIndex is not None:
            # number cards
            if self.valueIndex is not None:
                if (self.colorIndex == other.colorIndex) &\
                    (self.valueIndex != other.valueIndex):
                    # same color, different number
                    place = True
                    reason = 3
                elif (self.colorIndex != other.colorIndex) &\
                    (self.valueIndex == other.valueIndex):
                    # different color, same number
                    place = True
                    reason = 4
            # special cards
            elif self.specialIndex is not None:
                if (self.colorIndex == other.colorIndex) &\
                    (self.specialIndex != other.specialIndex):
                    # same color, different special
                    place = True
                    reason = 5
                elif (self.colorIndex != other.colorIndex) &\
                    (self.specialIndex == other.specialIndex):
                    # different color, same special (not wild)
                    place = True
                    reason = 6

        return place, reason


class Deck():
    def __init__(self, cards:list[Card]=None, shuffle:bool=True):
        '''
        Build the deck of Uno cards. For each color in a standard deck, there
        are 1 0 2 1-9's, and 2 non-wild special cards. There are 4 of each wild.
        This is a total of 108 cards. The deck is defined as a list of cards.
        :param cards: optional (default=None) list of cards for this deck; if
            None, the deck is built
        :param shuffle: optional (default=True) flag indicating whether or not
            to shuffle the cards generated
        '''

        if cards is None:
            # define possible indices
            colors = list(range(LENCOLORS))
            values = list(range(LENVALUES))
            specials = list(range(LENSPECIALS))
            wilds = list(range(LENWILDS))

            # get 1 of each number card == 0 & 2 of each number card > 1
            numbers = [Card(colr, valu, None) for (colr, valu) in product(colors,
                sorted([values[0]]*ZEROS_COUNT + values[1:]*NUMBERS_COUNT))]
            # get 2 of each special card
            nonwilds = [Card(colr, None, spec) for (colr, spec) in product(colors,
                sorted(specials*SPECIALS_COUNT))]
            # get 4 each of wilds
            wilds = [Card(None, None, wld+LENSPECIALS) for wld in
                sorted(wilds*WILDS_COUNT)]

            # shuffle or put in nice order
            if shuffle:
                self.cards = numbers + nonwilds + wilds
                np.random.shuffle(self.cards)
            else:
                self.cards = sorted(numbers + nonwilds,
                    key=lambda x:x.colorIndex) + wilds
        else:
            # used the passed-in list of cards
            self.cards = cards

        # add the size count
        self.size = len(self.cards)

        # initialize the card to next come off the deck
        self.topCard = 0

    def deal(self, cards:int=1, thisGame=None):
        '''
        Deal a specified number of cards off the top of the deck.
        :param cards: optional (default=1) integer number of cards to deal
        :param thisGame: optional (default=None) game object for possibly
            resetting the deck
        :return dealt: list of tuple (deck index, card) of cards dealt
        '''

        rebuild = False

        # only check for enough cards if game passed in
        if thisGame is not None:
            # ensure there are enough cards in the deck
            if (thisGame.discardPile is not None) & (self.size <= cards):
                # not enough cards, so reset the deck
                logg.info('Only %d card(s), so rebuilding the deck from discard pile',
                    self.size)
                thisGame.rebuildDeck()
                rebuild = True
            elif self.size <= cards:
                # this should never happen
                logg.debug('Only %d cards, but discard is empty',
                    self.size)
                raise ValueError('Not enough cards to deal')

        # if deck was rebuilt, need to refer, this time, to thisGame
        if rebuild:
            deck = thisGame.deck
        else:
            deck = self

        # get the cards to deal
        deal = [(indx, deck.cards[indx]) for indx in range(deck.topCard,
            deck.topCard + cards)]

        # update the topCard & size of the deck
        deck.topCard += cards
        deck.size -= cards

        return deal


class Hand():
    def __init__(self, cards:list[Card]):
        '''
        Build a hand of cards. The hand is an enumeration-based dict of cards.
        :param cards: list of cards initially dealt to this hand; each card
            should be represented by a tuple of (index into the deck, the
            actual card).
        '''

        # initialize the hand
        self.currCards = OrderedDict()
        self.playedCards = OrderedDict()
        self.nextIndex = 0

        # add dealt cards to this hand
        for card in cards:
            self.addCard(card, False)
        self.cardCount = len(self.currCards)

        # update hand statistics - just init with blank dicts
        self.colors = {colr:[] for colr in range(LENCOLORS)}
        self.values = {valu:[] for valu in range(LENVALUES)}
        self.specials = {spec:[] for spec in range(LENSPECIALS)}
        self.wilds = {wild:[] for wild in range(LENWILDS)}
        self.colorCounts = [0]*LENCOLORS
        self.__handSummarize__()

    def __handSummarize__(self):
        '''
        Generate a summary of the current hand, storing the index into the hand
        in colors, values, specials, and wilds (all index-based) dicts. Each
        entry in these dicts is a list holding indices into the hand. This also
        computes the points value for the hand.
        '''

        self.points = 0
        self.colors = {colr:[] for colr in range(LENCOLORS)}
        self.values = {valu:[] for valu in range(LENVALUES)}
        self.specials = {spec:[] for spec in range(LENSPECIALS)}
        self.wilds = {wild:[] for wild in range(LENWILDS)}

        if self.cardCount > 0:
            # iterate over cards in the hand now
            for (index, card) in self.currCards.items():
                # points
                self.points += card[1].points
                # color
                if card[1].colorIndex is not None:
                    self.colors[card[1].colorIndex].append(index)
                # value
                if card[1].valueIndex is not None:
                    self.values[card[1].valueIndex].append(index)
                # special
                if card[1].specialIndex is not None:
                    self.specials[card[1].specialIndex].append(index)
                # wild
                if card[1].wildIndex is not None:
                    self.wilds[card[1].wildIndex].append(index)

            # get number of cards for each color & get them sorted in descending
            self.colorCounts = [len(self.colors[colr]) for colr in range(LENCOLORS)]
            self.colorOrders = np.argsort(self.colorCounts)[::-1]

            # determine which cards could be stacked
            # get curr card indices from 0 to max, though all won't be there;
            # this is a bit wasteful, but I don't think it will really matter
            rng = range(max(self.currCards.keys())+1)
            self.canStack = np.ndarray(shape=(len(rng), len(rng)), dtype=object)
            for row in rng:
                for col in rng:
                    if row==col:
                        self.canStack[row, col] = (None, None)
                    else:
                        try:
                            self.canStack[row, col] = self.currCards[row][1].\
                                canPlace(self.currCards[col][1])
                        except KeyError:
                            # this card is no longer current, so just skip
                            pass
        else:
            self.colorCounts = self.colorOrders = [0]*LENCOLORS

    def addCard(self, card:tuple[int, Card], updateSummary:bool=True):
        '''
        Add a new card to this hand and possibly update the stats.
        :param card: tuple holding index of this card into the deck and the
            actual card
        :param update: optional (default=True) flag to update the hand summary
        '''

        # add the card
        self.currCards[self.nextIndex] = card
        # update the next index
        self.nextIndex += 1
        # update number of Cards
        self.cardCount = len(self.currCards)

        if updateSummary:
            self.__handSummarize__()

    def playCard(self, card:int, colorIndex:int=None):
        '''
        Play a card, moving it from the current hand, and updating the summary.
        :param card: index into the hand of the card to play
        :param colorIndex: optional (default=None) color of card chosen if
            played card is a wild
        :return card: tuple of card index and card to play
        :return remain: number of cards remaining
        '''

        # talk
        logg.info('\nPlaying %s', self.currCards[card][1])
        if colorIndex is not None:
            logg.info('\nNew color = %s', COLORS[colorIndex])
        # move to played cards
        this = self.currCards.pop(card)
        self.playedCards[card] = this
        # update number of Cards
        self.cardCount = len(self.currCards)

        # update the summaries
        self.__handSummarize__()
        self.__playedSummarize__()

        return this, self.cardCount

    def showHand(self, summary:bool=True, played:bool=False):
        '''
        Show the cards in the hand.
        :param summary: optional (default=True) boolean flag to show the summary
        :param played: optional (default False) boolean flag to also show played
            cards.
        :return prt: string showing the hand
        '''

        # iterate and print cards
        prt = 'Current Hand Points = %d'%self.points
        for (index, card) in self.currCards.items():
            prt += '\nCard %d = %s'%(index, card)

        # maybe show the summary
        if summary:
            prt += '\nHand Summary'
            # colors
            for (colorIndex, cards) in self.colors.items():
                if len(self.colors[colorIndex]) > 0:
                    prt += '\nColor = ' + COLORS[colorIndex]
                    for card in cards:
                        prt += '\n\tCard %d = %s'%(card, self.currCards[card])
            # values
            for (valueIndex, cards) in self.values.items():
                if len(self.values[valueIndex]) > 0:
                    prt += '\nValue = %d'%valueIndex
                    for card in cards:
                        prt += '\n\tCard %d = %r'%(card, self.currCards[card])
            # specials
            for (specialIndex, cards) in self.specials.items():
                if len(self.specials[specialIndex]) > 0:
                    prt += '\nSpecial = ' + SPECIALS[specialIndex]
                    for card in cards:
                        prt += '\n\tCard %d = %s'%(card, self.currCards[card])
            # wilds
            for (wildIndex, cards) in self.wilds.items():
                if len(self.wilds[wildIndex]) > 0:
                    prt += '\nWild = ' + WILDS[wildIndex]
                    for card in cards:
                        prt += '\n\tCard %d = %s'%(card, self.currCards[card])
            # show the stackables
            prt += '\nStackable Matrix\n%s'%\
                np.array([str(s[0])[0] for s in
                          thisGame.players[0].hand.canStack.flatten()]).\
                reshape(self.cardCount, self.cardCount)

        # maybe iterate and print played cards
        if played & (self.playedCards != {}):
            prt += '\nPlayed Cards'
            for (index, card) in self.playedCards.items():
                prt += '\nCard %d = %s'%(index, card)

        return prt

    def __playedSummarize__(self):
        '''
        Generate a summary of all played cards, storing the index into the hand
        in colors, values, specials, and wilds (all index-based) dicts. Each
        entry in these dicts is a list holding indices into the hand. This also
        computes the points value.
        '''

        self.pointsPlayd = 0
        self.colorsPlayd = {colr:[] for colr in range(LENCOLORS)}
        self.valuesPlayd = {valu:[] for valu in range(LENVALUES)}
        self.specialsPlayd = {spec:[] for spec in range(LENSPECIALS)}
        self.wildsPlayd = {wild:[] for wild in range(LENWILDS)}

        if self.playedCards != {}:
            # iterate over cards in the hand now
            for (index, card) in self.currCards.items():
                # points
                self.pointsPlayd += card[1].points
                # color
                if card[1].colorIndex is not None:
                    self.colorsPlayd[card[1].colorIndex].append(index)
                # value
                if card[1].valueIndex is not None:
                    self.valuesPlayd[card[1].valueIndex].append(index)
                # special
                if card[1].specialIndex is not None:
                    self.specialsPlayd[card[1].specialIndex].append(index)
                # wild
                if card[1].wildIndex is not None:
                    self.wildsPlayd[card[1].wildIndex].append(index)


class Player():
    def __init__(self, name:str, strategy, hand:Hand=None):
        '''
        Define an Uno player.
        :param name: string name of player
        :param strategy: dict holding 'strategy': callable implementing player's
            strategy, 'hurtFirst': flag for callable, 'hailMary': flag for
            callable (see strategy function)
        :param hand: optional (default=None) Hand object for player
        '''
        self.name = name
        self.strategy = strategy['strategy']
        self.strategyHF = strategy.get('hurtFirst', False)
        self.strategyHM = strategy.get('hailMary', False)
        self.hand = None

    def __str__(self):
        return '%s:\n%s'%(self.name,
            self.hand.showHand(summary=False, played=False))

    def addHand(self, hand:Hand):
        '''
        Give player a hand of Cards
        :param hand: hand object
        '''

        self.hand = hand

    def takeTurn(self, thisGame):
        '''
        Take a turn
        :param thisGame: game this player is in
        '''

        # if discard started with wild, must choose the color
        if thisGame.currWild is not None:
            if thisGame.currColor is None:
                # choose color as most frequent in hand
                bestColor = self.hand.colorOrders[0]
                logg.debug('\nWild on discard with no color, so choosing %s, %r',
                    COLORS[bestColor], self.hand.colorOrders[0])
                thisGame.currColor = bestColor

        # determine playable cards, and maybe draw one
        whilePass = 0
        while whilePass <= 1:
            playableCards = []
            # determine playable cards - same color, same number, same special
            if thisGame.currColor is not None:
                playableCards.extend(self.hand.colors[thisGame.currColor])
            if thisGame.currValue is not None:
                playableCards.extend(self.hand.values[thisGame.currValue])
            if thisGame.currSpecial is not None:
                playableCards.extend(self.hand.specials[thisGame.currSpecial])
            # wild cards are always playable
            playableCards.extend(self.hand.wilds[0])
            # wild +4s only playable if none of the current color in the hand
            if len(self.hand.colors[thisGame.currColor]) == 0:
                playableCards.extend(self.hand.wilds[1])
                logg.debug('\nNo %s color cards, so wild +4 is playable',
                    COLORS[thisGame.currColor])
            # get uniques
            playableCards = set(playableCards)

            # do we need to draw a card?
            if (len(playableCards) == 0) & (whilePass == 0):
                logg.info('\nNo cards to play, drawing 1')
                card = thisGame.deck.deal(1, thisGame)[0]
                self.hand.addCard(card)
                logg.info('\nDealt %s', card[1])
            else:
                break
            whilePass += 1

        # show hand
        logg.debug('\n%s', self.hand.showHand(summary=False))

        # can we play?
        if len(playableCards) > 0:
            # find why each of the playable cards are playable
            '''
            0 = same card
            1 = this is a wild card
            2 = other is a wild card
            3 = same color, different number
            4 = different color, same number
            5 = same color, different special
            6 = different color, same special
            '''

            playables = dict.fromkeys(playableCards, None)
            sameColorPlay, sameColorSpecialPlay = [], []
            diffColorPlay, wildPlay = [], []

            # iterate over cards
            for handIndx in playableCards:
                # get the card in the hand & determine if it's placeable
                card = self.hand.currCards[handIndx]
                playables[handIndx] = card[1].canPlace(thisGame.discardPile[-1][1])
                # summarize
                # get same colors
                if (playables[handIndx][1] in [0, 2, 3, 5]) &\
                    (card[1].wildIndex is None):
                    if card[1].specialIndex is None:
                        sameColorPlay.append((handIndx, card))
                    elif card[1].specialIndex is not None:
                        sameColorSpecialPlay.append((handIndx, card))
                # get wilds
                if (playables[handIndx][1] in [0, 1, 2]) &\
                    (card[1].wildIndex is not None):
                    wildPlay.append((handIndx, card))
                # get different colors
                if playables[handIndx][1] in [4, 6]:
                    diffColorPlay.append((handIndx, card))
                # talk
                logg.debug('\nPlayable %d = (%d, %s): reason = %d', handIndx,
                    *self.hand.currCards[handIndx], playables[handIndx][1])
            # summarize the summary
            sameColor = len(sameColorPlay)
            sameColorSpecial = len(sameColorSpecialPlay)
            diffColor = len(diffColorPlay)
            wilds = len(wildPlay)
            # talk
            logg.debug('\nPlayable: %d same colors, %d same color specials, %d wilds, %d different colors',
                sameColor, sameColorSpecial, wilds, diffColor)
            logg.debug('\nPlayable same colors: %s\nPlayable same color specials, %s',
                       sameColorPlay, sameColorSpecialPlay)
            logg.debug('\nPlayable wilds: %s\nPlayable different colors: %s',
                       wildPlay, diffColorPlay)

            ''' now determine the best card to play '''
            # execute the strategy
            bestCard, bestColor = self.strategy(self, thisGame,
                                                sameColorPlay, sameColorSpecialPlay,
                                                wildPlay, diffColorPlay,
                                                self.strategyHF, self.strategyHM)

            # just pick the highest value playable card - should not happen
            if bestCard is None:
                playMe = [(handIndx, self.hand.currCards[handIndx]) for handIndx
                          in playables.keys()]
                playMe.sort(key=lambda x: x[1][1].points)
                bestCard = playMe[-1][0]
                logg.debug('\nPlay highest value playable card: %s',
                           self.hand.currCards[bestCard][1])

            # play the best card
            if bestCard is not None:
                logg.info('\nBest card = %s', self.hand.currCards[bestCard][1])
                thisGame.addToDiscard(self.hand.currCards[bestCard], bestColor)
                _ = self.hand.playCard(bestCard, bestColor)
        else:
            logg.info('\nNo cards to play - skipping turn')        

        # yell Uno!
        if self.hand.cardCount == 1:
            logg.info('\nUno!')
        elif self.hand.cardCount == 0:
            logg.info('\nI won!')

        # update the card count for this player
        thisGame.playerCardsCounts[thisGame.currPlayer] = self.hand.cardCount
        logg.debug('\nPlayer %s now has %d cards, %d points',
            thisGame.players[thisGame.currPlayer].name, self.hand.cardCount, self.hand.points)


class Game():
    def __init__(self, descrip:str, players:list, start:int=0, rndSeed:int=None):
        '''
        Initialize a game of Uno.
        :parm descrip: string description of game
        :param players: list of player objects
        :param start: optional (default=None) index of starting player; if not
            provided, starter is randomly selected
        :param rndSeed: optional (default=None) seed for numpy prng
        '''

        # set the random state
        if rndSeed is None:
            rndSeed = dt.datetime.now()
            rndSeed = rndSeed.hour*10000 + rndSeed.minute*100 + rndSeed.second +\
                rndSeed.microsecond
        self.rndSeed = rndSeed
        np.random.seed(rndSeed)

        # get inputs & set players data
        self.name = descrip
        self.players = players
        self.playersCount = len(players)
        self.playersOrder = 1 # 1 or -1

        # get the Deck & put the top card on the pile
        self.rebuilt = 0
        self.rebuiltCards = []
        self.deck = Deck()
        self.discardPile = []
        self.addToDiscard(self.deck.deal(1)[0])

        # deal 7 cards to each player
        for player in self.players:
            player.addHand(Hand(self.deck.deal(7)))
        # remember each player's number of Cards
        self.playerCardsCounts = [7]*self.playersCount

        # set the starting & next player
        if start is None:
            self.currPlayer = np.random.randint(self.playersCount)
            self.start = self.currPlayer
        else:
            self.start = start
            self.currPlayer = start

        # prepare to handle first discard card = skip;
        # reverse, wilds, and +2 handled elsewhere
        if self.currSpecial is not None:
            if SPECIALS[self.currSpecial] == 'skp':
                self.currPlayer = (self.currPlayer + 1) % self.playersCount
        self.nextPlayer = self.__nextPlayer__()

        # talk
        logg.info('\n%s initialized with players', self.name)
        for player in self.players:
            logg.info(player)
        logg.info('\nInitial discard card = (%d = %s)', *self.discardPile[0])
        logg.info('\n%s goes first', self.players[self.currPlayer].name)

    def __nextPlayer__(self):
        '''
        Deterimine the next player, considering the current player, play order,
        and card currently on the discard pile. This method should be
        exexuted during initialization & at the end of each player's turn.
        '''

        # save current player so can edit if needed to handle skip
        curr = self.currPlayer

        # handle special cards on the discard pile
        if self.currSpecial is not None:
            if (SPECIALS[self.currSpecial] == 'skp') |\
                ((SPECIALS[self.currSpecial] == 'rev') & (self.playersCount == 2)):
                # skip - pretend current is next
                curr += 1
            elif SPECIALS[self.currSpecial] == 'rev':
                # reverse - change direction
                self.playersOrder *= -1

        # set the next player
        nxt = (curr + self.playersOrder) % self.playersCount
        logg.debug('\nNext player = %s', self.players[nxt].name)
        return nxt

    def addToDiscard(self, card:Card, colorIndex:int=None):
        '''
        Add a card to the discard pile.
        :param card: tuple of the index in the deck and the card object to add
        :param colorIndex: optional (default=None) color of card chosen if
            played card is a wild
        '''
        # add
        self.discardPile.append(card)
        logg.debug('\n%s added to discard', card[1])

        # define the deck index of the current card
        self.currCardIndex = card[0]

        # define current color if not provided
        if colorIndex is None:
            self.currColor = card[1].colorIndex
        else:
            self.currColor = colorIndex
        if self.currColor is not None:
            logg.debug('\nCurrent color = %s', COLORS[self.currColor])
        else:
            logg.debug('\nNo current color')
        # define current special & value
        self.currSpecial = card[1].specialIndex
        self.currValue = card[1].valueIndex
        self.currWild = card[1].wildIndex

    def playOne(self):
        '''
        Make the current player take a turn and advance to the next player.
        '''

        # talk
        logg.info('\n%s playing', self.players[self.currPlayer].name)
        # make player draw cards if necessary
        if self.currSpecial is not None:
            if self.currSpecial == 2:
                # +2 so draw 2
                logg.info('\n +2 on discard pile, so drawing 2')
                for (indx, card) in enumerate(self.deck.deal(2, self)):
                    logg.info('\nDealt %s', card[1])
                    self.players[self.currPlayer].hand.addCard(card,
                                                               updateSummary=(indx==1))
        elif self.currWild is not None:
            if self.currWild == 1:
                # wild+4, so draw 4
                logg.info('\nWild +4 on discard pile, so drawing 4')
                for (indx, card) in enumerate(self.deck.deal(4, self)):
                    logg.info('\nDealt %s', card[1])
                    self.players[self.currPlayer].hand.addCard(card,
                                                               updateSummary=(indx==3))
        # take the turn
        self.players[self.currPlayer].takeTurn(self)
        # set the next player if nobody won
        if min(self.playerCardsCounts) > 0:
            self.nextPlayer = self.__nextPlayer__()
            # set the new current player
            self.currPlayer = self.nextPlayer
        # talk about all players' card counts & points
        status = '; '.join(['%s has %d cards worth %d points'%\
                            (player.name, player.hand.cardCount, player.hand.points)
                            for player in self.players])
        logg.info('\n'+status)

    def play(self):
        '''
        Let's play Uno!.
        :return results: dictionary of some summary results with keys 'timing',
            'winner', 'played summary', 'player remaining summary', 'player played summary',
            'random seed', and 'times rebuilt'.
        '''

        # timing
        self.gameTimeStt = dt.datetime.now()
        self.gamePerfStt = time.perf_counter()
        logg.debug('\n%s started on %s', self.name, self.gameTimeStt.isoformat()[:16])

        # iterate over players, each taking their turn
        while min(self.playerCardsCounts) > 0:
            self.playOne()

        # post-game summary
        self.postGameSummary()
        logg.info('\n%s won!', self.players[self.winner].name)

        # timing
        self.gameTimeStp = dt.datetime.now()
        self.gamePerfStp = time.perf_counter()

        # talk
        logg.info('%s ended on %s (%0.3f(s)): %s won in %d turns, having played %d points; %d total cards played!',
                  self.name, self.gameTimeStp, self.gamePerfStp - self.gamePerfStt,
                  self.players[self.winner].name, *self.playerPlayed[self.winner][:2],
                  self.discardSummary[0])

        return {'timing':(self.gameTimeStt, self.gamePerfStt, self.gameTimeStp,
                          self.gamePerfStp), 'winner':self.winner,
                          'discard summary':self.discardSummary,
                          'player remaining summary':self.playerRemain,
                          'player played summary':self.playerPlayed,
                          'random seed':rndSeed, 'times rebuilt':self.rebuilt,
                          'start':self.start}

    def rebuildDeck(self):
        '''
        Deck is too short to deal cards to player, so rebuild it.
        '''

        # update the rebuildt counters
        self.rebuiltCards.extend(self.discardPile[:-1])
        self.rebuilt += 1

        # get discards sans top card for new deck & shuffle
        logg.debug('\nAdding & shuffling discard pile sans top (%d)',
                   len(self.discardPile[:-1]))
        cards = self.discardPile[:-1]
        np.random.shuffle(cards)
        # add the remainder of the deck: 42 is just a placeholder, as it'll be stripped
        logg.debug('\nAdding remainder of deck (%d)', self.deck.size)
        cards.extend([(42, card) for card in self.deck.cards[-self.deck.size:]])
        # take all cards from each player *in reverse order* and add
        for player in self.players[::-1]:
            logg.debug('\nAdding %s cards (%d)', player.name,
                       len(player.hand.currCards))
            cards.extend(player.hand.currCards.values())
            player.hand.currCards = {}
        # add the top card
        logg.debug('\nAdding the top discard (1)')
        cards += [self.discardPile[-1]]
        # remember the current color if top card is a wild
        if self.discardPile[-1][1].wildIndex is not None:
            currColor = thisGame.currColor
        else:
            currColor = None

        # create new deck object
        self.deck = Deck(cards=[card[1] for card in cards[::-1]], shuffle=False)

        # add top card back to discard and ensure the current color is there
        # if top card is a wild
        self.discardPile = []
        logg.debug('\nPutting back top discard')
        self.addToDiscard(self.deck.deal(1)[0], currColor)

        # deal back all cards
        for (pindx, player) in enumerate(self.players):
            # get the cards dealt & add them
            cards = self.deck.deal(self.playerCardsCounts[pindx])
            logg.debug('\nAdding back %d cards to %s', len(cards), player.name)
            for (cindx, card) in enumerate(cards):
                summary = (cindx == self.playerCardsCounts[pindx]-1)
                player.hand.addCard(card, updateSummary=summary)
        
        # talk
        logg.info('\nDeck rebuilt')

    def cardsSummary(self, cards):
        '''
        Generate a summary of a collection of cards, storing the count of cards
        in colors, values, specials, and wilds (all index-based) dicts. This also
        computes the points value for the hand and the total number of cards.
        :param cards: iterable of cards, as represented by a tuple of deck index
            and actual card
        :return cardCount: number of cards in collection
        :return points: total number of points
        :return colors: color index-keyed dict of card counts
        :return values: value index-keyed dict of card counts
        :return specials: special index-keyed dict of card counts
        :return wilds: wild index-keyed dict of card counts
        '''

        # init
        cardCount = len(cards)
        points = 0
        colors = {colr:0 for colr in range(LENCOLORS)}
        values = {valu:0 for valu in range(LENVALUES)}
        specials = {spec:0 for spec in range(LENSPECIALS)}
        wilds = {wild:0 for wild in range(LENWILDS)}

        if cardCount > 0:
            # iterate over cards in the hand now
            for card in cards:
                # points
                points += card[1].points
                # colors
                if card[1].colorIndex is not None:
                    colors[card[1].colorIndex] += 1
                # values
                if card[1].valueIndex is not None:
                    values[card[1].valueIndex] += 1
                # specials
                if card[1].specialIndex is not None:
                    specials[card[1].specialIndex] += 1
                # wild
                if card[1].wildIndex is not None:
                    wilds[card[1].wildIndex] += 1

        return cardCount, points, colors, values, specials, wilds

    def postGameSummary(self):
        '''
        Summarize a game after finishing
        '''

        # summary of all played cards
        self.discardSummary = self.cardsSummary(self.discardPile + self.rebuiltCards)

        # summaries by player: cardCount, points, colors, values, specials, wilds
        self.playerPlayed = [None]*len(self.players)
        self.playerRemain = [None]*len(self.players)
        for (indx, player) in enumerate(self.players):
            # played cards
            self.playerPlayed[indx] = self.cardsSummary(player.hand.playedCards.values())
            # remaining cards
            self.playerRemain[indx] = self.cardsSummary(player.hand.currCards.values())
            # winner
            if self.playerRemain[indx][0] == 0:
                self.winner = indx


def stratFinishCurrentColor(thisPlayer:Player, thisGame:Game, sameColorPlay,
                            sameColorSpecialPlay, wildPlay, diffColorPlay,
                            hurtFirst:bool=False, hailMary:bool=False):
    '''
    Implement the "finish current color" strategy. This will first try to play
    two cards (if two players can use skip / reverse to get a 2nd turn). Then it
    will try to play a special card, then a number card. Otherwise, it will try
    to play a wild.
    :param thisPlayer: current player
    :param thisGame: current game
    :param sameColorPlay: list of (hand index, card) playable same color value
        cards
    :param sameColorSpecialPlay: list of (hand index, card) playable same color
        special cards
    :param wildPlay: list of (hand index, card) playable wild cards
    :param diffColorPlay: list of (hand index, card) playable different color
        cards
    :param hurtFirst: optional boolean (default=False) flag to indicate that a
        draw 2 or draw 4 should be used before other specials or wilds; otherwise,
        use them last
    :param hailMary: optional boolean (default=False) flag to indicate that a
        player with <= 2 cards should be defended against by playing, in preference
        order: same color draw 2 same color special, draw 4, diff color draw 2,
        diff color special
    :return bestCard: integer index into the hand of the best playable card
    :return bestColor: integer color index of the color to set if wild played; 
        None if wild not played
    '''

    bestCard = bestColor = None

    # count the cards in each bin
    sameColor = len(sameColorPlay)
    sameColorSpecial = len(sameColorSpecialPlay)
    wilds = len(wildPlay)
    diffColor = len(diffColorPlay)

    # does any player have 2 or fewer cards?
    if (min(thisGame.playerCardsCounts) <= 2) & hailMary:
        playersLT2 = len([c for c in thisGame.playerCardsCounts if c <= 2])
        logg.info('%s players have <= 2 cards', playersLT2)
        # determine what special / wild cards can defend
        sameDraw2s = [play for play in sameColorSpecialPlay if play[1][1].specialIndex == 2]
        sameRevSkps = [play for play in sameColorSpecialPlay if play[1][1].specialIndex < 2]
        draw4s = [play for play in wildPlay if play[1][1].wildIndex==1]
        diffDraw2s = [play for play in diffColorPlay if play[1][1].specialIndex == 2]
        diffRevSkps = [play for play in diffColorPlay if (play[1][1].specialIndex == 0) |\
                       (play[1][1].specialIndex == 1)]
        defenseCards = len(sameDraw2s) + len(sameRevSkps) + len(draw4s) +\
            len(diffDraw2s) + len(diffRevSkps)

        logg.debug('%d cards to defend against <= 2 player', defenseCards)
        # get the next player & their card count
        nxt = thisGame.nextPlayer
        nxtCards = thisGame.playerCardsCounts[nxt]
        if nxtCards >= 2:
            # next player has <= 2 cards
            logg.info('Next player has %d cards', nxtCards)
            # preference order: current color draw 2, skip/reverse, draw 4,
            # other color draw 2, skip / reverse
            if defenseCards > 0:
                if len(sameDraw2s) > 0:
                    # ensure draw 2 same color is played
                    hurtFirst = True
                    logg.debug('Setting hurtFirst to True so draw 2 is played')
                elif len(sameRevSkps) > 0:
                    # don't need to do anything
                    pass
                elif len(draw4s) > 0:
                    # pretend no same color value cards available
                    ipdb.set_trace() # debug testing
                    sameColor = 0
                    hurtFirst = True
                    logg.debug('Setting hurtFirst to True and ignoring same color value cards so draw 4 is played')
                elif len(diffDraw2s) > 0:
                    # pretend no same color value cards & no wilds; ensure draw 2 diff color is played
                    # this might not work if the color of this card is not the max color
                    ipdb.set_trace() # debug testing
                    hurtFirst = True
                    sameColor = 0
                    wilds = 0
                    logg.debug('Setting hurtFirst to True and ignoring same color value cards & wilds so draw 2 is played')
                elif len(diffRevSkps) > 0:
                    # pretend no same color value cards & no wilds
                    # this might not work if the color of this card is not the max color
                    ipdb.set_trace() # debug testing
                    sameColor = 0
                    wilds = 0
                    logg.debug('Ignoring same color value cards & wilds so diff color special is played')
            else:
                logg.debug("Can't defend against next player with %d cards", nxtCards)
        else:
            # someone else has <= 2 cards
            logg.info('Unsure how to defend against non-next player with <= 2 cards')


    # choose the best card
    if (sameColor > 0) | (sameColorSpecial > 0):
        # can we place a skip or reverse to play an extra card?
        if (sameColorSpecial > 0) & (sameColor > 1) &\
            (thisGame.playersCount == 2):
            # get any skips or reverses, then take the first if extant
            playMe = [(handIndx, card) for (handIndx, card)
                        in sameColorSpecialPlay if card[1].specialIndex < 2]
            if len(playMe) > 0:
                bestCard = playMe[0][0]
                logg.debug('\n2 players, play same color: %s', playMe[0][1][1])
            else:
                # can't get an extra turn :-(
                pass
        elif sameColorSpecial > 0:
            playMe = [(handIndx, card) for (handIndx, card) in sameColorSpecialPlay]
            # check if any draw 2s playable
            draw2s = [play for play in playMe if play[1][1].specialIndex == 2]
            if (len(draw2s) > 0) and hurtFirst:
                # get the first draw 2
                bestCard = draw2s[0][0]
                logg.debug('\nPlay same color special (hurt first): %s',
                           draw2s[0][1][1])
            else:
                # sort specials so draw 2s are last
                playMe.sort(key=lambda x: x[1][1].specialIndex)
                # get the first same color special card to play            
                bestCard = playMe[0][0]
                logg.debug('\nPlay same color special: %s', playMe[0][1][1])
        else:
            # just play a same color number card
            playMe = [(handIndx, card) for (handIndx, card) in sameColorPlay]
            playMe.sort(key=lambda x: x[1][1].points)
            bestCard = playMe[-1][0]
            logg.debug('\nPlay same color: %s', playMe[-1][1][1])
    elif wilds > 0:
        playMe = [(handIndx, card) for (handIndx, card) in wildPlay]
        # check if any draw 4s playable
        draw4s = [play for play in playMe if play[1][1].wildIndex == 1]
        if (len(draw4s) > 0) and hurtFirst:
            # get the first draw 4
            bestCard = draw4s[0][0]
            logg.debug('\nPlay wild (hurt first): %s', draw4s[0][1][1])
        else:
            # sort wilds so draw 4s are last
            playMe.sort(key=lambda x: x[1][1].points)
            # get the first wild to play
            bestCard = playMe[0][0]
            logg.debug('\nPlay wild: %s', playMe[0][1][1])

        # choose the best color - the color with the most playable points
        points = [0]*len(COLORS)
        # iterate over cards and count total points
        for card in thisPlayer.hand.currCards.values():
            if card[1].colorIndex is not None:
                points[card[1].colorIndex] += card[1].points
        # get the color with the most points, then choose that color card with the most points
        logg.debug('\nPoints per color: '+','.join(['%s = %d'%(COLORS[cIndx], pts)
                                                    for (cIndx, pts) in enumerate(points)]))
        bestColor = np.argsort(points)[-1]
    # choose best diff color by total playable points
    elif diffColor > 0:
        # get possible colors
        colrs = set([card[1][1].colorIndex for card in diffColorPlay])
        # total the cards of each color
        points = [-1]*len(COLORS) # init to -1 so 0's can be differentiated from no cards
        for card in thisPlayer.hand.currCards.values():
            if (card[1].colorIndex is not None) & (card[1].colorIndex in colrs):
                # get rid of the -1 first
                if points[card[1].colorIndex] == -1:
                    points[card[1].colorIndex] = 0
                # add the points
                points[card[1].colorIndex] += card[1].points
        logg.debug('\nPoints per color: '+','.join(['%s = %d'%(COLORS[cIndx], pts)
                                                    for (cIndx, pts) in enumerate(points)]))
        # get the color with the most points, then choose that color's diffColor card
        bestColor = np.argsort(points)[-1]
        playMe = [play for play in diffColorPlay if play[1][1].colorIndex==bestColor]
        # check if any draw 2s playable
        draw2s = [play for play in playMe if play[1][1].specialIndex == 2]
        if (len(draw2s) > 0) & hurtFirst:
            # get the first diff color draw 2
            bestCard = draw2s[0][0]
            logg.debug('\nPlay different color (hurt first): %s', draw2s[0][1][1])
        elif (len(draw2s) > 0) & (len(draw2s) < len(playMe)) & (not hurtFirst):
            ipdb.set_trace()
            # there are more diff color playables than draw 2s, so drop them
            playMe = [play for play in playMe if play[1][1].specialIndex != 2]
            playMe.sort(key=lambda x: x[1][1].points)
            bestCard = playMe[-1][0]
            logg.debug('\nPlay different color: %s', playMe[-1][1][1])
        else:
            # not hurting first, no draw 2s, or only draw 2s available
            playMe.sort(key=lambda x: x[1][1].points)
            bestCard = playMe[-1][0]
            logg.debug('\nPlay different color: %s', playMe[-1][1][1])
    else:
        # should not happen
        logg.debug('\nNo card to play - impossible?!')
        ipdb.set_trace()

    return bestCard, bestColor


''' EXECUTE '''
# setup Monte Carlo simulation
logLevel = 10 # 10=DEBUG+, 20=INFO+
MCSims = 10
configs = [{'players':['Ben Dover', 'Mike Rotch', 'Hugh Jass', 'Eileen Dover'],
            'strats':[{'strategy':stratFinishCurrentColor, 'hurtFirst':False, 'hailMary':False},
                      {'strategy':stratFinishCurrentColor, 'hurtFirst':True, 'hailMary':False},
                      {'strategy':stratFinishCurrentColor, 'hurtFirst':False, 'hailMary':True},
                      {'strategy':stratFinishCurrentColor, 'hurtFirst':True, 'hailMary':True}],
            'start':None, 'descrip':'Test Game'}]*MCSims
allResults = [None]*MCSims
resultsDF = pd.DataFrame(index=range(MCSims))

# timing
MCTimeStt = dt.datetime.now()
MCPerfStt = time.perf_counter()

# run games serially
for (indx, gameCFG) in enumerate(configs):
    # talk
    print('Game %d of %d'%(indx+1, MCSims))

    # setup a game with this config
    sttTS = dt.datetime.now()
    gameDescrip = gameCFG['descrip']
    logGameName = 'Uno_'+re.sub(pattern='[^a-zA-Z0-9]', repl='_', string=gameDescrip) +\
        '_' + sttTS.strftime('%Y%m%d_%H%M%S_%f')[:-3]    
    players = [Player(name, strat) for (name, strat) in
               zip(gameCFG['players'], gameCFG['strats'])]
    #rndSeed = 607729
    rndSeed = None

    # start logger
    loggFilName = './output/%s.log'%logGameName
    print('Logging game to %s', loggFilName)
    logg = getCreateLogger(name=logGameName, file=loggFilName, level=logLevel)

    # run the game
    thisGame = Game(logGameName, players, gameCFG['start'], rndSeed)
    allResults[indx] = thisGame.play()

    # update the results dataframe
    resultsDF.loc[indx, 'winner'] = allResults[indx]['winner']
    resultsDF.loc[indx, 'num_players'] = len(players)
    resultsDF.loc[indx, 'start'] = allResults[indx]['start']
    resultsDF.loc[indx, 'rebuilt'] = allResults[indx]['times rebuilt']
    resultsDF.loc[indx, 'time_sec'] = allResults[indx]['timing'][3] -\
        allResults[indx]['timing'][1]
    resultsDF.loc[indx, 'cards_played'] = allResults[indx]['discard summary'][0]
    resultsDF.loc[indx, 'points_played'] = allResults[indx]['discard summary'][1]
    resultsDF.loc[indx, 'wilds_played'] = allResults[indx]['discard summary'][-1][0]
    resultsDF.loc[indx, 'wildplus4s_played'] = allResults[indx]['discard summary'][-1][1]
    resultsDF.loc[indx, 'revs_played'] = allResults[indx]['discard summary'][-2][0]
    resultsDF.loc[indx, 'skps_played'] = allResults[indx]['discard summary'][-2][1]
    resultsDF.loc[indx, 'plus2s_played'] = allResults[indx]['discard summary'][-2][2]
    # add player-specific data
    for (pindx, _) in enumerate(players):
        resultsDF.loc[indx, 'player%d_strat'%pindx] = gameCFG['strats'][pindx]['strategy'].__name__
        resultsDF.loc[indx, 'player%d_stratHF'%pindx] = gameCFG['strats'][pindx].get('hurtFirst', False)
        resultsDF.loc[indx, 'player%d_stratHM'%pindx] = gameCFG['strats'][pindx].get('hailMary', False)
        resultsDF.loc[indx, 'player%d_cards_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][0]
        resultsDF.loc[indx, 'player%d_points_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][1]
        resultsDF.loc[indx, 'player%d_wilds_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][-1][0]
        resultsDF.loc[indx, 'player%d_wildplus4s_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][-1][1]
        resultsDF.loc[indx, 'player%d_revs_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][-2][0]
        resultsDF.loc[indx, 'player%d_skps_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][-2][1]
        resultsDF.loc[indx, 'player%d_plus2s_played'%pindx] = allResults[indx]\
            ['player played summary'][pindx][-2][2]
    # add winner data again as separate featues
    winr = allResults[indx]['winner']
    resultsDF.loc[indx, 'winner_strat'] = gameCFG['strats'][winr]['strategy'].__name__
    resultsDF.loc[indx, 'winner_stratHF'] = gameCFG['strats'][winr].get('hurtFirst', False)
    resultsDF.loc[indx, 'winner_stratHM'] = gameCFG['strats'][winr].get('hailMary', False)
    resultsDF.loc[indx, 'winner_cards_played'] = allResults[indx]\
        ['player played summary'][winr][0]
    resultsDF.loc[indx, 'winner_points_played'] = allResults[indx]\
        ['player played summary'][winr][1]
    resultsDF.loc[indx, 'winner_wilds_played'] = allResults[indx]\
        ['player played summary'][winr][-1][0]
    resultsDF.loc[indx, 'winner_wildplus4s_played'] = allResults[indx]\
        ['player played summary'][winr][-1][1]
    resultsDF.loc[indx, 'winner_revs_played'] = allResults[indx]\
        ['player played summary'][winr][-2][0]
    resultsDF.loc[indx, 'winner_skps_played'] = allResults[indx]\
        ['player played summary'][winr][-2][1]
    resultsDF.loc[indx, 'winner_plus2s_played'] = allResults[indx]\
        ['player played summary'][winr][-2][2]

    # serialize results
    filName = loggFilName[:-4] + '.p'
    pickle.dump({'results':allResults[indx], 'game':thisGame, 'log file':loggFilName},
                file=open(filName, 'wb'))
    logg.info('\nResults serialized to %s', filName)

# timing
MCTimeStp = dt.datetime.now()
MCPerfStp = time.perf_counter()
print('Monte Carlo simulation with %d runs completed in %s(m)'%(MCSims, (MCPerfStp - MCPerfStt)/60))
display(resultsDF.head())