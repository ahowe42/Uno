'''
# TODO: 2. how to reset deck when empty?
# TODO: 3 playerCardsCounts not being updated
# TODO: define player strategies
# TODO: finish turn processing
'''
from itertools import product
import logging
import time
import datetime as dt
import multiprocessing
import numpy as np
import ipdb


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


def getCreateLogger(name:str, file:str=None, level:int=10):
    '''
    Get a logging object, creating it if non-existent.
    :param name: name of the logger
    :param file: optional file to where to store the log; required
        if creating a new logger object
    :param level: optional (default = 10) min level of logging
    :return logger: logging object
    '''

    logger = logging.getLogger(name)

    if len(logger.handlers) == 0:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(file)

        # Set handler levels
        c_handler.setLevel(0)
        f_handler.setLevel(0)

        # Create formatters and add it to handlers
        dfmt = '%Y%m%d_%H%M%S'
        c_format = logging.Formatter(fmt='%(name)s@%(asctime)s@%(levelname)s@%(message)s',
            datefmt=dfmt)
        f_format = logging.Formatter(fmt='%(process)d@%(asctime)s@%(name)s@%(levelname)s@%(message)s',
            datefmt=dfmt)
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
                raise ValueError('Special/Wild must in [0...,%d]'%(LENSPECIALSWILDS-1))
            if specialWild >= LENSPECIALS:
                self.wild = WILDS[specialWild-LENSPECIALS]
                self.wildIndex = specialWild-LENSPECIALS
            else:
                self.special = SPECIALS[specialWild]
                self.specialIndex = specialWild

        # give the card a name
        if self.colorIndex is not None:
            self.name = self.color + ' '
        else:
            self.name = ''
        if self.valueIndex is None:
            self.name += SPECIALSWILDS[specialWild]
        else:
            self.name += str(self.valueIndex)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Card(%r, %r, %r, %s)'%(self.colorIndex, self.valueIndex,
            self.specialWildIndex, self.name)

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
                if (self.colorIndex == other.colorIndex) & (self.valueIndex != other.valueIndex):
                    # same color, different number
                    place = True
                    reason = 3
                elif (self.colorIndex != other.colorIndex) & (self.valueIndex == other.valueIndex):
                    # different color, same number
                    place = True
                    reason = 4
            # special cards
            elif self.specialIndex is not None:
                if (self.colorIndex == other.colorIndex) & (self.specialIndex != other.specialIndex):
                    # same color, different special
                    place = True
                    reason = 5
                elif (self.colorIndex != other.colorIndex) & (self.specialIndex == other.specialIndex):
                    # different color, same special (not wild)
                    place = True
                    reason = 6

        return place, reason


class Deck():
    def __init__(self, cards:list[Card]=None, shuffle:bool=True):
        '''
        Build the deck of Uno cards. For each color in a standard deck, there are
        1 0's, 2 1-9's, and 2 non-wild special cards. There are 4 of each wild.
        This is a total of 108 cards. The deck is defined as a list of cards.
        :param cards: optional (default=None) list of cards for this deck; if
            None, the deck is built
        :param shuffle: optional (default=True) flag indicating whether or not to
            shuffle the cards generated; only used if cards is None
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
                self.cards = sorted(numbers + nonwilds, key=lambda x:x.colorIndex)\
                    + wilds
        else:
            # used the passed-in list of discards
            self.cards = cards

        # add the size count
        self.size = len(self.cards)
        # keep track of all cards put in the deck for any resets
        self.allCardCount = self.size

        # build the placeables matrix
        self.__buildPlacesMatrix__()

        # initialize the card to next come off the deck
        self.topCard = 0

    def __buildPlacesMatrix__(self):
        '''
        Build the matrix of placeable cards, as a square matrix.
        param cards: list of cards for which to build the matrix
        '''

        # pre-allocate the array
        self.placeables = np.ndarray(shape=(self.size, self.size), dtype=object)
        # iterate over cards to form the matrix - this could be made more efficient,
        # but the matrix is only built rarely, and the duplication does not consume
        # *so much* memory.
        rng = range(self.size)
        for row in rng:
            for col in rng:
                self.placeables[row, col] = self.cards[row].canPlace(self.cards[col])

    def canPlaceThis(self, this:int, that:int):
        '''
        Use the pre-built matrix to define if this card can be place on that card.
        :param this: index of this card
        :param that: index of that card
        :return result: tuple of boolean placement flag and reason
        '''

        return self.placeables[this, that]

    def deal(self, cards:int=1, thisGame=None):
        '''
        Deal a specified number of cards off the top of the deck.
        :param cards: optional (default=1) integer number of cards to deal
        :param thisGame: optional (default=None) game object for possibly
            resetting the deck
        :return dealt: list of tuples of deck indices and cards dealt
        '''

        # only check for enough cards if game passed in
        if thisGame is not None:
            # ensure there are enough cards in the deck
            if (thisGame.discardPile is not None) & (self.size <= cards):
                # not enough cards, so reset the deck
                logg.info('Only %d cards, so resetting deck from discard pile'%self.size)
                thisGame.rebuildDeck()
            elif self.size <= cards:
                # this should never happen
                logg.debug('Only %d cards, but discard is empty'%self.size)
                raise ValueError('Not enough cards to deal')

        # get the cards to deal
        deal = [(indx, self.cards[indx]) for indx in range(self.topCard,
            self.topCard + cards)]

        # update the topCard
        self.topCard += cards
        return deal


class Hand():
    def __init__(self, cards:list[Card]):
        '''
        Build a hand of cards. The hand is an enumeration-based dict of cards.
        :param cards: list of 7 cards initially dealt to this hand; each card
            should be represented by a tuple of (index into the deck, the
            actual card).
        '''

        # initialize the hand
        self.currCards = {}
        self.playedCards = {}
        self.nextIndex = 0

        # add dealt cards to this hand
        for card in cards:
            self.addCard(card, False)
        self.cardCount = len(self.currCards)

        # update hand statistics
        self.colors = {colr:[] for colr in range(LENCOLORS)}
        self.values = {valu:[] for valu in range(LENVALUES)}
        self.specials = {spec:[] for spec in range(LENSPECIALS)}
        self.wilds = {wild:[] for wild in range(LENWILDS)}
        self.colorCounts = [0]*LENCOLORS
        self.__handSummarize__()

        # update the points
        self.__computePoints__()

    def __computePoints__(self):
        '''
        Compute the (losing) points for this hand.
        '''

        self.points = 0

        # number cards
        self.points += sum([key*len(val) for (key, val) in self.values.items()
            if val != {}])
        # special cards
        for (index, cards) in self.specials.items():
            self.points += SPECIALS_POINTS[index]*len(cards)
        # wild cards
        for (index, cards) in self.wilds.items():
            self.points += WILDS_POINTS[index]*len(cards)

    def __handSummarize__(self):
        '''
        Generate a summary of the current hand, storing the index into the hand
        in colors, values, specials, and wilds (all index-based) dicts. Each entry
        in these dicts is a list holding indices into the hand.
        '''

        # iterate over cards in the hand now
        for (index, card) in self.currCards.items():
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
        self.canStack = np.ndarray(shape=(self.cardCount, self.cardCount),
            dtype=object)
        rng = range(self.cardCount)
        for row in rng:
            for col in rng:
                if row==col:
                    self.canStack[row, col] = (None, None)
                else:
                    self.canStack[row, col] = self.currCards[row][1].\
                        canPlace(self.currCards[col][1])

    def addCard(self, card:tuple[int, Card], updateSummary:bool=True):
        '''
        Add a new card to this hand and possibly update the stats.
        :param card: tuple holding index of this card into the deck and
            the actual card
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
            self.__computePoints__()

    def playCard(self, card:int):
        '''
        Play a card, moving it from the current hand, and updating the summary.
        :param card: index into the hand of the card to play
        :return card: tuple of card index and card to play
        :return remain: number of cards remaining
        '''

        # move to played cards
        this = self.currCards.pop(card)
        self.playedCards[card] = this
        # update number of Cards
        self.cardCount = len(self.currCards)

        # update the summary
        self.__handSummarize__()

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
            prt += '\nCard %d = %r'%(index, card)

        # maybe show the summary
        if summary:
            prt += '\nHand Summary'
            # colors
            for (colorIndex, cards) in self.colors.items():
                if len(self.colors[colorIndex]) > 0:
                    prt += '\nColor = ' + COLORS[colorIndex]
                    for card in cards:
                        prt += '\n\tCard %d = %r'%(card, self.currCards[card])
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
                        prt += '\n\tCard %d = %r'%(card, self.currCards[card])
            # wilds
            for (wildIndex, cards) in self.wilds.items():
                if len(self.wilds[wildIndex]) > 0:
                    prt += '\nWild = ' + WILDS[wildIndex]
                    for card in cards:
                        prt += '\n\tCard %d = %r'%(card, self.currCards[card])
            # show the stackables
            prt += '\nStackable Matrix\n%s'%\
                np.array([str(s[0])[0] for s in thisGame.players[0].hand.canStack.flatten()]).\
                reshape(self.cardCount, self.cardCount)

        # maybe iterate and print played cards
        if played & (self.playedCards != {}):
            prt += '\nPlayed Cards'
            for (index, card) in self.playedCards.items():
                prt += '\nCard %d = %r'%(index, card)

        return prt


class Player():
    def __init__(self, name:str, strategy, hand:Hand=None):
        '''
        Define an Uno player.
        :param name: string name of player
        :param strategy: ??? strategy of player
        :param hand: optional (default=None) Hand object for player
        '''
        self.name = name
        self.strategy = strategy
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
        '''

        ''' strategy ideas: prefer finish color, switch max color,
        hurt next player, change player, wait to hurt next player,
        first blood '''

        # if discard started with wild, must choose the color
        if thisGame.currSpecial is not None:
            if (thisGame.currSpecial > 2) & (thisGame.currColor is None):
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
            playableCards.extend(self.hand.specials[3])
            # wild +4s only playable if none of the current color in the hand
            if len(self.hand.colors[thisGame.currColor]) == 0:
                playableCards.extend(self.hand.specials[4])
                logg.debug('\nNo %d color cards, so wild +4 is playable',
                    thisGame.currColor)
            # get uniques
            playableCards = set(playableCards)

            # do we need to draw a card?
            if (len(playableCards) == 0) & (whilePass == 0):
                logg.info('\n\tNo cards to play, drawing 1')
                self.hand.addCard(self.deck.deal(1, thisGame)[0])
            whilePass += 1

        # find why each of the playable cards are playable
        '''
        0 = same card
        1 = same color, different number
        2 = different color, same number
        3 = wild card
        4 = same color, different special (not wild)
        5 = different color, same special (not wild)
        '''
        sameColor = diffColor = wilds = sameColorSpecial = 0
        playables = dict.fromkeys(playableCards)
        for handIndx in playableCards:
            # get the card in the hand
            card = self.hand.currCards[handIndx]
            playables[handIndx] = thisGame.deck.canPlaceThis(card[0],
                thisGame.currCardIndex)
            # summarize
            sameColor += playables[handIndx][1] in [0, 1, 4]
            diffColor += playables[handIndx][1] in [2, 5]
            wilds += (playables[handIndx][1] == 3)
            sameColorSpecial += (playables[handIndx][1] in [0, 4])
            logg.debug('\nPlayable %d = (%d, %r): reason = %d', handIndx,
                *self.hand.currCards[handIndx], playables[handIndx][1])
        logg.debug('\nPlayable: %d same colors, %d different colors, %d wilds, %d same color specials',
            sameColor, diffColor, wilds, sameColorSpecial)

        ''' now determine the best card to play '''
        '''  TODO need to rethink this about rating plays'''

        bestCard = None

        # for now implement "prefer finish color"
        if sameColor > 0:
            # can we place a skip or reverse to play an extra card
            if sameColorSpecial > 0:
                # iterate over playables and find the relevant Cards
                playSpecials = {spec:[] for spec in range(SPECIALS_COUNT)}
                for (handIndx, result) in playables.items():
                    if result[1] in [0, 4]:
                        playSpecials[self.hand.currCards[handIndx][1].specialIndex] = 1
                # now that we have the playable specials cards, what to do
                if (len(playSpecials[0]) > 0) | (len(playSpecials[1]) > 0) &\
                    (sameColor > 1) & (thisGame.playersCount == 2):
                    # same color skip or reverse & thisGame.playersCount==2
                    pass
                else:
                    pass
                # what about skip/rev + +2?
                # what about avoiding next player if has uno?
            else:
                # just play a color card
                for (handIndx, result) in playables.items():
                    if result[1] in [0, 1, 4]:
                        bestCard = handIndx
                        break
        elif diffColor > 0:
            pass
        elif wild > 0:
            ''' TODO: choose best color '''
            bestColor = None
        else:
            logg.info('\nNo card to play')

        # play the best card
        if bestCard is not None:
            self.hand.playCard(bestCard)
            thisGame.addToDiscard(bestCard, bestColor)

        # yell Uno!
        if self.hand.cardCount == 1:
            logg.info('\nUno!')
        elif self.hand.cardCount == 0:
            logg.info('\nI won!')


class Game():
    def __init__(self, descrip:str, players:list, start:int=0):
        '''
        Initialize a game of Uno.
        :parm descrip: string description of game
        :param players: list of player objects
        :param start: optional (default=None) index of starting player; if not
            provided, starter is randomly selected
        '''

        self.name = descrip
        self.players = players
        self.playersCount = len(players)
        self.playersOrder = 1 # 1 or -1

        # get the Deck & put the top card on the pile
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
        else:
            self.currPlayer = start
        # handle first discard card = skip
        if self.currSpecial is not None:
            if SPECIALS[self.currSpecial] == 'skp':
                self.currPlayer = (self.currPlayer + 1) % self.playersCount
        self.nextPlayer = self.__nextPlayer__()

        # talk
        logg.info('\nUno game %s initialized with players', self.name)
        for player in self.players:
            logg.info(player)
        logg.info('\nInitial discard card = (%d = %r)', *self.discardPile[0])
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
            if SPECIALS[self.currSpecial] == 'rev':
                # reverse - change direction
                self.playersOrder *= -1
            elif SPECIALS[self.currSpecial] == 'skp':
                # skip - pretend current is next
                curr += 1

        # set the next player
        return (curr + self.playersOrder) % self.playersCount

    def addToDiscard(self, card:Card, colorIndex:int=None):
        '''
        Add a card to the discard pile.
        :param card: tuple of the index in the deck and the card object to add
        :param colorIndex: optional (default=None) color of card
        '''
        # add
        self.discardPile.append(card)

        # define the deck index of the current card
        self.currCardIndex = card[0]

        # define current color if not provided
        if colorIndex is None:
            self.currColor = card[1].colorIndex
        else:
            self.currColor = colorIndex
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
                logg.info('\n\t +2 on discard pile, so drawing 2')
                for (indx, cards) in enumerate(self.deck.deal(2, self)):
                    self.currPlayer.hand.addCard(card, updateSummary=(indx==1))
        elif self.currWild is not None:
            if self.currWild == 1:
                # wild+4, so draw 4
                logg.info('\n\t Wild +4 on discard pile, so drawing 4')
                for (indx, cards) in enumerate(self.deck.deal(4, self)):
                    self.currPlayer.hand.addCard(card, updateSummary=(indx==3))
        # take the turn
        self.players[self.currPlayer].takeTurn()
        # add the card(s) played to the discard pile
        # TODO: how does this work?
        # set the next player
        self.nextPlayer = self.__nextPlayer__()
        # set the new current player
        self.currPlayer = self.nextPlayer

    def play(self):
        '''
        Let's play Uno!.
        '''

        # timing
        gameTimeStt = dt.datetime.now()
        gamePerfStt = time.perf_counter()

        # iterate over players, each taking their turn
        while min(self.playerCardsCounts) > 0:
            self.playOne()

    def rebuildDeck(self):
        '''
        Deck is too short to deal cards to player, so rebuild it.
        '''

        # get discards sans top card for new deck & shuffle
        cards = self.discardPile[:-1]
        np.random.shuffle(cards)
        # take all cards from each player *in reverse order* and add
        for player in self.players[::-1]:
            cards.extend(player.hand.currCards.values())
            player.hand.currCards = {}
        # add the top card
        cards += [self.discardPile[-1]]
        # create new deck object
        self.deck = Deck(cards=[card[1] for card in cards])
        # add top card back to discard
        self.discardPile = []
        self.addToDiscard(self.deck.deal(1)[0])
        # deal back all cards
        for (pindx, player) in enumerate(self.players):
            # get the cards dealt & add them
            cards = self.deck.deal(self.playerCardsCounts[pindx])
            for card in cards[:-1]:
                player.hand.addCard(card, updateSummary=False)
            player.hand.addCard(card, updateSummary=False)


# testing code
# start the log
sttTS = dt.datetime.now()
loggFilName = './Uno_%s.log'%(sttTS.strftime('%Y%m%d_%H%M%S'))
logg = getCreateLogger(name='UNO', file=loggFilName)
# setup the game
np.random.seed(42)
thisGame = Game('test', [Player('Andrew', None), Player('Resham', None),
    Player('Ma', None)])


'''# process
with multiprocessing.Pool() as pool:
    for (indx, expd) in pool.imap(expandDates, thisData.itertuples(name=None)):
        results[indx] = expd'''
