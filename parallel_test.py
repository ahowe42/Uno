import logging
import multiprocessing
import numpy.random as rnd


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

class test():
    def __init__(self, indx:int):
        '''
        blah
        '''
       
        self.indx = indx
        self.logg = getCreateLogger(name='test%d'%indx,
                                    file='./output/test%d.log'%indx, level=10)
        self.logg.info(indx)
   
    def log(self, msg:str):
        '''
        blah
        '''

        self.logg.info(msg)


def testMe(indx:int, rndCnts:list):
    '''
    blah
    '''

    rVal = rnd.randint(3)
    this = test(indx)
    this.log('message 1')
    this.log('message 2')
    this.log('message 3')
    this.log(rVal)
    rndCnts[rVal] += 1

    return 'Test %d complete'%indx

if __name__ == '__main__':
    INDX = 10
    eLogg = getCreateLogger(name='exper', file='./output/experiment.log',
                            level=10)
    rndCnts = [0]*4

    def tm(indx:int):
        return testMe(indx, rndCnts)

    with multiprocessing.Pool() as pool:
        for retVal in pool.imap(tm, range(INDX)):
            eLogg.info('Parallel run %s', retVal)
    eLogg.info(rndCnts)