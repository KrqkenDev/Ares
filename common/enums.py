from enum import Enum

class Exchange(Enum):
    """
    Supported stock exchanges for trading assets. 
    This enum is used to specify the exchange on which an asset is traded.
    """

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    EURONEXT = "EURONEXT"
    TSE = "TSE"

class OrderAction(Enum):
    """
    Supported order actions for trading assets. 
    This enum is used to specify whether an order is a buy or sell action.
    """
    BUY = "BUY"
    SELL = "SELL"



class AgentAction(Enum):

    HOLD = 0
    BUY = 1
    SELL = 2

    @classmethod
    def count(cls):
        return len(cls)
