from assets.asset import Asset
from common.enums import Exchange, OrderAction
from portfolio.portfolio import Portfolio

from dataclasses import dataclass
    

@dataclass(frozen=True)
class Order:
    asset: Asset
    shares: int
    price: float
    action: OrderAction


class Broker:
    """
    The broker class is responsible for executing trades on behalf of the portfolio. 
    It calculates the total cost of a trade, including commissions and any applicable 
    taxes, and validates orders before execution.
    Args:
        - portfolio (Portfolio) : The portfolio instance that the broker will manage trades for.
        - commission_rate (float) : The commission rate charged by the broker for each trade.
        - uk_stamp_duty_rate (float) : The stamp duty rate for trades executed on the London Stock Exchange (LSE).
        - Note : Later more markets and their respective taxes can be added as needed.
    """
    def __init__(self, portfolio: Portfolio, 
                 commission_rate: float = 0.0, 
                 uk_stamp_duty_rate: float = 0.005):
        
        self.portfolio = portfolio
        self.commission_rate = commission_rate
        self.uk_stamp_duty_rate = uk_stamp_duty_rate

    def _calculate_commission(self, order:Order) -> float:
        return self._trade_value(order)*self.commission_rate   

    def _calculate_uk_stamp_duty(self, order: Order)-> float:
        if order.action != OrderAction.BUY:
            return 0.0

        if order.asset.exchange != Exchange.LSE:
            return 0.0
        
        trade_value = order.shares * order.price

        return trade_value * self.uk_stamp_duty_rate
       

    def _calculate_buy_cost(self, order: Order)-> float:
        """
        Calculates the total cost of a buy order, including trade value,
        commission, and UK stamp duty if applicable.
        """
        trade_value = self._trade_value(order)
        commission = self._calculate_commission(order)
        uk_stamp_duty = self._calculate_uk_stamp_duty(order)

        return trade_value + commission + uk_stamp_duty
    
    def _calculate_sell_proceeds(self, order:Order) -> float:
        trade_value = self._trade_value(order)
        commission = self._calculate_commission(order)
        net_proceeds = trade_value - commission
        return net_proceeds

    def _validate_order(self, order: Order)-> None:
        """
        Validates the order to ensure that it has a valid ticker, positive shares and price, 
        and a valid action (BUY or SELL). 
        Raises ValueError if any validation fails.
        """
        if not order.asset.ticker.strip():
            raise ValueError("Order ticker must be provided.")
        
        if not isinstance(order.asset.exchange, Exchange):
            raise ValueError("Asset exchange must be a valid Exchange")
        
        if order.shares <= 0:
            raise ValueError("Order shares must be positive.")
        
        if order.price <= 0:
            raise ValueError("Order price must be positive.")
        
        if order.action not in (OrderAction.BUY, OrderAction.SELL):
            raise ValueError("Order action must be 'BUY' or 'SELL'.")
        
    def _trade_value(self, order: Order) -> float:
        return order.shares * order.price
    
   
    def execute_order(self, order:Order) -> None:
        """
        Executes the valid order through the portfolio.

        BUY:
            - Checks the total cost and fees
            - Buys shares
            - Removes fees from cash
        SELL:
            Sells shares
            Applies the commission.
        Also it:
            - Validates the order
            - Calculates costs/proceeds
            - Updates holdings
        """
        self._validate_order(order)

        if order.action == OrderAction.BUY:
            total_cost = self._calculate_buy_cost(order)

            if self.portfolio.get_cash() < total_cost:
                raise ValueError("Insufficient funds for this trade.")
            
            self.portfolio.withdraw_cash(total_cost)
            
            #Buying shares
            self.portfolio.buy_stock(ticker=order.asset.ticker,
                                     shares=order.shares, 
                                     price=order.price)

        elif order.action == OrderAction.SELL:

            proceeds = self._calculate_sell_proceeds(order)
            #Sells shares
            self.portfolio.sell_stock(ticker=order.asset.ticker,
                                      shares=order.shares,
                                      price=order.price)
            #Then, updates money.
            self.portfolio.deposit_cash(proceeds)
