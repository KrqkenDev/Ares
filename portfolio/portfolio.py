from datetime import UTC, datetime


class Portfolio:
    """
    Manages the user's portfolio, including cash balance, stock holdings, and trade history.
    Args:
        starting_cash (float): The initial cash balance for the portfolio. Defaults to 100,000.00.
    Uses:
        - cash (float): The current cash balance in the portfolio.
        - holdings (dict): A dictionary containing stock tickers as keys and their corresponding
          details (shares owned and average purchase price) as values.
        - trade_history (list): A list of dictionaries representing the history of trades made
          in the portfolio.
    """
    def __init__(self, starting_cash: float = 100000.00):
        self.starting_cash = starting_cash
        self.cash = starting_cash

        self.holdings: dict[str, dict[str, float | int]] = {}
        self.trade_history: list[dict] = []

    def reset(self) -> None:
        """
        Reset portfolio to its initial state
        """
        self.cash = self.starting_cash
        self.holdings.clear()
        self.trade_history.clear()


    def get_cash(self) -> float:
        return self.cash
    
    def get_position(self, ticker: str) -> dict | None:
        """
        Retrieves the position details for a specific stock ticker.
        """
        return self.holdings.get(ticker)
    
    def get_trade_history(self) -> list:
        """
        Fetches the trade history of the portfolio.
        """
        return self.trade_history

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """
        Calculates total value of the portfolio.
        
        Args:
            current_prices
                - Dictionary containing ticker prices.
        
        Returns:
            - Cash and value of all total holdings.
        """

        total_value = self.cash

        for ticker, position in self.holdings.items():
            if ticker not in current_prices:
                raise KeyError(f"Missing price data for {ticker}")

            shares = position["shares"]
            price = current_prices[ticker]

            total_value += shares * price

        return float(total_value)

    
    def owns_stock(self, ticker: str) -> bool:
        """
        Checks if the portfolio owns any shares of a specific stock.
        Returns: True if the stock is owned, False otherwise.
        """
        return ticker in self.holdings

    
    def _remove_empty_position(self, ticker: str):
        """
        Removes a stock from the holdings if the number of shares owned is zero.
        """
        if ticker in self.holdings and self.holdings [ticker]["shares"] == 0:
            del self.holdings[ticker]

    def withdraw_cash(self, amount: float) -> None:
        if self.cash >= amount and amount > 0:
            self.cash -= amount
        else:
            raise ValueError("Not enough cash for transaction.")

    def deposit_cash(self, amount: float) -> None:
        if amount > 0:
            self.cash += amount
        else:
            raise ValueError("Not enough cash for transaction.")

    def sell_stock(self, ticker: str, shares: int, price: float):
        """
        Sells a specified number of shares of a stock at a given price.
        Updates the holdings and trade history accordingly.
        Args:
            ticker (str): The stock ticker symbol.
            shares (int): The number of shares to sell.
            price (float): The price at which to sell the shares.
        """
        if shares <= 0:
            raise ValueError("Number of shares to sell must be positive.")
        if price <= 0:
            raise ValueError("Price must be positive.")

        if ticker not in self.holdings or self.holdings[ticker]["shares"] < shares:
            raise ValueError(f"Not enough shares of {ticker} to sell.")
        
        average_cost = self.holdings[ticker]["average_cost"]

        trade_value = shares * price

        #Calculate profit from the sale
        profit = (price - average_cost) * shares


        # Update holdings
        self.holdings[ticker]["shares"] -= shares

        # Remove the stock from holdings if no shares are left
        self._remove_empty_position(ticker)
        
        # Record the trade in trade history
        trade_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": "SELL",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "total_value": trade_value,
            "profit": profit
        }

        self.trade_history.append(trade_record)
        
    def buy_stock(self, ticker: str, shares: int, price: float):
        """
        Buys a specified number of shares of a stock at a given price.
        Updates the holdings and trade history accordingly.
        Args:
            ticker (str): The stock ticker symbol.
            shares (int): The number of shares to buy.
            price (float): The price at which to buy the shares.
        """
        if shares <= 0:
            raise ValueError("Number of shares to buy must be positive.")
        if price <= 0:
            raise ValueError("Price must be positive.")

        trade_value = shares * price

        # Update holdings
        if ticker in self.holdings:
            current_shares = self.holdings[ticker]["shares"]
            current_average_cost = self.holdings[ticker]["average_cost"]

            new_total_shares = current_shares + shares
            new_average_cost = ((current_average_cost * current_shares) + trade_value) / new_total_shares

            self.holdings[ticker]["shares"] = new_total_shares
            self.holdings[ticker]["average_cost"] = new_average_cost
        else:
            self.holdings[ticker] = {
                "shares": shares,
                "average_cost": price
            }

        # Record the trade in trade history
        trade_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "total_value": trade_value
        }

        self.trade_history.append(trade_record)