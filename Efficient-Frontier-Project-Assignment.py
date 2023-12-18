import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class EfficientFrontier:
    def __init__(self, assets, start_date, end_date):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data()
        self.returns = self.calculate_returns()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def get_data(self):
        data = yf.download(self.assets, start=self.start_date, end=self.end_date)['Adj Close']
        return data

    def calculate_returns(self):
        returns = self.data.pct_change()
        return returns

    def generate_random_portfolios(self, num_portfolios):
        results = np.zeros((3, num_portfolios))
        risk_free_rate = 0.02  # Change this to your risk-free rate

        all_weights = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.assets))
            weights /= np.sum(weights)
            all_weights.append(weights)

            portfolio_return = np.sum(weights * self.mean_returns)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

            results[0, i] = portfolio_return
            results[1, i] = portfolio_stddev
            results[2, i] = portfolio_sharpe_ratio

        return results, all_weights

    def calculate_efficient_frontier(self):
        num_portfolios = 10000  # You can adjust this value

        results, all_weights = self.generate_random_portfolios(num_portfolios)

        return results, all_weights

    def plot_efficient_frontier(self, results):
        plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
        plt.title('Efficient Frontier')
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

    def main(self):
        efficient_frontier_data, all_weights = self.calculate_efficient_frontier()
        self.plot_efficient_frontier(efficient_frontier_data)

if __name__ == "__main__":
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'JPM', 'GE']  # Replace with your chosen assets
    start_date = '2021-10-31'
    end_date = '2023-10-31'

    ef = EfficientFrontier(assets, start_date, end_date)
    ef.main()
