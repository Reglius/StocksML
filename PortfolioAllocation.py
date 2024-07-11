# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define stocks and timeframe
stocks = [
'PARA',
'PWR',
'ORCL',
'VST',
'AAPL',
'RL',
'NTAP',
'XOM',
'EMN',
'CTRA'
]
start_date = '2022-07-01'
end_date = '2024-07-01'


# Fetching stock data
stock_data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

# Calculating Log Returns
log_returns = np.log(stock_data / stock_data.shift(1))

# Covariance Matrix
cov_matrix = log_returns.cov()

# Annualized Returns
annual_returns = log_returns.mean() * 250

# Number of assets
num_assets = len(stocks)

# Function to generate random portfolios
def generate_portfolios(num_portfolios, num_assets, cov_matrix, annual_returns):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(weights * annual_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = results[0,i] / results[1,i] # Sharpe Ratio
        weights_record.append(weights)
    return results, weights_record

# Number of portfolios to simulate
num_portfolios = 10000

# Generating portfolios
results, weights = generate_portfolios(num_portfolios, num_assets, cov_matrix, annual_returns)

# Convert results to DataFrame
portfolios = pd.DataFrame({'Return': results[0], 'Volatility': results[1], 'Sharpe Ratio': results[2]})

# Find portfolio with maximum Sharpe Ratio
max_sharpe_port = portfolios.iloc[portfolios['Sharpe Ratio'].idxmax()]

# Find portfolio with minimum volatility
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]

# Plotting efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker='*', color='r', s=500, label='Max Sharpe Ratio')
plt.scatter(min_vol_port[1], min_vol_port[0], marker='*', color='g', s=500, label='Min Volatility')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.legend(labelspacing=0.8)
plt.grid(True)
plt.show()

# Print the optimal portfolios
print("Portfolio with Maximum Sharpe Ratio:")
print(max_sharpe_port)
print("\nPortfolio with Minimum Volatility:")
print(min_vol_port)

# Print the weights of each stock in the portfolios
print("Weights of each stock in the portfolio with Maximum Sharpe Ratio:")
max_sharpe_weights = weights[portfolios['Sharpe Ratio'].idxmax()]
max_sharpe_weights_dict = {stocks[i]: max_sharpe_weights[i] for i in range(len(stocks))}
print(max_sharpe_weights_dict)

print("\nWeights of each stock in the portfolio with Minimum Volatility:")
min_vol_weights = weights[portfolios['Volatility'].idxmin()]
min_vol_weights_dict = {stocks[i]: min_vol_weights[i] for i in range(len(stocks))}
print(min_vol_weights_dict)



for i in range(len(stocks)):
    print(stocks[i], max_sharpe_weights[i])

print(max_sharpe_port)
print(portfolios['Sharpe Ratio'].idxmax())