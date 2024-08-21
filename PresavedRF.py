import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pytz
import time
import joblib

def get_valuation_measures(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history()
    if df.empty:
        return None

    # Fetch financial data
    market_cap = df['Close'].iloc[-1] * stock.info['sharesOutstanding']
    pe_ratio = stock.info.get('forwardPE', np.nan)
    pb_ratio = stock.info.get('priceToBook', np.nan)
    dividend_yield = stock.info.get('dividendYield', np.nan)
    ps_ratio = stock.info.get('priceToSalesTrailing12Months', np.nan)
    ev = stock.info.get('enterpriseValue', np.nan)
    ev_ebitda = stock.info.get('enterpriseToEbitda', np.nan)
    gross_margin = stock.info.get('grossMargins', np.nan)
    operating_margin = stock.info.get('operatingMargins', np.nan)
    net_profit_margin = stock.info.get('profitMargins', np.nan)
    roe = stock.info.get('returnOnEquity', np.nan)
    roa = stock.info.get('returnOnAssets', np.nan)

    next_month_return = np.nan

    valuation_measures = {
        'Ticker': ticker,
        'Market Cap': market_cap,
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Dividend Yield': dividend_yield,
        'P/S Ratio': ps_ratio,
        'Enterprise Value': ev,
        'EV/EBITDA Ratio': ev_ebitda,
        'Gross Margin': gross_margin,
        'Operating Margin': operating_margin,
        'Net Profit Margin': net_profit_margin,
        'ROE': roe,
        'ROA': roa,
        'Date': '2024-07-31',
        'Next_Month_Return': next_month_return
    }

    return valuation_measures

# Load S&P 500 companies
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()
tickers = [tick.replace('.', '-') for tick in tickers]

all_data = []
for ticker in tqdm(tickers, desc="Getting Stock Data"):
    measures = get_valuation_measures(ticker)
    if measures is not None:
        all_data.append(measures)

# Convert the list of dictionaries to a DataFrame
data = pd.DataFrame(all_data)

# Load the pre-trained Random Forest model
rf_model = joblib.load(r"C:\Users\pawc3\Desktop\random_forest.joblib")

# Predict the next month's returns for all stocks
final_predictions = {}
for ticker in tqdm(tickers, desc="Predicting next month's returns"):
    df = data[data['Ticker'] == ticker]
    if not df.empty:
        X = df[['Market Cap', 'P/E Ratio', 'P/B Ratio', 'Dividend Yield', 'P/S Ratio',
                'Enterprise Value', 'EV/EBITDA Ratio', 'Gross Margin', 'Operating Margin',
                'Net Profit Margin', 'ROE', 'ROA']].mean().to_frame().T
        final_predictions[ticker] = rf_model.predict(X)[0]

# Convert final predictions to DataFrame and get the top 10 performers
final_predictions_df = pd.DataFrame(list(final_predictions.items()), columns=['Ticker', 'Predicted_Return'])
top_10_performers = final_predictions_df.nlargest(10, 'Predicted_Return')

# Display the top 10 performers and their performance
print("Top 10 performance predictions for the next month:")
print(top_10_performers)
