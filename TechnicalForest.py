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

# Define function to get valuation measures
def get_valuation_measures(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
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

    # Calculate next month's return
    next_month_start = end_date + timedelta(days=1)
    next_month_end = end_date + relativedelta(months=1)
    next_month_df = stock.history(start=next_month_start, end=next_month_end)
    if not next_month_df.empty:
        next_month_return = (next_month_df['Close'].iloc[-1] / df['Close'].iloc[-1]) - 1
    else:
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
        'Date': end_date,
        'Next_Month_Return': next_month_return
    }

    return valuation_measures

# Load S&P 500 companies
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()
tickers = [tick.replace('.', '-') for tick in tickers]

# Get historical data and valuation measures for each stock
def get_stock_data(ticker):
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            end_date = datetime.strptime('2024-05-31', "%Y-%m-%d").date()
            start_date = end_date - relativedelta(years=2)
            all_data = []

            current_start_date = start_date
            while current_start_date < end_date:
                current_end_date = current_start_date + relativedelta(months=1)
                measures = get_valuation_measures(ticker, current_start_date, current_end_date)
                if measures is not None:
                    all_data.append(measures)
                current_start_date += relativedelta(months=1)

            if not all_data:
                print(f"No data found for {ticker}")
                tickers.remove(ticker)
                return None

            return all_data
        except Exception as e:
            print(f"Error fetching data for {ticker} (attempt {attempts + 1}/{max_attempts}): {e}")
            attempts += 1
            time.sleep(2)  # Wait for 2 seconds before retrying

    tickers.remove(ticker)
    return None

# Fetch data for all tickers with progress bar
data = []
for ticker in tqdm(tickers, desc="Fetching stock data"):
    stock_data = get_stock_data(ticker)
    if stock_data is not None:
        data.extend(stock_data)

# Convert to DataFrame
valuation_df = pd.DataFrame(data).dropna()
print(valuation_df)

# Prepare data for model
combined_df = valuation_df.dropna(subset=['Next_Month_Return'])
X = combined_df[['Market Cap', 'P/E Ratio', 'P/B Ratio', 'Dividend Yield', 'P/S Ratio',
                 'Enterprise Value', 'EV/EBITDA Ratio', 'Gross Margin', 'Operating Margin',
                 'Net Profit Margin', 'ROE', 'ROA']]
y = combined_df['Next_Month_Return']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

try:
    joblib.dump(rf_model, r"C:\Users\pawc3\Desktop\random_forest.joblib")
except Exception as e:
    print("Error saving randomforest")

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Predict the next month's returns for all stocks
final_predictions = {}
for ticker in tqdm(tickers, desc="Predicting next month's returns"):
    if ticker in combined_df['Ticker'].values:
        df = combined_df[combined_df['Ticker'] == ticker]
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
