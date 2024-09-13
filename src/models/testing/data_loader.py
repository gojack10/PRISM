import pandas as pd
from sqlalchemy import create_engine
import os

def load_data():
    # Database connection
    db_path = "/mnt/SSD_Storage/Documents/coding-projects/PRISM/data/processed/market_data.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    tickers = ['AAPL', 'NVDA', 'GOOG']
    all_data = []

    for ticker in tickers:
        query = f"""
        WITH latest_company_overview AS (
            SELECT *
            FROM company_overview
            WHERE ticker = '{ticker}'
        ),
        latest_stock_sentiment AS (
            SELECT *
            FROM stock_sentiment
            WHERE ticker = '{ticker}'
        )
        SELECT 
            i.date, 
            '{ticker}' AS ticker,
            i.open, 
            i.high, 
            i.low, 
            i.close, 
            i.volume,
            s.sentiment_score,
            s.sentiment_confidence,
            s.key_topics,
            s.sentiment_change,
            s.financial_metrics,
            s.short_term_outlook,
            s.long_term_outlook,
            co.PERatio,
            co.PEGRatio,
            co.PriceToBookRatio,
            co.OperatingMarginTTM,
            co.ReturnOnEquityTTM,
            co.QuarterlyEarningsGrowthYOY,
            co.QuarterlyRevenueGrowthYOY,
            co.DividendYield,
            co.AnalystTargetPrice,
            co."50DayMovingAverage",
            co."200DayMovingAverage",
            co.Beta,
            co.MarketCapitalization,
            sma.value AS sma_value,
            macd.macd AS macd_line,
            macd.macd_signal AS signal_line,
            macd.macd_hist AS macd_histogram,
            rsi.value AS rsi_value,
            bb.real_upper_band,
            bb.real_middle_band,
            bb.real_lower_band,
            obv.value AS obv_value,
            cci.value AS cci_value
        FROM intraday_{ticker} i
        LEFT JOIN latest_stock_sentiment s ON s.date <= i.date
        LEFT JOIN latest_company_overview co ON co.date <= i.date
        LEFT JOIN indicator_sma_{ticker} sma ON i.date = sma.date
        LEFT JOIN indicator_macd_{ticker} macd ON i.date = macd.date
        LEFT JOIN indicator_rsi_{ticker} rsi ON i.date = rsi.date
        LEFT JOIN indicator_bbands_{ticker} bb ON i.date = bb.date
        LEFT JOIN indicator_obv_{ticker} obv ON i.date = obv.date
        LEFT JOIN indicator_cci_{ticker} cci ON i.date = cci.date
        ORDER BY i.date ASC
        """
        
        df = pd.read_sql(query, engine)
        all_data.append(df)

    # Combine data from all tickers
    combined_data = pd.concat(all_data, ignore_index=True)

    # Convert date to datetime and set as index
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    combined_data.set_index('date', inplace=True)

    # Handle missing values
    combined_data = handle_missing_values(combined_data)

    # Add any additional derived features
    combined_data = add_derived_features(combined_data)

    # Drop rows with NaN values created by rolling computations
    combined_data.dropna(inplace=True)

    # Verify data sufficiency
    print(f"Total rows in combined dataset: {len(combined_data)}")
    print(f"Number of unique dates: {combined_data.index.nunique()}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"Columns in the dataset: {combined_data.columns.tolist()}")
    print(combined_data.info())

    return combined_data

def handle_missing_values(df):
    # Fill missing values for numerical columns with group-wise mean
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df.groupby('ticker')[numeric_columns].transform(lambda x: x.fillna(x.mean()))
    
    # For sentiment-related columns, fill with neutral values
    sentiment_columns = ['sentiment_score', 'sentiment_confidence']
    df[sentiment_columns] = df[sentiment_columns].fillna(0)
    
    # For text columns, fill with 'Unknown'
    text_columns = ['key_topics', 'sentiment_change', 'financial_metrics', 'short_term_outlook', 'long_term_outlook']
    df[text_columns] = df[text_columns].fillna('Unknown')
    
    return df

def add_derived_features(df):
    # Add percentage change in close price
    df['close_pct_change'] = df.groupby('ticker')['close'].pct_change()
    
    # Add rolling mean of volume (e.g., 5-day rolling mean)
    df['volume_rolling_mean'] = df.groupby('ticker')['volume'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Add a binary column for whether close price is above SMA
    df['above_sma'] = (df['close'] > df['sma_value']).astype(int)
    
    # Add volatility (e.g., 20-day rolling standard deviation of returns)
    df['volatility'] = df.groupby('ticker')['close_pct_change'].rolling(window=20, min_periods=1).std().reset_index(0, drop=True)
    
    # Add MACD crossover signal
    df['macd_crossover'] = ((df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) <= df['signal_line'].shift(1))).astype(int)
    
    return df

if __name__ == "__main__":
    data = load_data()
    print(data.head())
