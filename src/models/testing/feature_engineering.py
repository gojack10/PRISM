import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_lagged_features(df, columns, lags):
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('ticker')[col].shift(lag)
        else:
            logger.warning(f"Column '{col}' not found. Skipping lagged features for this column.")
    return df

def calculate_returns(df, column='close'):
    if column in df.columns:
        df[f'{column}_return'] = df.groupby('ticker')[column].pct_change()
    else:
        logger.warning(f"Column '{column}' not found. Skipping returns calculation.")
    return df

def create_rolling_features(df, columns, windows):
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_{window}'] = df.groupby('ticker')[col].rolling(window=window).mean().reset_index(0, drop=True)
        else:
            logger.warning(f"Column '{col}' not found. Skipping rolling features for this column.")
    return df

def normalize_features(df, columns):
    columns_to_normalize = [col for col in columns if col in df.columns]
    if columns_to_normalize:
        scaler = StandardScaler()
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    else:
        logger.warning("No columns to normalize.")
    return df

def engineer_features(df):
    logger.info("Columns at the start of feature engineering:")
    logger.info(df.columns)

    # Create lagged features
    df = create_lagged_features(df, ['volume'], [1, 7])  # Removed 'close' as it's not in the input
    
    # Calculate returns (skip this step as 'close' is not in the input)
    # df = calculate_returns(df)
    
    # Create rolling features
    df = create_rolling_features(df, ['volume'], [7, 30])  # Removed 'close' as it's not in the input
    
    # Add time-based features
    if 'date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['date'], unit='s').dt.dayofweek
        df['month'] = pd.to_datetime(df['date'], unit='s').dt.month
        
        # Apply cyclical encoding to time-based features
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
        
        # Drop original day_of_week, month, and date columns
        df = df.drop(['day_of_week', 'month', 'date'], axis=1)
    else:
        logger.warning("'date' column not found. Skipping time-based features.")
    
    # One-hot encode the 'ticker' column
    if 'ticker' in df.columns:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ticker_encoded = encoder.fit_transform(df[['ticker']])
            
            # Try to use get_feature_names_out for newer scikit-learn versions
            try:
                feature_names = encoder.get_feature_names_out(['ticker'])
            except AttributeError:
                # Fall back to get_feature_names for older versions
                feature_names = encoder.get_feature_names(['ticker'])
            
            ticker_columns = pd.DataFrame(ticker_encoded, columns=feature_names, index=df.index)
            df = pd.concat([df.drop('ticker', axis=1), ticker_columns], axis=1)
        except Exception as e:
            logger.error(f"Error during one-hot encoding: {str(e)}")
            # If one-hot encoding fails, we'll just drop the 'ticker' column
            df = df.drop('ticker', axis=1)
    else:
        logger.warning("'ticker' column not found. Skipping one-hot encoding for tickers.")
    
    # Normalize numerical features
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df = normalize_features(df, numerical_columns)
    
    logger.info("Columns at the end of feature engineering:")
    logger.info(df.columns)
    logger.info("Data types after feature engineering:")
    logger.info(df.dtypes)
    
    return df
