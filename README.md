# PRISM

Predictive Research Integrating Sentiment and Market indicators

PRISM is a data-driven stock analysis and prediction system that combines traditional technical indicators with advanced sentiment analysis. The project aims to provide a comprehensive framework for collecting, processing, and analyzing financial data to generate insights and predictions about stock market movements.

## Key Components

1. **Data Collection**
   - `alpha_vantage_intraday.py`: Fetches daily stock price data
   - `alpha_vantage_indicators.py`: Retrieves technical indicators (SMA, MACD, RSI, BBANDS, OBV, CCI)
   - `alpha_vantage_overview.py`: Collects company overview data (P/E ratio, market cap, etc.)
   - `alpha_vantage_sentiment.py`: Gathers news sentiment data

2. **Data Processing**
   - `sentiment_json_to_claude.py`: Processes raw sentiment data using the Claude AI model
   - `database_builder.py`: Consolidates all collected data into a SQLite database

3. **Machine Learning**
   - `data_loader.py`: Loads processed data from the SQLite database
   - `model.py`: Defines and trains an XGBoost model for stock price prediction
   - `eval.py`: Evaluates model performance and performs hyperparameter tuning
   - `run_model.py`: Orchestrates the entire modeling process

4. **Utilities**
   - `database_update.py`: Manages the overall data update process

## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the project:
   - Set up your Alpha Vantage API key in `config/.env`
   - Adjust settings in `config/config.yaml`:
```
   tickers:
  - EX
  - EX
  - EX
```

3. Run the data update process:
   ```
   python src/database_update.py
   ```

4. Train and evaluate the model:
   ```
   python src/run_model.py
   ```

## Current State and Limitations

- The project successfully collects and processes various types of financial data.
- An XGBoost model is implemented for stock price prediction, with basic feature importance analysis and hyperparameter tuning.
- The system is set up to handle multiple stock tickers as defined in the configuration.
- Sentiment analysis using the Claude AI model is integrated, providing an additional layer of analysis.

Please note:
- The project is still in development and may require further refinement and testing.
- Advanced features like automated trading or real-time predictions are not yet implemented.
- The current model's performance and reliability for actual trading decisions have not been thoroughly validated.

## Disclaimer

This project is for educational and research purposes only. It should not be used as financial advice or for making real investment decisions without proper validation and risk assessment.
