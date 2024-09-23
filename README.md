# PRISM

Predictive Research Integrating Sentiment and Market indicators

PRISM is a data-driven stock analysis and prediction system that combines traditional technical indicators with advanced sentiment analysis. The project aims to provide a comprehensive framework for collecting, processing, and analyzing financial data to generate insights and predictions about stock market movements.

## Key Components

1. **Data Collection**
   - `market_data_collection.py`: Fetches various financial data including intraday prices, earnings, income statements, and technical indicators
   - `alpha_vantage_sentiment.py`: Gathers news sentiment data (not shown in the provided code snippets)

2. **Data Processing**
   - `database_builder.py`: Consolidates all collected data into a PostgreSQL database
   - `data_loader.py`: Loads and preprocesses data from the database for model training

3. **Machine Learning**
   - `model_training.py`: Defines and trains an N-BEATS model for stock price prediction
   - `eval.py`: Evaluates model performance (not shown in the provided code snippets)

4. **Utilities**
   - `database_update.py`: Manages the overall data update process (not shown in the provided code snippets)

## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the project:
   - Set up your Alpha Vantage API key and database credentials in `config/.env`
   - Adjust settings in `config/config.yaml` to specify the tickers you want to analyze

3. Run the data collection process:
   ```
   python src/data_collection/market_data_collection.py
   ```

4. Build the database:
   ```
   python src/database_builder.py
   ```

5. Train and evaluate the model:
   ```
   python src/models/testing/model_training.py
   ```

## Current State and Limitations

- Collects and processes various types of financial data, including intraday prices, technical indicators, earnings, and economic indicators.
- An N-BEATS model is implemented for stock price prediction, leveraging the Darts library for time series forecasting.
- The system is set up to handle multiple stock tickers as defined in the configuration.
- Data is stored in a PostgreSQL database for efficient retrieval and processing.
- Implements rate limiting to respect API constraints.

Please note:
- The project is still in development and may require further refinement and testing.
- Advanced features like automated trading or real-time predictions are not yet implemented.
- The current model's performance and reliability for actual trading decisions have not been thoroughly validated.

## Disclaimer

This project is for educational and research purposes only. It should not be used as financial advice or for making real investment decisions without proper validation and risk assessment.
