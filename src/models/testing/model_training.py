import os
import logging
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
import torch
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer  # <-- Added import
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError

# Import functions from data_loader.py
from data_loader import (
    load_environment_variables,
    create_db_engine,
    fetch_and_process_data,
)

# Import custom callbacks
from callbacks import LossHistoryCallback  # Ensure correct import path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def parse_timestamp(ts):
    """
    Attempts to parse a timestamp string into a pandas Timestamp object.
    Supports multiple datetime formats:
    - "%Y-%m-%d %H:%M:%S"
    - "%Y-%m-%d %H:%M"
    - "%Y-%m-%d"

    Parameters:
        ts (str): The timestamp string to parse.

    Returns:
        pd.Timestamp or pd.NaT: The parsed Timestamp or Not-a-Time if parsing fails.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return pd.to_datetime(ts, format=fmt)
        except ValueError:
            continue
    # If all formats fail, return NaT
    return pd.NaT

class CustomNBEATSModel(NBEATSModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mape_metric = MeanAbsolutePercentageError()
        self.mae_metric = MeanAbsoluteError()
        self.mse_metric = MeanSquaredError()

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = super().validation_step(batch, batch_idx)
        val_loss = output['loss'] if isinstance(output, dict) else output
        
        # Calculate predictions
        val_pred = self(batch)
        val_target = batch['y']
        
        # Log metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mape', self.mape_metric(val_pred, val_target), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mae', self.mae_metric(val_pred, val_target), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('mse', self.mse_metric(val_pred, val_target), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        rmse = torch.sqrt(self.mse_metric(val_pred, val_target))
        self.log('rmse', rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return val_loss

def main():
    # Calculate the current directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dotenv_path = os.path.join(project_root, 'config', '.env')

    # Load environment variables
    load_environment_variables(dotenv_path)

    try:
        # Create database engine
        engine = create_db_engine()
        logging.info(f"Engine type in main: {type(engine)}")  # Added logging

        # Fetch and process all data
        processed_data = fetch_and_process_data(engine)

        # Access the preprocessed intraday data
        df_intraday = processed_data.get('intraday')
        if df_intraday is None:
            logging.error("Intraday data is not available. Exiting.")
            return

        # Verify the frequency
        logging.info(f"Intraday data frequency before setting: {df_intraday.index.freq}")

        # Ensure the frequency is set to hourly
        if df_intraday.index.freq is None:
            df_intraday = df_intraday.asfreq('h')
            logging.info("Intraday data frequency set to hourly ('h').")

        # Create TimeSeries object
        try:
            series = TimeSeries.from_dataframe(
                df_intraday,
                value_cols='close',
                freq='h',  # Ensure the frequency is set to hourly
                fill_missing_dates=True
            )
            logging.info("TimeSeries object created successfully.")
        except ValueError as ve:
            logging.error(f"Error creating TimeSeries: {ve}")
            return

        # Split data into training and validation sets
        train, val = series.split_before(0.8)
        logging.info(f"Training set size: {len(train)}")
        logging.info(f"Validation set size: {len(val)}")

        # Determine the device to use for training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")  # Optional: Log the device being used

        # Initialize the custom callback
        loss_history = LossHistoryCallback()

        # Define callbacks for the main Trainer, including the custom callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            filename='best_model-{epoch:02d}-{val_loss:.2f}'
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            mode='min'
        )

        # Initialize Trainer with corrected precision
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision='64-true', 
            max_epochs=100,
            enable_progress_bar=True,
            logger=True,
            callbacks=[checkpoint_callback, early_stopping_callback, loss_history]
        )
        
        # Instantiate the custom model
        model = CustomNBEATSModel(
            input_chunk_length=14, 
            output_chunk_length=7, 
            n_epochs=100,
        )

        # Define independent learning rates
        learning_rates = [0.001, 0.002]

        # Define parameter grid as a dictionary
        parameter_grid = {
            "input_chunk_length": [14, 30],
            "output_chunk_length": [7, 14],
            "num_stacks": [2, 3],
            "num_blocks": [1, 2],
            "batch_size": [256, 512],
            "random_state": [42],
            "optimizer_kwargs": [{"lr": lr} for lr in learning_rates],
            "pl_trainer_kwargs": [{
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "precision": '64-true',
                "enable_progress_bar": True,
                "logger": True,
            }],
        }

        # Perform grid search with updated model
        logging.info("Starting hyperparameter tuning...")
        from darts.metrics import mape

        # Unpack the grid search result
        best_model, best_parameters, best_score = NBEATSModel.gridsearch(
            parameters=parameter_grid,
            series=train,
            val_series=val,
            metric=mape,  # Use the callable metric function
            reduction=np.mean,
            verbose=True,
            n_jobs=1
        )

        logging.info(f"Best parameters found: {best_parameters}")
        logging.info(f"Best score (MAPE): {best_score}")
        logging.info("Hyperparameter tuning completed.")
        
        # Train the best model (since gridsearch returns an untrained model)
        best_model.fit(train)
        
        # Use the best model for forecasting
        forecast = best_model.predict(n=7)

        # Evaluate the best model
        from darts.metrics import mae, mape, rmse
        
        evaluation_mae = mae(val, forecast)
        evaluation_mape = mape(val, forecast)
        evaluation_rmse = rmse(val, forecast)
        
        logging.info(f"Best Model Evaluation MAE: {evaluation_mae}")
        logging.info(f"Best Model Evaluation MAPE: {evaluation_mape}")
        logging.info(f"Best Model Evaluation RMSE: {evaluation_rmse}")
        
        # After training, access the stored losses
        epochs = range(1, len(loss_history.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss_history.train_losses, label='Training Loss')
        plt.plot(epochs, loss_history.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Perform backtesting
        logging.info("Starting backtesting...")
        backtest_forecasts = best_model.historical_forecasts(
            series=series,
            start=0.8,
            forecast_horizon=7,
            stride=1,
            retrain=False,
            verbose=True
        )
        
        # Evaluate backtest performance
        backtest_mae = mae(series.drop_before(backtest_forecasts.start_time()), backtest_forecasts)
        backtest_mape = mape(series.drop_before(backtest_forecasts.start_time()), backtest_forecasts)
        backtest_rmse = rmse(series.drop_before(backtest_forecasts.start_time()), backtest_forecasts)
        
        logging.info(f"Backtest MAE: {backtest_mae}")
        logging.info(f"Backtest MAPE: {backtest_mape}")
        logging.info(f"Backtest RMSE: {backtest_rmse}")
        
        # Plot backtest forecasts
        plt.figure()
        series.plot(label='Actual')
        backtest_forecasts.plot(label='Backtest Forecast')
        plt.legend()
        plt.title('Backtesting Results')
        plt.show()

        # Calculate residuals
        residuals = (val - forecast).pd_series()
        
        # Plot residuals over time
        plt.figure()
        residuals.plot()
        plt.title('Residuals Over Time')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.show()
        
        # Plot histogram of residuals
        plt.figure()
        residuals.hist(bins=30)
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.show()

        # Save the model
        best_model.save_model("nbeats_model.pth.tar")
        logging.info("Model saved successfully.")

        # Plot Actual vs Forecasted Values
        plt.figure(figsize=(12, 6))
        series.plot(label='Actual')
        forecast.plot(label='Forecast', lw=2)
        plt.title('Actual vs Forecasted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # After training, you can access all logged metrics
        for metric_name, metric_value in best_model.trainer.callback_metrics.items():
            print(f"Final {metric_name}: {metric_value}")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()