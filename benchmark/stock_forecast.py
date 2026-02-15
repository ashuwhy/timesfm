
import yfinance as yf
import pandas as pd
import timesfm
import torch
import matplotlib.pyplot as plt
import numpy as np
import datetime

def fetch_and_forecast(ticker="AAPL", context_len=512, horizon_len=30):
    print(f"Fetching data for {ticker}...")
    
    # Fetch data - get enough history for context
    # context_len days + some buffer
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=context_len * 2) # *2 to account for weekends/holidays
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if len(data) < context_len:
        print(f"Not enough data fetched. Got {len(data)}, need {context_len}.")
        return

    # Use 'Close' price
    # yfinance might return MultiIndex columns if multiple tickers, but we request one.
    # It usually returns a DataFrame with 'Close', 'Open', etc.
    if isinstance(data.columns, pd.MultiIndex):
        ts_data = data['Close'][ticker].values
    else:
        ts_data = data['Close'].values

    # Take the last context_len points
    context_data = ts_data[-context_len:]
    
    print(f"Data shape: {context_data.shape}")
    print(f"Last observed price: {context_data[-1]:.2f}")

    print("Loading TimesFM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False, 
            force_flip_invariance=True,
        )
    )
    
    print("Forecasting...")
    # TimesFM expects a list of arrays
    point_forecast, _ = model.forecast(
        horizon=horizon_len, 
        inputs=[context_data]
    )
    
    forecast_values = point_forecast[0]
    
    print("Forecast complete.")
    print(f"Forecast for next {horizon_len} days:")
    print(forecast_values)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot context (last 100 points only for clarity)
    plot_context_len = 100
    plt.plot(range(plot_context_len), context_data[-plot_context_len:], label='History (Last 100 days)', color='blue')
    
    # Plot forecast
    # Start forecast plot from the last context point
    forecast_x = range(plot_context_len - 1, plot_context_len + horizon_len)
    # Prepend last context value to forecast to make lines connect
    forecast_plot_data = np.concatenate(([context_data[-1]], forecast_values))
    
    plt.plot(forecast_x, forecast_plot_data, label='TimesFM Forecast', color='red', linestyle='--')
    
    plt.title(f"TimesFM Forecast for {ticker}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    output_file = "stock_forecast.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    fetch_and_forecast()
