import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

def run_arima(df):
    tickers = df['ticker'].unique()
    forecasts = {}
    models = {}

    for t in tickers:
        print(f"\nProcessing ticker: {t}")

        temp = df[df['ticker'] == t].sort_index()
        series = temp['close'].dropna()
        series.index = temp.index

        # Step 1: fit ARIMA
        model = auto_arima(series,
                           seasonal=False,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True)

        models[t] = model

        # Step 2: forecast next 10 days
        future_values = model.predict(n_periods=10)

        last_date = series.index[-1]
        forecast_index = pd.date_range(last_date + pd.Timedelta(days=1),
                                       periods=10, freq='D')

        forecast_series = pd.Series(future_values, index=forecast_index)
        forecasts[t] = forecast_series

        print(f"\nForecast for ticker {t}:\n", forecast_series)

        # plotting (unchanged)
        plt.figure(figsize=(10, 6))
        plt.plot(series, label="Historical")
        plt.plot(forecast_series, label="Forecast", linestyle='--')
        plt.title(f'Ticker {t} Forecast')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.show()

    return forecasts, models
