import pandas as pd
from preprocessing import preprocess_data
from stationarity import plot_series, check_stationarity
from arima_model import run_arima

def main():
    # load your input CSV
    df = pd.read_csv("data/sample_input.csv")

    # preprocess
    df = preprocess_data(df)

    # stationarity checks
    plot_series(df)
    check_stationarity(df)

    # forecasting
    run_arima(df)

if __name__ == "__main__":
    main()
