import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def plot_series(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['close'])
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()

def check_stationarity(df):
    tickers = df['ticker'].unique()
    adf_results = {}

    for t in tickers:
        print(f"\nProcessing ticker: {t}")

        temp = df[df['ticker'] == t].sort_index()
        series = temp['close'].dropna()

        # original ADF logic
        adf_result = adfuller(series)
        p_value = adf_result[1]

        adf_results[t] = p_value

        if p_value < 0.05:
            print(f"  ✓ Series is stationary (p={p_value:.4f})")
        else:
            print(f"  ✗ Series is NOT stationary (p={p_value:.4f})")

    return adf_results
