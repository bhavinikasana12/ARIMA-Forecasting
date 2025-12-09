import pandas as pd

def preprocess_data(df):
    # Your original missing-value logic
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # aggregate by date (original logic)
    df = df.groupby('date').mean()

    df = df.sort_index()
    df = df.asfreq('D')                 # daily frequency
    df = df.interpolate()               # fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df
