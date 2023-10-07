import os
import click
import pickle
import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)

def load_data() -> pd.DataFrame:
    '''
    Load CSV data as a dataframe

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    '''
    # List of file names
    file_names = [
        "data/AAPL_2023-05-01_2023-09-30.csv",
        "data/AMZN_2023-05-01_2023-09-30.csv",
        "data/GOOGL_2023-05-01_2023-09-30.csv",
        "data/IBM_2023-05-01_2023-09-30.csv",
        "data/JPM_2023-05-01_2023-09-30.csv",
        "data/META_2023-05-01_2023-09-30.csv",
        "data/MSFT_2023-05-01_2023-09-30.csv",
        "data/NFLX_2023-05-01_2023-09-30.csv",
        "data/NVDA_2023-05-01_2023-09-30.csv",
        "data/TSLA_2023-05-01_2023-09-30.csv",
        "data/V_2023-05-01_2023-09-30.csv"
    ]

    # Create an empty list to store the DataFrames
    dfs = []

    # Loop through the file names and read each CSV file into a DataFrame
    for file_name in file_names:
        df = pd.read_csv(file_name)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract the symbol from the file name (assuming the format "symbol_start_date_end_date.csv")
        symbol = file_name.split("_")[0].split("/")[-1]
        
        # Add the "symbol" column
        df["symbol"] = symbol
        
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all the DataFrames in the list into one DataFrame
    df = pd.concat(dfs, ignore_index=True)
    
    return df

def create_features(df, rsi_period=10, macd_short_window=12, 
                    macd_long_window=20, macd_signal_window=9, 
                    stochastic_k_period=14, stochastic_d_period=3, 
                    bollinger_window=10, num_std_dev=2):
    # Sort df by timestamp
    df = df.sort_values(by='timestamp')

    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change() * 100

    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff(1)
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    average_up = up.rolling(window=rsi_period, min_periods=1).mean()
    average_down = down.rolling(window=rsi_period, min_periods=1).mean()
    rs = average_up / average_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculate MACD (Moving Average Convergence Divergence)
    short_ema = df['close'].ewm(span=macd_short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=macd_long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=macd_signal_window, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal

    # Calculate Stochastic Oscillator
    lowest_low = df['low'].rolling(window=stochastic_k_period, min_periods=1).min()
    highest_high = df['high'].rolling(window=stochastic_k_period, min_periods=1).max()
    k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=stochastic_d_period, min_periods=1).mean()
    df['stochastic_k'] = k
    df['stochastic_d'] = d

    # Calculate Bollinger Bands
    middle_band = df['close'].rolling(window=bollinger_window).mean()
    std_dev = df['close'].rolling(window=bollinger_window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    df['bollinger_upper'] = upper_band
    df['bollinger_middle'] = middle_band
    df['bollinger_lower'] = lower_band


    # drop the null values
    df = df.dropna()

    return df

def data_splitting(df, train_ratio=0.7, val_ratio=0.15, seed=None):
    # Data splitting and sort by timestamp
    df = df.sort_values(by='timestamp')

    # Determine the split indices based on proportions
    total_length = len(df)
    train_end_index = int(total_length * train_ratio)
    val_end_index = train_end_index + int(total_length * val_ratio)

    # Split the data
    train_data = df.iloc[:val_end_index]  # Include 'daily_return' in training data

    # Check if 'daily_return' is in train_data
    if 'daily_return' in train_data.columns:
        print("The 'daily_return' column is in train_data.")
    else:
        print("The 'daily_return' column is NOT in train_data.")

    val_data = df.iloc[train_end_index:val_end_index]
    test_data = df.iloc[val_end_index:]

    # Exclude the 'timestamp' column
    X_train = train_data.drop(['daily_return', 'timestamp'], axis=1)
    y_train = train_data['daily_return']

    X_val = val_data.drop(['daily_return', 'timestamp'], axis=1)
    y_val = val_data['daily_return']

    X_test = test_data.drop(['daily_return', 'timestamp'], axis=1)
    y_test = test_data['daily_return']

    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot(X_train, X_val, X_test):
    # Categorical transformation
    # Apply one-hot encoding to 'symbol' column in each dataset
    X_train = pd.concat([X_train, pd.get_dummies(X_train['symbol'], prefix='symbol', drop_first=False)], axis=1)
    X_val = pd.concat([X_val, pd.get_dummies(X_val['symbol'], prefix='symbol', drop_first=False)], axis=1)
    X_test = pd.concat([X_test, pd.get_dummies(X_test['symbol'], prefix='symbol', drop_first=False)], axis=1)

        # Drop the original 'symbol' column
    X_train.drop('symbol', axis=1, inplace=True)
    X_val.drop('symbol', axis=1, inplace=True)
    X_test.drop('symbol', axis=1, inplace=True)

    return X_train, X_val, X_test


def normalization(X_train, X_val, X_test):
    # Numerical scaling
    scaler = MinMaxScaler()
    num_features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 
                    'macd', 'macd_signal', 'stochastic_k', 'stochastic_d', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower']

    # Fit and transform on the training set
    X_train[num_features] = scaler.fit_transform(X_train[num_features])

    # Transform the validation and test sets using the scaler fitted on the training data
    X_val[num_features] = scaler.transform(X_val[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    return X_train, X_val, X_test

@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option("--dest_path", help="Location where the resulting files will be saved")
def run_data_prep(wandb_project: str, wandb_entity: str, dest_path: str):
    # Initialize a Weights & Biases run
    wandb.init(project=wandb_project, entity=wandb_entity, job_type="preprocess")

    # Load the data
    df = load_data()  # Call the load_data function to load the data

    # Create features
    df = create_features(df)  # Call the create_features function to add features

    # Split  the data
    X_train, y_train, X_val, y_val, X_test, y_test = data_splitting(df)

    # Apply one-hot encoding to 'symbol' column
    X_train, X_val, X_test = one_hot(X_train, X_val, X_test)

    # normalize the data
    X_train, X_val, X_test = normalization(X_train, X_val, X_test)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save scaler and datasets
    scaler = MinMaxScaler()  # Create a scaler
    X_train, X_val, X_test = normalization(X_train, X_val, X_test) 
    dump_pickle(scaler, os.path.join(dest_path, "scaler.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    artifact = wandb.Artifact("Stock_data", type="preprocessed_dataset")
    artifact.add_dir(dest_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    run_data_prep()


