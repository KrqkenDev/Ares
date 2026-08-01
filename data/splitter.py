import pandas as pd


def split_data(historical_data: dict[str, pd.DataFrame], training_ratio: float=0.7, training: bool = True) -> dict[str, pd.DataFrame]:
    """
    Splits historical market data into training and testing datasets.

    Args:
        historical_data: Dictionary mapping ticker symbols to DataFrames.
        train_ratio: Percentage of data used for training.
        training: True returns the training set, False returns the testing set.

    Returns:
        Dictionary containing the requested portion of each DataFrame.
    """
    if not 0 < training_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_dataset = {}

    for ticker, dataframe in historical_data.items():
        split_index = int(len(dataframe)* training_ratio)

        if training:
            split_dataset[ticker] = dataframe.iloc[:split_index].copy()
        else:
            split_dataset[ticker] = dataframe.iloc[split_index:].copy()

    return split_dataset