import os
import pickle
import click
import numpy as np

import wandb
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score, median_absolute_error




import pandas as pd

def load_csv(filename):
    """
    Load a CSV file into a DataFrame.
    
    Args:
        filename (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filename)



@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--data_artifact",
    help="Address of the Weights & Biases artifact holding the preprocessed data",
)
@click.option("--random_state", default=0, help="Random state")
def run_train(
    wandb_project: str,
    wandb_entity: str,
    data_artifact: str,
    random_state: int,
):
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name="xgb_regressor",
        job_type="train",
        config={"random_state": random_state},
    )

    # Fetch the preprocessed dataset from artifacts
    artifact = wandb.use_artifact(data_artifact, type="preprocessed_dataset")
    artifact_dir = artifact.download()

# Load the CSV files into DataFrames
    train_data = load_csv(os.path.join(artifact_dir, "train.csv"))
    val_data = load_csv(os.path.join(artifact_dir, "val.csv"))

    # Separate the features (X) and the target variable (y)
    X_train = train_data.drop("daily_return", axis=1)
    y_train = train_data["daily_return"]

    X_val = val_data.drop("daily_return", axis=1)
    y_val = val_data["daily_return"]


    model = xgb.XGBRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Log Metrics to Weights & Biases
    wandb.log({
    "MAE (Train)": mean_absolute_error(y_train, y_pred_train),
    "MAE (Validation)": mean_absolute_error(y_val, y_pred_val),
    "MSE (Train)": mean_squared_error(y_train, y_pred_train),
    "MSE (Validation)": mean_squared_error(y_val, y_pred_val),
    "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    "RMSE (Validation)": np.sqrt(mean_squared_error(y_val, y_pred_val)),
    "R-squared (Train)": r2_score(y_train, y_pred_train),
    "R-squared (Validation)": r2_score(y_val, y_pred_val),
    "Explained Variance (Train)": explained_variance_score(y_train, y_pred_train),
    "Explained Variance (Validation)": explained_variance_score(y_val, y_pred_val),
    "MedAE (Train)": median_absolute_error(y_train, y_pred_train),
    "MedAE (Validation)": median_absolute_error(y_val, y_pred_val),
})


    # Save your model
    with open("xgb_regressor.pkl", "wb") as f:
        pickle.dump(model, f)

    
    # Log your model as a versioned file to Weights & Biases Artifact
    artf = wandb.Artifact(f"Stock-xbg-model", type="model")
    artf.add_file("xgb_regressor.pkl")
    wandb.log_artifact(artf)



if __name__ == "__main__":
    run_train()
