import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_score_graphs(training_epoch_scores, validation_epoch_scores):
    scores_to_plot = [
        "Loss",
        "Accuracy",
        "Balanced Accuracy",
        "F1-score",
        "Kappa",
        "Top 2 Accuracy",
        "Top 3 Accuracy",
    ]
    for score_type in scores_to_plot:
        the_df = create_score_df(
            training_epoch_scores, validation_epoch_scores, score_type
        )
        fig = px.line(the_df, x="Epochs", y=score_type, color="Stage")
        fig.show()

def create_score_df(training_epoch_scores, validation_epoch_scores, score_type):
    train_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    epochs = np.arange(1, training_epoch_scores.shape[0] + 1, 1)
    stage = ["Train"] * training_epoch_scores.shape[0]
    train_df["Epochs"] = epochs
    train_df["Stage"] = stage
    train_df[score_type] = training_epoch_scores[score_type]
    validation_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    stage = ["Validation"] * training_epoch_scores.shape[0]
    validation_df["Epochs"] = epochs
    validation_df["Stage"] = stage
    validation_df[score_type] = validation_epoch_scores[score_type]
    score_df = pd.concat([train_df, validation_df])
    return score_df
