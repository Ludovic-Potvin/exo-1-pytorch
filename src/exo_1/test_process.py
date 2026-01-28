from tqdm import tqdm
from .other_tools import show_compute_model_performances
import pandas as pd
import numpy as np
from .train_process import compute_model_outputs


def test_model(test_loader, model, loss_function, device, classes):
    print()
    print()
    test_scores = pd.DataFrame(
        columns=[
            "Loss",
            "Accuracy",
            "Balanced Accuracy",
            "F1-score",
            "Kappa",
            "Top 2 Accuracy",
            "Top 3 Accuracy",
        ]
    )
    model.eval()
    mini_batch_counter = 0
    running_loss = 0.0
    running_accuracy = 0.0
    all_outputs = []
    all_labels = []
    with tqdm(test_loader, unit=" mini-batch") as progress_testing:
        for inputs, labels in progress_testing:
            progress_testing.set_description("Testing the training model")
            all_labels, all_outputs, loss, accuracy = compute_model_outputs(
                inputs, labels, device, model, all_labels, all_outputs, loss_function
            )
            running_loss += loss.item()
            running_accuracy += accuracy
            progress_testing.set_postfix(
                testing_loss=running_loss / (mini_batch_counter + 1),
                testing_accuracy=100.0 * (running_accuracy / (mini_batch_counter + 1)),
            )
            mini_batch_counter += 1
    test_scores = show_compute_model_performances(
        np.array(all_labels),
        np.array(all_outputs),
        running_loss / mini_batch_counter,
        test_scores,
        classes,
    )
    return test_scores
