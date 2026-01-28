import pandas as pd
import numpy as np
from tqdm import tqdm

from .other_tools import compute_accuracy, model_performances
from .graphs import plot_score_graphs


def train_model(
    epoch_number,
    train_loader,
    validation_loader,
    model,
    optimizer,
    loss_function,
    device,
):
    training_epoch_scores = pd.DataFrame(
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
    validation_epoch_scores = pd.DataFrame(
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
    model.train()
    for epoch in range(epoch_number):
        mini_batch_counter = 0
        running_loss = 0.0
        running_accuracy = 0.0
        all_outputs = []
        all_labels = []
        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:
            for inputs, labels in progress_epoch:
                progress_epoch.set_description(f"Epoch {epoch + 1}/{epoch_number}")
                all_labels, all_outputs, loss, accuracy = compute_model_outputs(
                    inputs,
                    labels,
                    device,
                    model,
                    all_labels,
                    all_outputs,
                    loss_function,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy
                progress_epoch.set_postfix(
                    train_loss=running_loss / (mini_batch_counter + 1),
                    train_accuracy=100.0
                    * (running_accuracy / (mini_batch_counter + 1)),
                )
                mini_batch_counter += 1
        training_epoch_scores = model_performances(
            np.array(all_labels),
            np.array(all_outputs),
            running_loss / mini_batch_counter,
            training_epoch_scores,
        )
        validation_epoch_scores = validate_model(
            validation_loader, model, loss_function, device, validation_epoch_scores
        )
    plot_score_graphs(training_epoch_scores, validation_epoch_scores)
    return model


def validate_model(
    validation_loader, model, loss_function, device, validation_epoch_scores
):
    model.eval()
    mini_batch_counter = 0
    running_loss = 0.0
    running_accuracy = 0.0
    all_labels = []
    all_outputs = []
    with tqdm(validation_loader, unit=" mini-batch") as progress_validation:
        for inputs, labels in progress_validation:
            progress_validation.set_description(" Validation step")
            all_labels, all_outputs, loss, accuracy = compute_model_outputs(
                inputs, labels, device, model, all_labels, all_outputs, loss_function
            )
            running_loss += loss.item()
            running_accuracy += accuracy
            progress_validation.set_postfix(
                validation_loss=running_loss / (mini_batch_counter + 1),
                validation_accuracy=100.0
                * (running_accuracy / (mini_batch_counter + 1)),
            )
            mini_batch_counter += 1
    validation_epoch_scores = model_performances(
        np.array(all_labels),
        np.array(all_outputs),
        running_loss / mini_batch_counter,
        validation_epoch_scores,
    )
    return validation_epoch_scores


def compute_model_outputs(
    inputs, labels, device, model, all_labels, all_outputs, loss_function
):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    all_labels.extend(np.array(labels.cpu()))
    all_outputs.extend(np.array(outputs.detach().cpu()))
    loss = loss_function(outputs, labels)
    accuracy = compute_accuracy(labels, outputs)
    return all_labels, all_outputs, loss, accuracy
