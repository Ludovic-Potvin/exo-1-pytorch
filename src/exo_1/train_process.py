from tqdm import tqdm
from .other_tools import compute_accuracy


def train_model(
    epoch_number,
    train_loader,
    validation_loader,
    model,
    optimizer,
    loss_function,
    device,
):
    model.train()
    for epoch in range(epoch_number):
        mini_batch_counter = 0
        running_loss = 0.0
        running_accuracy = 0.0
        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:
            for inputs, labels in progress_epoch:
                progress_epoch.set_description(f"Epoch {epoch + 1}/{epoch_number}")
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                accuracy = compute_accuracy(labels, outputs)
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
    validation_loss, validation_accuracy = validate_model(
        validation_loader, model, loss_function, device
    )
    return model


def validate_model(validation_loader, model, loss_function, device):
    model.eval()
    mini_batch_counter = 0
    running_loss = 0.0
    running_accuracy = 0.0
    with tqdm(validation_loader, unit=" mini-batch") as progress_validation:
        for inputs, labels in progress_validation:
            progress_validation.set_description(" Validation step")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            accuracy = compute_accuracy(labels, outputs)
            running_loss += loss.item()
            running_accuracy += accuracy
            progress_validation.set_postfix(
                validation_loss=running_loss / (mini_batch_counter + 1),
                validation_accuracy=100.0
                * (running_accuracy / (mini_batch_counter + 1)),
            )
            mini_batch_counter += 1
    return running_loss / mini_batch_counter, 100.0 * (
        running_accuracy / mini_batch_counter
    )
