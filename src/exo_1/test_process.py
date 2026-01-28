from tqdm import tqdm
from .other_tools import compute_accuracy


def test_model(test_loader, model, loss_function, device):
    print()
    print()
    model.eval()
    mini_batch_counter = 0
    running_loss = 0.0
    running_accuracy = 0.0
    with tqdm(test_loader, unit=" mini-batch") as progress_testing:
        for inputs, labels in progress_testing:
            progress_testing.set_description("Testing the training 10/11 model")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            accuracy = compute_accuracy(labels, outputs)
            running_loss += loss.item()
            running_accuracy += accuracy
            progress_testing.set_postfix(
                testing_loss=running_loss / (mini_batch_counter + 1),
                testing_accuracy=100.0 * (running_accuracy / (mini_batch_counter + 1)),
            )
            mini_batch_counter += 1
    return running_loss / mini_batch_counter, 100.0 * (
        running_accuracy / mini_batch_counter
    )
