import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    top_k_accuracy_score,
    confusion_matrix,
    classification_report,
)


def get_model_information(model):
    print()
    print("Model Summary")
    model_parameters = [layer for layer in model.parameters()]
    layer_name = [child for child in model.children()]
    column_name = [
        "Layer Name",
        "Number of Trainable Parameters",
        "Number of (non trainable) Parameters",
    ]
    table_information = pd.DataFrame(columns=column_name)
    character_counts = [len(str(string_element)) for string_element in layer_name]
    max_character_number = max(character_counts)
    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    j = 0
    total_trainable_params = 0
    total_params = 0
    for i in layer_name:
        tmp_list = []
        try:
            bias = i.bias is not None
            if bias is True:
                if model_parameters[j].requires_grad is True:
                    trainable_params = (
                        model_parameters[j].numel() + model_parameters[j + 1].numel()
                    )
                    tmp_list.append(str(i))
                    tmp_list.append(trainable_params)
                    tmp_list.append(0)
                    total_trainable_params += trainable_params
                else:
                    params = (
                        model_parameters[j].numel() + model_parameters[j + 1].numel()
                    )
                    tmp_list.append(str(i))
                    tmp_list.append(0)
                    tmp_list.append(params)
                    total_params += params
                j = j + 2
            else:
                if model_parameters[j].requires_grad is True:
                    trainable_params = model_parameters[j].numel()
                    tmp_list.append(str(i))
                    tmp_list.append(trainable_params)
                    tmp_list.append(0)
                    total_trainable_params += trainable_params
                else:
                    params = model_parameters[j].numel()
                    tmp_list.append(str(i))
                    tmp_list.append(0)
                    tmp_list.append(params)
                    total_params += params
                j = j + 1
        except:
            tmp_list.append(str(i))
            tmp_list.append(0)
            tmp_list.append(0)
        table_information.loc[len(table_information)] = tmp_list
    print(table_information.to_string(index=False, justify="center"))
    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    print(f"Total")
    print(f" Trainable Parameters: {total_trainable_params}")
    print(f" Non Trainable Parameters: {total_params}")
    print("=" * (max_character_number + 2 + 30 + 2 + 36))
    print()


def compute_accuracy(outputs, labels):
    outputs = outputs.argmax(dim=1)
    labels = labels.argmax(dim=1)

    correct = (outputs == labels).sum().float()
    total_size = float(labels.size(0))

    accuracy = correct / total_size

    return accuracy.item()

def vec_to_int(y_true, y_predicted):
    y_true = np.argmax(y_true, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)
    return y_true, y_predicted

def model_performances(y_true, y_predicted, loss, my_score_df):
    scores = []
    y_int_true, y_int_predicted = vec_to_int(y_true, y_predicted)
    scores.extend([loss])
    scores.extend([accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([balanced_accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([f1_score(y_int_true, y_int_predicted, average="micro")])
    scores.extend([cohen_kappa_score(y_int_true, y_int_predicted)])
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=2)])
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=3)])
    my_score_df.loc[len(my_score_df)] = scores
    return my_score_df
def show_compute_model_performances(y_true, y_predicted, loss, my_score_df, classes):
    scores = []
    y_int_true, y_int_predicted = vec_to_int(y_true, y_predicted)
    scores.extend([loss])
    accuracy = accuracy_score(y_int_true, y_int_predicted)
    print("Accuracy: " + str(accuracy))
    scores.extend([accuracy])
    balanced_accuracy = balanced_accuracy_score(y_int_true, y_int_predicted)
    print("Balanced Accuracy: " + str(balanced_accuracy))
    scores.extend([balanced_accuracy])
    f1 = f1_score(y_int_true, y_int_predicted, average="micro")
    print("F1-score: " + str(f1))
    scores.extend([f1])
    kappa = cohen_kappa_score(y_int_true, y_int_predicted)
    print("Kappa: " + str(kappa))
    scores.extend([kappa])
    if y_true.shape[1] <= 10:
        print(confusion_matrix(y_int_true, y_int_predicted))
    print(classification_report(y_int_true, y_int_predicted, target_names=classes))
    top_2 = top_k_accuracy_score(y_int_true, y_predicted, k=2)
    print("Top 2 Accuracy: " + str(top_2))
    scores.extend([top_2])
    top_3 = top_k_accuracy_score(y_int_true, y_predicted, k=3)
    print("Top 3 Accuracy: " + str(top_3))
    scores.extend([top_3])
    my_score_df.loc[len(my_score_df)] = scores
    return my_score_df
