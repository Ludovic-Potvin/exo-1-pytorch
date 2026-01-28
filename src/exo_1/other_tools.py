import pandas as pd


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
    accuracy = 0

    labels = labels.argmax(dim=1)
    outputs = outputs.argmax(dim=1)

    corrects = outputs == labels
    accuracy += corrects.sum().float() / float(labels.size(0))

    return accuracy.item()

