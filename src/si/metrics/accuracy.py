def accuracy(y_true, y_pred):
    """
    Calculates the accuracy by comparing the true labels with the predicted labels.
    """
    if len(y_pred) == len(y_true):
        total_label = len(y_true)
    else:
        return None

    well_labels = 0
    for i in range(total_label):
        if y_true[i] == y_pred[i]:
            well_labels += 1

    return well_labels / total_label