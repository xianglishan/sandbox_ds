from sklearn.metrics import confusion_matrix


def score(y_valid, y_pred):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

    # Function to calculate F1 score
    def calculate_f1(TP, FP, FN):
        # no positive case
        if TP + FP == 0 or TP + FN == 0:
            return 0

        # else we calculate
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if (precision + recall) != 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0

    # Calculating F1 score for original classes
    F1_original = calculate_f1(tp, fp, fn)

    # Calculating F1 score for reversed classes
    F1_reversed = calculate_f1(tn, fn, fp)

    # Calculating mean F1 Score
    return (F1_original + F1_reversed) / 2
