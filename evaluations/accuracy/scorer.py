from sklearn.metrics import accuracy_score


def scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
