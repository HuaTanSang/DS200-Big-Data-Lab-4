from sklearn.metrics import f1_score, accuracy_score

def compute_scores(predicted_label: list, true_label: list) -> dict:
    metrics = {
        "f1": f1_score,
        "accuracy": accuracy_score,
    }

    scores = {}
    for name, fn in metrics.items():
        if fn is accuracy_score:
            scores[name] = fn(true_label, predicted_label)
        else:
            scores[name] = fn(true_label, predicted_label, average='weighted')
    return scores
