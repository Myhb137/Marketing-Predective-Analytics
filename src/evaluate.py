
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix
)


def validate_model(model, X_val, y_val):
    

    probabilities = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y_val, probabilities)
    loss = log_loss(y_val, probabilities)

    print("\nValidation Results")
    print("=" * 40)
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Log Loss:                   {loss:.4f}")

    return ap


def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    print("\nFinal Test Results")
    print("=" * 40)
    print(f"Average Precision (PR-AUC): "
        f"{average_precision_score(y_test, probabilities):.4f}")
    print(f"ROC-AUC:                    "
        f"{roc_auc_score(y_test, probabilities):.4f}")
    print(f"Log Loss:                   "
        f"{log_loss(y_test, probabilities):.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))
