
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
import time


def TrainModels(X_train, y_train, X_val, y_val):

    models = {
        
        "XGB": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            device = 'cuda'

        ),
        "LGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1, 
            device_type='cpu'
            
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        probabilities = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, probabilities)

        results[name] = {
            "model": model,
            "score": score,
            "training_time": training_time
        }

        print(f"Average Precision: {score:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

    # Select the best model AFTER evaluating all models
    best_name = max(
        results,
        key=lambda name: results[name]["score"]
    )

    print(f"\nSelected model: {best_name}")
    print(f"Validation score: {results[best_name]['score']:.4f}")

    return results


if __name__ == "__main__":
    TrainModels()