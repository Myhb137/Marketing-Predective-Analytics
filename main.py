
import time
from pathlib import Path
import joblib

from src.preprocessing import (
    load_data,
    select_features,
    splitData
)
from src.train import TrainModels
from src.evaluate import evaluate_model
from src.explainer import explain_model


def main():
    start_time = time.time()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("       CAMPAIGN CONVERSION PREDICTION")
    print("=" * 60)

    data = load_data()

    X, y = select_features(data)
    del data

    X_train, X_val, X_test, y_train, y_val, y_test = splitData(X, y)
    del X, y

    results = TrainModels(
        X_train, y_train,
        X_val, y_val
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, result in results.items():
        print(
            f"{name:10} | "
            f"Average Precision: {result['score']:.4f} | "
            f"Training time: {result['training_time']:.2f}s"
        )

    best_name = max(
        results,
        key=lambda name: results[name]["score"]
    )

    best_result = results[best_name]
    best_model = best_result["model"]

    print("\n" + "=" * 60)
    print("SELECTED MODEL")
    print("=" * 60)
    print(f"Model: {best_name}")
    print(f"Validation Average Precision: {best_result['score']:.4f}")

    model_path = artifacts_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nModel saved to: {model_path}")

    evaluate_model(best_model, X_test, y_test)

    if best_name in ["XGB", "LGBM"]:
        print("\nGenerating model explanations...")

        X_sample = X_val.sample(
            n=min(300, len(X_val)),
            random_state=42
        )

        feature_importance = explain_model(best_model, X_sample)

        print("\nFeature importance (mean absolute SHAP value):")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")
    else:
        print("\nSHAP TreeExplainer skipped: selected model is not tree-based.")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Pipeline completed in {elapsed_time / 60:.2f} minutes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
