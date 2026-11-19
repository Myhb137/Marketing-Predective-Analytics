
import shap
import matplotlib.pyplot as plt


def explain_model(model, X_sample):
    print("\nGenerating SHAP explanations...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.show()
    plt.savefig('plot/explainer.png')
    
    
    importance = abs(shap_values.values).mean(axis=0)

    feature_importance = dict(
        sorted(
            zip(X_sample.columns, importance),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return feature_importance
