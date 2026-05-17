import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

def evaluate_model(model, X_test, Y_test, model_name):
    preds   = model.predict(X_test)
    proba   = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(Y_test, preds) * 100
    roc_auc  = roc_auc_score(Y_test, proba) * 100

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  ROC-AUC  : {roc_auc:.2f}%")
    print(f"\n{classification_report(Y_test, preds, target_names=['Normal', 'Flagged'])}")

    return {'Model': model_name, 'Accuracy': accuracy, 'ROC_AUC': roc_auc}

def plot_confusion_matrices(models_dict, X_test, Y_test):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 5))

    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        preds = model.predict(X_test)
        cm    = confusion_matrix(Y_test, preds)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Flagged'],
            yticklabels=['Normal', 'Flagged'],
            ax=ax
        )
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_names):
    fi = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fi, x='Importance', y='Feature', ax=ax)
    ax.set_title('Feature Importance — Random Forest')
    plt.tight_layout()

    # ✅ Only one savefig — correct file name
    plt.savefig('outputs/feature_importance.png', bbox_inches='tight')
    plt.show()
    plt.close()

    return fi
