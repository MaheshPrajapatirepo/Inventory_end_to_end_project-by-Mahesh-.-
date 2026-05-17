import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import os

def evaluate(model_name, Y_true, Y_pred, split="Test"):
    rmse = root_mean_squared_error(Y_true, Y_pred)
    mae  = mean_absolute_error(Y_true, Y_pred)
    r2   = r2_score(Y_true, Y_pred)

    print(f"\n=== {split} Metrics — {model_name} ===")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R2   : {r2:.3f}")

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def plot_actual_vs_predicted(Y_true, Y_pred, model_name):
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_true, Y_pred, alpha=0.5)
    plt.plot(
        [Y_true.min(), Y_true.max()],
        [Y_true.min(), Y_true.max()],
        color='red', linewidth=2, label='Perfect Prediction'
    )
    plt.xlabel('Actual Freight')
    plt.ylabel('Predicted Freight')
    plt.title(f'Actual vs Predicted — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../outputs/regression_actual_vs_predicted.png"))
    plt.show()
