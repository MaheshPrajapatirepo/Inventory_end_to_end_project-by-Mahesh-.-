"""
main_classification.py
Run: python main_classification.py
Trains invoice approval flag model and saves classification_model.pkl + scaler.pkl
"""

from sklearn.model_selection import train_test_split

from src.classification.data_loader import load_classification_data, prepare_features
from src.classification.train       import scale_data, train_all_models, train_best_model
from src.classification.evaluate    import evaluate_model, plot_confusion_matrices, plot_feature_importance
from src.export_model               import save_model

def main():
    print("📦 Loading classification data...")
    df   = load_classification_data()
    X, y = prepare_features(df)

    print(f"\n📊 Label distribution:\n{y.value_counts()}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n⚙️  Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("\n🚀 Training all models...")
    trained_models = train_all_models(X_train_scaled, Y_train)

    print("\n📊 Evaluating all models...")
    for name, model in trained_models.items():
        evaluate_model(model, X_test_scaled, Y_test, name)

    plot_confusion_matrices(trained_models, X_test_scaled, Y_test)

    print("\n🏆 Training best model with optimal params...")
    best_rf = train_best_model(X_train_scaled, Y_train)
    evaluate_model(best_rf, X_test_scaled, Y_test, 'Tuned Random Forest')

    print("\n📈 Feature Importance...")
    plot_feature_importance(best_rf, X_train.columns)

    print("\n💾 Saving models...")
    save_model(best_rf, "classification_model.pkl")
    save_model(scaler,  "scaler.pkl")
    print("\n✅ Classification pipeline complete!")

if __name__ == "__main__":
    main()