"""
main_regression.py
Run: python main_regression.py
Trains freight cost prediction model and saves freight_model.pkl
"""

from sklearn.model_selection import train_test_split
from src.regression.data_loader import load_vendor_data, engineer_features, get_features_and_target
from src.regression.train       import run_cross_validation, train_best_model
from src.regression.evaluate    import evaluate, plot_actual_vs_predicted
from src.export_model           import save_model

def main():
    print("📦 Loading vendor data...")
    df = load_vendor_data()
    df = engineer_features(df)

    X, y = get_features_and_target(df)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🔁 Cross Validation...")
    cv_df      = run_cross_validation(X_train, Y_train)
    best_model = cv_df.iloc[0]['Model']
    print(f"\n🏆 Best Model: {best_model}")

    print("\n🚀 Training best model...")
    pipeline = train_best_model(X_train, Y_train, best_model)

    print("\n📊 Evaluating...")
    preds = pipeline.predict(X_test)
    evaluate(best_model, Y_test, preds)
    plot_actual_vs_predicted(Y_test, preds, best_model)

    print("\n💾 Saving model...")
    save_model(pipeline, "freight_model.pkl")
    print("\n✅ Regression pipeline complete!")

if __name__ == "__main__":
    main()
