import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

MODELS = {
    'LogisticRegression'    : LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42, max_depth=5),
    'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
}

def scale_data(X_train, X_test):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_all_models(X_train_scaled, Y_train):
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train_scaled, Y_train)
        trained[name] = model
        print(f"✅ Trained: {name}")
    return trained

def train_best_model(X_train_scaled, Y_train):
    # ✅ Best params directly applied — no GridSearch needed
    best_rf = RandomForestClassifier(
        criterion         = 'gini',
        max_depth         = 6,
        min_samples_leaf  = 1,
        min_samples_split = 3,
        n_estimators      = 200,
        random_state      = 42
    )
    best_rf.fit(X_train_scaled, Y_train)
    print("✅ Best Random Forest trained with optimal params!")
    return best_rf