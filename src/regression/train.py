import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

SCORING = {
    'rmse': 'neg_root_mean_squared_error',
    'mae' : 'neg_mean_absolute_error',
    'r2'  : 'r2'
}

MODELS = {
    'LinearRegression'     : LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
}

def build_pipeline(model):
    return Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model',  model)
    ])

def run_cross_validation(X_train, Y_train, cv=5):
    results = []
    for name, model in MODELS.items():
        pipe       = build_pipeline(model)
        cv_results = cross_validate(pipe, X_train, Y_train, cv=cv, scoring=SCORING)
        results.append({
            'Model'  : name,
            'CV RMSE': -cv_results['test_rmse'].mean(),
            'CV MAE' : -cv_results['test_mae'].mean(),
            'CV R2'  :  cv_results['test_r2'].mean(),
        })
    df = pd.DataFrame(results).sort_values(by='CV RMSE').reset_index(drop=True)
    print("\n=== Regression CV Results ===")
    print(df.to_string(index=False))
    return df

def train_best_model(X_train, Y_train, model_name):
    model = MODELS[model_name]
    pipe  = build_pipeline(model)
    pipe.fit(X_train, Y_train)
    print(f"\n✅ Trained: {model_name}")
    return pipe
