from src.db_connect import run_query

def load_vendor_data():
    df = run_query("SELECT * FROM vendor_invoice")
    return df

def engineer_features(df):
    df = df.copy()
    df['Freight_per_unit'] = df['Freight'] / df['Quantity']
    return df

def get_features_and_target(df):
    X = df[['Quantity', 'Dollars']]
    y = df['Freight']
    return X, y
