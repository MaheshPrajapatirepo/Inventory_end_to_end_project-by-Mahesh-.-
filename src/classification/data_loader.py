from src.db_connect import run_query

def load_classification_data():
    df = run_query("""
        WITH purchase_agg AS (
            SELECT
                p.PONumber,
                COUNT(DISTINCT p.Brand)                             AS total_brands,
                SUM(p.Quantity)                                     AS total_item_quantity,
                SUM(p.Dollars)                                      AS total_item_dollars,
                AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
            FROM purchases p
            GROUP BY p.PONumber
        )
        SELECT
            vi.PONumber,
            vi.Quantity                                             AS invoice_quantity,
            vi.Dollars                                              AS invoice_dollars,
            vi.Freight,
            (julianday(vi.InvoiceDate) - julianday(vi.PODate))      AS days_po_to_invoice,
            (julianday(vi.PayDate)    - julianday(vi.InvoiceDate))  AS days_to_pay,
            pa.total_brands,
            pa.total_item_quantity,
            pa.total_item_dollars,
            pa.avg_receiving_delay
        FROM vendor_invoice vi
        LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber
    """)
    return df

def create_risk_labels(row):
    if abs(row['invoice_dollars'] - row['total_item_dollars']) > 5:
        return 1
    if row['avg_receiving_delay'] > 10:
        return 1
    return 0

def prepare_features(df):
    df = df.copy()
    df['flag_invoice'] = df.apply(create_risk_labels, axis=1)

    # Features — leaking columns removed
    X = df[['PONumber',
            'invoice_quantity',
            'Freight',
            'days_po_to_invoice',
            'total_item_quantity']]

    y = df['flag_invoice']
    return X, y
