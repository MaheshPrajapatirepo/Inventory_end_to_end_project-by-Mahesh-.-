# 📦 Inventory Intelligence App

An end-to-end Machine Learning project that predicts **freight cost** and flags invoices that require **manual approval** — built with scikit-learn and Streamlit.

---

## 📸 App Screenshot

![App Screenshot](/Users/admin/Documents/project/outputs/app_ss1.png)
![App Screenshot](/Users/admin/Documents/project/outputs/app_ss2.png)
![App Screenshot](/Users/admin/Documents/project/outputs/app_ss3.png)
---

## 🎯 Project Goals

| Goal | Type | Model Used |
|------|------|-----------|
| Predict freight cost | Regression | Random Forest / Linear Regression |
| Flag invoices for manual approval | Classification | Random Forest Classifier |

---

## 🗂️ Project Structure

```
project/
│
├── app.py                          ← Streamlit app (both tools in one)
├── main_regression.py              ← Run regression pipeline
├── main_classification.py          ← Run classification pipeline
├── requirements.txt                ← All dependencies
├── inventory.db                    ← SQLite database
│
├── src/
│   ├── db_connect.py               ← Shared SQLite connection
│   ├── export_model.py             ← Save & load .pkl files
│   │
│   ├── regression/
│   │   ├── data_loader.py          ← Load vendor data + feature engineering
│   │   ├── train.py                ← Cross validation + train best model
│   │   └── evaluate.py             ← RMSE, MAE, R2 + plots
│   │
│   └── classification/
│       ├── data_loader.py          ← SQL join + risk label creation
│       ├── train.py                ← Train all models + best model
│       └── evaluate.py             ← Accuracy, ROC-AUC, confusion matrix
│
├── models/                         ← Saved .pkl files (auto-created)
│   ├── freight_model.pkl           ← Regression model
│   ├── classification_model.pkl    ← Classification model
│   └── scaler.pkl                  ← StandardScaler
│
└── outputs/                        ← All saved plots
    ├── regression_actual_vs_predicted.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

## ⚙️ Setup & Installation

### 1. Clone or download the project
```bash
cd /Users/admin/Documents/project
```

### 2. Activate your environment
```bash
conda activate /Users/admin/miniconda/envs/ml_env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🗄️ Database Setup

`inventory.db` is not included in this repo due to file size (405MB).

**Download from Kaggle:**
👉 [inventory.db — Kaggle Dataset](https://www.kaggle.com/datasets/mahixprajapati/inventory-raw-data)

**After downloading:**

0. Place `inventory.db` in the project root:

Inventory_end_to_end_project/
└── inventory.db

---

## 🚀 How to Run

### Step 1 — Train Regression Model
```bash
python main_regression.py
```
Trains freight cost prediction model and saves `models/freight_model.pkl`

### Step 2 — Train Classification Model
```bash
python main_classification.py
```
Trains invoice approval classifier and saves `models/classification_model.pkl` + `models/scaler.pkl`

### Step 3 — Launch Streamlit App
```bash
streamlit run app.py
```
Opens the app at `http://localhost:8501`

---

## 🧠 Models & Results

### Regression — Freight Cost Prediction

| Model | RMSE | MAE | R2 |
|-------|------|-----|----|
| Linear Regression | 114.584663 | 25.410102 | 0.969535 |
| Decision Tree | 150.420292 | 30.611609 | 0.949976 |
| Random Forest | 189.999106 | 36.612372 | 0.917824 |

> 📌 Fill in your results after running `main_regression.py`

---

### Classification — Invoice Approval Flag

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 82.87% | 85.54% |
| Decision Tree | 91.70% | 92.25% |
| Random Forest | 91.16% | 94.53% |


**Best Model Parameters (Random Forest):**
```python
{
    'criterion'        : 'gini',
    'max_depth'        : 6,
    'min_samples_leaf' : 1,
    'min_samples_split': 3,
    'n_estimators'     : 200
}
```

---

## 🔍 Features Used

### Regression Features
| Feature | Description |
|---------|-------------|
| `Quantity` | Number of units ordered |
| `Dollars` | Invoice dollar amount |

### Classification Features
| Feature | Description |
|---------|-------------|
| `PONumber` | Purchase order number |
| `invoice_quantity` | Quantity on invoice |
| `Freight` | Freight cost |
| `days_po_to_invoice` | Days between PO and invoice date |
| `total_item_quantity` | Total quantity across PO |

### Risk Label Logic
An invoice is flagged (`flag_invoice = 1`) if:
- `|invoice_dollars - total_item_dollars| > 5` — dollar mismatch
- `avg_receiving_delay > 10` — late delivery

---

## 🖥️ Streamlit App Features

### Tab 1 — 🚚 Freight Cost Predictor
- Input: Quantity + Invoice Amount
- Output: Estimated freight cost + cost per unit

### Tab 2 — 🔍 Invoice Approval Checker
- Input: PO Number, Invoice Quantity, Freight, Days to Invoice, Total Quantity
- Output: Approval status + risk score percentage

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| pandas | Data manipulation |
| scikit-learn | ML models & pipelines |
| SQLite | Database |
| Streamlit | Web application |
| seaborn / matplotlib | Visualizations |
| scipy | Statistical testing (t-test) |

---

## 👤 Author

**Mahesh .**  
P.Email: **mk5035008@gmail.com**
College Email: **m.mahesh@op.iitg.ac.in**

---

## 📝 Notes

- Models must be trained (`main_regression.py` and `main_classification.py`) before running the app
- Database path is set to `/Users/admin/Documents/project/inventory.db`
- All plots are saved to the `outputs/` folder automatically
