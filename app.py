"""
app.py — Streamlit Application
Run: streamlit run app.py

Two tools in one app:
  1. Freight Cost Predictor   (Regression)
  2. Invoice Approval Checker (Classification)
"""

import streamlit as st
import pickle
import numpy as np
import os

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Intelligence",
    page_icon="📦",
    layout="centered"
)

# ── Load Models ───────────────────────────────────────────────────
def load_pkl(filename):
    path = os.path.join("models", filename)
    with open(path, "rb") as f:
        return pickle.load(f)

freight_model         = load_pkl("freight_model.pkl")
classification_model  = load_pkl("classification_model.pkl")
scaler                = load_pkl("scaler.pkl")

# ── Header ────────────────────────────────────────────────────────
st.title("📦 Inventory Intelligence App")
st.markdown("Predict **freight cost** and check if an invoice needs **manual approval**.")
st.divider()

# ══════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🚚 Freight Cost Predictor", "🔍 Invoice Approval Checker"])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — FREIGHT COST PREDICTOR (Regression)
# ──────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("🚚 Predict Freight Cost")
    st.markdown("Enter order details to estimate the freight cost.")

    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input(
            "Quantity (units)",
            min_value=1,
            value=100,
            step=1
        )

    with col2:
        dollars = st.number_input(
            "Invoice Amount ($)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

    if st.button("Predict Freight Cost", use_container_width=True):
        input_data = np.array([[quantity, dollars]])
        prediction = freight_model.predict(input_data)[0]

        st.success(f"### 💰 Estimated Freight Cost: **${prediction:.2f}**")

        st.info(
            f"📦 For **{quantity} units** worth **${dollars:.2f}**, "
            f"the estimated freight cost is **${prediction:.2f}** "
            f"(**${prediction/quantity:.4f}** per unit)."
        )

# ──────────────────────────────────────────────────────────────────
# TAB 2 — INVOICE APPROVAL CHECKER (Classification)
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Invoice Manual Approval Checker")
    st.markdown("Enter invoice details to check if it needs manual review.")

    col1, col2 = st.columns(2)

    with col1:
        po_number = st.number_input(
            "PO Number",
            min_value=1,
            value=1001,
            step=1
        )
        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=1,
            value=50,
            step=1
        )
        freight = st.number_input(
            "Freight Cost ($)",
            min_value=0.0,
            value=200.0,
            step=10.0
        )

    with col2:
        days_po_to_invoice = st.number_input(
            "Days from PO to Invoice",
            min_value=0,
            value=5,
            step=1
        )
        total_item_quantity = st.number_input(
            "Total Item Quantity (PO)",
            min_value=1,
            value=200,
            step=1
        )

    if st.button("Check Approval Status", use_container_width=True):
        input_data = np.array([[
            po_number,
            invoice_quantity,
            freight,
            days_po_to_invoice,
            total_item_quantity
        ]])

        input_scaled  = scaler.transform(input_data)
        prediction    = classification_model.predict(input_scaled)[0]
        proba         = classification_model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            st.error(f"### 🚨 Manual Approval Required")
            st.warning(
                f"This invoice has a **{proba:.1f}% risk** of needing manual review. "
                f"Please verify the PO details before processing."
            )
        else:
            st.success(f"### ✅ Auto Approval — No Issues Found")
            st.info(
                f"This invoice looks clean with only **{proba:.1f}% risk**. "
                f"It can be processed automatically."
            )

        st.metric(label="Risk Score", value=f"{proba:.1f}%")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("Built by Mahesh · DSAI Project · Powered by scikit-learn + Streamlit")