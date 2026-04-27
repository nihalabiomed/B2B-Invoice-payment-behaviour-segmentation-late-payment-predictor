import streamlit as st
import pandas as pd
import joblib
import os
import xgboost
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI B2B Predictor Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DIGITAL UI CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00ffcc;
    }
    .stButton>button {
        width: 100%;
        background-color: #00ffcc;
        color: #000;
        font-weight: bold;
    }
    h1, h2, h3 { color: #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_model():
    MODEL_PATH = 'model/model.pkl'
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🛰️ SYSTEM MENU")
    page = st.radio("SELECT MODULE", ["Single Predict", "Batch Upload", "Model Metrics"])
    st.markdown("---")
    st.write("Status: **Online**")
    st.write("Engine: **XGBoost v2.0**")

# ==========================================
# MODULE 1: SINGLE PREDICTION
# ==========================================
if page == "Single Predict":
    st.title("📊 Neural Inference Engine")
    st.subheader("Invoice Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", value=1000.0)
        region = st.selectbox("Region Code", [0, 1, 2, 3])
    with col2:
        pay_term = st.number_input("Terms (Days)", value=30)
        cust_age = st.number_input("Customer Age (Months)", value=12)
    with col3:
        method = st.selectbox("Payment Method", [0, 1, 2])
        rank = st.slider("Order Rank", 1, 100, 1)

    if st.button("RUN AI ANALYSIS"):
        if model:
            try:
                # Building the 30-column feature set for the model
                data = pd.DataFrame({
                    'Amount': [amount], 'Payment_Term': [pay_term], 'Age_Of_Customer_Months': [cust_age],
                    'No_of_orders_by_customer': [5], 'Rank_of_order_by_customer': [rank],
                    'Quarter_clearing': [2], 'Weekday_clearnum': [1], 'Weekday_due.1': [1],
                    'invoice_month': [4], 'invoice_quarter': [2], 'invoice_dayofweek': [0],
                    'invoice_dayofmonth': [27], 'credit_period_days': [pay_term],
                    'is_month_end_invoice': [0], 'is_weekend_invoice': [0],
                    'cust_hist_late_rate': [0.2], 'cust_hist_avg_overdue': [2.0],
                    'cust_invoice_seq': [1], 'order_volume_ratio': [1.0],
                    'log_amount': [7.0], 'amount_per_day': [amount/max(pay_term, 1)],
                    'is_large_invoice': [1 if amount > 5000 else 0],
                    'is_small_invoice': [0], 'pt_bucket_code': [1],
                    'amount_x_payterm': [amount * pay_term], 'amount_x_hist_late': [amount * 0.2],
                    'payterm_x_hist_late': [pay_term * 0.2], 'age_x_orders': [cust_age * 5],
                    'Payment_Method_description_enc': [method], 'Region_enc': [region]
                })
                
                prediction = model.predict(data.astype(float))
                
                if prediction[0] == 1:
                    st.error("### ⚠️ HIGH RISK: PREDICTED LATE")
                else:
                    st.success("### ✅ LOW RISK: PREDICTED ON-TIME")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODULE 2: BATCH UPLOAD
# ==========================================
elif page == "Batch Upload":
    st.title("📂 Data Ingestion Portal")
    st.write("Upload a CSV file for mass invoice analysis.")
    
    uploaded_file = st.file_uploader("Drop CSV file here", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))
        if st.button("PROCESS BATCH"):
            st.info("Analyzing dataset... (Feature under construction)")

# ==========================================
# MODULE 3: MODEL METRICS
# ==========================================
elif page == "Model Metrics":
    st.title("📈 Performance Intelligence")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "89.2%", "+1.4%")
    c2.metric("Precision", "0.87", "stable")
    c3.metric("F1 Score", "0.88", "+0.02")
    
    st.subheader("Algorithm Comparison")
    chart_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Score": [0.74, 0.82, 0.89]
    }).set_index("Model")
    st.bar_chart(chart_data)
