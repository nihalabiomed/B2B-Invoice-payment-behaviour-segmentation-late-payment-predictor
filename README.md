https://b2b-invoice-payment-behaviour-segmentation-late-payment-predic.streamlit.app

Dataset: https://www.kaggle.com/datasets/sonalisingh1411/accounts-receivable-and-payment-delay-analysis?select=Dataset.csv   
## Business problem
#### Every company that sells to other businesses (B2B) struggles with 2 questions:
 * Who are the customers that chronically pay late? → Clustering
 * Which specific invoice will be paid late before it's even due? → Prediction
   
This project answers both.
 The final product is a Streamlit app where a finance team uploads invoices and gets live risk scores.
________________________________________
 #### 30 Features  Built and Why Each Exists
### Date-derived (7 features)
•	invoice_month, invoice_quarter, invoice_dayofweek, invoice_dayofmonth —  late payments spike in certain months/quarters

•	credit_period_days — longer credit = more risk

•	is_month_end_invoice — month-end invoices often get delayed

•	is_weekend_invoice — weekend invoices get processed later

### Customer history lag features (4 features) — most important
•	cust_hist_late_rate — this customer's past late payment % (shifted to prevent data leakage)

•	cust_hist_avg_overdue — how many days late on average historically

•	cust_invoice_seq — which invoice number this is for that customer

•	order_volume_ratio — rank of order / total orders (measures order regularity)

### Amount features (4 features)
•	log_amount — raw amount is skewed; log makes it normally distributed

•	amount_per_day — amount ÷ credit period = daily financial pressure

•	is_large_invoice — top 25% by value (large invoices paid differently)

•	is_small_invoice — bottom 25% by value

### Payment term bucket (1 feature)
•	pt_bucket_code — groups 0–30, 31–60, 61–90, 91–120, 120+ days

### Interaction features (4 features)
•	amount_x_payterm — high amount AND long term together

•	amount_x_hist_late — large invoice from a historically late customer

•	payterm_x_hist_late — long credit term given to a bad payer

•	age_x_orders — older customer with more orders (loyalty proxy)

### Encoded categoricals (2 features)
•	Payment_Method_description_enc, Region_enc
________________________________________
### deliverables for Member 2 are two files that get auto-saved when da.py finishes running:
* artifacts/invoices_clean.csv — the fully cleaned dataset with all 30 engineered features already added as columns. Member 2 reads this directly. They do not need to re-clean or re-engineer anything.
* artifacts/feature_names.pkl — a Python list of the exact 30 feature column names in the correct order. Member 2 loads this with joblib.load() so they use exactly the same features that have been built, in the same order.
________________________________________

