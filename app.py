import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the model and data
filename = 'models/classification_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("data/Clustered_Data.csv")

# Page title and description
st.title("Marketing Segmentation - Customer Clustering")

st.subheader("Enter Customer Data")

# Input form for customer data
with st.form("form"):
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(
        label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(
        label='One-off Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(
        label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(
        label='Cash Advance', step=0.01, format="%.2f")
    purchases_frequency = st.number_input(
        label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(
        label='One-off Purchases Frequency', step=0.1, format="%.6f")
    purchases_installments_frequency = st.number_input(
        label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(
        label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(
        label='Cash Advance Transactions', step=1)
    purchases_trx = st.number_input(label='Purchases Transactions', step=1)
    credit_limit = st.number_input(
        label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(
        label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(
        label='Percent Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance, purchases_frequency, oneoff_purchases_frequency,
             purchases_installments_frequency, cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    submitted = st.form_submit_button("Get Cluster")

# Display the cluster prediction and visualizations
if submitted:
    clust = loaded_model.predict(data)[0]
    st.success(f'The data belongs to Cluster {clust}')

    st.subheader(f'Distribution of Features for Cluster {clust}')
    cluster_df = df[df['CLUSTER'] == clust]

    # Improved visualization for each feature in the cluster
    for c in cluster_df.drop(['CLUSTER'], axis=1).columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(cluster_df[c], kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Distribution of {c}', fontsize=15)
        ax.set_xlabel(c, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)
