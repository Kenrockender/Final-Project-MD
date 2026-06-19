import os
import json

import boto3
import pandas as pd
import streamlit as st

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'credit-score-endpoint')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

st.set_page_config(page_title="Credit Score Classification", layout="wide")

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer"
]
CREDIT_MIX = ["Good", "Standard", "Bad"]
MIN_AMOUNT = ["No", "Yes", "NM"]
PAYMENT_BEHAVIOUR = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
]
RISK_LEVEL = {"Good": "Low", "Standard": "Moderate", "Poor": "High"}


@st.cache_resource
def get_runtime():
    return boto3.client('sagemaker-runtime', region_name=AWS_REGION)


def predict(record):
    response = get_runtime().invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(record)
    )
    result = json.loads(response['Body'].read().decode('utf-8'))
    return result[0] if isinstance(result, list) else result


st.title("Credit Score Classification")
st.caption(f"SageMaker endpoint: {ENDPOINT_NAME} ({AWS_REGION})")

with st.sidebar:
    st.header("Customer Profile")

    st.subheader("Demographics & Income")
    age = st.number_input("Age", 0, 100, 33)
    occupation = st.selectbox("Occupation", OCCUPATIONS, index=OCCUPATIONS.index("Engineer"))
    annual_income = st.number_input("Annual Income", 0.0, 300000.0, 37000.0)
    monthly_salary = st.number_input("Monthly Inhand Salary", 0.0, 50000.0, 3100.0)

    st.subheader("Accounts & Loans")
    num_bank = st.number_input("Num Bank Accounts", 0, 20, 5)
    num_card = st.number_input("Num Credit Cards", 0, 20, 5)
    interest = st.number_input("Interest Rate (%)", 0, 50, 13)
    num_loan = st.number_input("Num of Loans", 0, 20, 3)
    total_emi = st.number_input("Total EMI per month", 0.0, 100000.0, 70.0)

    st.subheader("Payment Behaviour")
    delay = st.number_input("Delay from due date (days)", 0, 100, 18)
    num_delayed = st.number_input("Num of Delayed Payments", 0, 100, 14)
    changed_limit = st.number_input("Changed Credit Limit", -50.0, 50.0, 9.4)
    num_inquiry = st.number_input("Num Credit Inquiries", 0, 50, 6)
    min_amount = st.selectbox("Payment of Min Amount", MIN_AMOUNT, index=0)
    pay_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR, index=4)

    st.subheader("Credit Profile")
    credit_mix = st.selectbox("Credit Mix", CREDIT_MIX, index=1)
    outstanding = st.number_input("Outstanding Debt", 0.0, 10000.0, 1170.0)
    utilization = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 32.3)
    history_age = st.text_input("Credit History Age", "18 Years and 5 Months")
    invested = st.number_input("Amount Invested Monthly", 0.0, 10000.0, 136.0)
    balance = st.number_input("Monthly Balance", 0.0, 100000.0, 336.0)

input_data = {
    'Age': age, 'Occupation': occupation, 'Annual_Income': annual_income,
    'Monthly_Inhand_Salary': monthly_salary, 'Num_Bank_Accounts': num_bank,
    'Num_Credit_Card': num_card, 'Interest_Rate': interest, 'Num_of_Loan': num_loan,
    'Delay_from_due_date': delay, 'Num_of_Delayed_Payment': num_delayed,
    'Changed_Credit_Limit': changed_limit, 'Num_Credit_Inquiries': num_inquiry,
    'Credit_Mix': credit_mix, 'Outstanding_Debt': outstanding,
    'Credit_Utilization_Ratio': utilization, 'Credit_History_Age': history_age,
    'Payment_of_Min_Amount': min_amount, 'Total_EMI_per_month': total_emi,
    'Amount_invested_monthly': invested, 'Payment_Behaviour': pay_behaviour,
    'Monthly_Balance': balance
}

try:
    result = predict(input_data)
    score = result['credit_score']
    proba = result['probabilities']

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Credit Score", score)
    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    col3.metric("Risk Level", RISK_LEVEL.get(score, "-"))

    st.divider()

    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({'Probability': proba}).reindex(["Poor", "Standard", "Good"])
    st.bar_chart(prob_df)
except Exception as exc:
    st.error(f"Gagal memanggil endpoint '{ENDPOINT_NAME}': {exc}")

st.divider()
st.subheader("Input Summary")
st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
