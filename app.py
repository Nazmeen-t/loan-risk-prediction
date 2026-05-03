
import streamlit as st
import pandas as pd 
import plotly.express as px

USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login Page")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# STOP app if not logged in
if not st.session_state.logged_in:
    login()
    st.stop()

# Load data
df = pd.read_csv("loan_data.csv")
def risk_group(x):
    if x == 0:
        return "No Delinquency"
    elif x <= 2:
        return "Low (1-2)"
    elif x <= 5:
        return "Medium (3-5)"
    else:
        return "High (5+)"

df['Delinquency_Group'] = df['num_times_delinquent'].apply(risk_group)

st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>Loan Risk & Approval Dashboard</h1>",
    unsafe_allow_html=True
)

#st.subheader("Key Metrices")

df['Approved_Flag_Num'] = df['Approved_Flag'].apply(lambda x: 1 if x in ['P1', 'P2'] else 0)

 # Sidebar Filters
st.sidebar.header("Filters")

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['GENDER'].unique(),
    default=df['GENDER'].unique()
)

selected_risk = st.sidebar.multiselect(
    "Select Risk Level",
    options=df['Delinquency_Group'].unique(),
    default=df['Delinquency_Group'].unique()
)

# Apply filters
df_filtered = df[
    (df['GENDER'].isin(selected_gender)) &
    (df['Delinquency_Group'].isin(selected_risk))
]

#st.markdown("## Key Metrices")

st.markdown("### <span style='color:#1ABC9C'>Key Metrics</span>", unsafe_allow_html=True)
# KPIs
total = df_filtered.shape[0]
approved = df_filtered[df_filtered['Approved_Flag_Num'] == 1].shape[0]
rejected = total - approved

approval_rate = approved / total
rejection_rate = rejected / total

avg_credit = df_filtered['Credit_Score'].mean()
avg_income = df_filtered['NETMONTHLYINCOME'].mean()

high_risk = df_filtered[df_filtered['Delinquency_Group'] == 'High (5+)'].shape[0]
high_risk_pct = high_risk / total

col1, col2, col3, col4 = st.columns(4)
col5, col6 = st.columns(2)

col1.metric("Total Customers", f"{total:,}")
col2.metric("Approval Rate", f"{approval_rate:.2%}")
col3.metric("Rejection Rate", f"{rejection_rate:.2%}")
col4.metric("Avg Credit Score", f"{avg_credit:.0f}")

col5.metric("Avg Income", f"{avg_income:,.0f}")
col6.metric("High Risk %", f"{high_risk_pct:.2%}")

# Charts row 1
col1, col2 = st.columns([1.2,1])

# LEFT SIDE
with col1:
    st.subheader("Customer Distribution by Risk")

    risk_order = ["No Delinquency", "Low (1-2)", "Medium (3-5)", "High (5+)"]

    risk_counts = (
        df_filtered['Delinquency_Group']
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ['Risk Level', 'Count']

    fig1 = px.bar(
    risk_counts,
    x='Risk Level',
    y='Count',
    color='Risk Level',
    color_discrete_map={
        "No Delinquency": "#27AE60",   # Green
        "Low (1-2)": "#3498DB",        # Blue
        "Medium (3-5)": "#F39C12",     # Orange
        "High (5+)": "#E74C3C"         # Red
    },
    category_orders={"Risk Level": risk_order}
)

    fig1.update_traces(width=0.6)

    st.plotly_chart(fig1, use_container_width=True)


# RIGHT SIDE 
with col2:
    st.subheader("Approval Status Distribution")

    approval_order = ['P1', 'P2', 'P3', 'P4']

    approval_counts = (
        df_filtered['Approved_Flag']
        .value_counts()
        .reindex(approval_order, fill_value=0)
        .reset_index()
    )

    approval_counts.columns = ['Status', 'Count']

    fig2 = px.bar(
    approval_counts,
    x='Status',
    y='Count',
    color='Status',
    color_discrete_map={
        "P1": "#2ECC71",
        "P2": "#3498DB",
        "P3": "#F39C12",
        "P4": "#E74C3C"
    },
    category_orders={"Status": approval_order}
)

    fig2.update_traces(width=0.6)

    st.plotly_chart(fig2, use_container_width=True)

#charts row2

st.subheader("Loan Approval by Gender")
pivot = pd.crosstab(df_filtered['GENDER'], df_filtered['Approved_Flag'])
fig3 = px.bar(
    pivot,
    barmode='stack',
    color_discrete_sequence=["#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]
)

st.plotly_chart(fig3, use_container_width=True)
st.markdown("### <span style='color:#8E44AD'>Data Preview</span>", unsafe_allow_html=True)
st.dataframe(df_filtered.head(10))

st.markdown("""
### Key Insights:
- Majority customers have no delinquency
- Approval rate is high (~74%)
- Low risk customers dominate approvals
""")

st.markdown("---")
st.caption("Dashboard created using Streamlit | Data Analysis Project")

# =========================
# Loan Prediction Section
# =========================

import joblib

# Load trained model
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names=joblib.load("features.pkl")

st.markdown("---")
st.subheader(" Loan Approval Prediction")

# User inputs
credit_score = st.number_input("Enter Credit Score", min_value=300)
income = st.number_input("Enter Monthly Income", min_value=0)
delinq = st.number_input("Enter Number of Delinquencies", min_value=0)
recent_delinq = st.number_input("Enter Recent Level of Delinquency", min_value=0)
enq_l3m = st.number_input("Enter Enquiries in Last 3 Months", min_value=0)
pct_pl_enq = st.number_input("Enter % PL Enquiries (Last 6M of Ever)", min_value=0.0)

# Predict button
if st.button("Predict Loan Approval"):

    # Step 1: Create input 
    input_df = pd.DataFrame(
    [[
        credit_score,
        income,
        delinq,
        recent_delinq,
        enq_l3m,
        pct_pl_enq
    ]],
    columns=feature_names
)

    # Step 2: Apply scaling
    input_scaled = scaler.transform(input_df)

    # Step 3: Predict
    prediction = model.predict(input_scaled)

    # Step 4: Show result
    if prediction[0] == 1:
        st.success("Loan Likely to be Approved")
    else:
        st.error("Loan Likely to be Rejected")
