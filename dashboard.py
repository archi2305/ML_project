# ---------------------------------------------
# Corporate Resource Optimization Dashboard
# ---------------------------------------------
import streamlit as st
import requests
import pandas as pd
import altair as alt

# Streamlit page config
st.set_page_config(page_title="Corporate Resource Optimization", layout="wide")

st.title("🏢 Corporate Resource Optimization Dashboard")
st.markdown("Predict employee performance based on workload and other parameters.")

# ---------------- USER INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", [1, 2, 3, 4, 5])
    hours = st.number_input("Work Hours Per Week", min_value=20, max_value=80, value=40)
    budget = st.number_input("Monthly Salary", min_value=10000, max_value=150000, value=50000)
    education = st.number_input("Education Level (1-3)", min_value=1, max_value=3, value=2)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    resigned = st.selectbox("Resigned", ["No", "Yes"])
    team_size = st.number_input("Team Size", min_value=1, max_value=20, value=5)
    overtime = st.number_input("Overtime Hours", min_value=0, max_value=50, value=5)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Performance"):
    # Prepare JSON
    input_data = {
        "Department": department,
        "Gender": gender,
        "Work_Hours_Per_Week": hours,
        "Monthly_Salary": budget,
        "Education_Level": education,
        "Resigned": resigned,
        "Team_Size": team_size,
        "Overtime_Hours": overtime
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        result = response.json()

        if "prediction" in result:
            st.success(f"✅ Predicted Performance Score: {result['prediction']}")
        else:
            st.error(f"⚠️ API Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"❌ Connection error: {e}")

# ---------------- OPTIONAL: VISUALIZATION ----------------
st.markdown("---")
st.subheader("📈 Example Past Data (Simulated)")
example_data = pd.DataFrame({
    "Department": [1, 2, 3, 2, 1],
    "Work_Hours_Per_Week": [40, 38, 42, 36, 45],
    "Monthly_Salary": [60000, 75000, 55000, 68000, 72000],
    "Performance_Score": [3, 4, 5, 3, 4]
})
st.dataframe(example_data)

try:
    chart = (
        alt.Chart(example_data)
        .mark_bar()
        .encode(x="Work_Hours_Per_Week", y="Performance_Score", color="Department:N")
        .properties(width=600, height=300)
    )
    st.altair_chart(chart)
except Exception as e:
    st.warning(f"Visualization Error: {e}")
