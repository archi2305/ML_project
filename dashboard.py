import streamlit as st
import pandas as pd
import requests
import altair as alt

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Corporate Resource Optimization Dashboard",
    page_icon="💼",
    layout="centered",
)

# -----------------------------
# Title
# -----------------------------
st.title("💼 Corporate Resource Optimization Dashboard")
st.markdown("### Predict employee performance based on workload and other parameters.")

# -----------------------------
# Input Form
# -----------------------------
st.markdown("---")
st.subheader("🧾 Enter Employee Data")

col1, col2 = st.columns(2)

with col1:
    department = st.number_input("Department", min_value=1, max_value=10, step=1)
    hours_worked = st.number_input("Hours Worked", min_value=1, max_value=100, step=1)
    budget_used = st.number_input("Budget Used (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    education_level = st.number_input("Education Level (1-3)", min_value=1, max_value=3, step=1)
    resigned = st.selectbox("Resigned", ["No", "Yes"])

# Convert gender/resigned to numeric
gender_val = 1 if gender == "Male" else 0
resigned_val = 1 if resigned == "Yes" else 0

# Button
if st.button("🚀 Predict Performance"):
    input_data = {
        "Department": department,
        "HoursWorked": hours_worked,
        "BudgetUsed": budget_used,
        "Gender": gender_val,
        "Education_Level": education_level,
        "Resigned": resigned_val
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        result = response.json()

        if result["status"] == "success":
            st.success(
                f"✅ Predicted Performance Score: {result['prediction']} ({result['performance_label']})"
            )
        else:
            st.warning(f"⚠️ API Error: {result['message']}")

    except Exception as e:
        st.error(f"❌ Error connecting to API: {e}")

# -----------------------------
# Example Visualization
# -----------------------------
st.markdown("---")
st.subheader("📊 Example Past Data (Simulated)")

# Example Dataset
data = {
    "Department": [1, 2, 3, 2, 4, 1],
    "HoursWorked": [40, 38, 45, 50, 42, 36],
    "BudgetUsed": [0.6, 0.7, 0.8, 0.9, 0.5, 0.4],
    "PerformanceScore": [3, 4, 5, 4, 2, 3]
}
df = pd.DataFrame(data)

st.dataframe(df)

# Chart
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x="HoursWorked:Q",
        y="PerformanceScore:Q",
        color="Department:N",
        tooltip=["Department", "HoursWorked", "PerformanceScore"]
    )
    .properties(width=600, height=300)
)

st.altair_chart(chart, use_container_width=True)

st.caption("📈 The chart shows how working hours affect performance scores.")
