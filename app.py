import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("placement_model.pkl")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

.probability {
    font-size: 42px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 Student Placement Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based student placement prediction system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎯 Student Details")
st.sidebar.write("Enter the student's details below.")

cgpa = st.sidebar.slider(
    "CGPA", 5.0, 10.0, 7.5, 0.1
)

attendance = st.sidebar.slider(
    "Attendance (%)", 50, 100, 75
)

coding_score = st.sidebar.slider(
    "Coding Score", 0, 100, 60
)

aptitude_score = st.sidebar.slider(
    "Aptitude Score", 0, 100, 60
)

communication_score = st.sidebar.slider(
    "Communication Score", 0, 100, 65
)

internships = st.sidebar.number_input(
    "Internships", 0, 5, 1
)

projects = st.sidebar.number_input(
    "Projects", 0, 10, 2
)

# --------------------------------------------------
# INPUT DATA
# --------------------------------------------------

student_data = pd.DataFrame({
    "CGPA": [cgpa],
    "Attendance": [attendance],
    "Coding_Score": [coding_score],
    "Aptitude_Score": [aptitude_score],
    "Communication_Score": [communication_score],
    "Internships": [internships],
    "Projects": [projects]
})

# --------------------------------------------------
# STUDENT PROFILE
# --------------------------------------------------

st.subheader("📋 Student Profile")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CGPA", cgpa)
col2.metric("Attendance", f"{attendance}%")
col3.metric("Coding Score", coding_score)
col4.metric("Aptitude Score", aptitude_score)

col5, col6, col7 = st.columns(3)

col5.metric("Communication", communication_score)
col6.metric("Internships", internships)
col7.metric("Projects", projects)

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.subheader("🔮 Placement Prediction")

predict_button = st.button(
    "🚀 Predict Placement",
    use_container_width=True
)

if predict_button:

    prediction = model.predict(student_data)[0]

    probability = model.predict_proba(student_data)[0][1]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    if prediction == 1:
        st.success(
            "🎉 Student is likely to be placed based on the model prediction."
        )
    else:
        st.warning(
            "⚠️ Student may need to improve placement readiness."
        )

    st.markdown(
        f"""
        <div class="result-box">
            <div>Placement Probability</div>
            <div class="probability">{probability * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # READINESS
    # --------------------------------------------------

    st.subheader("📊 Placement Readiness")

    if probability >= 0.80:
        readiness = "Excellent 🟢"
    elif probability >= 0.60:
        readiness = "Good 🟢"
    elif probability >= 0.40:
        readiness = "Moderate 🟡"
    else:
        readiness = "Needs Improvement 🔴"

    st.info(f"Placement Readiness Level: **{readiness}**")

    # --------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------

    st.subheader("🧠 Why did the model make this prediction?")

    # Get Logistic Regression model and scaler
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["model"]

    # Standardize student's inputs
    scaled_data = scaler.transform(student_data)

    # Get model coefficients
    coefficients = classifier.coef_[0]

    # Calculate contribution of each feature
    contributions = scaled_data[0] * coefficients

    feature_names = student_data.columns

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    explanation["Impact"] = explanation["Contribution"].apply(
        lambda x: "Positive 📈" if x > 0 else "Negative 📉"
    )

    explanation["Contribution"] = explanation["Contribution"].round(3)

    # Sort by absolute importance
    explanation["Absolute_Impact"] = explanation["Contribution"].abs()

    explanation = explanation.sort_values(
        "Absolute_Impact",
        ascending=False
    )

    explanation_display = explanation[
        ["Feature", "Contribution", "Impact"]
    ]

    st.dataframe(
        explanation_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Positive values indicate factors contributing toward placement, "
        "while negative values indicate factors reducing the predicted placement probability."
    )

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": abs(coefficients)
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------

    st.subheader("💡 Improvement Suggestions")

    suggestions = []

    if cgpa < 7.5:
        suggestions.append("📚 Improve CGPA to strengthen academic performance.")

    if attendance < 75:
        suggestions.append("🕐 Improve attendance to at least 75%.")

    if coding_score < 60:
        suggestions.append("💻 Practice coding and improve problem-solving skills.")

    if aptitude_score < 60:
        suggestions.append("🧮 Practice quantitative and logical aptitude.")

    if communication_score < 65:
        suggestions.append("🗣️ Improve communication and interview skills.")

    if internships == 0:
        suggestions.append("💼 Consider gaining internship experience.")

    if projects < 2:
        suggestions.append("🚀 Build more practical projects.")

    if suggestions:
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.success(
            "🌟 Great profile! Keep improving your technical and communication skills."
        )

    # --------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------

    st.subheader("📌 Prediction Input")

    st.dataframe(
        student_data,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("🤖 Model Information")

info1, info2, info3 = st.columns(3)

info1.metric("Best Model", "Logistic Regression")
info2.metric("Test Accuracy", "92.5%")
info3.metric("Training Samples", "800")

st.caption(
    "Features used: CGPA, attendance, coding, aptitude, communication, "
    "internships and projects."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Placement Prediction System • Machine Learning Project"
)