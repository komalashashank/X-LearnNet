
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="X-LearnNet | AI Learning Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

# --------------------------------------------------
# PROFESSIONAL DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111c30;
}
[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #293a55;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #a8b8d0;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
h1, h2, h3 {
    color: #f1f5f9;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
}
div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
[data-testid="stDataFrame"] {
    border: 1px solid #293a55;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return {
        "risk": pd.read_csv(BASE / "powerbi_data/student_risk_overview.csv"),
        "predictions": pd.read_csv(BASE / "powerbi_data/model_predictions.csv"),
        "performance": pd.read_csv(BASE / "powerbi_data/model_performance.csv"),
        "shap": pd.read_csv(BASE / "powerbi_data/shap_global_importance.csv"),
        "gnn": pd.read_csv(BASE / "powerbi_data/graphsage_feature_importance.csv"),
        "lime": pd.read_csv(BASE / "powerbi_data/lime_student_explanation.csv")
    }

data = load_data()
risk = data["risk"]
predictions = data["predictions"]
performance = data["performance"]
shap = data["shap"]
gnn = data["gnn"]
lime = data["lime"]

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def percent(value):
    return f"{value * 100:.1f}%"

def performance_explanation(student):
    """Generate a readable interpretation from student records."""

    findings = []
    strengths = []
    concerns = []

    gpa = float(student["gpa"])
    prior_gpa = float(student["prior_gpa"])
    attendance = float(student["recorded_attendance_rate"])
    calculated_attendance = float(student["calculated_attendance_rate"])
    assignment = float(student["assignment_submission_rate"])
    video = float(student["video_completion_rate"])
    logins = int(student["lms_logins_past_month"])
    grade = float(student["avg_course_grade"])

    # Academic performance
    if gpa >= 3.5:
        strengths.append("The student's GPA indicates strong academic performance.")
    elif gpa >= 3.0:
        findings.append("The student's GPA indicates a satisfactory academic standing.")
    else:
        concerns.append("The student's GPA is relatively low and may require academic support.")

    if gpa > prior_gpa:
        strengths.append("The GPA has improved compared with the previous GPA.")
    elif gpa < prior_gpa:
        concerns.append("The GPA has declined compared with the previous GPA.")
    else:
        findings.append("The GPA has remained unchanged compared with the previous GPA.")

    if grade >= 80:
        strengths.append("Average course grades indicate good subject performance.")
    elif grade < 60:
        concerns.append("Average course grades indicate possible difficulty with course material.")

    # Learning behaviour
    if attendance >= 0.80:
        strengths.append("Recorded attendance is good.")
    elif attendance < 0.60:
        concerns.append("Low recorded attendance is an area that needs attention.")

    if assignment >= 0.80:
        strengths.append("Assignment submission behaviour is consistent.")
    elif assignment < 0.60:
        concerns.append("Assignment submission is low and may affect academic progress.")

    if video >= 0.70:
        strengths.append("The student demonstrates good video learning engagement.")
    elif video < 0.40:
        concerns.append("Video completion is low, suggesting an opportunity to improve learning engagement.")

    if logins >= 20:
        strengths.append("The student has regular LMS login activity.")
    elif logins < 10:
        concerns.append("LMS login activity is relatively low.")

    # Attendance data quality
    attendance_gap = abs(attendance - calculated_attendance)
    if attendance_gap >= 0.10:
        findings.append(
            "There is a noticeable difference between recorded and calculated "
            "attendance. This data should be verified before making decisions."
        )

    if not strengths:
        strengths.append("No strong positive indicator was identified from the selected measures.")

    if not concerns:
        concerns.append("No major concern was identified using the selected indicator thresholds.")

    findings.extend(strengths)
    findings.extend(concerns)

    return findings, strengths, concerns


def student_recommendations(student, concerns):
    recommendations = []

    if float(student["gpa"]) < 3.0:
        recommendations.append("Arrange academic mentoring or subject-specific tutoring.")

    if float(student["recorded_attendance_rate"]) < 0.75:
        recommendations.append("Review attendance patterns and encourage regular class participation.")

    if float(student["assignment_submission_rate"]) < 0.70:
        recommendations.append("Provide assignment planning support and monitor submission deadlines.")

    if float(student["video_completion_rate"]) < 0.50:
        recommendations.append("Encourage completion of course videos and learning resources.")

    if int(student["lms_logins_past_month"]) < 10:
        recommendations.append("Encourage more consistent use of the LMS.")

    if abs(
        float(student["recorded_attendance_rate"])
        - float(student["calculated_attendance_rate"])
    ) >= 0.10:
        recommendations.append("Verify attendance records before taking intervention decisions.")

    if not recommendations:
        recommendations.append("Continue monitoring performance and maintain current learning habits.")

    return recommendations


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("# 🎓 X-LearnNet")
st.markdown("### Intelligent Student Success Prediction & Explainable Learning Analytics")
st.markdown(
    "An AI-driven framework combining educational big data, "
    "Graph Neural Networks and Explainable AI."
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("# X-LearnNet")
st.sidebar.caption("Learning Analytics Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Student Analysis",
        "Model Performance",
        "Explainable AI"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Framework Components**")
st.sidebar.write("• Educational Big Data")
st.sidebar.write("• Apache Spark Analytics")
st.sidebar.write("• Logistic Regression")
st.sidebar.write("• GraphSAGE")
st.sidebar.write("• SHAP and LIME")

st.sidebar.caption("Academic Capstone Project")

# --------------------------------------------------
# DASHBOARD OVERVIEW
# --------------------------------------------------
if page == "Dashboard Overview":

    st.subheader("Institutional Performance Overview")

    total = len(risk)
    high = (risk["risk_level"].str.lower() == "high").sum()
    medium = (risk["risk_level"].str.lower() == "medium").sum()
    low = (risk["risk_level"].str.lower() == "low").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("High Risk", f"{high:,}")
    c3.metric("Medium Risk", f"{medium:,}")
    c4.metric("Low Risk", f"{low:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Student Risk Distribution")
        st.bar_chart(risk["risk_level"].value_counts())

    with right:
        st.markdown("#### Academic Indicators")
        indicators = pd.Series({
            "GPA (%)": risk["gpa"].mean() / 4 * 100,
            "Attendance (%)": risk["recorded_attendance_rate"].mean() * 100,
            "Assignments (%)": risk["assignment_submission_rate"].mean() * 100,
            "Video Completion (%)": risk["video_completion_rate"].mean() * 100
        })
        st.bar_chart(indicators)

    st.divider()
    st.markdown("#### Student Records")

    selected = st.selectbox(
        "Filter student risk",
        ["All", "High", "Medium", "Low"]
    )

    filtered = risk if selected == "All" else risk[
        risk["risk_level"].str.lower() == selected.lower()
    ]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

# --------------------------------------------------
# STUDENT ANALYSIS
# --------------------------------------------------
elif page == "Student Analysis":

    st.subheader("Individual Student Analysis")

    ids = sorted(risk["student_id"].dropna().unique())
    default_id = "STU012586" if "STU012586" in ids else ids[0]

    student_id = st.selectbox(
        "Search / Select Student ID",
        ids,
        index=ids.index(default_id)
    )

    student = risk[risk["student_id"] == student_id].iloc[0]

    st.markdown(f"### Student Profile: `{student_id}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GPA", f"{student['gpa']:.2f} / 4.0")
    c2.metric("Risk Level", student["risk_level"])
    c3.metric("Attendance", percent(student["recorded_attendance_rate"]))
    c4.metric("Assignment Rate", percent(student["assignment_submission_rate"]))

    st.divider()

    st.markdown("#### Student Academic Information")

    details = {
        "Major": student["major"],
        "Program": student["program_level"],
        "Semester": student["semester"],
        "Enrollment": student["enrollment_status"],
        "Previous GPA": student["prior_gpa"],
        "Average Course Grade": student["avg_course_grade"],
        "LMS Logins": student["lms_logins_past_month"],
        "Video Completion": percent(student["video_completion_rate"]),
        "Forum Participation": student["forum_participation_count"],
        "Learning Hours": student["avg_learning_hours"]
    }

    st.dataframe(
        pd.DataFrame(details.items(), columns=["Parameter", "Value"]),
        use_container_width=True,
        hide_index=True
    )

    p = predictions[predictions["student_id"] == student_id]

    if not p.empty:
        p = p.iloc[0]
        st.markdown("#### AI Model Predictions")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Logistic Regression**")
            st.metric("Failure Probability", percent(p["lr_failure_probability"]))
            st.write("Prediction:", "Failure" if p["lr_prediction"] == 1 else "No Failure")
            st.write("Test result:", p["lr_result"])

        with c2:
            st.markdown("**GraphSAGE**")
            st.metric("Failure Probability", percent(p["graphsage_failure_probability"]))
            st.write("Prediction:", "Failure" if p["graphsage_prediction"] == 1 else "No Failure")
            st.write("Test result:", p["graphsage_result"])

    st.caption("Model probabilities are estimates, not certainty or confirmed future outcomes.")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":

    st.subheader("Machine Learning Model Evaluation")

    formatted = performance.copy()
    for col in ["accuracy", "precision", "recall", "f1_score"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.2%}")

    for col in ["roc_auc", "pr_auc"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")

    st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.divider()

    metric = st.selectbox(
        "Choose evaluation metric",
        ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    )

    st.bar_chart(performance.set_index("model")[[metric]])

    st.markdown("#### Performance Interpretation")
    st.write(
        "Logistic Regression has higher accuracy, precision, ROC-AUC and PR-AUC "
        "in the reported test results. GraphSAGE has higher recall and a slightly "
        "higher F1-score. GraphSAGE therefore identifies more actual failure cases, "
        "while both models have trade-offs."
    )

# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------
elif page == "Explainable AI":

    st.subheader("Explainable AI – Student Performance Interpretation")
    st.write(
        "Select a student to understand academic performance, learning behaviour, "
        "model predictions and possible support actions."
    )

    ids = sorted(risk["student_id"].dropna().unique())
    default_id = "STU012586" if "STU012586" in ids else ids[0]

    student_id = st.selectbox(
        "Select Student for Explanation",
        ids,
        index=ids.index(default_id),
        key="explain_student"
    )

    student = risk[risk["student_id"] == student_id].iloc[0]

    # Student-level narrative
    findings, strengths, concerns = performance_explanation(student)
    recommendations = student_recommendations(student, concerns)

    st.divider()
    st.markdown(f"### 📋 Performance Explanation: {student_id}")

    a, b, c = st.columns(3)
    a.metric("GPA", f"{student['gpa']:.2f}/4.0")
    b.metric("Risk Category", student["risk_level"])
    c.metric("Average Course Grade", f"{student['avg_course_grade']:.1f}")

    st.markdown("#### Overall Interpretation")

    if concerns:
        st.warning(
            f"This student's records show {len(concerns)} area(s) "
            "that may need attention."
        )
    else:
        st.success("No major concern was identified from the selected indicators.")

    st.markdown("#### Detailed Performance Findings")

    for item in findings:
        st.markdown(f"- {item}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### ✅ Strengths")
        for item in strengths:
            st.write("•", item)

    with right:
        st.markdown("#### ⚠️ Areas for Improvement")
        for item in concerns:
            st.write("•", item)

    st.divider()

    st.markdown("#### Recommended Student Support")

    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"{i}. {recommendation}")

    st.caption(
        "These recommendations are generated from transparent indicator thresholds. "
        "They are suggestions for human review, not automated decisions."
    )

    # Individual AI predictions
    st.divider()
    st.markdown("### AI Prediction Explanation")

    p = predictions[predictions["student_id"] == student_id]

    if not p.empty:
        p = p.iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Logistic Regression**")
            st.metric("Failure Probability", percent(p["lr_failure_probability"]))
            st.write("Predicted outcome:", "Failure" if p["lr_prediction"] == 1 else "No Failure")
            st.write("Evaluation result:", p["lr_result"])

        with c2:
            st.markdown("**GraphSAGE**")
            st.metric("Failure Probability", percent(p["graphsage_failure_probability"]))
            st.write("Predicted outcome:", "Failure" if p["graphsage_prediction"] == 1 else "No Failure")
            st.write("Evaluation result:", p["graphsage_result"])

        st.info(
            "The two models may produce different predictions because they use "
            "different learning approaches. Their outputs should be reviewed "
            "alongside student records."
        )
    else:
        st.info("No model prediction is available for this student.")

    # Local report when it exists
    report = (
        BASE / "outputs/unified_explainability"
        / f"{student_id}_unified_explainability_report.md"
    )

    if report.exists():
        st.divider()
        st.markdown("### Saved Individual XAI Report")
        with st.expander("Open detailed student explanation"):
            st.markdown(report.read_text(encoding="utf-8"))

    # Explainability methods
    st.divider()
    st.markdown("### Explainability Methods")

    tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "GraphSAGE"])

    with tab1:
        st.markdown("#### Global SHAP Feature Importance")
        st.caption(
            "This chart shows global Logistic Regression feature importance, "
            "not the selected student's individual SHAP contribution."
        )
        st.bar_chart(
            shap.sort_values("mean_abs_shap", ascending=False)
                .head(12)
                .set_index("feature")[["mean_abs_shap"]]
        )

        shap_image = BASE / "outputs/xlearnnet_lr_shap_summary.png"
        if shap_image.exists():
            st.image(str(shap_image), caption="Global SHAP Summary")

        if student_id == "STU012586":
            local_shap = BASE / "outputs/xlearnnet_lr_shap_STU012586.csv"
            if local_shap.exists():
                st.markdown("#### Individual SHAP Results")
                st.dataframe(
                    pd.read_csv(local_shap),
                    use_container_width=True,
                    hide_index=True
                )

    with tab2:
        st.markdown("#### LIME Local Explanation")
        if student_id == "STU012586":
            local_lime = BASE / "outputs/xlearnnet_lr_lime_STU012586.csv"
            if local_lime.exists():
                st.dataframe(
                    pd.read_csv(local_lime),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(lime, use_container_width=True, hide_index=True)

            lime_image = BASE / "outputs/xlearnnet_lr_lime_student.png"
            if lime_image.exists():
                st.image(str(lime_image), caption="LIME Student Explanation")
        else:
            st.info(
                "The saved individual LIME explanation currently belongs to "
                "STU012586. A local LIME explanation for this selected student "
                "has not been saved yet."
            )

    with tab3:
        st.markdown("#### GraphSAGE Global Feature Importance")
        st.caption(
            "GraphSAGE feature-mask scores show importance in the saved graph "
            "explanation. They are not failure probabilities or signed effects."
        )

        st.bar_chart(
            gnn.sort_values("importance", ascending=False)
               .head(12)
               .set_index("feature")[["importance"]]
        )

        gnn_image = BASE / "outputs/graphsage/graphsage_feature_explanation.png"
        if gnn_image.exists():
            st.image(str(gnn_image), caption="GraphSAGE Feature Explanation")

        if student_id == "STU012586":
            report_file = (
                BASE / "outputs/unified_explainability"
                / f"{student_id}_unified_explainability_report.md"
            )
            if report_file.exists():
                st.info(
                    "The saved unified report above includes the local "
                    "GraphSAGE explanation for this student."
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "X-LearnNet | Explainable AI-Driven Learning Analytics | "
    "Academic Capstone Project"
)
