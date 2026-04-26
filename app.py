import sys
from pathlib import Path
import logging

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure src folder is available for imports
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from data_cleaning import load_and_clean_data
from scheduler import generate_schedule
from model_training import create_risk_labels
from utils import setup_logging, validate_dataframe, validate_csv_file

setup_logging()

st.set_page_config(page_title="Smart Study Planner", layout="wide")

# Basic access control
st.sidebar.title("Access")
app_password = st.sidebar.text_input("Enter dashboard password", type="password")

if app_password != "studyplanner123":
    st.warning("Please enter the dashboard password to access the application.")
    st.stop()

st.title("Smart Study Planner")
st.write(
    "A Python-based decision-support dashboard that helps users organize study tasks, "
    "prioritize deadlines, and identify tasks at risk of being missed."
)

st.info(
    "Security features included in this product: basic password protection, dataset validation, "
    "CSV file-type validation, and privacy-conscious handling of non-sensitive study-task data."
)

try:
    # Optional CSV upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file (optional)", type=["csv"])

    if uploaded_file is not None:
        if validate_csv_file(uploaded_file):
            df = pd.read_csv(uploaded_file)
            logging.info("User uploaded a CSV dataset.")
        else:
            st.error("Invalid file type. Please upload a CSV file.")
            logging.error("Invalid file upload attempted.")
            st.stop()
    else:
        df = pd.read_csv("data/raw_study_tasks.csv")
        logging.info("Default dataset loaded.")

    if not validate_dataframe(df):
        st.error("Dataset is missing one or more required columns.")
        logging.error("Dataset validation failed.")
        st.stop()

    # Save uploaded data temporarily if used, then clean through normal pipeline
    temp_path = "data/temp_uploaded_tasks.csv" if uploaded_file is not None else "data/raw_study_tasks.csv"
    if uploaded_file is not None:
        df.to_csv(temp_path, index=False)

    df = load_and_clean_data(temp_path)
    df = create_risk_labels(df)

    model = joblib.load("models/risk_model.pkl")
    logging.info("Model loaded successfully.")

    feature_columns = [
        "estimated_hours",
        "difficulty",
        "study_hours_available",
        "days_until_due",
        "priority_score"
    ]

    df["predicted_risk"] = model.predict(df[feature_columns])
    logging.info("Risk predictions generated successfully.")

    # Sidebar filters
    st.sidebar.header("Filter Tasks")

    selected_courses = st.sidebar.multiselect(
        "Select Course(s)",
        options=sorted(df["course_name"].unique()),
        default=sorted(df["course_name"].unique())
    )

    selected_priorities = st.sidebar.multiselect(
        "Select Priority Level(s)",
        options=sorted(df["priority"].unique()),
        default=sorted(df["priority"].unique())
    )

    selected_completion = st.sidebar.selectbox(
        "Completion Status",
        options=["All", "Completed", "Not Completed"]
    )

    filtered_df = df[
        (df["course_name"].isin(selected_courses)) &
        (df["priority"].isin(selected_priorities))
    ]

    if selected_completion == "Completed":
        filtered_df = filtered_df[filtered_df["completed"] == 1]
    elif selected_completion == "Not Completed":
        filtered_df = filtered_df[filtered_df["completed"] == 0]

    # KPI metrics
    total_tasks = len(filtered_df)
    completed_tasks = int(filtered_df["completed"].sum())
    high_priority_tasks = len(filtered_df[filtered_df["priority"] == "High"])
    at_risk_tasks = int(filtered_df["predicted_risk"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tasks", total_tasks)
    col2.metric("Completed Tasks", completed_tasks)
    col3.metric("High-Priority Tasks", high_priority_tasks)
    col4.metric("Predicted At-Risk Tasks", at_risk_tasks)

    st.markdown("---")

    # Visualizations
    st.subheader("Task Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        bar_fig = px.bar(
            filtered_df.groupby("course_name", as_index=False)["estimated_hours"].sum(),
            x="course_name",
            y="estimated_hours",
            title="Estimated Study Hours by Course"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with viz_col2:
        pie_fig = px.pie(
            filtered_df,
            names="priority",
            title="Task Distribution by Priority"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    line_fig = px.line(
        filtered_df.sort_values("due_date"),
        x="due_date",
        y="estimated_hours",
        color="course_name",
        title="Estimated Study Hours Over Time"
    )
    st.plotly_chart(line_fig, use_container_width=True)

    st.markdown("---")

    # Recommended schedule
    st.subheader("Recommended Study Schedule")
    schedule_df = generate_schedule(filtered_df)
    st.dataframe(schedule_df, use_container_width=True)

    st.markdown("---")

    # Predicted at-risk tasks
    st.subheader("Predicted At-Risk Tasks")

    risk_view = filtered_df[
        [
            "course_name",
            "task_name",
            "due_date",
            "priority",
            "estimated_hours",
            "study_hours_available",
            "days_until_due",
            "predicted_risk"
        ]
    ].copy()

    risk_view["predicted_risk"] = risk_view["predicted_risk"].map({0: "No", 1: "Yes"})
    st.dataframe(risk_view, use_container_width=True)

    st.markdown("---")

    # Maintenance / monitoring section
    st.subheader("Monitoring and Maintenance")
    st.write(
        "This product supports ongoing maintenance through application logging, model persistence "
        "using a saved .pkl file, repeatable data cleaning workflows, and retraining capability "
        "through the model training script."
    )

except Exception as e:
    logging.exception("Application error occurred.")
    st.error(f"An unexpected error occurred: {e}")