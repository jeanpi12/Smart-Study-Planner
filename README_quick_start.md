# Smart Study Planner Quick-Start Guide

## Overview

The Smart Study Planner is a Python-based Streamlit application that helps users organize study tasks, generate optimized study schedules, and identify tasks at risk of being missed. The application uses a priority-based scheduling algorithm and a machine-learning model to support decision-making.

---

## Requirements

* Python 3.x
* PyCharm or another Python IDE
* Required Python libraries listed in `requirements.txt`

---

## Installation Steps

1. Open the project in PyCharm.
2. Create or activate the project virtual environment.
3. Install the required libraries by running:

   ```bash
   python -m pip install -r requirements.txt
   ```

---

## Running the Application

1. Ensure the dataset file exists at:

   ```
   data/raw_study_tasks.csv
   ```
2. Ensure the trained model file exists at:

   ```
   models/risk_model.pkl
   ```
3. Run the Streamlit application:

   ```bash
   python -m streamlit run app.py
   ```

---

## Dashboard Access

When prompted, enter the dashboard password:

```
studyplanner123
```

---

## Main Features

* Cleans and preprocesses study-task data
* Generates an optimized study schedule using a priority-based algorithm
* Predicts which tasks are at risk using a machine-learning model
* Displays interactive visualizations (bar chart, pie chart, line chart)
* Provides filtering by course, priority, and completion status
* Supports optional CSV upload with validation
* Includes basic security features such as password protection and input validation

---

## Supporting Scripts

* `src/data_cleaning.py` → Cleans and prepares the dataset
* `src/scheduler.py` → Generates the optimized study schedule
* `src/model_training.py` → Trains the machine-learning model
* `src/evaluation.py` → Evaluates model performance
* `src/utils.py` → Provides validation and logging functionality

---

## Notes

* Application logs are stored in:

  ```
  outputs/app.log
  ```
* Screenshots for documentation are stored in:

  ```
  outputs/screenshots
  ```
* The model can be retrained by running:

  ```bash
  python src/model_training.py
  ```

---

## Summary

The Smart Study Planner provides a complete data-driven solution for improving study organization. By combining data preprocessing, a priority-based scheduling algorithm, and predictive analytics, the application helps users make more informed decisions about how to allocate their study time effectively.
