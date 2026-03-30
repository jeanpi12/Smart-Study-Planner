import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_cleaning import load_and_clean_data


def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary target column called 'at_risk'.
    A task is at risk if it is due soon and not completed,
    or if the estimated workload is greater than the likely available time.
    """
    df = df.copy()

    df["at_risk"] = (
        ((df["days_until_due"] <= 2) & (df["completed"] == 0)) |
        (df["estimated_hours"] > (df["study_hours_available"] * (df["days_until_due"] + 1)))
    ).astype(int)

    return df


def train_model(df: pd.DataFrame):
    """
    Trains a Random Forest model to predict whether a task is at risk.
    """
    feature_columns = [
        "estimated_hours",
        "difficulty",
        "study_hours_available",
        "days_until_due",
        "priority_score"
    ]

    X = df[feature_columns]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, feature_columns


if __name__ == "__main__":
    df = load_and_clean_data("data/raw_study_tasks.csv")
    df = create_risk_labels(df)

    model, X_test, y_test, feature_columns = train_model(df)

    joblib.dump(model, "models/risk_model.pkl")
    print("Model trained and saved to models/risk_model.pkl")
    print("This script can be rerun whenever new study-task data becomes available.")
    print("Feature columns used:", feature_columns)