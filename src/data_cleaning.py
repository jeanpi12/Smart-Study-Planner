import pandas as pd


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw dataset and performs cleaning and feature engineering.
    """
    df = pd.read_csv(file_path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing text values
    df["course_name"] = df["course_name"].fillna("Unknown Course")
    df["task_name"] = df["task_name"].fillna("Unknown Task")
    df["priority"] = df["priority"].fillna("Medium")

    # Fill missing numeric values with medians
    df["difficulty"] = df["difficulty"].fillna(df["difficulty"].median())
    df["estimated_hours"] = df["estimated_hours"].fillna(df["estimated_hours"].median())
    df["study_hours_available"] = df["study_hours_available"].fillna(df["study_hours_available"].median())
    df["completed"] = df["completed"].fillna(0)

    # Convert due_date to actual datetime format
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")

    # Remove rows where due_date could not be parsed
    df = df.dropna(subset=["due_date"])

    # Create a days_until_due feature
    today = pd.Timestamp.today().normalize()
    df["days_until_due"] = (df["due_date"] - today).dt.days

    # Prevent negative values from causing issues
    df["days_until_due"] = df["days_until_due"].clip(lower=0)

    # Convert text priority into numeric values
    priority_map = {"Low": 1, "Medium": 2, "High": 3}
    df["priority_score"] = df["priority"].map(priority_map)

    # Normalize difficulty and estimated hours
    df["difficulty_norm"] = df["difficulty"] / df["difficulty"].max()
    df["estimated_hours_norm"] = df["estimated_hours"] / df["estimated_hours"].max()

    return df


if __name__ == "__main__":
    cleaned_df = load_and_clean_data("data/raw_study_tasks.csv")
    cleaned_df.to_csv("data/cleaned_study_tasks.csv", index=False)
    print("Cleaned dataset saved to data/cleaned_study_tasks.csv")