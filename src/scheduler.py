import pandas as pd


def generate_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an optimized study schedule using a priority-based scoring algorithm.
    """

    df = df.copy()

    # Create urgency score so that tasks due sooner get a higher value
    df["urgency_score"] = 1 / (df["days_until_due"] + 1)

    # Calculate final weighted score
    df["final_score"] = (
        df["priority_score"] * 0.4
        + df["urgency_score"] * 0.3
        + df["difficulty_norm"] * 0.2
        + df["estimated_hours_norm"] * 0.1
    )

    # Sort tasks from the highest score to the lowest score
    df = df.sort_values(by="final_score", ascending=False)

    # Create recommended study order
    df["recommended_order"] = range(1, len(df) + 1)

    return df[
        [
            "recommended_order",
            "course_name",
            "task_name",
            "due_date",
            "priority",
            "estimated_hours",
            "study_hours_available",
            "days_until_due",
            "final_score",
        ]
    ]