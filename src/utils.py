import logging
from pathlib import Path
import pandas as pd


def setup_logging(log_file: str = "outputs/app.log") -> None:
    """
    Sets up basic logging for monitoring application activity and errors.
    """
    Path("outputs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validates that the required columns exist in the dataset.
    """
    required_columns = {
        "task_id",
        "course_name",
        "task_name",
        "due_date",
        "estimated_hours",
        "priority",
        "difficulty",
        "study_hours_available",
        "completed"
    }

    return required_columns.issubset(df.columns)


def validate_csv_file(uploaded_file) -> bool:
    """
    Validates that the uploaded file is a CSV file.
    """
    if uploaded_file is None:
        return False

    return uploaded_file.name.lower().endswith(".csv")