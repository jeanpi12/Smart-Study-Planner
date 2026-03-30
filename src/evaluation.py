from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from model_training import create_risk_labels, train_model
from data_cleaning import load_and_clean_data


def evaluate_model():
    """
    Evaluates the trained machine learning model using standard classification metrics.
    """
    df = load_and_clean_data("data/raw_study_tasks.csv")
    df = create_risk_labels(df)

    model, X_test, y_test, feature_columns = train_model(df)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    evaluate_model()