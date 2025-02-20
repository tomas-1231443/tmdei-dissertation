import pandas as pd
import pytest
from src.preprocessing.preprocess import preprocess_single_alert, preprocess_bulk_alerts
from src.config import EXCEL_COLUMNS

def test_preprocess_single_alert():
    alert = {
        "Issue ID": "123",
        "Issue Type": "ALERT",
        "Status": "Resolved",
        "Description": "Error: Something went wrong!",
        "Custom field (Alert Technology)": "Tech-Name",
        "Custom field (Incident Description)": "Some Description.",
        "Custom field (Incident Resolution 1)": "Fixed!",
        "Custom field (Request Type)": "Update",
        "Custom field (Source)": "System",
        "Custom field (Source Alert Rule Name)": "Rule-1",
        "Custom field (Source_Alert_Id)": "A-456",
        "Custom field (Taxonomy)": "TypeA",
        "Priority": "High"
    }
    processed = preprocess_single_alert(alert.copy())
    # Expected: text should be lowercased and punctuation removed.
    assert processed["Description"] == "error something went wrong", f"Got: {processed['Description']}"

def test_preprocess_bulk_alerts_success():
    # Create a dummy DataFrame with all expected columns.
    data = {col: ["Test data!!"] for col in EXCEL_COLUMNS}
    df = pd.DataFrame(data)
    processed_df = preprocess_bulk_alerts(df)
    for col in processed_df.columns:
        # The string "Test data!!" should become "test data"
        assert processed_df[col].iloc[0] == "test data", f"Column {col} not processed correctly"

def test_preprocess_bulk_alerts_missing_column():
    # Create a DataFrame missing one expected column.
    data = {col: ["Test data!!"] for col in EXCEL_COLUMNS[:-1]}  # leaving out the last column
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        preprocess_bulk_alerts(df)
